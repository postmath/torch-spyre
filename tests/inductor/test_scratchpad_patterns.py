from dataclasses import dataclass
from unittest import TestCase, expectedFailure

# This appears to currently be necessary to make importing a torch_spyre module work.
import torch  # noqa: F401

from torch_spyre._inductor.scratchpad import scratchpad_planning, ScratchPadAllocator
from torch_spyre._inductor import config

# From scratchpad.py
AVAILABLE_LX_SIZE = int((2 << 20) * (1.0 - config.dxp_lx_frac_avail))


@dataclass
class ReadWrites:
    reads: list["Buffer"]
    writes: list["Buffer"]


@dataclass
class Operation:
    name: str
    inputs: list[str]
    outputs: list[str]

    # To make scratchpad.py work, we add an origin_node field that points to the op itself.
    origin_node = None

    def __post_init__(self):
        self.origin_node = self

    def get_read_writes(self) -> ReadWrites:
        # Returns a list of (buffer_name, "read" or "write") for all buffers used by this operation.
        reads = [all_buffers[buffer_name] for buffer_name in self.inputs]
        writes = [all_buffers[buffer_name] for buffer_name in self.outputs]
        return ReadWrites(reads=reads, writes=writes)


@dataclass
class Buffer:
    name: str
    size: int


all_buffers: dict[str, Buffer] = {}


class InstrumentedAllocator(ScratchPadAllocator):
    def __init__(self):
        super().__init__()
        self.allocations = {}

    def should_consider_op(self, op: Operation) -> bool:
        return True

    def allocate(self, tensor_name: str, addr: int):
        pass

    def mem_usage_by_op(self, op: Operation) -> dict[str, dict[str, bool | int]]:
        # Returns a dict mapping each buffer name to a dict with keys "is_input", "is_output", and "size".
        # is_input is True if the buffer is an input to the op, and False otherwise. is_output is True
        # if the buffer is an output of the op, and False otherwise. size is the size of the buffer.
        result = {}
        for tensor_name in op.inputs:
            result[tensor_name] = {
                "is_input": True,
                "size": all_buffers[tensor_name].size,
            }
        for tensor_name in op.outputs:
            result[tensor_name] = {
                "is_input": False,
                "size": all_buffers[tensor_name].size,
            }
        return result


# A type alias for the result of an allocation. The ith entry in the list is the state during
# the ith operation. It maps each allocated buffer to the scratch pad address where it is
# allocated at that point in time.
AllocationResult = list[dict[str, int]]


@dataclass
class AllocationTestCase:
    buffers: dict[str, Buffer]
    operations: list[Operation]
    # A "good" allocation pattern that we want to compare to. The test verifies that this pattern
    # is valid and that the current result is at least as good -- that is, the HBM usage of the
    # current result is no more than that of the good pattern.
    good_allocation: AllocationResult


class TestExamplePattern(TestCase):
    def assertIsValidAllocation(
        self, allocation: AllocationResult, operations: list[Operation]
    ):
        for i, op in enumerate(operations):
            # Check that each buffer that is used is allocated.
            for buffer_name in op.inputs + op.outputs:
                self.assertIn(buffer_name, allocation[i])

            # Check that buffers do not overlap.
            sorted_allocations = sorted(
                allocation[i].items(), key=lambda x: x[1]
            )  # sort by address
            for j in range(len(sorted_allocations) - 1):
                buffer_name_j, addr_j = sorted_allocations[j]
                buffer_name_next, addr_next = sorted_allocations[j + 1]
                size_j = all_buffers[buffer_name_j].size
                self.assertLessEqual(
                    addr_j + size_j,
                    addr_next,
                    f"Buffers {buffer_name_j} and {buffer_name_next} overlap during operation {op.name}",
                )
            self.assertLessEqual(
                sorted_allocations[-1][1] + all_buffers[sorted_allocations[-1][0]].size,
                AVAILABLE_LX_SIZE,
                f"Buffer {sorted_allocations[-1][0]} exceeds scratch pad size during operation {op.name}",
            )

    def hbm_usage_test(
        self, allocation: AllocationResult, operations: list[Operation]
    ) -> int:
        hbm_usage = 0
        for i, op in enumerate(operations):
            for buffer_name in op.inputs:
                if i == 0 or buffer_name not in allocation[i - 1]:
                    # This buffer is not allocated in the scratch pad before this operation, so it must be loaded from HBM.
                    hbm_usage += all_buffers[buffer_name].size
            for buffer_name in op.outputs:
                if i == len(operations) - 1 or buffer_name not in allocation[i + 1]:
                    # This buffer is not allocated in the scratch pad after this operation, so it must be stored to HBM.
                    hbm_usage += all_buffers[buffer_name].size
        return hbm_usage

    def hbm_usage_current_run(
        self, operations: list[Operation], alloc: InstrumentedAllocator
    ) -> int:
        hbm_usage = 0

        # Count all usage for buffers not allocated in the scratchpad.
        for i, op in enumerate(operations):
            for buffer_name in op.inputs:
                if i == 0 or buffer_name not in alloc.allocations:
                    # This buffer is not allocated in the scratch pad before this operation, so it must be loaded from HBM.
                    hbm_usage += all_buffers[buffer_name].size
            for buffer_name in op.outputs:
                if i == len(operations) - 1 or buffer_name not in alloc.allocations:
                    # This buffer is not allocated in the scratch pad after this operation, so it must be stored to HBM.
                    hbm_usage += all_buffers[buffer_name].size

        # All buffers allocated in the scratchpad are counted only once each.
        for buffer_name in alloc.allocations:
            hbm_usage += all_buffers[buffer_name].size

        return hbm_usage

    def run_test_case(self, test_case: AllocationTestCase):
        global all_buffers
        all_buffers = test_case.buffers

        # Assert that the "good allocation" is indeed valid
        self.assertIsValidAllocation(test_case.good_allocation, test_case.operations)

        alloc = InstrumentedAllocator()

        scratchpad_planning(test_case.operations, alloc)

        # Verify that the currently implemented allocation is indeed valid
        # (not implemented yet)

        # Verify that the currently implemented allocation is at least as good as the "good
        # allocation" in terms of HBM usage.
        current_hbm_usage = self.hbm_usage_current_run(test_case.operations, alloc)
        good_hbm_usage = self.hbm_usage_test(
            test_case.good_allocation, test_case.operations
        )

        self.assertLessEqual(
            current_hbm_usage,
            good_hbm_usage,
            f"Current allocation uses more HBM ({current_hbm_usage} bytes) than the good allocation ({good_hbm_usage} bytes). ",
        )

    @expectedFailure
    def test_simple_fragmentation_pattern(self):
        quarter_scratchpad_size = 1 << 19  # 512KB
        buffers = {}
        buffers["A"] = Buffer("A", quarter_scratchpad_size)
        buffers["B"] = Buffer("B", quarter_scratchpad_size)
        buffers["C"] = Buffer("C", 2 * quarter_scratchpad_size)
        # We can fit A and C together, or B and C together, but not all three, because of the space
        # reserved for the compiler.

        op1 = Operation("op1", inputs=["A"], outputs=["B"])
        op2 = Operation("op2", inputs=["B"], outputs=["A"])
        op3 = Operation("op3", inputs=["B"], outputs=["C"])

        testcase = AllocationTestCase(
            buffers,
            [op1, op2, op3],
            good_allocation=[
                # A is used only during op1 and op2, so we allocate it after B. This way we can
                # evict it after op2 and have enough space for C during op3.
                {"A": quarter_scratchpad_size, "B": 0},
                {"A": quarter_scratchpad_size, "B": 0},
                {"B": 0, "C": quarter_scratchpad_size},
            ],
        )
        self.run_test_case(testcase)


if __name__ == "__main__":
    import unittest

    unittest.main()
