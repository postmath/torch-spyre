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
    _buffer_registry: dict[str, "Buffer"]

    # To make scratchpad.py work, we add an origin_node field that points to the op itself.
    origin_node = None

    def __post_init__(self):
        self.origin_node = self

    def get_read_writes(self) -> ReadWrites:
        # Returns a list of (buffer_name, "read" or "write") for all buffers used by this operation.
        reads = [self._buffer_registry[buffer_name] for buffer_name in self.inputs]
        writes = [self._buffer_registry[buffer_name] for buffer_name in self.outputs]
        return ReadWrites(reads=reads, writes=writes)


@dataclass
class Buffer:
    name: str
    size: int


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
                "size": op._buffer_registry[tensor_name].size,
            }
        for tensor_name in op.outputs:
            result[tensor_name] = {
                "is_input": False,
                "size": op._buffer_registry[tensor_name].size,
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
            if allocation[i]:
                # Sort by address:
                sorted_allocations = sorted(allocation[i].items(), key=lambda x: x[1])
                for j in range(len(sorted_allocations) - 1):
                    buffer_name_j, addr_j = sorted_allocations[j]
                    buffer_name_next, addr_next = sorted_allocations[j + 1]
                    size_j = op._buffer_registry[buffer_name_j].size
                    self.assertLessEqual(
                        addr_j + size_j,
                        addr_next,
                        f"Buffers {buffer_name_j} and {buffer_name_next} overlap during operation {op.name}",
                    )

                self.assertLessEqual(
                    sorted_allocations[-1][1]
                    + op._buffer_registry[sorted_allocations[-1][0]].size,
                    AVAILABLE_LX_SIZE,
                    f"Buffer {sorted_allocations[-1][0]} exceeds scratch pad size during operation {op.name}",
                )

    def hbm_usage_for_good_allocation(
        self, allocation: AllocationResult, operations: list[Operation]
    ) -> int:
        hbm_usage = 0
        for i, op in enumerate(operations):
            for buffer_name in op.inputs:
                if i == 0 or buffer_name not in allocation[i - 1]:
                    # This buffer is not allocated in the scratch pad before this operation, so it must be loaded from HBM.
                    hbm_usage += op._buffer_registry[buffer_name].size
            for buffer_name in op.outputs:
                if i == len(operations) - 1 or buffer_name not in allocation[i + 1]:
                    # This buffer is not allocated in the scratch pad after this operation, so it must be stored to HBM.
                    hbm_usage += op._buffer_registry[buffer_name].size
        return hbm_usage

    def hbm_usage_for_actual_run(
        self, operations: list[Operation], alloc: InstrumentedAllocator
    ) -> int:
        if not operations:
            return 0

        hbm_usage = 0

        # Count all usage for buffers not allocated in the scratchpad.
        for i, op in enumerate(operations):
            for buffer_name in op.inputs:
                if i == 0 or buffer_name not in alloc.allocations:
                    # This buffer is not allocated in the scratch pad before this operation, so it must be loaded from HBM.
                    hbm_usage += op._buffer_registry[buffer_name].size
            for buffer_name in op.outputs:
                if i == len(operations) - 1 or buffer_name not in alloc.allocations:
                    # This buffer is not allocated in the scratch pad after this operation, so it must be stored to HBM.
                    hbm_usage += op._buffer_registry[buffer_name].size

        # All buffers allocated in the scratchpad are counted only once each.
        for buffer_name in alloc.allocations:
            hbm_usage += operations[0]._buffer_registry[buffer_name].size

        return hbm_usage

    def verify_test_case(self, test_case: AllocationTestCase):
        self.assertIsValidAllocation(test_case.good_allocation, test_case.operations)

    def run_test_case(self, test_case: AllocationTestCase):
        alloc = InstrumentedAllocator()

        scratchpad_planning(test_case.operations, alloc)

        # Verify that the currently implemented allocation is indeed valid
        # (not implemented yet)

        # Verify that the currently implemented allocation is at least as good as the "good
        # allocation" in terms of HBM usage.
        current_hbm_usage = self.hbm_usage_for_actual_run(test_case.operations, alloc)
        good_hbm_usage = self.hbm_usage_for_good_allocation(
            test_case.good_allocation, test_case.operations
        )

        self.assertLessEqual(
            current_hbm_usage,
            good_hbm_usage,
            f"Current allocation uses more HBM ({current_hbm_usage} bytes) than the good allocation ({good_hbm_usage} bytes). ",
        )

    def make_simple_fragmentation_pattern_test_case(self) -> AllocationTestCase:
        """Allocate two buffers A and B that are each a quarter of the scratchpad size, where the
        first buffer can be freed after the second operation. Then allocate a third buffer that is
        half the scratchpad size. This can only fit if B was allocated at the start or end of the
        scratchpad, leaving a contiguous region for C."""
        quarter_scratchpad_size = 1 << 19  # 512KB
        buffers = {}
        buffers["A"] = Buffer("A", quarter_scratchpad_size)
        buffers["B"] = Buffer("B", quarter_scratchpad_size)
        buffers["C"] = Buffer("C", 2 * quarter_scratchpad_size)
        # We can fit A and C together, or B and C together, but not all three, because of the space
        # reserved for the compiler.

        op1 = Operation("op1", inputs=["A"], outputs=["B"], _buffer_registry=buffers)
        op2 = Operation("op2", inputs=["B"], outputs=["A"], _buffer_registry=buffers)
        op3 = Operation("op3", inputs=["B"], outputs=["C"], _buffer_registry=buffers)

        return AllocationTestCase(
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

    def test_verify_simple_fragmentation_pattern(self):
        self.verify_test_case(self.make_simple_fragmentation_pattern_test_case())

    @expectedFailure
    def test_simple_fragmentation_pattern(self):
        self.run_test_case(self.make_simple_fragmentation_pattern_test_case())

    def make_staircase_pattern_test_case(self) -> AllocationTestCase:
        """Allocate N*2 buffers of sizes k, k, 2*k, 2*k, 3*k, 3*k, ..., N*k, N*k. After an
        even-numbered buffer is allocated, free the previous odd-numbered buffer. This creates a
        "staircase" pattern of allocations that can only be fit if the allocator is smart about
        fragmentation. In that case, the maximum scratchpad usage is
        k + 2*k + ... + N*k + N*k = k * N * (N + 1) / 2 + N * k = k * N * (N + 3) / 2, so we choose
        k such that this is just less than the available scratchpad size.

        The greedy allocator will always allocate the next buffer just after all other buffers,
        because no gap is big enough for the current size. So it uses
        2 * (k + 2*k + ... + N*k) = k * N * (N + 1) or roughly 2/3 times more."""
        N = 7
        k = (2 * AVAILABLE_LX_SIZE) // (N * (N + 3))
        k = (k // 128) * 128  # round down to a multiple of the stick size

        # This only works if the greedy allocator uses more than fits in the scratchpad, so we
        # assert that here.
        self.assertGreater(k * N * (N + 1), AVAILABLE_LX_SIZE)

        buffers = {
            f"{letter}{i}": Buffer(f"{letter}{i}", i * k)
            for i in range(1, N + 1)
            for letter in ["A", "B"]
        }

        def op_pair(i: int) -> tuple[Operation, Operation]:
            return (
                Operation(
                    f"op{i}_0",
                    inputs=[f"A{i}"],
                    outputs=[f"B{i}"],
                    _buffer_registry=buffers,
                ),
                Operation(
                    f"op{i}_1",
                    inputs=[f"B{i}"],
                    outputs=[f"A{i}"],
                    _buffer_registry=buffers,
                ),
            )

        ops = [op for i in range(1, N + 1) for op in op_pair(i)] + [
            Operation(
                "op_final",
                inputs=[f"B{i}" for i in range(1, N + 1)],
                outputs=[],
                _buffer_registry=buffers,
            )
        ]

        def good_allocation_pair(i: int) -> tuple[dict[str, int], dict[str, int]]:
            # Allocate A{i} at 0 and B{j} for j <= i at N*k, (N+1)*k, (N+3)*k, (N+6)*k, ...,
            # (N+i*(i-1)/2)*k.
            alloc = {f"B{j}": (N + j * (j - 1) // 2) * k for j in range(1, i + 1)}
            alloc[f"A{i}"] = 0
            return (alloc, alloc)

        good_allocations = [
            alloc for i in range(1, N + 1) for alloc in good_allocation_pair(i)
        ]
        good_allocations.append(
            {f"B{i}": (N + i * (i - 1) // 2) * k for i in range(1, N + 1)}
        )

        testcase = AllocationTestCase(buffers, ops, good_allocation=good_allocations)
        return testcase

    def test_verify_staircase_pattern(self):
        self.verify_test_case(self.make_staircase_pattern_test_case())

    @expectedFailure
    def test_staircase_pattern(self):
        self.run_test_case(self.make_staircase_pattern_test_case())

    def make_downward_staircase_pattern_test_case(self) -> AllocationTestCase:
        """Allocate 1+N*2 buffers of sizes k, N*k, N*k, (N-1)*k, (N-1)*k, ..., 2*k, 2*k, k, k.
        After an odd-numbered buffer (>1) is allocated, free the previous even-numbered buffer.
        This creates an easier "staircase" pattern of allocations than in
        `make_staircase_pattern_test_case`. Still, the greedy allocator will prefer to allocate
        buffers at the end if it can't allocate them at address 0. So we first allocate one small
        buffer at the start which will block address 0. In the optimal case, the maximum scratchpad
        usage is k + N*k + (N-1)*k + ... + 2*k + k + k = k * (4 + N * (N + 1)) / 2, so we choose k
        such that this is just less than the available scratchpad size.

        The greedy allocator will always allocate the next buffer just after all other buffers,
        up until the point where it fills up the cache and starts looking for gaps. The total usage
        is less clear to analyze."""
        N = 5
        k = (2 * AVAILABLE_LX_SIZE) // (4 + N * (N + 1))
        k = (k // 128) * 128  # round down to a multiple of the stick size

        buffers = {
            f"{letter}{i}": Buffer(f"{letter}{i}", (N + 1 - i) * k)
            for i in range(1, N + 1)
            for letter in ["A", "B"]
        }
        buffers["Z"] = Buffer("Z", k)

        def op_pair(i: int) -> tuple[Operation, Operation]:
            return (
                Operation(
                    f"op{i}_0",
                    inputs=[f"A{i}"],
                    outputs=[f"B{i}"],
                    _buffer_registry=buffers,
                ),
                Operation(
                    f"op{i}_1",
                    inputs=[f"B{i}"],
                    outputs=[f"A{i}"],
                    _buffer_registry=buffers,
                ),
            )

        ops = (
            [Operation("op_start", inputs=["Z"], outputs=[], _buffer_registry=buffers)]
            + [op for i in range(1, N + 1) for op in op_pair(i)]
            + [
                Operation(
                    "op_final",
                    inputs=[f"B{i}" for i in range(1, N + 1)],
                    outputs=["Z"],
                    _buffer_registry=buffers,
                )
            ]
        )

        def good_allocation_pair(i: int) -> tuple[dict[str, int], dict[str, int]]:
            # Allocate Z at 0, A{i} at k, and B{j} for j <= i as follows: B{N} at 2*k, B{N-1} at
            # 3*k, B{N-2} at 5*k, B{N-3} at 8*k, ..., B{N-j} at (j*(j+1)/2 + 2)*k. The gap
            # between A{i} and B{i} = B{N+1-(N+1-i)} is (N+1-i)*(N+2-i)/2*k, which is big enough
            # for A{i+1} of size (N-i)*k.
            alloc = {f"B{N - j}": (j * (j + 1) // 2 + 2) * k for j in range(N - i, N)}
            alloc["Z"] = 0
            alloc[f"A{i}"] = k
            return (alloc, alloc)

        good_allocations = [{"Z": 0}]
        good_allocations.extend(
            [alloc for i in range(1, N + 1) for alloc in good_allocation_pair(i)]
        )
        last_allocation = good_allocation_pair(N)[0]
        del last_allocation[f"A{N}"]  # A{N} is not needed for the final op
        good_allocations.append(last_allocation)

        testcase = AllocationTestCase(buffers, ops, good_allocation=good_allocations)
        return testcase

    def test_verify_downward_staircase_pattern(self):
        self.verify_test_case(self.make_downward_staircase_pattern_test_case())

    @expectedFailure
    def test_downward_staircase_pattern(self):
        self.run_test_case(self.make_downward_staircase_pattern_test_case())

    def make_simple_eviction_pattern_test_case(self) -> AllocationTestCase:
        """This pattern requires allocating a buffer, evicting it, and then reallocating it later.

        We use two buffers A and B that are each exactly the available LX size. We have six
        operations. The first two use A, the next two use B, and the last two use A again. Optimal
        use would allocate A and B for two ops each at alternate times."""
        buffers = {
            "A": Buffer("A", AVAILABLE_LX_SIZE),
            "B": Buffer("B", AVAILABLE_LX_SIZE),
        }
        ops = [
            Operation("op1", inputs=["A"], outputs=[], _buffer_registry=buffers),
            Operation("op2", inputs=["A"], outputs=[], _buffer_registry=buffers),
            Operation("op3", inputs=["B"], outputs=[], _buffer_registry=buffers),
            Operation("op4", inputs=["B"], outputs=[], _buffer_registry=buffers),
            Operation("op5", inputs=["A"], outputs=[], _buffer_registry=buffers),
            Operation("op6", inputs=["A"], outputs=[], _buffer_registry=buffers),
        ]

        good_allocation = [{"A": 0}] * 2 + [{"B": 0}] * 2 + [{"A": 0}] * 2

        testcase = AllocationTestCase(buffers, ops, good_allocation=good_allocation)
        return testcase

    def test_verify_simple_eviction_pattern(self):
        self.verify_test_case(self.make_simple_eviction_pattern_test_case())

    @expectedFailure
    def test_simple_eviction_pattern(self):
        self.run_test_case(self.make_simple_eviction_pattern_test_case())

    def make_eviction_reallocation_pattern_test_case(self) -> AllocationTestCase:
        """This pattern requires allocating a buffer, evicting it, and then reallocating it later
        at a different address to achieve optimality.

        We use four buffers total: A0, A1, A2 of size 1/3 the available size, and B of size twice
        that. We first ensure that A0, A1, and A2 must be allocated together, then A0 and B, then
        A1 and B, and finally A2 and B. Because B can fit only with one of the A buffers at the top
        or the bottom, whichever one was allocated in the middle must be moved.

        We ensure that any set is allocated together in an optimal allocation by using four ops
        in a row that use them all as input. This means that, whatever was in the scratchpad before
        and whatever is in it after, we can complete that phase with one full scratchpad worth of
        HBM transfers. On the other hand, if not everything is allocated on the scratchpad, then we
        have to stream at least one buffer four times, which entails at least 4/3 of the scratchpad
        size in HBM transfers."""
        A_size = AVAILABLE_LX_SIZE // 3
        A_size = (A_size // 128) * 128  # round down to a multiple of the stick size
        B_size = 2 * A_size

        # This will work if 4 * A_size > AVAILABLE_LX_SIZE.
        self.assertGreater(4 * A_size, AVAILABLE_LX_SIZE)

        buffers = {f"A{i}": Buffer(f"A{i}", A_size) for i in range(3)}
        buffers["B"] = Buffer("B", B_size)

        pattern = [["A0", "A1", "A2"], ["A0", "B"], ["A1", "B"], ["A2", "B"]]
        ops = [
            Operation(f"op{i}_{j}", inputs=group, outputs=[], _buffer_registry=buffers)
            for i, group in enumerate(pattern)
            for j in range(4)
        ]

        good_allocations = (
            [{"A0": 0, "A1": A_size, "A2": 2 * A_size}] * 4
            + [{"A0": 0, "B": A_size}] * 4
            + [{"A1": 0, "B": A_size}] * 4
            + [{"A2": 0, "B": A_size}] * 4
        )
        testcase = AllocationTestCase(buffers, ops, good_allocation=good_allocations)
        return testcase

    def test_verify_eviction_reallocation_pattern(self):
        self.verify_test_case(self.make_eviction_reallocation_pattern_test_case())

    @expectedFailure
    def test_eviction_reallocation_pattern(self):
        self.run_test_case(self.make_eviction_reallocation_pattern_test_case())


if __name__ == "__main__":
    import unittest

    unittest.main()
