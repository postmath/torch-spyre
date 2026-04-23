from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Optional, Iterable
from unittest import TestCase, expectedFailure
from enum import Enum

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


def make_buffer_registry(names_sizes: Iterable[tuple[str, int]]) -> dict[str, Buffer]:
    return {name: Buffer(name=name, size=size) for (name, size) in names_sizes}


class InstrumentedAllocator(ScratchPadAllocator):
    def __init__(self):
        super().__init__()
        self.allocations = {}

    def should_consider_op(self, op: Operation) -> bool:
        return True

    def allocate(self, tensor_name: str, addr: int):
        if tensor_name in self.allocations:
            # TODO: support this. We need to store allocations differently, and then modify the
            # logic for measuring HBM usage in TestExamplePattern.hbm_usage_for_actual_run to
            # account for this. Also update TestExamplePattern.verify_actual_run to account for
            # this.
            assert self.allocations[tensor_name] == addr, (
                f"Buffer {tensor_name} was already allocated at address "
                f"{self.allocations[tensor_name]}, but is being allocated again at address {addr}."
                f" That is probably a good improvement, but it means this test needs to be "
                f"adjusted."
            )
        self.allocations[tensor_name] = addr

    def mem_usage_by_op(self, op: Operation) -> dict[str, dict[str, bool | int]]:
        # Returns a dict mapping each buffer name to a dict with keys "is_input" and "size".
        # is_input is True if the buffer is an input to the op, and False otherwise. size is the
        # size of the buffer.
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


class Component(Enum):
    LX = "LX"
    HBM = "HBM"


@dataclass
class Allocation:
    buffer: str
    component: Component = Component.LX
    # If the component is LX, then the address must be an integer. If the component is HBM, we don't
    # care about the address; this is encoded by the address being None. (This is enforced in
    # TestExamplePattern.verify_test_case.)
    address: Optional[int] = None


# A type alias for the result of an allocation. The ith entry in the list is the state during
# the ith operation. It maps each allocated buffer to the scratch pad address where it is
# allocated at that point in time.
AllocationResult = list[list[Allocation]]


@dataclass
class AllocationTestCase:
    buffers: dict[str, Buffer]
    operations: list[Operation]
    # A "good" allocation pattern that we want to compare to. The test verifies that this pattern
    # is valid and that the current result is at least as good -- that is, the HBM usage of the
    # current result is no more than that of the good pattern.
    good_allocation: AllocationResult


class TestExamplePattern(TestCase):
    def find_single_use_buffers(
        self,
        operations: list[Operation],
        *,
        see_later: Optional[Callable[[Operation, str], None]] = None,
        see_first: Optional[Callable[[Operation, str], None]] = None,
    ) -> set[str]:
        """Returns the set of buffers that are used only once in the list of operations. see_first
        is called the first time any buffer is seen, and see_later is called any other time any
        buffer is seen."""
        single_use_buffers = set()
        seen_buffers = set()
        for op in operations:
            for buffer_name in op.inputs + op.outputs:
                if buffer_name in seen_buffers:
                    if see_later is not None:
                        see_later(op, buffer_name)
                    single_use_buffers.discard(buffer_name)
                else:
                    if see_first is not None:
                        see_first(op, buffer_name)
                    single_use_buffers.add(buffer_name)
                    seen_buffers.add(buffer_name)

        return single_use_buffers

    def verify_test_case(self, test_case: AllocationTestCase, *, inplace: bool = False):
        allocation = test_case.good_allocation
        operations = test_case.operations
        self.assertEqual(
            len(allocation),
            len(operations),
            f"Good allocation should have the same number of entries as the number of operations, "
            f"but found {len(allocation)} allocations and {len(operations)} operations.",
        )
        for alloc in allocation:
            for a in alloc:
                self.assertEqual(
                    a.address is not None,
                    a.component == Component.LX,
                    f"Buffers should have an address iff they are allocated in LX, but found {a}.",
                )

        # Buffers that are used only once need not be allocated in the scratchpad, because it
        # doesn't help reduce HBM transfers. In the meantime, verify that we didn't write any
        # operations that write to a buffer, except possibly the first time we see that buffer.
        def no_output(op: Operation, buffer_name: str):
            self.assertNotIn(
                buffer_name,
                op.outputs,
                f"Buffer {buffer_name} is written to in operation {op.name}, but accessed before "
                f"that operation. However, this test is case is not marked as in-place, so we "
                f"avoid in-place operations.",
            )

        single_use_buffers = self.find_single_use_buffers(
            operations, see_later=None if inplace else no_output
        )

        for i, op in enumerate(operations):
            # Check that each buffer that is used is allocated.
            for buffer_name in op.inputs + op.outputs:
                if buffer_name not in single_use_buffers:
                    self.assertTrue(
                        any(alloc.buffer == buffer_name for alloc in allocation[i]),
                        f"Buffer {buffer_name} used by operation {op.name} is not allocated at "
                        f"this point in the good allocation pattern, but it is used more than once.",
                    )

            # Check that there is at least one output.
            self.assertGreater(
                len(op.outputs),
                0,
                f"Operation {op.name} should have at least one output.",
            )

            # Check that allocated buffers do not overlap.
            allocated_buffers = [
                alloc for alloc in allocation[i] if alloc.address is not None
            ]
            if allocated_buffers:
                # Sort by address:
                sorted_allocations = sorted(
                    list(allocated_buffers),
                    key=lambda x: x.address,  # pyright: ignore[reportCallIssue, reportArgumentType]
                )
                for j in range(len(sorted_allocations) - 1):
                    buffer_name_j = sorted_allocations[j].buffer
                    addr_j = sorted_allocations[j].address
                    buffer_name_next = sorted_allocations[j + 1].buffer
                    addr_next = sorted_allocations[j + 1].address
                    size_j = op._buffer_registry[buffer_name_j].size
                    self.assertLessEqual(
                        addr_j + size_j,
                        addr_next,
                        f"Buffers {buffer_name_j} and {buffer_name_next} overlap during operation {op.name}",
                    )

                self.assertLessEqual(
                    sorted_allocations[-1].address
                    + op._buffer_registry[sorted_allocations[-1].buffer].size,
                    AVAILABLE_LX_SIZE,
                    f"Buffer {sorted_allocations[-1].buffer} exceeds scratch pad size during operation {op.name}",
                )

    def verify_actual_run(
        self, test_case: AllocationTestCase, alloc: InstrumentedAllocator
    ):
        # Verify that the actual run's allocation is valid. We assume that any allocation is "live"
        # during the entire liveness of the corresponding buffer.
        liveness_start = {}
        liveness_end = {}
        for i, op in enumerate(test_case.operations):
            for buffer_name in op.inputs + op.outputs:
                if buffer_name not in liveness_start:
                    liveness_start[buffer_name] = i
                liveness_end[buffer_name] = i

        # Sanity check -- every buffer should have a start and an end to its liveness.
        self.assertTrue(set(liveness_start.keys()) == set(liveness_end.keys()))

        allocate_at = defaultdict(list)
        deallocate_at = defaultdict(list)
        for buffer_name in liveness_start:
            allocate_at[liveness_start[buffer_name]].append(buffer_name)
            deallocate_at[liveness_end[buffer_name] + 1].append(buffer_name)

        live_buffers = set()
        for i, op in enumerate(test_case.operations):
            live_buffers.update(allocate_at[i])
            for buffer_name in op.inputs + op.outputs:
                # Verify that buffer_name does not overlap with any allocated buffers at this point.
                addr = alloc.allocations[buffer_name]
                size = op._buffer_registry[buffer_name].size
                self.assertLessEqual(
                    addr + size,
                    AVAILABLE_LX_SIZE,
                    f"Buffer {buffer_name} exceeds scratch pad size during operation {op.name}",
                )
                for other_buffer_name in live_buffers:
                    if other_buffer_name == buffer_name:
                        continue
                    other_addr = alloc.allocations[other_buffer_name]
                    other_size = op._buffer_registry[other_buffer_name].size
                    if addr <= other_addr:
                        self.assertLessEqual(
                            addr + size,
                            other_addr,
                            f"Buffers {buffer_name} and {other_buffer_name} overlap during "
                            f"operation {op.name}",
                        )
                    else:
                        self.assertLessEqual(
                            other_addr + other_size,
                            addr,
                            f"Buffers {buffer_name} and {other_buffer_name} overlap during "
                            f"operation {op.name}",
                        )
            live_buffers.difference_update(deallocate_at[i + 1])

    def hbm_usage_for_good_allocation(
        self, allocation: AllocationResult, operations: list[Operation]
    ) -> int:
        if not operations:
            return 0
        registry = operations[0]._buffer_registry

        single_use_buffers = self.find_single_use_buffers(operations)
        hbm_usage = sum(
            registry[buffer_name].size for buffer_name in single_use_buffers
        )

        for i, op in enumerate(operations):
            for buffer_name in op.inputs:
                if buffer_name not in single_use_buffers and (
                    i == 0 or buffer_name not in allocation[i - 1]
                ):
                    # This buffer is not allocated in the scratch pad before this operation, so it must be loaded from HBM.
                    hbm_usage += registry[buffer_name].size
            for buffer_name in op.outputs:
                if buffer_name not in single_use_buffers and (
                    i == len(operations) - 1 or buffer_name not in allocation[i + 1]
                ):
                    # This buffer is not allocated in the scratch pad after this operation, so it must be stored to HBM.
                    hbm_usage += registry[buffer_name].size
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

    def run_test_case(self, test_case: AllocationTestCase):
        alloc = InstrumentedAllocator()

        scratchpad_planning(test_case.operations, alloc)

        # Verify that the currently implemented allocation is indeed valid
        self.verify_actual_run(test_case, alloc)

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
        """Allocate two buffers A and B that are each a third of the available scratchpad size,
        where A can be freed after the second operation. Then allocate a third buffer C
        that is two thirds of the scratchpad size. This can only fit if B was allocated at the start
        or end of the scratchpad, leaving a contiguous region for C."""
        third_scratchpad_size = AVAILABLE_LX_SIZE // 3
        third_scratchpad_size = (
            third_scratchpad_size // 128
        ) * 128  # round down to a multiple of the stick size
        buffers = make_buffer_registry(
            [
                ("A", third_scratchpad_size),
                ("B", third_scratchpad_size),
                ("C", 2 * third_scratchpad_size),
                ("D", third_scratchpad_size),
                ("E", third_scratchpad_size),
            ]
        )

        op1 = Operation("op1", inputs=["A"], outputs=["B"], _buffer_registry=buffers)
        op2 = Operation(
            "op2", inputs=["A", "B"], outputs=["D"], _buffer_registry=buffers
        )
        op3 = Operation("op3", inputs=["B"], outputs=["C"], _buffer_registry=buffers)
        op4 = Operation(
            "op4", inputs=["B", "C"], outputs=["E"], _buffer_registry=buffers
        )

        # A is used only during op1 and op2, so we allocate it after B. This way we can
        # evict it after op2 and have enough space for C during op3.
        alloc_A = Allocation(buffer="A", address=third_scratchpad_size)
        alloc_B = Allocation(buffer="B", address=0)
        alloc_C = Allocation(buffer="C", address=third_scratchpad_size)
        alloc_D = Allocation(buffer="D", component=Component.HBM)
        alloc_E = Allocation(buffer="E", component=Component.HBM)
        return AllocationTestCase(
            buffers,
            [op1, op2, op3, op4],
            good_allocation=[
                [alloc_A, alloc_B],
                [alloc_A, alloc_B, alloc_D],
                [alloc_B, alloc_C],
                [alloc_B, alloc_C, alloc_E],
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
        } | {f"sink_{i}": Buffer(f"sink_{i}", 128) for i in range(1, N + 2)}

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
                    inputs=[f"A{i}", f"B{i}"],
                    outputs=[f"sink_{i}"],
                    _buffer_registry=buffers,
                ),
            )

        ops = [op for i in range(1, N + 1) for op in op_pair(i)] + [
            Operation(
                "op_final",
                inputs=[f"B{i}" for i in range(1, N + 1)],
                outputs=[f"sink_{N + 1}"],
                _buffer_registry=buffers,
            )
        ]

        def good_allocation_pair(i: int) -> tuple[list[Allocation], list[Allocation]]:
            # Allocate A{i} at 0 and B{j} for j <= i at N*k, (N+1)*k, (N+3)*k, (N+6)*k, ...,
            # (N+i*(i-1)/2)*k.
            alloc = [
                Allocation(buffer=f"B{j}", address=(N + j * (j - 1) // 2) * k)
                for j in range(1, i + 1)
            ]
            alloc.append(Allocation(buffer=f"A{i}", address=0))
            return (alloc, alloc)

        good_allocations = [
            alloc for i in range(1, N + 1) for alloc in good_allocation_pair(i)
        ]
        good_allocations.append(
            [
                Allocation(buffer=f"B{i}", address=(N + i * (i - 1) // 2) * k)
                for i in range(1, N + 1)
            ]
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
        for i in range(0, N + 2):
            buffers[f"sink_{i}"] = Buffer(f"sink_{i}", 128)

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
                    inputs=[f"A{i}", f"B{i}"],
                    outputs=[f"sink_{i}"],
                    _buffer_registry=buffers,
                ),
            )

        ops = (
            [
                Operation(
                    "op_start",
                    inputs=["Z"],
                    outputs=["sink_0"],
                    _buffer_registry=buffers,
                )
            ]
            + [op for i in range(1, N + 1) for op in op_pair(i)]
            + [
                Operation(
                    "op_final",
                    inputs=[f"B{i}" for i in range(1, N + 1)] + ["Z"],
                    outputs=[f"sink_{N + 1}"],
                    _buffer_registry=buffers,
                )
            ]
        )

        def good_allocation_pair(i: int) -> tuple[list[Allocation], list[Allocation]]:
            # Allocate Z at 0, A{i} at k, and B{j} for j <= i as follows: B{N} at 2*k, B{N-1} at
            # 3*k, B{N-2} at 5*k, B{N-3} at 8*k, ..., B{N-j} at (j*(j+1)/2 + 2)*k. The gap
            # between A{i} and B{i} = B{N+1-(N+1-i)} is (N+1-i)*(N+2-i)/2*k, which is big enough
            # for A{i+1} of size (N-i)*k.
            alloc = [
                Allocation(buffer=f"B{N - j}", address=(j * (j + 1) // 2 + 2) * k)
                for j in range(N - i, N)
            ]
            alloc.append(Allocation(buffer="Z", address=0))
            alloc.append(Allocation(buffer=f"A{i}", address=k))
            return (alloc, alloc)

        good_allocations = [[Allocation(buffer="Z", address=0)]]
        good_allocations.extend(
            [alloc for i in range(1, N + 1) for alloc in good_allocation_pair(i)]
        )
        last_allocation = good_allocation_pair(N)[0]
        del last_allocation[-1]  # A{N} is not needed for the final op
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
        } | {f"sink_{i}": Buffer(f"sink_{i}", 128) for i in range(1, 7)}
        ops = [
            Operation(
                "op1", inputs=["A"], outputs=["sink_1"], _buffer_registry=buffers
            ),
            Operation(
                "op2", inputs=["A"], outputs=["sink_2"], _buffer_registry=buffers
            ),
            Operation(
                "op3", inputs=["B"], outputs=["sink_3"], _buffer_registry=buffers
            ),
            Operation(
                "op4", inputs=["B"], outputs=["sink_4"], _buffer_registry=buffers
            ),
            Operation(
                "op5", inputs=["A"], outputs=["sink_5"], _buffer_registry=buffers
            ),
            Operation(
                "op6", inputs=["A"], outputs=["sink_6"], _buffer_registry=buffers
            ),
        ]

        good_allocation = (
            [[Allocation(buffer="A", address=0)]] * 2
            + [[Allocation(buffer="B", address=0)]] * 2
            + [[Allocation(buffer="A", address=0)]] * 2
        )

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
        for i in range(4):
            for j in range(4):
                buffers[f"sink_{i}_{j}"] = Buffer(f"sink_{i}_{j}", 128)

        pattern = [["A0", "A1", "A2"], ["A0", "B"], ["A1", "B"], ["A2", "B"]]
        ops = [
            Operation(
                f"op{i}_{j}",
                inputs=group,
                outputs=[f"sink_{i}_{j}"],
                _buffer_registry=buffers,
            )
            for i, group in enumerate(pattern)
            for j in range(4)
        ]

        good_allocations = (
            [
                [
                    Allocation(buffer="A0", address=0),
                    Allocation(buffer="A1", address=A_size),
                    Allocation(buffer="A2", address=2 * A_size),
                ]
            ]
            * 4
            + [
                [
                    Allocation(buffer="A0", address=0),
                    Allocation(buffer="B", address=A_size),
                ]
            ]
            * 4
            + [
                [
                    Allocation(buffer="A1", address=0),
                    Allocation(buffer="B", address=A_size),
                ]
            ]
            * 4
            + [
                [
                    Allocation(buffer="A2", address=0),
                    Allocation(buffer="B", address=A_size),
                ]
            ]
            * 4
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
