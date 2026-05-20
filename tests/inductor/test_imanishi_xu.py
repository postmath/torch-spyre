# Importing torch appears to be necessary in order to import some torch_spyre members.
import torch  # noqa: F401
from torch_spyre._inductor.imanishi_xu import (
    AllocationBuilder,
    Allocations,
    BestFit,
    Buffer,
    BufferList,
    CoolingScheduleFromPaper,
    ExponentialCoolingSchedule,
    FirstFit,
    ImanishiXuAllocator,
    MaxRangeTree_Array,
    MaxRangeTree_List,
    MaxRangeTree_Tree,
    overlaps,
)
import unittest
import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    _MixinBase = unittest.TestCase
else:
    _MixinBase = object


class _TestMaxRangeTree(_MixinBase):
    MaxRangeTree_cls: type

    def test_max_16(self):
        m = self.MaxRangeTree_cls(16)
        m.increase_values(2, 6, 3)
        expected = [0] * 2 + [3] * 4 + [0] * 10
        for j in range(16):
            for i in range(j):
                self.assertEqual(m.max(i, j), max(expected[i:j]))

        m.increase_values(4, 10, 4)
        expected[4:10] = [4] * 6
        for j in range(16):
            for i in range(j):
                self.assertEqual(m.max(i, j), max(expected[i:j]))

    def test_max_rand_31(self):
        m = self.MaxRangeTree_cls(31)
        expected = [0] * 31
        random.seed(0)

        for _ in range(100):
            left = random.randrange(31)
            right = random.randrange(30)
            if left <= right:
                right += 1
            else:
                left, right = right, left
            # Now we have uniformly random left < right.

            val = m.max(left, right) + 1
            m.increase_values(left, right, val)
            expected[left:right] = [val] * (right - left)
            for j in range(31):
                for i in range(j):
                    self.assertEqual(m.max(i, j), max(expected[i:j]))


class TestMaxRangeTree_Array(_TestMaxRangeTree, unittest.TestCase):
    MaxRangeTree_cls = MaxRangeTree_Array


class TestMaxRangeTree_List(_TestMaxRangeTree, unittest.TestCase):
    MaxRangeTree_cls = MaxRangeTree_List


class TestMaxRangeTree_Tree(_TestMaxRangeTree, unittest.TestCase):
    MaxRangeTree_cls = MaxRangeTree_Tree


class _TestMaxRangeTreeInit(_MixinBase):
    MaxRangeTree_cls: type

    def test_single_element(self):
        t = self.MaxRangeTree_cls(1)
        self.assertEqual(t.max(0, 1), 0)

    def test_power_of_two_size(self):
        t = self.MaxRangeTree_cls(8)
        self.assertEqual(t.max(0, 8), 0)

    def test_non_power_of_two_size(self):
        t = self.MaxRangeTree_cls(5)
        self.assertEqual(t.max(0, 5), 0)

    def test_large_size(self):
        t = self.MaxRangeTree_cls(1000)
        self.assertEqual(t.max(0, 1000), 0)


class TestMaxRangeTree_ArrayInit(_TestMaxRangeTreeInit, unittest.TestCase):
    MaxRangeTree_cls = MaxRangeTree_Array


class TestMaxRangeTree_ListInit(_TestMaxRangeTreeInit, unittest.TestCase):
    MaxRangeTree_cls = MaxRangeTree_List


class TestMaxRangeTree_TreeInit(_TestMaxRangeTreeInit, unittest.TestCase):
    MaxRangeTree_cls = MaxRangeTree_Tree


class _TestMaxRangeTreeMax(_MixinBase):
    MaxRangeTree_cls: type

    def setUp(self):
        self.t = self.MaxRangeTree_cls(8)

    def test_max_whole_array_initially_zero(self):
        self.assertEqual(self.t.max(0, 8), 0)

    def test_max_single_element(self):
        self.assertEqual(self.t.max(3, 4), 0)

    def test_max_subrange(self):
        self.assertEqual(self.t.max(2, 6), 0)

    def test_max_after_single_increase(self):
        self.t.increase_values(2, 5, 10)
        self.assertEqual(self.t.max(0, 8), 10)
        self.assertEqual(self.t.max(2, 5), 10)
        self.assertEqual(self.t.max(3, 4), 10)

    def test_max_outside_updated_range(self):
        self.t.increase_values(2, 5, 10)
        self.assertEqual(self.t.max(0, 2), 0)
        self.assertEqual(self.t.max(5, 8), 0)

    def test_max_spanning_boundary(self):
        self.t.increase_values(2, 5, 10)
        self.assertEqual(self.t.max(1, 6), 10)
        self.assertEqual(self.t.max(4, 7), 10)

    def test_max_size_one_tree(self):
        t = self.MaxRangeTree_cls(1)
        t.increase_values(0, 1, 42)
        self.assertEqual(t.max(0, 1), 42)


class TestMaxRangeTree_ArrayMax(_TestMaxRangeTreeMax, unittest.TestCase):
    MaxRangeTree_cls = MaxRangeTree_Array


class TestMaxRangeTree_ListMax(_TestMaxRangeTreeMax, unittest.TestCase):
    MaxRangeTree_cls = MaxRangeTree_List


class TestMaxRangeTree_TreeMax(_TestMaxRangeTreeMax, unittest.TestCase):
    MaxRangeTree_cls = MaxRangeTree_Tree


class _TestMaxRangeTreeIncreaseValues(_MixinBase):
    MaxRangeTree_cls: type

    def setUp(self):
        self.t = self.MaxRangeTree_cls(8)

    def test_increase_full_range(self):
        self.t.increase_values(0, 8, 5)
        self.assertEqual(self.t.max(0, 8), 5)

    def test_increase_partial_range(self):
        self.t.increase_values(3, 6, 7)
        self.assertEqual(self.t.max(3, 6), 7)
        self.assertEqual(self.t.max(0, 3), 0)
        self.assertEqual(self.t.max(6, 8), 0)

    def test_increase_single_element(self):
        self.t.increase_values(4, 5, 99)
        self.assertEqual(self.t.max(4, 5), 99)
        self.assertEqual(self.t.max(3, 5), 99)
        self.assertEqual(self.t.max(5, 8), 0)

    def test_increase_non_overlapping_ranges(self):
        self.t.increase_values(0, 3, 5)
        self.t.increase_values(5, 8, 10)
        self.assertEqual(self.t.max(0, 3), 5)
        self.assertEqual(self.t.max(3, 5), 0)
        self.assertEqual(self.t.max(5, 8), 10)
        self.assertEqual(self.t.max(0, 8), 10)

    def test_increase_overlapping_ranges_second_higher(self):
        self.t.increase_values(1, 5, 4)
        self.t.increase_values(3, 7, 9)
        self.assertEqual(self.t.max(1, 3), 4)
        self.assertEqual(self.t.max(3, 5), 9)
        self.assertEqual(self.t.max(5, 7), 9)
        self.assertEqual(self.t.max(0, 8), 9)

    def test_increase_same_range_twice(self):
        self.t.increase_values(2, 6, 3)
        self.t.increase_values(2, 6, 8)
        self.assertEqual(self.t.max(2, 6), 8)

    def test_increase_nested_ranges(self):
        self.t.increase_values(0, 8, 2)
        self.t.increase_values(2, 6, 7)
        self.t.increase_values(3, 5, 15)
        self.assertEqual(self.t.max(0, 2), 2)
        self.assertEqual(self.t.max(2, 3), 7)
        self.assertEqual(self.t.max(3, 5), 15)
        self.assertEqual(self.t.max(5, 6), 7)
        self.assertEqual(self.t.max(6, 8), 2)

    def test_increase_left_boundary(self):
        self.t.increase_values(0, 1, 42)
        self.assertEqual(self.t.max(0, 1), 42)
        self.assertEqual(self.t.max(1, 8), 0)

    def test_increase_right_boundary(self):
        self.t.increase_values(7, 8, 42)
        self.assertEqual(self.t.max(7, 8), 42)
        self.assertEqual(self.t.max(0, 7), 0)


class TestMaxRangeTree_ArrayIncreaseValues(
    _TestMaxRangeTreeIncreaseValues, unittest.TestCase
):
    MaxRangeTree_cls = MaxRangeTree_Array


class TestMaxRangeTree_ListIncreaseValues(
    _TestMaxRangeTreeIncreaseValues, unittest.TestCase
):
    MaxRangeTree_cls = MaxRangeTree_List


class TestMaxRangeTree_TreeIncreaseValues(
    _TestMaxRangeTreeIncreaseValues, unittest.TestCase
):
    MaxRangeTree_cls = MaxRangeTree_Tree


class _TestMaxRangeTreeNonPowerOfTwo(_MixinBase):
    """Test correctness when n is not a power of two."""

    MaxRangeTree_cls: type

    def test_size_5_full_range(self):
        t = self.MaxRangeTree_cls(5)
        t.increase_values(0, 5, 10)
        self.assertEqual(t.max(0, 5), 10)

    def test_size_5_partial(self):
        t = self.MaxRangeTree_cls(5)
        t.increase_values(2, 4, 7)
        self.assertEqual(t.max(0, 2), 0)
        self.assertEqual(t.max(2, 4), 7)
        self.assertEqual(t.max(4, 5), 0)

    def test_size_3(self):
        t = self.MaxRangeTree_cls(3)
        t.increase_values(1, 3, 6)
        self.assertEqual(t.max(0, 1), 0)
        self.assertEqual(t.max(1, 3), 6)
        self.assertEqual(t.max(0, 3), 6)

    def test_size_7_last_element(self):
        t = self.MaxRangeTree_cls(7)
        t.increase_values(6, 7, 50)
        self.assertEqual(t.max(6, 7), 50)
        self.assertEqual(t.max(0, 6), 0)


class TestMaxRangeTree_ArrayNonPowerOfTwo(
    _TestMaxRangeTreeNonPowerOfTwo, unittest.TestCase
):
    MaxRangeTree_cls = MaxRangeTree_Array


class TestMaxRangeTree_ListNonPowerOfTwo(
    _TestMaxRangeTreeNonPowerOfTwo, unittest.TestCase
):
    MaxRangeTree_cls = MaxRangeTree_List


class TestMaxRangeTree_TreeNonPowerOfTwo(
    _TestMaxRangeTreeNonPowerOfTwo, unittest.TestCase
):
    MaxRangeTree_cls = MaxRangeTree_Tree


class _TestMaxRangeTreeLazyPropagation(_MixinBase):
    """Tests specifically targeting lazy propagation correctness."""

    MaxRangeTree_cls: type

    def test_lazy_pushed_down_on_subquery(self):
        # Set the whole range, then query a subrange to trigger push-down
        t = self.MaxRangeTree_cls(8)
        t.increase_values(0, 8, 5)
        self.assertEqual(t.max(0, 4), 5)
        self.assertEqual(t.max(4, 8), 5)
        self.assertEqual(t.max(2, 6), 5)

    def test_multiple_levels_of_lazy(self):
        t = self.MaxRangeTree_cls(8)
        t.increase_values(0, 8, 3)
        t.increase_values(0, 4, 6)
        t.increase_values(0, 2, 9)
        self.assertEqual(t.max(0, 2), 9)
        self.assertEqual(t.max(2, 4), 6)
        self.assertEqual(t.max(4, 8), 3)

    def test_increase_after_full_set_subrange(self):
        t = self.MaxRangeTree_cls(16)
        t.increase_values(0, 16, 1)
        t.increase_values(4, 12, 5)
        t.increase_values(6, 10, 10)
        self.assertEqual(t.max(0, 4), 1)
        self.assertEqual(t.max(4, 6), 5)
        self.assertEqual(t.max(6, 10), 10)
        self.assertEqual(t.max(10, 12), 5)
        self.assertEqual(t.max(12, 16), 1)

    def test_interleaved_increases_and_queries(self):
        t = self.MaxRangeTree_cls(8)
        t.increase_values(0, 8, 2)
        self.assertEqual(t.max(0, 8), 2)
        t.increase_values(2, 6, 5)
        self.assertEqual(t.max(0, 2), 2)
        self.assertEqual(t.max(2, 6), 5)
        t.increase_values(4, 8, 7)
        self.assertEqual(t.max(0, 4), 5)
        self.assertEqual(t.max(4, 8), 7)
        self.assertEqual(t.max(0, 8), 7)


class TestMaxRangeTree_ArrayLazyPropagation(
    _TestMaxRangeTreeLazyPropagation, unittest.TestCase
):
    MaxRangeTree_cls = MaxRangeTree_Array


class TestMaxRangeTree_ListLazyPropagation(
    _TestMaxRangeTreeLazyPropagation, unittest.TestCase
):
    MaxRangeTree_cls = MaxRangeTree_List


class TestMaxRangeTree_TreeLazyPropagation(
    _TestMaxRangeTreeLazyPropagation, unittest.TestCase
):
    MaxRangeTree_cls = MaxRangeTree_Tree


class _TestMaxRangeTreeExhaustive(_MixinBase):
    """Brute-force cross-validation against a plain list for small sizes."""

    MaxRangeTree_cls: type

    def _brute_max(self, arr, left, right):
        return max(arr[left:right])

    def _run_scenario(self, n, ops):
        """ops is a list of (l, r, val) increase_values calls."""
        arr = [0] * n
        t = self.MaxRangeTree_cls(n)
        for left, right, val in ops:
            t.increase_values(left, right, val)
            for i in range(left, right):
                arr[i] = val  # valid since val >= current (enforced by test design)
        for left in range(n):
            for right in range(left + 1, n + 1):
                with self.subTest(n=n, l=left, r=right):
                    self.assertEqual(
                        t.max(left, right), self._brute_max(arr, left, right)
                    )

    def test_size_4_single_op(self):
        self._run_scenario(4, [(1, 3, 5)])

    def test_size_4_two_non_overlapping(self):
        self._run_scenario(4, [(0, 2, 3), (2, 4, 7)])

    def test_size_4_two_overlapping(self):
        self._run_scenario(4, [(0, 3, 4), (1, 4, 9)])

    def test_size_6_nested(self):
        self._run_scenario(6, [(0, 6, 1), (1, 5, 4), (2, 4, 8)])

    def test_size_5_full_coverage(self):
        self._run_scenario(5, [(0, 5, 2), (1, 4, 5), (2, 3, 10)])

    def test_size_8_many_ops(self):
        ops = [
            (0, 8, 1),
            (2, 7, 3),
            (0, 4, 5),
            (3, 6, 7),
            (1, 5, 9),
        ]
        self._run_scenario(8, ops)


class TestMaxRangeTree_ArrayExhaustive(_TestMaxRangeTreeExhaustive, unittest.TestCase):
    MaxRangeTree_cls = MaxRangeTree_Array


class TestMaxRangeTree_ListExhaustive(_TestMaxRangeTreeExhaustive, unittest.TestCase):
    MaxRangeTree_cls = MaxRangeTree_List


class TestMaxRangeTree_TreeExhaustive(_TestMaxRangeTreeExhaustive, unittest.TestCase):
    MaxRangeTree_cls = MaxRangeTree_Tree


def _make_simple_bl() -> BufferList:
    return BufferList.from_buffers([Buffer("A", 10, 0, 5), Buffer("B", 20, 3, 8)])


def _make_inplace_bl() -> BufferList:
    parent = Buffer("P", 10, 0, 3, in_place_children=["C"])
    child = Buffer("C", 10, 3, 6, in_place_parents=["P"])
    return BufferList.from_buffers([parent, child])


class TestOverlaps(unittest.TestCase):
    def test_disjoint(self):
        self.assertFalse(overlaps((0, 3), (5, 8)))

    def test_touching_not_overlapping(self):
        self.assertFalse(overlaps((0, 5), (5, 8)))

    def test_overlapping(self):
        self.assertTrue(overlaps((0, 5), (3, 8)))

    def test_contained(self):
        self.assertTrue(overlaps((2, 6), (1, 8)))

    def test_identical(self):
        self.assertTrue(overlaps((3, 7), (3, 7)))


class TestBuffer(unittest.TestCase):
    def test_construction(self):
        b = Buffer("A", 10, 2, 7)
        self.assertEqual(b.name, "A")
        self.assertEqual(b.size, 10)
        self.assertEqual(b.first_use, 2)
        self.assertEqual(b.last_use, 7)
        self.assertEqual(b.in_place_parents, [])
        self.assertEqual(b.in_place_children, [])

    def test_zero_duration_allowed(self):
        Buffer("A", 10, 3, 3)

    def test_invalid_duration_raises(self):
        with self.assertRaises(AssertionError):
            Buffer("A", 10, 5, 3)

    def test_overlaps_in_time(self):
        self.assertTrue(Buffer("A", 10, 0, 5).overlaps_in_time(Buffer("B", 10, 3, 8)))

    def test_no_overlap_in_time(self):
        self.assertFalse(Buffer("A", 10, 0, 3).overlaps_in_time(Buffer("B", 10, 5, 8)))

    def test_touching_not_overlapping(self):
        self.assertFalse(Buffer("A", 10, 0, 3).overlaps_in_time(Buffer("B", 10, 4, 8)))

    def test_in_place_touching_suppresses_overlap(self):
        parent = Buffer("P", 10, 0, 5, in_place_children=["C"])
        child = Buffer("C", 10, 5, 10, in_place_parents=["P"])
        self.assertFalse(parent.overlaps_in_time(child))
        self.assertFalse(child.overlaps_in_time(parent))

    def test_in_place_non_touching_still_overlaps(self):
        a = Buffer("A", 10, 0, 5, in_place_children=["B"])
        b = Buffer("B", 10, 3, 8, in_place_parents=["A"])
        self.assertTrue(a.overlaps_in_time(b))

    def test_random_produces_valid_buffer(self):
        b = Buffer.random("X", 1000, 50, random.Random(42))
        self.assertIsInstance(b, Buffer)
        self.assertGreaterEqual(b.last_use, b.first_use)

    def test_random_in_place(self):
        parent = Buffer("P", 1000, 0, 10)
        child = Buffer.random(
            "C", parent.size, 50, random.Random(0), [parent], in_place_probability=1.0
        )
        self.assertIn("P", child.in_place_parents)
        self.assertIn("C", parent.in_place_children)
        self.assertEqual(parent.last_use, child.first_use)


class TestBufferList(unittest.TestCase):
    def test_from_buffers(self):
        bl = BufferList.from_buffers([Buffer("A", 10, 0, 5), Buffer("B", 20, 3, 8)])
        self.assertEqual(len(bl), 2)
        self.assertEqual(bl.max_time, 8)
        self.assertEqual(bl._dict["A"], 0)
        self.assertEqual(bl._dict["B"], 1)

    def test_from_buffers_empty(self):
        bl = BufferList.from_buffers([])
        self.assertEqual(len(bl), 0)
        self.assertEqual(bl.max_time, 0)

    def test_iter(self):
        bl = BufferList.from_buffers([Buffer("A", 10, 0, 5), Buffer("B", 20, 3, 8)])
        self.assertEqual([b.name for b in bl], ["A", "B"])

    def test_in_place_mismatch_raises(self):
        bad = Buffer("P", 10, 0, 4, in_place_children=["C"])
        child = Buffer("C", 10, 5, 8, in_place_parents=["P"])  # 5 != 4
        with self.assertRaises(AssertionError):
            BufferList.from_buffers([bad, child])


class TestAllocations(unittest.TestCase):
    def test_non_overlapping_both_at_zero(self):
        bl = BufferList.from_buffers([Buffer("A", 10, 0, 3), Buffer("B", 20, 5, 8)])
        height, alloc = Allocations.from_order(bl, [0, 1])
        self.assertEqual(alloc.addresses[0], 0)
        self.assertEqual(alloc.addresses[1], 0)
        self.assertEqual(height, 20)

    def test_overlapping_stacked(self):
        bl = _make_simple_bl()
        height, alloc = Allocations.from_order(bl, [0, 1])
        self.assertEqual(alloc.addresses[0], 0)
        self.assertEqual(alloc.addresses[1], 10)
        self.assertEqual(height, 30)

    def test_in_place_parent_first(self):
        bl = _make_inplace_bl()
        height, alloc = Allocations.from_order(bl, [0, 1])
        self.assertEqual(alloc.addresses[0], 0)
        self.assertEqual(alloc.addresses[1], 0)
        self.assertEqual(height, 10)

    def test_in_place_child_first(self):
        bl = _make_inplace_bl()
        height, alloc = Allocations.from_order(bl, [1, 0])
        self.assertEqual(alloc.addresses[0], 0)
        self.assertEqual(alloc.addresses[1], 0)
        self.assertEqual(height, 10)

    def test_blocker_raises_height(self):
        parent = Buffer("P", 10, 0, 3, in_place_children=["C"])
        child = Buffer("C", 10, 3, 6, in_place_parents=["P"])
        blocker = Buffer("X", 10, 1, 5)
        bl_clean = BufferList.from_buffers([parent, child])
        bl_blocked = BufferList.from_buffers([parent, child, blocker])
        h_clean, _ = Allocations.from_order(bl_clean, [0, 1])
        h_blocked, _ = Allocations.from_order(bl_blocked, [2, 0, 1])
        self.assertEqual(h_clean, 10)
        self.assertGreater(h_blocked, h_clean)

    def test_in_place_chain(self):
        p = Buffer("P", 10, 0, 3, in_place_children=["B"])
        b = Buffer("B", 10, 3, 6, in_place_parents=["P"], in_place_children=["C"])
        c = Buffer("C", 10, 6, 9, in_place_parents=["B"])
        bl = BufferList.from_buffers([p, b, c])
        height, alloc = Allocations.from_order(bl, [0, 2, 1])
        self.assertEqual(alloc.addresses[0], 0)
        self.assertEqual(alloc.addresses[1], 0)
        self.assertEqual(alloc.addresses[2], 0)
        self.assertEqual(height, 10)

    def test_partial_order(self):
        bl = _make_simple_bl()
        height, alloc = Allocations.from_order(bl, [0])
        self.assertEqual(alloc.addresses[0], 0)
        self.assertEqual(height, 10)

    def test_order_too_long_raises(self):
        bl = _make_simple_bl()
        with self.assertRaises(ValueError):
            Allocations.from_order(bl, [0, 1, 2])

    def test_to_order_sorted_by_address(self):
        bl = _make_simple_bl()
        _, alloc = Allocations.from_order(bl, [0, 1])
        order = alloc.to_order()
        addrs = [alloc.addresses[i] for i in order]
        self.assertEqual(addrs, sorted(addrs))


class TestAllocationBuilder(unittest.TestCase):
    def test_matches_from_order(self):
        bl = _make_simple_bl()
        height_ref, alloc_ref = Allocations.from_order(bl, [0, 1])
        builder = AllocationBuilder(bl)
        builder.emplace(0)
        builder.emplace(1)
        self.assertEqual(builder.peak_height(), height_ref)
        self.assertEqual(builder.build().addresses, alloc_ref.addresses)

    def test_emplace_returns_address(self):
        bl = _make_simple_bl()
        builder = AllocationBuilder(bl)
        self.assertEqual(builder.emplace(0), 0)
        self.assertEqual(builder.emplace(1), 10)

    def test_peak_height_increments(self):
        bl = _make_simple_bl()
        builder = AllocationBuilder(bl)
        builder.emplace(0)
        self.assertEqual(builder.peak_height(), 10)
        builder.emplace(1)
        self.assertEqual(builder.peak_height(), 30)

    def test_matches_from_order_in_place(self):
        bl = _make_inplace_bl()
        height_ref, alloc_ref = Allocations.from_order(bl, [0, 1])
        builder = AllocationBuilder(bl)
        builder.emplace(0)
        builder.emplace(1)
        self.assertEqual(builder.peak_height(), height_ref)
        self.assertEqual(builder.build().addresses, alloc_ref.addresses)


class TestExponentialCoolingSchedule(unittest.TestCase):
    def _temps(self, s):
        return list(iter(s))

    def test_starts_at_t0(self):
        s = ExponentialCoolingSchedule(
            t0=100.0, t_end=10.0, steps_per_epoch=2, epochs=3
        )
        self.assertAlmostEqual(self._temps(s)[0], 100.0)

    def test_monotone_decreasing(self):
        temps = self._temps(
            ExponentialCoolingSchedule(t0=100.0, t_end=1.0, steps_per_epoch=2, epochs=4)
        )
        for i in range(len(temps) - 1):
            self.assertLessEqual(temps[i + 1], temps[i])

    def test_total_steps(self):
        s = ExponentialCoolingSchedule(
            t0=100.0, t_end=10.0, steps_per_epoch=3, epochs=4
        )
        self.assertEqual(len(self._temps(s)), 3 * 4 - 1)

    def test_iter_returns_independent_copy(self):
        s = ExponentialCoolingSchedule(
            t0=100.0, t_end=10.0, steps_per_epoch=2, epochs=3
        )
        self.assertEqual(self._temps(s), self._temps(s))

    def test_epoch_temperature_drop(self):
        s = ExponentialCoolingSchedule(
            t0=100.0, t_end=10.0, steps_per_epoch=2, epochs=2
        )
        temps = self._temps(s)
        alpha = (10.0 / 100.0) ** 0.5
        self.assertAlmostEqual(temps[0], 100.0)
        self.assertAlmostEqual(temps[1], 100.0 * alpha)
        self.assertAlmostEqual(temps[2], 100.0 * alpha)


class TestCoolingScheduleFromPaper(unittest.TestCase):
    def _make_bl(self):
        return BufferList.from_buffers(
            [Buffer("A", 30000, 0, 5), Buffer("B", 30000, 3, 8)]
        )

    def test_yields_n_temperatures(self):
        s = CoolingScheduleFromPaper(buffers=self._make_bl(), n=10)
        self.assertEqual(len(list(iter(s))), 10)

    def test_monotone_decreasing(self):
        temps = list(iter(CoolingScheduleFromPaper(buffers=self._make_bl(), n=100)))
        self.assertGreater(temps[0], temps[-1])

    def test_all_positive(self):
        for t in iter(CoolingScheduleFromPaper(buffers=self._make_bl(), n=10)):
            self.assertGreater(t, 0)

    def test_iter_returns_independent_copy(self):
        s = CoolingScheduleFromPaper(buffers=self._make_bl(), n=5)
        self.assertEqual(list(iter(s)), list(iter(s)))


class TestFirstFitAllMinus(unittest.TestCase):
    def test_interval_in_middle(self):
        self.assertEqual(
            FirstFit.all_minus([(0, 100)], (30, 60), 10), [(0, 30), (60, 100)]
        )

    def test_interval_at_left(self):
        self.assertEqual(FirstFit.all_minus([(0, 100)], (0, 50), 10), [(50, 100)])

    def test_interval_at_right(self):
        self.assertEqual(FirstFit.all_minus([(0, 100)], (50, 100), 10), [(0, 50)])

    def test_minimum_size_filters_small_piece(self):
        # left piece (0, 5) is too small; right piece (15, 100) is kept
        self.assertEqual(FirstFit.all_minus([(0, 100)], (5, 15), 10), [(15, 100)])

    def test_gap_entirely_left_of_interval(self):
        self.assertEqual(FirstFit.all_minus([(0, 20)], (30, 60), 5), [(0, 20)])

    def test_gap_entirely_right_of_interval(self):
        self.assertEqual(FirstFit.all_minus([(60, 100)], (0, 30), 5), [(60, 100)])

    def test_multiple_gaps(self):
        self.assertEqual(
            FirstFit.all_minus([(0, 50), (70, 100)], (20, 80), 5),
            [(0, 20), (80, 100)],
        )

    def test_empty_intervals(self):
        self.assertEqual(FirstFit.all_minus([], (10, 20), 5), [])


class TestFirstFit(unittest.TestCase):
    def test_non_overlapping(self):
        buffers = [Buffer("A", 10, 0, 3), Buffer("B", 10, 5, 8)]
        ff = FirstFit(buffers)
        bl = BufferList.from_buffers(buffers)
        height, _ = Allocations.from_order(bl, ff.order)
        self.assertLessEqual(height, 10)

    def test_overlapping(self):
        buffers = [Buffer("A", 10, 0, 5), Buffer("B", 10, 3, 8)]
        ff = FirstFit(buffers)
        bl = BufferList.from_buffers(buffers)
        height, _ = Allocations.from_order(bl, ff.order)
        self.assertEqual(height, 20)

    def test_order_is_permutation(self):
        buffers = [Buffer(f"B{i}", 10, i, i + 3) for i in range(5)]
        ff = FirstFit(buffers)
        self.assertEqual(sorted(ff.order), list(range(5)))


class TestBestFit(unittest.TestCase):
    def test_non_overlapping(self):
        buffers = [Buffer("A", 10, 0, 3), Buffer("B", 10, 5, 8)]
        bf = BestFit(buffers)
        bl = BufferList.from_buffers(buffers)
        height, _ = Allocations.from_order(bl, bf.order)
        self.assertLessEqual(height, 10)

    def test_overlapping(self):
        buffers = [Buffer("A", 10, 0, 5), Buffer("B", 10, 3, 8)]
        bf = BestFit(buffers)
        bl = BufferList.from_buffers(buffers)
        height, _ = Allocations.from_order(bl, bf.order)
        self.assertEqual(height, 20)

    def test_order_is_permutation(self):
        buffers = [Buffer(f"B{i}", 10, i, i + 3) for i in range(5)]
        bf = BestFit(buffers)
        self.assertEqual(sorted(bf.order), list(range(5)))


class _AlwaysRandom(random.Random):
    """Deterministic stub: randrange always returns val % n, random() always returns 0.0."""

    def __init__(self, val: int) -> None:
        super().__init__()
        self._val = val

    def randrange(self, start, stop=None, step=1):  # type: ignore[override]
        return self._val % start  # ImanishiXuAllocator only calls randrange(n)

    def random(self) -> float:  # type: ignore[override]
        return 0.0


class TestImanishiXuAllocator(unittest.TestCase):
    def _make_inplace3(self) -> BufferList:
        return BufferList.from_buffers(
            [
                Buffer("P", 10, 0, 5, in_place_children=["C"]),
                Buffer("C", 10, 5, 10, in_place_parents=["P"]),
                Buffer("X", 15, 0, 10),
            ]
        )

    def _short_schedule(self) -> ExponentialCoolingSchedule:
        return ExponentialCoolingSchedule(
            t0=10.0, t_end=1.0, steps_per_epoch=2, epochs=2
        )

    def test_constructor_computes_best_height(self):
        bl = _make_simple_bl()
        allocator = ImanishiXuAllocator(
            buffers=bl, order=[0, 1], schedule=self._short_schedule()
        )
        self.assertEqual(allocator.best_height, 30)

    def test_annealing_step_rotate_single_buffer_returns_none(self):
        bl = BufferList.from_buffers([Buffer("A", 10, 0, 5)])
        allocator = ImanishiXuAllocator(
            buffers=bl, order=[0], schedule=self._short_schedule()
        )
        self.assertIsNone(allocator.annealing_step_rotate(100.0))

    def test_annealing_step_rotate_parent_discovers_inplace_with_child(self):
        """Rotating P from position 2 to position 0 in [C, X, P] gives [P, C, X].
        C is then placed after P and discovers in-place reuse with parent P,
        reducing height from 35 to 25."""
        bl = self._make_inplace3()
        # order [C, X, P] = [1, 2, 0]: height 35, no in-place used
        allocator = ImanishiXuAllocator(
            buffers=bl,
            order=[1, 2, 0],
            schedule=self._short_schedule(),
            ordering_fuzz_factor=0.0,
            random=_AlwaysRandom(2),  # always pick i=2 (P at position 2)
        )
        result = allocator.annealing_step_rotate(1.0)
        # Sort key is -htb[j]: htb=[35,25,35] → sorted ascending gives j=0 first (j=2 is
        # skipped as ==i), improvement=0 accepted via Boltzmann exp(0)=1.
        self.assertEqual(result, (2, 0))
        # new order [P, C, X] = [0, 1, 2]: C is placed after P and discovers in-place
        self.assertEqual(allocator.order, [0, 1, 2])
        height, _ = Allocations.from_order(bl, allocator.order)
        self.assertEqual(height, 25)

    def test_annealing_step_rotate_child_discovers_inplace_with_parent(self):
        """Rotating X from position 1 to position 0 in [P, X, C] yields [X, P, C],
        where C is placed last and discovers in-place reuse with parent P."""
        bl = self._make_inplace3()
        # order [P, X, C] = [0, 2, 1]: height 35, no in-place used
        allocator = ImanishiXuAllocator(
            buffers=bl,
            order=[0, 2, 1],
            schedule=self._short_schedule(),
            ordering_fuzz_factor=0.0,
            random=_AlwaysRandom(1),  # always pick i=1 (X at position 1)
        )
        result = allocator.annealing_step_rotate(1.0)
        self.assertEqual(result, (1, 0))
        # new order [X, P, C] = [2, 0, 1]: C is placed last and finds P as parent
        self.assertEqual(allocator.order, [2, 0, 1])
        height, _ = Allocations.from_order(bl, allocator.order)
        self.assertEqual(height, 25)

    def test_solve_discovers_inplace(self):
        """solve() starting from a suboptimal order should reach best_height 25."""
        bl = self._make_inplace3()
        schedule = ExponentialCoolingSchedule(
            t0=100.0, t_end=0.1, steps_per_epoch=10, epochs=10
        )
        # order [C, X, P] = [1, 2, 0]: height 35
        allocator = ImanishiXuAllocator(
            buffers=bl,
            order=[1, 2, 0],
            schedule=schedule,
            ordering_fuzz_factor=0.0,
            random=random.Random(0),
        )
        allocator.solve()
        self.assertEqual(allocator.best_height, 25)


if __name__ == "__main__":
    unittest.main()
