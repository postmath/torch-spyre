# Importing torch appears to be necessary in order to import some torch_spyre members.
import torch  # noqa: F401
from torch_spyre._inductor.imanishi_xu import MaxRangeTree_Tree
import unittest
import random


class TestMaxRangeTree_Tree(unittest.TestCase):
    def test_max_16(self):
        m = MaxRangeTree_Tree(16)
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
        m = MaxRangeTree_Tree(31)
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


class TestMaxRangeTree_TreeInit(unittest.TestCase):
    def test_single_element(self):
        t = MaxRangeTree_Tree(1)
        self.assertEqual(t.max(0, 1), 0)

    def test_power_of_two_size(self):
        t = MaxRangeTree_Tree(8)
        self.assertEqual(t.max(0, 8), 0)

    def test_non_power_of_two_size(self):
        t = MaxRangeTree_Tree(5)
        self.assertEqual(t.max(0, 5), 0)

    def test_zero_assert(self):
        with self.assertRaises(AssertionError):
            MaxRangeTree_Tree(0)

    def test_large_size(self):
        t = MaxRangeTree_Tree(1000)
        self.assertEqual(t.max(0, 1000), 0)


class TestMaxRangeTree_TreeMax(unittest.TestCase):
    def setUp(self):
        self.t = MaxRangeTree_Tree(8)

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
        t = MaxRangeTree_Tree(1)
        t.increase_values(0, 1, 42)
        self.assertEqual(t.max(0, 1), 42)


class TestMaxRangeTree_TreeIncreaseValues(unittest.TestCase):
    def setUp(self):
        self.t = MaxRangeTree_Tree(8)

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


class TestMaxRangeTree_TreeNonPowerOfTwo(unittest.TestCase):
    """Test correctness when n is not a power of two."""

    def test_size_5_full_range(self):
        t = MaxRangeTree_Tree(5)
        t.increase_values(0, 5, 10)
        self.assertEqual(t.max(0, 5), 10)

    def test_size_5_partial(self):
        t = MaxRangeTree_Tree(5)
        t.increase_values(2, 4, 7)
        self.assertEqual(t.max(0, 2), 0)
        self.assertEqual(t.max(2, 4), 7)
        self.assertEqual(t.max(4, 5), 0)

    def test_size_3(self):
        t = MaxRangeTree_Tree(3)
        t.increase_values(1, 3, 6)
        self.assertEqual(t.max(0, 1), 0)
        self.assertEqual(t.max(1, 3), 6)
        self.assertEqual(t.max(0, 3), 6)

    def test_size_7_last_element(self):
        t = MaxRangeTree_Tree(7)
        t.increase_values(6, 7, 50)
        self.assertEqual(t.max(6, 7), 50)
        self.assertEqual(t.max(0, 6), 0)


class TestMaxRangeTree_TreeLazyPropagation(unittest.TestCase):
    """Tests specifically targeting lazy propagation correctness."""

    def test_lazy_pushed_down_on_subquery(self):
        # Set the whole range, then query a subrange to trigger push-down
        t = MaxRangeTree_Tree(8)
        t.increase_values(0, 8, 5)
        self.assertEqual(t.max(0, 4), 5)
        self.assertEqual(t.max(4, 8), 5)
        self.assertEqual(t.max(2, 6), 5)

    def test_multiple_levels_of_lazy(self):
        t = MaxRangeTree_Tree(8)
        t.increase_values(0, 8, 3)
        t.increase_values(0, 4, 6)
        t.increase_values(0, 2, 9)
        self.assertEqual(t.max(0, 2), 9)
        self.assertEqual(t.max(2, 4), 6)
        self.assertEqual(t.max(4, 8), 3)

    def test_increase_after_full_set_subrange(self):
        t = MaxRangeTree_Tree(16)
        t.increase_values(0, 16, 1)
        t.increase_values(4, 12, 5)
        t.increase_values(6, 10, 10)
        self.assertEqual(t.max(0, 4), 1)
        self.assertEqual(t.max(4, 6), 5)
        self.assertEqual(t.max(6, 10), 10)
        self.assertEqual(t.max(10, 12), 5)
        self.assertEqual(t.max(12, 16), 1)

    def test_interleaved_increases_and_queries(self):
        t = MaxRangeTree_Tree(8)
        t.increase_values(0, 8, 2)
        self.assertEqual(t.max(0, 8), 2)
        t.increase_values(2, 6, 5)
        self.assertEqual(t.max(0, 2), 2)
        self.assertEqual(t.max(2, 6), 5)
        t.increase_values(4, 8, 7)
        self.assertEqual(t.max(0, 4), 5)
        self.assertEqual(t.max(4, 8), 7)
        self.assertEqual(t.max(0, 8), 7)


class TestMaxRangeTree_TreeExhaustive(unittest.TestCase):
    """Brute-force cross-validation against a plain list for small sizes."""

    def _brute_max(self, arr, left, right):
        return max(arr[left:right])

    def _run_scenario(self, n, ops):
        """ops is a list of (l, r, val) increase_values calls."""
        arr = [0] * n
        t = MaxRangeTree_Tree(n)
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


if __name__ == "__main__":
    unittest.main()
