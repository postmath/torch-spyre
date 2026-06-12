# Copyright 2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the capacity-bounded allocation plans."""

import itertools
import os
import random
import unittest
from unittest import TestCase
from typing import TYPE_CHECKING

from torch_spyre._inductor.scratchpad.plan_solver import (
    LifetimeBoundBuffer,
)
from torch_spyre._inductor.scratchpad.permutation_layout import (
    PermutationBasedLayoutSolver,
    Profile,
    ReferencePermutationBasedLayoutSolver,
)

ALIGNMENT = 128

# Exhaustive randomized differential runs: thousands of seeds, larger problems,
# dense in-place wiring. Skipped by default (slow); opt in with the env var.
_STRESS = os.environ.get("TORCH_SPYRE_STRESS_SCRATCHPAD") == "1"


def _random_buffers(rng, n, horizon=12, max_size=200, inplace_prob=0.25):
    """Generate ``n`` random buffers, occasionally wiring in-place pairs.

    Lifetimes are half-open ``[start, end)`` and non-empty (end > start).
    """
    buffers = []
    for i in range(n):
        start = rng.randint(0, horizon)
        end = rng.randint(start + 1, horizon + 1)
        size = rng.randint(1, max_size)
        buffers.append(_buf(f"b{i}", size, start, end))
    # Turn a few buffers into in-place children of an earlier buffer: the child
    # starts at the parent's last live tick (parent.end - 1, so that
    # parent.end == child.start + 1) and clamps its size to fit.
    for child_i in range(1, n):
        if rng.random() < inplace_prob:
            parent_i = rng.randrange(child_i)
            parent = buffers[parent_i]
            child = buffers[child_i]
            child.start_time = parent.end_time - 1
            child.end_time = max(child.end_time, parent.end_time)
            child.size = rng.randint(1, parent.size)
            child.in_place_parents = [parent.name]
    return buffers


def _check_consistency(test, plan, tag=""):
    """Verify the live contact profiles: each is canonical and spans its
    buffer's lifetime, and the reverse-pointer invariant holds in both
    directions (iterating segments, not sampling columns):
    ``below_profile[x] == z`` over a segment iff ``above_profile[z] == x`` there.
    """
    n = len(plan.buffers)
    for i in range(n):
        for prof in (plan.below_profile[i], plan.above_profile[i]):
            prof.validate()
            test.assertEqual(prof.span_start, plan.buffers[i].start_time, tag)
            test.assertEqual(prof.span_end, plan.buffers[i].end_time, tag)
    for primary, reverse in (
        (plan.below_profile, plan.above_profile),
        (plan.above_profile, plan.below_profile),
    ):
        for x in range(n):
            p = primary[x]
            for k, z in enumerate(p.labels):
                if z is None:
                    continue
                a, b = p.starts[k], p.starts[k + 1]
                _, rev_labels = reverse[z].segments(a, b)
                for lab in rev_labels:
                    test.assertEqual(lab, x, f"{tag} reverse-pointer x={x} z={z}")


def _check_contact_faithful(test, plan, tag=""):
    """Verify ``contact_at(c, t)`` describes the slot directly below ``c`` at
    every column ``t``. In all cases ``c`` rests on the named buffer (the
    ``parent`` of a tuple), i.e. the max-top among ``c``'s earlier-positioned
    candidates alive there. A ``(parent, child)`` tuple appears iff that slot is
    shared by an in-place pair at its transition column: ``parent`` and ``child``
    are co-located (same address), both alive at ``t``, linked by reuse, and
    ``c``'s order-below neighbour is the higher-positioned of the two (which one
    that is depends on the reuse direction).
    """
    n = len(plan.buffers)
    pos = plan.position

    def top(w):
        return plan.addresses[w] + plan.buffers[w].size

    def alive(w):
        return plan.buffers[w].start_time <= t < plan.buffers[w].end_time

    for c in range(n):
        bc = plan.buffers[c]
        for t in range(bc.start_time, bc.end_time):
            contact = plan.contact_at(c, t)
            cand = [w for w in plan.overlaps[c] if pos[w] < pos[c] and alive(w)]
            if not cand:
                test.assertIsNone(contact, f"{tag} c={c} t={t}")
                continue
            max_top = max(top(w) for w in cand)
            if isinstance(contact, tuple):
                parent, child = contact
                # c rests on the parent (the larger, max-top co-located buffer).
                test.assertEqual(top(parent), max_top, f"{tag} c={c} t={t} parent")
                # parent/child are an in-place pair, co-located, both alive at t,
                # linked by reuse, and one of them is c's order-below neighbour.
                test.assertEqual(
                    plan._in_place_pair(parent, child),
                    (parent, child),
                    f"{tag} c={c} t={t} pair",
                )
                test.assertEqual(
                    plan.addresses[parent], plan.addresses[child], f"{tag} colocated"
                )
                test.assertTrue(alive(parent) and alive(child), f"{tag} both-alive")
                test.assertTrue(
                    plan.inplace_reuse.get(parent) == child
                    or plan.inplace_reuse.get(child) == parent,
                    f"{tag} c={c} t={t} reuse-link",
                )
                test.assertIn(
                    plan.below_profile[c].label_at(t),
                    (parent, child),
                    f"{tag} c={c} t={t} order-below",
                )
            else:
                test.assertEqual(top(contact), max_top, f"{tag} c={c} t={t} int")


def _named_segments(plan, profile):
    """A profile as a readable list of (start, end, neighbour-name-or-None)."""
    return [
        (
            profile.starts[k],
            profile.starts[k + 1],
            None if profile.labels[k] is None else plan.buffers[profile.labels[k]].name,
        )
        for k in range(len(profile.labels))
    ]


def _below_named(plan, name):
    return _named_segments(plan, plan.below_profile[plan._name_to_idx[name]])


def _above_named(plan, name):
    return _named_segments(plan, plan.above_profile[plan._name_to_idx[name]])


def _buf(name, size, start, end, in_place_parents=None):
    return LifetimeBoundBuffer(
        name=name,
        size=size,
        start_time=start,
        end_time=end,
        in_place_parents=in_place_parents or [],
    )


# Both concrete plans share PermutationBasedLayoutSolverBase, so the Step 1 skeleton
# behaviour (field setup, helpers, finalize) is identical and tested for both.
if TYPE_CHECKING:
    MixinBase = TestCase
else:
    MixinBase = object


class SkeletonTestsMixin(MixinBase):
    plan_class: type = None  # type: ignore[assignment]

    def make_plan(self, buffers, permutation, capacity, alignment=ALIGNMENT):
        return self.plan_class(buffers, permutation, capacity, alignment)

    def test_init_stores_fields(self):
        buffers = [_buf("a", 64, 0, 2), _buf("b", 64, 1, 3)]
        plan = self.make_plan(buffers, [1, 0], capacity=256, alignment=ALIGNMENT)

        self.assertIs(plan.buffers, buffers)
        self.assertEqual(plan.permutation, [1, 0])
        self.assertEqual(plan.capacity, 256)
        self.assertEqual(plan.alignment, ALIGNMENT)
        self.assertEqual(plan._name_to_idx, {"a": 0, "b": 1})
        # addresses has one slot per buffer; its contents depend on _build.
        self.assertEqual(len(plan.addresses), 2)

    def test_permutation_is_copied(self):
        buffers = [_buf("a", 64, 0, 1)]
        perm = [0]
        plan = self.make_plan(buffers, perm, capacity=128)
        perm.append(99)
        self.assertEqual(plan.permutation, [0])

    def test_invalid_permutation_rejected(self):
        buffers = [_buf("a", 64, 0, 1), _buf("b", 64, 0, 1)]
        with self.assertRaises(AssertionError):
            self.make_plan(buffers, [0, 0], capacity=128)
        with self.assertRaises(AssertionError):
            self.make_plan(buffers, [0], capacity=128)

    def test_align_up(self):
        buffers = [_buf("a", 64, 0, 1)]
        plan = self.make_plan(buffers, [0], capacity=128, alignment=128)
        self.assertEqual(plan._align_up(0), 0)
        self.assertEqual(plan._align_up(1), 128)
        self.assertEqual(plan._align_up(128), 128)
        self.assertEqual(plan._align_up(129), 256)

    def test_top(self):
        buffers = [_buf("a", 64, 0, 1)]
        plan = self.make_plan(buffers, [0], capacity=256)
        plan.addresses[0] = 128
        self.assertEqual(plan._top(0), 192)

    def test_is_fully_allocated(self):
        buffers = [_buf("a", 64, 0, 1)]
        plan = self.make_plan(buffers, [0], capacity=100)
        # Fits exactly at the boundary.
        plan.addresses[0] = 36
        self.assertTrue(plan._is_fully_allocated(0))
        # One byte over capacity.
        plan.addresses[0] = 37
        self.assertFalse(plan._is_fully_allocated(0))

    def test_quality_accessor(self):
        buffers = [_buf("a", 64, 0, 1)]
        plan = self.make_plan(buffers, [0], capacity=128)
        plan.total_allocated_size = 42
        plan.total_allocated_count = 3
        self.assertEqual(plan.quality(), 42)
        self.assertEqual(plan.count_allocated(), 3)

    def test_finalize_writes_back_only_fully_allocated(self):
        buffers = [
            _buf("fits", 64, 0, 1),
            _buf("over_cap", 64, 0, 1),
            _buf("also_over", 64, 0, 1),
        ]
        plan = self.make_plan(buffers, [0, 1, 2], capacity=128)
        # Every buffer has a notional address; only those fitting below capacity
        # are committed. 100 + 64 = 164 > 128 and 200 + 64 = 264 > 128.
        plan.addresses = [0, 100, 200]

        plan.finalize()

        self.assertEqual(buffers[0].address, 0)
        self.assertIsNone(buffers[1].address)
        self.assertIsNone(buffers[2].address)


class ReferenceSolverSkeletonTests(SkeletonTestsMixin, TestCase):
    plan_class = ReferencePermutationBasedLayoutSolver


class PermutationBasedLayoutSolverSkeletonTests(SkeletonTestsMixin, TestCase):
    plan_class = PermutationBasedLayoutSolver


def _addr(plan, name):
    return plan.addresses[plan._name_to_idx[name]]


class ReferencePlacementTests(TestCase):
    """Step 2: O(n^2) placement in ReferencePermutationBasedLayoutSolver."""

    def plan(self, buffers, permutation, capacity=10_000, alignment=1):
        return ReferencePermutationBasedLayoutSolver(
            buffers, permutation, capacity, alignment
        )

    def test_disjoint_lifetimes_all_at_zero(self):
        # No two buffers are ever alive together, so each reuses address 0.
        buffers = [_buf("a", 64, 0, 1), _buf("b", 64, 2, 3), _buf("c", 64, 4, 5)]
        plan = self.plan(buffers, [0, 1, 2])
        self.assertEqual([_addr(plan, n) for n in "abc"], [0, 0, 0])
        self.assertEqual(plan.quality(), 192)

    def test_overlapping_lifetimes_stack(self):
        buffers = [_buf("a", 64, 0, 2), _buf("b", 50, 1, 3)]
        plan = self.plan(buffers, [0, 1])
        self.assertEqual(_addr(plan, "a"), 0)
        self.assertEqual(_addr(plan, "b"), 64)  # stacked on top of a
        self.assertEqual(plan.quality(), 114)

    def test_permutation_order_changes_layout(self):
        buffers = [_buf("a", 64, 0, 2), _buf("b", 50, 1, 3)]
        plan = self.plan(buffers, [1, 0])  # b placed first
        self.assertEqual(_addr(plan, "b"), 0)
        self.assertEqual(_addr(plan, "a"), 50)  # a stacked on top of b

    def test_alignment_rounds_up(self):
        buffers = [_buf("a", 64, 0, 2), _buf("b", 64, 1, 3)]
        plan = self.plan(buffers, [0, 1], alignment=128)
        self.assertEqual(_addr(plan, "a"), 0)
        self.assertEqual(_addr(plan, "b"), 128)  # ceil(64/128)*128

    def test_freed_low_space_reused_by_later_disjoint_buffer(self):
        # a dies before c starts; c does not overlap a, so it drops back to 0
        # even though b (overlapping both) sits above.
        buffers = [
            _buf("a", 64, 0, 2),
            _buf("b", 64, 1, 5),
            _buf("c", 64, 3, 5),
        ]
        plan = self.plan(buffers, [0, 1, 2])
        self.assertEqual(_addr(plan, "a"), 0)
        self.assertEqual(_addr(plan, "b"), 64)
        # c overlaps b (not a) -> stacks only on b.
        self.assertEqual(_addr(plan, "c"), 128)

    def test_in_place_child_reuses_parent_address(self):
        parent = _buf("p", 128, 0, 5)
        child = _buf("c", 64, 4, 10, in_place_parents=["p"])
        plan = self.plan([parent, child], [0, 1])
        self.assertEqual(_addr(plan, "p"), 0)
        self.assertEqual(_addr(plan, "c"), 0)  # reuses parent's address
        self.assertEqual(plan.quality(), 192)

    def test_in_place_parent_placed_after_child_reuses_address(self):
        # Symmetric case: child allocated first, parent reuses its address.
        parent = _buf("p", 128, 0, 5)
        child = _buf("c", 64, 4, 10, in_place_parents=["p"])
        plan = self.plan([parent, child], [1, 0])  # child first
        self.assertEqual(_addr(plan, "c"), 0)
        self.assertEqual(_addr(plan, "p"), 0)  # parent reuses child's address

    def test_in_place_blocked_when_child_larger_than_parent(self):
        parent = _buf("p", 64, 0, 5)
        child = _buf("c", 128, 4, 10, in_place_parents=["p"])
        plan = self.plan([parent, child], [0, 1])
        self.assertEqual(_addr(plan, "p"), 0)
        self.assertEqual(_addr(plan, "c"), 64)  # cannot reuse; stacks on top

    def test_in_place_blocked_by_intruding_buffer(self):
        # The collision case from the design discussion: Z coexists with the
        # child but not the parent, so reusing the parent's address would
        # overlap Z. Placement must fall back to stacking.
        parent = _buf("p", 50, 0, 5)
        child = _buf("c", 30, 4, 10, in_place_parents=["p"])
        z = _buf("z", 20, 6, 10)
        plan = self.plan([parent, child, z], [0, 2, 1])  # order: p, z, c
        self.assertEqual(_addr(plan, "p"), 0)
        self.assertEqual(_addr(plan, "z"), 0)  # z does not overlap p
        # c overlaps both p (top 50) and z (top 20); p is topmost and is the
        # in-place partner, but reusing addr 0 would hit z -> stack at 50.
        self.assertEqual(_addr(plan, "c"), 50)

    def test_over_capacity_buffer_not_counted_but_still_addressed(self):
        # b stacks above a and crosses the capacity line: it keeps an address
        # (so later buffers stack on it) but is excluded from total_size.
        buffers = [_buf("a", 64, 0, 3), _buf("b", 64, 1, 3)]
        plan = self.plan(buffers, [0, 1], capacity=100)
        self.assertEqual(_addr(plan, "a"), 0)
        self.assertEqual(_addr(plan, "b"), 64)  # 64 + 64 = 128 > 100
        self.assertEqual(plan.quality(), 64)  # only a counts

    def test_finalize_after_build(self):
        buffers = [_buf("a", 64, 0, 3), _buf("b", 64, 1, 3)]
        plan = self.plan(buffers, [0, 1], capacity=100)
        plan.finalize()
        self.assertEqual(buffers[0].address, 0)
        self.assertIsNone(buffers[1].address)  # over capacity, not committed


class ContactProfileTests(TestCase):
    """Order-based contact profiles in PermutationBasedLayoutSolver."""

    def plan(self, buffers, permutation, capacity=10_000, alignment=1):
        return PermutationBasedLayoutSolver(buffers, permutation, capacity, alignment)

    def test_simple_stack(self):
        buffers = [_buf("a", 64, 0, 3), _buf("b", 50, 1, 3)]
        plan = self.plan(buffers, [0, 1])
        self.assertEqual(_below_named(plan, "b"), [(1, 3, "a")])
        self.assertEqual(_above_named(plan, "a"), [(0, 1, None), (1, 3, "b")])
        self.assertEqual(_below_named(plan, "a"), [(0, 3, None)])
        self.assertEqual(_above_named(plan, "b"), [(1, 3, None)])

    def test_air_gap_neighbor(self):
        # low spans the whole range; tall is between low and high in order over
        # [0,3); after tall dies, high's below-neighbour is low (over an address
        # air gap -- the order-based relation does not care about the gap).
        low = _buf("low", 64, 0, 10)
        tall = _buf("tall", 256, 0, 3)
        high = _buf("high", 64, 0, 10)
        plan = self.plan([low, tall, high], [0, 1, 2])
        self.assertEqual(_below_named(plan, "high"), [(0, 3, "tall"), (3, 10, "low")])

    def test_in_place_pair_ordered_by_position(self):
        # Parent before child in the permutation -> parent below child over the
        # shared boundary tick, even though the child reuses the parent's slot.
        parent = _buf("p", 128, 0, 5)
        child = _buf("c", 64, 4, 10, in_place_parents=["p"])
        plan = self.plan([parent, child], [0, 1])
        self.assertEqual(_below_named(plan, "c"), [(4, 5, "p"), (5, 10, None)])
        self.assertEqual(_above_named(plan, "p"), [(0, 4, None), (4, 5, "c")])

    def test_contact_at_floor_and_plain_stack(self):
        # a is on the floor; b rests on a (plain int, no in-placement).
        buffers = [_buf("a", 64, 0, 3), _buf("b", 50, 1, 3)]
        plan = self.plan(buffers, [0, 1])
        a, b = plan._name_to_idx["a"], plan._name_to_idx["b"]
        self.assertIsNone(plan.contact_at(a, 0))
        self.assertIsNone(plan.contact_at(a, 2))
        self.assertEqual(plan.contact_at(b, 1), a)
        self.assertEqual(plan.contact_at(b, 2), a)

    def test_contact_at_poke_through_tuple(self):
        # Order p, c, h. c in-places onto p; p (taller) pokes through, so h
        # rests on p at the shared tick (4) and is reported as (p, c). After p
        # dies (t >= 5) h rests on the now-plain child c.
        p = _buf("p", 128, 0, 5)
        c = _buf("c", 64, 4, 10, in_place_parents=["p"])
        h = _buf("h", 32, 4, 10)
        plan = self.plan([p, c, h], [0, 1, 2])
        ip, ic, ih = (plan._name_to_idx[n] for n in ("p", "c", "h"))
        self.assertEqual(_addr(plan, "c"), 0)  # reuses p
        self.assertEqual(_addr(plan, "h"), 128)  # rests on p, not on c
        self.assertEqual(plan.contact_at(ih, 4), (ip, ic))  # poke-through tuple
        self.assertEqual(plan.contact_at(ih, 5), ic)  # p gone -> plain child
        self.assertEqual(plan.contact_at(ih, 9), ic)
        self.assertEqual(plan.contact_at(ic, 4), ip)  # c's own contact is p
        self.assertIsNone(plan.contact_at(ip, 0))  # p on the floor

    def test_contact_at_parent_reused_child_tuple(self):
        # The other reuse direction: the child is placed first and the parent
        # reuses its slot, so the parent (larger) is h's order-below and the
        # child is buried inside it. At the transition tick the slot below h is
        # the pair (parent, child) -- the case the old semantics hid behind a
        # bare int. h still rests on the parent.
        p = _buf("p", 128, 0, 5)
        c = _buf("c", 64, 4, 10, in_place_parents=["p"])
        h = _buf("h", 32, 4, 6)
        plan = self.plan([p, c, h], [1, 0, 2])  # order: c, then p, then h
        ip, ic, ih = (plan._name_to_idx[n] for n in ("p", "c", "h"))
        self.assertEqual(_addr(plan, "c"), 0)
        self.assertEqual(_addr(plan, "p"), 0)  # parent reuses child's slot
        self.assertEqual(_addr(plan, "h"), 128)  # rests on the parent p
        self.assertEqual(plan.contact_at(ih, 4), (ip, ic))  # slot below = pair
        self.assertEqual(plan.contact_at(ih, 5), ic)  # p dead -> plain child
        self.assertEqual(plan.contact_at(ip, 4), ic)  # p's own order-below is c

    def test_contact_candidates_sufficient_exhaustive(self):
        """Exhaustively (all small configurations) verify the property that
        justifies the contact-based ``_recompute_address``: the candidate set
        built from ``contact_at`` over ``z``'s below-profile breakpoints yields
        the same ``_placement_decision`` as the full earlier-overlapping set.

        Placement reads neither capacity nor alignment, so the equivalence is
        independent of both and one of each suffices. ~40k configs, well under
        a second. The heavier validation (n up to 4, larger sizes, live swap
        propagation) lives in StressTests and the one-off exhaustive sweep.
        """

        def contact_cands(plan, z):
            s: set[int] = set()
            for t in plan.below_profile[z].starts[:-1]:
                co = plan.contact_at(z, t)
                if isinstance(co, tuple):
                    s.update(co)
                elif co is not None:
                    s.add(co)
            return list(s)

        def full_cands(plan, z):
            pz = plan.position[z]
            return [w for w in plan.overlaps[z] if plan.position[w] < pz]

        horizon = 3
        lifetimes = [(s, e) for s in range(horizon) for e in range(s + 1, horizon + 1)]
        for n in (2, 3):
            for life in itertools.product(lifetimes, repeat=n):
                starts = [s for s, _ in life]
                ends = [e for _, e in life]
                # in-place parent options: any other buffer whose lifetime makes
                # it a geometrically valid parent (parent.end == child.start + 1).
                parent_opts = [
                    [None]
                    + [j for j in range(n) if j != i and ends[j] == starts[i] + 1]
                    for i in range(n)
                ]
                for sizes in itertools.product((1, 2), repeat=n):
                    for wiring in itertools.product(*parent_opts):
                        ipp = [[f"b{j}"] if j is not None else [] for j in wiring]
                        for perm in itertools.permutations(range(n)):
                            bufs = [
                                _buf(
                                    f"b{i}",
                                    sizes[i],
                                    starts[i],
                                    ends[i],
                                    in_place_parents=list(ipp[i]),
                                )
                                for i in range(n)
                            ]
                            plan = PermutationBasedLayoutSolver(
                                bufs, list(perm), 10**9, 1
                            )
                            for z in range(n):
                                tag = (
                                    f"life={life} sizes={sizes} wiring={wiring} "
                                    f"perm={perm} z={z}"
                                )
                                self.assertEqual(
                                    plan._placement_decision(z, contact_cands(plan, z)),
                                    plan._placement_decision(z, full_cands(plan, z)),
                                    tag,
                                )

    # --- randomized differential checks ------------------------------------

    def _cases(self, seeds=300, max_n=8):
        for seed in range(seeds):
            rng = random.Random(seed)
            n = rng.randint(1, max_n)
            buffers = _random_buffers(rng, n)
            perm = list(range(n))
            rng.shuffle(perm)
            capacity = rng.choice([200, 600, 10_000])
            alignment = rng.choice([1, 64, 128])
            yield seed, buffers, perm, capacity, alignment

    def test_addresses_match_reference(self):
        for seed, buffers, perm, cap, align in self._cases():
            ref = ReferencePermutationBasedLayoutSolver(buffers, perm, cap, align)
            fast = PermutationBasedLayoutSolver(buffers, perm, cap, align)
            self.assertEqual(fast.addresses, ref.addresses, f"seed={seed}")
            self.assertEqual(fast.quality(), ref.quality(), f"seed={seed}")

    def test_profiles_consistent(self):
        for seed, buffers, perm, cap, align in self._cases():
            fast = PermutationBasedLayoutSolver(buffers, perm, cap, align)
            _check_consistency(self, fast, f"seed={seed}")
            _check_contact_faithful(self, fast, f"seed={seed}")


class RegressionFixtureTests(TestCase):
    """The exact contact-profile fixture from the spec."""

    def _fixture(self):
        # index: z=0, w=1, x=2, y=3; per-column order bottom->top: z, w, x, y.
        buffers = [
            _buf("z", 10, 0, 15),
            _buf("w", 10, 0, 10),
            _buf("x", 10, 0, 10),
            _buf("y", 10, 5, 15),
        ]
        return PermutationBasedLayoutSolver(buffers, [0, 1, 2, 3], 10_000, 1)

    def test_initial_profiles(self):
        plan = self._fixture()
        self.assertEqual(_below_named(plan, "z"), [(0, 15, None)])
        self.assertEqual(_below_named(plan, "w"), [(0, 10, "z")])
        self.assertEqual(_below_named(plan, "x"), [(0, 10, "w")])
        self.assertEqual(_below_named(plan, "y"), [(5, 10, "x"), (10, 15, "z")])
        _check_consistency(self, plan)

    def test_after_swapping_x_and_y(self):
        plan = self._fixture()
        plan.swap(2)  # swap x (pos 2) and y (pos 3); shared range I = [5, 10)
        self.assertEqual(_below_named(plan, "x"), [(0, 5, "w"), (5, 10, "y")])
        self.assertEqual(_below_named(plan, "y"), [(5, 10, "w"), (10, 15, "z")])
        self.assertEqual(_above_named(plan, "w"), [(0, 5, "x"), (5, 10, "y")])
        self.assertEqual(_above_named(plan, "z"), [(0, 10, "w"), (10, 15, "y")])
        self.assertEqual(_above_named(plan, "y"), [(5, 10, "x"), (10, 15, None)])
        self.assertEqual(_above_named(plan, "x"), [(0, 10, None)])
        _check_consistency(self, plan)


class SwapTests(TestCase):
    """Incremental swap in PermutationBasedLayoutSolver."""

    def plan(self, buffers, permutation, capacity=10_000, alignment=1):
        return PermutationBasedLayoutSolver(buffers, permutation, capacity, alignment)

    def test_overlapping_swap_relayouts(self):
        buffers = [_buf("a", 64, 0, 2), _buf("b", 50, 0, 2)]
        plan = self.plan(buffers, [0, 1])
        self.assertEqual([_addr(plan, "a"), _addr(plan, "b")], [0, 64])
        delta = plan.swap(0)  # -> [b, a]
        self.assertEqual([_addr(plan, "b"), _addr(plan, "a")], [0, 50])
        self.assertEqual(delta, 0)  # both still fit

    def test_non_overlapping_swap_is_noop(self):
        buffers = [_buf("a", 64, 0, 1), _buf("b", 64, 2, 3)]
        plan = self.plan(buffers, [0, 1])
        before = list(plan.addresses)
        delta = plan.swap(0)
        self.assertEqual(delta, 0)
        self.assertEqual(plan.addresses, before)
        self.assertEqual(plan.permutation, [1, 0])

    def test_swap_changes_total_size(self):
        # Only one of the two can fit fully below capacity; swapping which one
        # is placed first changes the total.
        buffers = [_buf("a", 30, 0, 2), _buf("b", 90, 0, 2)]
        plan = self.plan(buffers, [0, 1], capacity=100)
        self.assertEqual(plan.quality(), 30)  # a@0 fits, b@30 (->120) does not
        delta = plan.swap(0)  # -> [b, a]: b@0 fits, a@90 (->120) does not
        self.assertEqual(plan.quality(), 90)
        self.assertEqual(delta, 60)

    def test_swap_back_restores(self):
        buffers = [_buf("a", 30, 0, 2), _buf("b", 90, 0, 2)]
        plan = self.plan(buffers, [0, 1], capacity=100)
        d1 = plan.swap(0)
        d2 = plan.swap(0)
        self.assertEqual(d1 + d2, 0)
        self.assertEqual(plan.quality(), 30)
        self.assertEqual([_addr(plan, "a"), _addr(plan, "b")], [0, 30])

    def test_finalize_after_swaps_end_to_end(self):
        # Build, optimize via swaps, then commit: only buffers that fit below
        # capacity get an address written back.
        buffers = [_buf("a", 30, 0, 2), _buf("b", 90, 0, 2)]
        plan = self.plan(buffers, [0, 1], capacity=100)
        plan.swap(0)  # -> [b, a]: b@0 fits, a@90 (-> 120) does not
        plan.finalize()
        self.assertEqual(buffers[1].address, 0)  # b committed
        self.assertIsNone(buffers[0].address)  # a over capacity, dropped

    def test_random_swap_sequences_match_reference(self):
        for seed in range(3000):
            rng = random.Random(seed)
            n = rng.randint(2, 9)
            buffers = _random_buffers(rng, n)
            perm = list(range(n))
            rng.shuffle(perm)
            cap = rng.choice([150, 400, 10_000])
            align = rng.choice([1, 64, 128])
            fast = PermutationBasedLayoutSolver(buffers, perm, cap, align)

            for step in range(rng.randint(1, 2 * n)):
                i = rng.randrange(n - 1)
                before = fast.quality()
                delta = fast.swap(i)
                tag = f"seed={seed} step={step}"

                # Ground truth: a fresh reference build of the new permutation.
                ref = ReferencePermutationBasedLayoutSolver(
                    buffers, list(fast.permutation), cap, align
                )
                self.assertEqual(fast.addresses, ref.addresses, tag)
                self.assertEqual(fast.quality(), ref.quality(), tag)
                self.assertEqual(delta, fast.quality() - before, tag)

                # The incrementally maintained contact profiles match a
                # from-scratch rebuild of the same permutation, exactly, and are
                # internally consistent.
                rebuilt = PermutationBasedLayoutSolver(
                    buffers, list(fast.permutation), cap, align
                )
                self.assertEqual(fast.below_profile, rebuilt.below_profile, tag)
                self.assertEqual(fast.above_profile, rebuilt.above_profile, tag)
                self.assertEqual(fast.inplace_reuse, rebuilt.inplace_reuse, tag)
                _check_consistency(self, fast, tag)
                _check_contact_faithful(self, fast, tag)


class RotateTests(TestCase):
    """rotate(i, j) and the single-element sweep it enables."""

    def plan(self, buffers, permutation, capacity=10_000, alignment=1):
        return PermutationBasedLayoutSolver(buffers, permutation, capacity, alignment)

    def test_rotate_noop(self):
        buffers = [_buf("a", 64, 0, 2), _buf("b", 50, 0, 2)]
        plan = self.plan(buffers, [0, 1])
        before = list(plan.addresses)
        self.assertEqual(plan.rotate(1, 1), 0)
        self.assertEqual(plan.permutation, [0, 1])
        self.assertEqual(plan.addresses, before)

    def test_rotate_moves_element(self):
        # Three mutually overlapping buffers; move the first to the end.
        buffers = [_buf("a", 10, 0, 3), _buf("b", 20, 0, 3), _buf("c", 30, 0, 3)]
        plan = self.plan(buffers, [0, 1, 2])  # a@0, b@10, c@30
        plan.rotate(0, 2)  # -> [b, c, a]: b@0, c@20, a@50
        self.assertEqual(plan.permutation, [1, 2, 0])
        self.assertEqual([_addr(plan, n) for n in "abc"], [50, 0, 20])

    def test_random_rotations_match_reference(self):
        for seed in range(3000):
            rng = random.Random(seed)
            n = rng.randint(2, 9)
            buffers = _random_buffers(rng, n)
            perm = list(range(n))
            rng.shuffle(perm)
            cap = rng.choice([150, 400, 10_000])
            align = rng.choice([1, 64, 128])
            fast = PermutationBasedLayoutSolver(buffers, perm, cap, align)
            # Force the remove/reinsert fast path on every rotation (small n
            # gives small distances otherwise, so the chain would always win).
            fast._rotate_remove_insert_threshold = 1

            for step in range(rng.randint(1, 2 * n)):
                i, j = rng.randrange(n), rng.randrange(n)
                before = fast.quality()
                delta = fast.rotate(i, j)
                tag = f"seed={seed} step={step} i={i} j={j}"

                ref = ReferencePermutationBasedLayoutSolver(
                    buffers, list(fast.permutation), cap, align
                )
                self.assertEqual(fast.addresses, ref.addresses, tag)
                self.assertEqual(fast.quality(), ref.quality(), tag)
                self.assertEqual(fast.count_allocated(), ref.count_allocated(), tag)
                self.assertEqual(delta, fast.quality() - before, tag)

                rebuilt = PermutationBasedLayoutSolver(
                    buffers, list(fast.permutation), cap, align
                )
                self.assertEqual(fast.below_profile, rebuilt.below_profile, tag)
                self.assertEqual(fast.above_profile, rebuilt.above_profile, tag)
                self.assertEqual(fast.inplace_reuse, rebuilt.inplace_reuse, tag)
                _check_consistency(self, fast, tag)
                _check_contact_faithful(self, fast, tag)

    def test_single_element_sweep_matches_reference(self):
        # Sweep one element across every position (rotate it to 0, then bubble
        # it right), reading quality() at each stop. Each stop must match a
        # fresh build of that permutation, and a round trip must restore the
        # original state exactly -- the contract the annealing sweep relies on.
        for seed in range(500):
            rng = random.Random(seed)
            n = rng.randint(2, 9)
            buffers = _random_buffers(rng, n)
            perm = list(range(n))
            rng.shuffle(perm)
            cap = rng.choice([150, 400, 10_000])
            align = rng.choice([1, 64, 128])
            fast = PermutationBasedLayoutSolver(buffers, perm, cap, align)
            fast._rotate_remove_insert_threshold = 1  # exercise the fast path

            orig_perm = list(fast.permutation)
            orig_addr = list(fast.addresses)
            i = rng.randrange(n)
            x = orig_perm[i]
            others = [b for b in orig_perm if b != x]

            qualities = {}
            fast.rotate(i, 0)  # x to the front
            qualities[0] = fast.quality()
            for p in range(1, n):
                fast.swap(p - 1)  # bubble x from p-1 to p
                qualities[p] = fast.quality()

            # Every recorded objective matches a fresh build of "x inserted at p".
            for p in range(n):
                test_perm = others[:p] + [x] + others[p:]
                ref = ReferencePermutationBasedLayoutSolver(
                    buffers, test_perm, cap, align
                )
                self.assertEqual(qualities[p], ref.quality(), f"seed={seed} p={p}")

            # Round trip restores the exact original state (no hysteresis).
            fast.rotate(n - 1, i)
            self.assertEqual(fast.permutation, orig_perm, f"seed={seed}")
            self.assertEqual(fast.addresses, orig_addr, f"seed={seed}")
            rebuilt = PermutationBasedLayoutSolver(buffers, orig_perm, cap, align)
            self.assertEqual(fast.below_profile, rebuilt.below_profile, f"{seed}")
            self.assertEqual(fast.above_profile, rebuilt.above_profile, f"{seed}")
            self.assertEqual(fast.inplace_reuse, rebuilt.inplace_reuse, f"{seed}")
            _check_consistency(self, fast, f"seed={seed}")


class FastRotateTests(TestCase):
    """The remove-one / reinsert-elsewhere fast rotate in
    :class:`PermutationBasedLayoutSolver`, forced on for every rotation.

    The fast path is distance-gated in production; here we pin
    ``_rotate_remove_insert_threshold = 1`` so it fires on *every* rotate --
    including the large ``|i - j|`` moves it exists for -- and check it against
    both oracles (a from-scratch reference build and a fresh rebuild). Both
    profile modes are covered: ``"patch"`` (the incremental single-move splice)
    and ``"rebuild"`` (full ``_build_profiles``), which must agree exactly.
    """

    def _assert_matches(self, fast, cap, align, delta, before, tag):
        ref = ReferencePermutationBasedLayoutSolver(
            fast.buffers, list(fast.permutation), cap, align
        )
        self.assertEqual(fast.addresses, ref.addresses, tag)
        self.assertEqual(fast.quality(), ref.quality(), tag)
        self.assertEqual(fast.count_allocated(), ref.count_allocated(), tag)
        self.assertEqual(delta, fast.quality() - before, tag)
        rebuilt = PermutationBasedLayoutSolver(
            fast.buffers, list(fast.permutation), cap, align
        )
        self.assertEqual(fast.below_profile, rebuilt.below_profile, tag)
        self.assertEqual(fast.above_profile, rebuilt.above_profile, tag)
        self.assertEqual(fast.inplace_reuse, rebuilt.inplace_reuse, tag)
        _check_consistency(self, fast, tag)
        _check_contact_faithful(self, fast, tag)

    def test_long_rotations_both_directions(self):
        # A handful of mutually overlapping buffers; sweep every (i, j) pair,
        # which includes the full-distance moves in both directions, for both
        # profile modes. Each rotate is applied to a fresh plan.
        n = 7
        buffers = [_buf(f"b{k}", 10 * (k + 1), 0, 5) for k in range(n)]
        for mode in ("patch", "rebuild"):
            for i in range(n):
                for j in range(n):
                    for cap, align in ((10_000, 1), (250, 64)):
                        fast = PermutationBasedLayoutSolver(
                            buffers, list(range(n)), cap, align
                        )
                        fast._rotate_remove_insert_threshold = 1
                        fast._rotate_profile_mode = mode
                        before = fast.quality()
                        delta = fast.rotate(i, j)
                        expected = list(range(n))
                        x = expected.pop(i)
                        expected.insert(j, x)
                        self.assertEqual(fast.permutation, expected)
                        self._assert_matches(
                            fast, cap, align, delta, before, f"{mode} i={i} j={j}"
                        )

    def test_dense_inplace_random(self):
        # Dense in-place wiring (inplace_prob up to ~0.7), random rotations
        # forced through the fast path, both profile modes.
        for mode in ("patch", "rebuild"):
            for seed in range(1500):
                rng = random.Random(seed)
                n = rng.randint(2, 12)
                buffers = _random_buffers(
                    rng, n, horizon=15, max_size=300, inplace_prob=0.7
                )
                perm = list(range(n))
                rng.shuffle(perm)
                cap = rng.choice([150, 400, 800, 10**9])
                align = rng.choice([1, 32, 64, 128])
                fast = PermutationBasedLayoutSolver(buffers, perm, cap, align)
                fast._rotate_remove_insert_threshold = 1
                fast._rotate_profile_mode = mode
                for step in range(rng.randint(1, 2 * n)):
                    i, j = rng.randrange(n), rng.randrange(n)
                    before = fast.quality()
                    delta = fast.rotate(i, j)
                    tag = f"{mode} seed={seed} step={step} i={i} j={j}"
                    self._assert_matches(fast, cap, align, delta, before, tag)

    def test_threshold_dispatch_agrees_with_chain(self):
        # The fast path and the swap-chain must produce identical results for
        # the same move. Run the same rotation on two clones, one forced to the
        # fast path and one forced to the chain, and compare.
        for seed in range(800):
            rng = random.Random(seed)
            n = rng.randint(3, 12)
            buffers = _random_buffers(rng, n, horizon=15, inplace_prob=0.5)
            perm = list(range(n))
            rng.shuffle(perm)
            cap = rng.choice([150, 400, 10**9])
            align = rng.choice([1, 64, 128])
            base = PermutationBasedLayoutSolver(buffers, perm, cap, align)
            i, j = rng.randrange(n), rng.randrange(n)

            chain = base.copy()
            chain._rotate_remove_insert_threshold = n + 1  # never fast
            d_chain = chain.rotate(i, j)

            fast = base.copy()
            fast._rotate_remove_insert_threshold = 1  # always fast
            d_fast = fast.rotate(i, j)

            tag = f"seed={seed} i={i} j={j}"
            self.assertEqual(fast.permutation, chain.permutation, tag)
            self.assertEqual(fast.addresses, chain.addresses, tag)
            self.assertEqual(d_fast, d_chain, tag)
            self.assertEqual(fast.below_profile, chain.below_profile, tag)
            self.assertEqual(fast.above_profile, chain.above_profile, tag)
            self.assertEqual(fast.inplace_reuse, chain.inplace_reuse, tag)


class CopyTests(TestCase):
    """copy() makes an independent layout snapshot sharing static structures."""

    def plan(self, buffers, permutation, capacity=10_000, alignment=1):
        return PermutationBasedLayoutSolver(buffers, permutation, capacity, alignment)

    def test_static_shared_dynamic_independent(self):
        buffers = [_buf("a", 64, 0, 3), _buf("b", 50, 0, 3), _buf("c", 40, 1, 3)]
        plan = self.plan(buffers, [0, 1, 2])
        clone = plan.copy()
        # Static structures are shared by reference.
        self.assertIs(clone.buffers, plan.buffers)
        self.assertIs(clone.overlaps, plan.overlaps)
        self.assertIs(clone._name_to_idx, plan._name_to_idx)
        # Dynamic state is equal but independent.
        self.assertEqual(clone.addresses, plan.addresses)
        self.assertEqual(clone.below_profile, plan.below_profile)
        self.assertIsNot(clone.permutation, plan.permutation)
        self.assertIsNot(clone.below_profile, plan.below_profile)
        self.assertIsNot(clone.below_profile[0], plan.below_profile[0])

    def test_mutating_copy_leaves_original_intact(self):
        for seed in range(2000):
            rng = random.Random(seed)
            n = rng.randint(2, 9)
            buffers = _random_buffers(rng, n)
            perm = list(range(n))
            rng.shuffle(perm)
            cap = rng.choice([150, 400, 10_000])
            align = rng.choice([1, 64, 128])
            plan = PermutationBasedLayoutSolver(buffers, perm, cap, align)

            orig_perm = list(plan.permutation)
            orig_addr = list(plan.addresses)
            orig_below = {
                k: Profile(list(p.starts), list(p.labels))
                for k, p in plan.below_profile.items()
            }
            orig_quality = plan.quality()

            clone = plan.copy()
            for _ in range(rng.randint(1, 2 * n)):
                clone.swap(rng.randrange(n - 1))

            # Original is untouched by mutations on the clone.
            self.assertEqual(plan.permutation, orig_perm, seed)
            self.assertEqual(plan.addresses, orig_addr, seed)
            self.assertEqual(plan.below_profile, orig_below, seed)
            self.assertEqual(plan.quality(), orig_quality, seed)

            # The mutated clone is a valid plan: matches a fresh build.
            rebuilt = PermutationBasedLayoutSolver(
                buffers, list(clone.permutation), cap, align
            )
            self.assertEqual(clone.addresses, rebuilt.addresses, seed)
            self.assertEqual(clone.quality(), rebuilt.quality(), seed)
            self.assertEqual(clone.below_profile, rebuilt.below_profile, seed)
            self.assertEqual(clone.above_profile, rebuilt.above_profile, seed)


@unittest.skipUnless(
    _STRESS, "set TORCH_SPYRE_STRESS_SCRATCHPAD=1 to run scratchpad stress tests"
)
class StressTests(TestCase):
    """Exhaustive randomized differential coverage. Not run by default; these
    are the heavy versions of the SwapTests / RotateTests / CopyTests checks --
    thousands of seeds, larger n, dense in-place wiring -- against from-scratch
    reference and rebuild oracles."""

    def _stress_buffers(self, rng, n):
        return _random_buffers(rng, n, horizon=15, max_size=300, inplace_prob=0.4)

    def _cases(self, seeds, max_n=13):
        for seed in range(seeds):
            rng = random.Random(seed)
            n = rng.randint(2, max_n)
            buffers = self._stress_buffers(rng, n)
            perm = list(range(n))
            rng.shuffle(perm)
            cap = rng.choice([150, 400, 800, 10**9])
            align = rng.choice([1, 32, 64, 128])
            yield seed, rng, n, buffers, perm, cap, align

    def _assert_matches_rebuild(self, fast, cap, align, tag):
        ref = ReferencePermutationBasedLayoutSolver(
            fast.buffers, list(fast.permutation), cap, align
        )
        self.assertEqual(fast.addresses, ref.addresses, tag)
        self.assertEqual(fast.quality(), ref.quality(), tag)
        self.assertEqual(fast.count_allocated(), ref.count_allocated(), tag)
        rebuilt = PermutationBasedLayoutSolver(
            fast.buffers, list(fast.permutation), cap, align
        )
        self.assertEqual(fast.below_profile, rebuilt.below_profile, tag)
        self.assertEqual(fast.above_profile, rebuilt.above_profile, tag)
        self.assertEqual(fast.inplace_reuse, rebuilt.inplace_reuse, tag)
        _check_contact_faithful(self, fast, tag)

    def test_swap_sequences(self):
        for seed, rng, n, buffers, perm, cap, align in self._cases(20000):
            fast = PermutationBasedLayoutSolver(buffers, perm, cap, align)
            for step in range(rng.randint(1, 3 * n)):
                i = rng.randrange(n - 1)
                before = fast.quality()
                delta = fast.swap(i)
                tag = f"seed={seed} step={step}"
                self.assertEqual(delta, fast.quality() - before, tag)
                self._assert_matches_rebuild(fast, cap, align, tag)

    def test_rotation_sequences(self):
        for seed, rng, n, buffers, perm, cap, align in self._cases(10000):
            fast = PermutationBasedLayoutSolver(buffers, perm, cap, align)
            fast._rotate_remove_insert_threshold = 1  # force the fast path
            for step in range(rng.randint(1, 3 * n)):
                i, j = rng.randrange(n), rng.randrange(n)
                before = fast.quality()
                delta = fast.rotate(i, j)
                tag = f"seed={seed} step={step} i={i} j={j}"
                self.assertEqual(delta, fast.quality() - before, tag)
                self._assert_matches_rebuild(fast, cap, align, tag)

    def test_single_element_sweeps(self):
        for seed, rng, n, buffers, perm, cap, align in self._cases(3000):
            fast = PermutationBasedLayoutSolver(buffers, perm, cap, align)
            fast._rotate_remove_insert_threshold = 1  # force the fast path
            orig_perm = list(fast.permutation)
            orig_addr = list(fast.addresses)
            i = rng.randrange(n)
            x = orig_perm[i]
            others = [b for b in orig_perm if b != x]

            qualities = {}
            fast.rotate(i, 0)
            qualities[0] = fast.quality()
            for p in range(1, n):
                fast.swap(p - 1)
                qualities[p] = fast.quality()
            for p in range(n):
                test_perm = others[:p] + [x] + others[p:]
                ref = ReferencePermutationBasedLayoutSolver(
                    buffers, test_perm, cap, align
                )
                self.assertEqual(qualities[p], ref.quality(), f"seed={seed} p={p}")

            fast.rotate(n - 1, i)
            self.assertEqual(fast.permutation, orig_perm, seed)
            self.assertEqual(fast.addresses, orig_addr, seed)

    def test_copy_isolation(self):
        for seed, rng, n, buffers, perm, cap, align in self._cases(10000):
            plan = PermutationBasedLayoutSolver(buffers, perm, cap, align)
            orig_addr = list(plan.addresses)
            orig_below = {
                k: Profile(list(p.starts), list(p.labels))
                for k, p in plan.below_profile.items()
            }
            clone = plan.copy()
            for _ in range(rng.randint(1, 3 * n)):
                clone.swap(rng.randrange(n - 1))
            self.assertEqual(plan.addresses, orig_addr, seed)
            self.assertEqual(plan.below_profile, orig_below, seed)
            self._assert_matches_rebuild(clone, cap, align, f"seed={seed} (clone)")


class ProfileTests(TestCase):
    """Unit tests for the Profile step-function, in isolation."""

    def test_uniform_and_label_at(self):
        p = Profile.uniform(0, 10, 7)
        self.assertEqual(p.span_start, 0)
        self.assertEqual(p.span_end, 10)
        self.assertEqual(p.label_at(0), 7)
        self.assertEqual(p.label_at(9), 7)
        p.validate()

    def test_from_segments_coalesces(self):
        p = Profile.from_segments([0, 3, 5, 9], [1, 1, 2])
        self.assertEqual(p, Profile([0, 5, 9], [1, 2]))
        p.validate()

    def test_segments_clips_and_copies(self):
        p = Profile([0, 5, 10, 15], [1, 2, 3])
        starts, labels = p.segments(3, 12)
        self.assertEqual(starts, [3, 5, 10, 12])
        self.assertEqual(labels, [1, 2, 3])
        # returned data must not alias internal state
        starts[0] = -999
        self.assertEqual(p.starts[0], 0)
        # whole-span and empty range
        self.assertEqual(p.segments(0, 15), ([0, 5, 10, 15], [1, 2, 3]))
        self.assertEqual(p.segments(7, 7), ([7], []))

    def test_splice_at_exact_breakpoints(self):
        p = Profile([0, 5, 10, 15], [1, 2, 3])
        p.splice(5, 10, [5, 10], [9])
        self.assertEqual(p, Profile([0, 5, 10, 15], [1, 9, 3]))
        p.validate()

    def test_splice_inside_one_segment(self):
        p = Profile([0, 10], [1])
        p.splice(3, 7, [3, 7], [2])
        self.assertEqual(p, Profile([0, 3, 7, 10], [1, 2, 1]))
        p.validate()

    def test_splice_spanning_several_segments(self):
        p = Profile([0, 5, 10, 15, 20], [1, 2, 3, 4])
        p.splice(3, 17, [3, 17], [9])
        self.assertEqual(p, Profile([0, 3, 17, 20], [1, 9, 4]))
        p.validate()

    def test_splice_coalesces_both_seams(self):
        p = Profile([0, 5, 10, 15], [1, 2, 1])
        # replace the middle [5,10) with label 1 -> whole thing coalesces to one
        p.splice(5, 10, [5, 10], [1])
        self.assertEqual(p, Profile([0, 15], [1]))
        p.validate()

    def test_splice_multi_segment_replacement(self):
        p = Profile([0, 10], [1])
        p.splice(2, 8, [2, 4, 6, 8], [2, 3, 2])
        self.assertEqual(p, Profile([0, 2, 4, 6, 8, 10], [1, 2, 3, 2, 1]))
        p.validate()

    def test_relabel_splits_straddling_segment(self):
        p = Profile([0, 10], [1])
        p.relabel(3, 7, {1: 5})
        self.assertEqual(p, Profile([0, 3, 7, 10], [1, 5, 1]))
        p.validate()

    def test_relabel_only_matching_labels(self):
        p = Profile([0, 5, 10, 15], [1, 2, 3])
        p.relabel(0, 15, {1: 9, 3: 9})  # 1->9, 3->9, 2 untouched
        self.assertEqual(p, Profile([0, 5, 10, 15], [9, 2, 9]))
        p.validate()

    def test_empty_range_noops(self):
        p = Profile([0, 5, 10], [1, 2])
        before = Profile(list(p.starts), list(p.labels))
        p.splice(5, 5, [5], [])
        self.assertEqual(p, before)
        p.relabel(7, 7, {2: 9})
        self.assertEqual(p, before)

    def test_label_set(self):
        p = Profile([0, 5, 10], [1, None])
        self.assertEqual(p.label_set(), {1, None})
        self.assertEqual(p.label_set() - {None}, {1})

    def test_validate_catches_corruption(self):
        bad_order = Profile([0, 5, 5, 10], [1, 2, 3])  # not strictly increasing
        with self.assertRaises(AssertionError):
            bad_order.validate()
        bad_adjacent = Profile([0, 5, 10], [1, 1])  # adjacent equal labels
        with self.assertRaises(AssertionError):
            bad_adjacent.validate()
        bad_len = Profile([0, 5, 10], [1])  # length mismatch
        with self.assertRaises(AssertionError):
            bad_len.validate()

    def test_none_labels_round_trip(self):
        p = Profile.uniform(0, 10, None)
        self.assertIsNone(p.label_at(4))
        p.splice(3, 7, [3, 7], [2])
        self.assertEqual(p, Profile([0, 3, 7, 10], [None, 2, None]))
        p.validate()
