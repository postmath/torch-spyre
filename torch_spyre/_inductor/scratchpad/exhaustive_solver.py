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

"""Exhaustive branch-and-bound memory plan solver.

This solver finds the maximum-weight set of :class:`LifetimeBoundBuffer`\\ s that
can be co-resident in scratchpad, by enumerating one representative per
equivalence class of buffer drop orders and packing each with a Tetris-style
drop (identical geometry to the other solvers in this package, including the
in-place exception).

Weight is currently the buffer ``size`` -- i.e. the solver maximises the total
number of bytes pinned, the same quantity the first-fit/best-fit solvers
effectively maximise. The weight function is isolated in :meth:`_weight` so it
can be swapped for a richer cost model later.

Background
----------
Two buffers *interact* iff their half-open lifetime intervals
``[start_time, end_time)`` share at least one time column. Two drop orders
produce the same packing iff one can be reached from the other by repeatedly
swapping adjacent *non*-interacting buffers, so the equivalence classes are in
bijection with the acyclic orientations of the interaction graph, of which an
interval graph has ``prod(1 + d_i)`` (see :meth:`class_count`).

We enumerate exactly one representative per class using the standard
lexicographic-canonical (trace monoid) rule under the start-time order: a buffer
``b`` may be placed immediately after the last-placed buffer ``last`` iff ``b``
sorts after ``last`` *or* ``b`` interacts with ``last``. Equivalently we forbid
an adjacent "descent" between two non-interacting buffers, which is exactly the
condition that makes a drop order the lexicographically-smallest member of its
class. (Any fixed total order works for this; start time is convenient. The
perfect-elimination order needed for the *count* formula is a different,
end-time order -- see :meth:`class_count`.)

Pruning keeps the search tractable up to roughly ``n = 30`` even with dense
interaction:

* **Upper bound** -- a completion can add at most the weight of the remaining
  buffers that are not already provably unplaceable, so we abandon a node whose
  optimistic total cannot beat the incumbent.
* **Strong (doesn't-fit) prune** -- a buffer that does not fit on the current
  skyline cannot fit deeper in the same subtree (the skyline only grows), so it
  is dropped from the subtree's candidate set. The arrangement that defers it is
  reached as the representative of a different class, so nothing is lost.

The incumbent is recorded at *every* node, not only at full-placement leaves,
because the optimum generally evicts some buffers (the cache is finite).
"""

import logging
from dataclasses import dataclass
from typing import Optional, Union

from torch_spyre._inductor.scratchpad.plan_solver import (
    LifetimeBoundBuffer,
    MemoryPlanSolver,
)
from torch_spyre._inductor.scratchpad.firstfit_bestfit_solver import (
    _assert_in_place_relationships,
    round_up_to_alignment,
)

logger = logging.getLogger(__name__)

# Empirical throughput model. Per-node cost grows with n because each node drops
# every still-live buffer (~O(n) drops, each an O(n) skyline scan), so the node
# rate decays roughly geometrically with n: measured ~200k nodes/s at n=10, ~85k
# at n=20, ~47k at n=30, fit well by _RATE_AT_N0 / _RATE_DECAY ** n.
_RATE_AT_N0 = 400_000  # nodes/sec extrapolated to n = 0
_RATE_DECAY = 1.068  # per-buffer decay of the node rate


@dataclass(frozen=True)
class TimeBudget:
    """A search cap expressed as target wall-clock seconds rather than a fixed
    node count.

    :meth:`node_cap` converts the target seconds into a node count for an
    ``n``-buffer problem using the throughput model above, so the same budget
    yields a smaller node cap (but similar wall time) as ``n`` grows. Pass an
    instance as ``ExhaustiveLayoutSolver``'s ``node_budget`` to opt into
    runtime-scaled capping.
    """

    seconds: float = 1.0

    def node_cap(self, n: int) -> int:
        return max(1, round(self.seconds * _RATE_AT_N0 / _RATE_DECAY**n))


# Default cap: scale to roughly a second of wall time at the problem's n.
DEFAULT_BUDGET = TimeBudget(1.0)

# A search cap: a fixed node count, a runtime-scaled TimeBudget, or None for an
# uncapped (provably-optimal) search.
NodeBudget = Union[int, TimeBudget, None]


def _interacts(a: LifetimeBoundBuffer, b: LifetimeBoundBuffer) -> bool:
    """True iff the half-open lifetimes of ``a`` and ``b`` share a time column.

    Touching at a boundary (``a.end_time == b.start_time``) is not interaction.
    """
    return a.start_time < b.end_time and b.start_time < a.end_time


class ExhaustiveLayoutSolver(MemoryPlanSolver):
    """Maximum-weight scratchpad placement via exhaustive branch-and-bound.

    The solver assigns an aligned address to every buffer it can pin and leaves
    the rest evicted (``address=None``), maximising the total pinned weight
    (currently ``size``). Given a fixed input it is fully deterministic.
    """

    def __init__(
        self,
        size: int,
        alignment: int = 128,
        node_budget: NodeBudget = DEFAULT_BUDGET,
        verbose: bool = False,
    ):
        """Initialise the solver.

        Args:
            size: Total scratchpad capacity in bytes (the cache height ``H``).
            alignment: Byte alignment for placed addresses. Defaults to 128
                (one Spyre stick).
            node_budget: Cap on the number of search nodes explored. When the
                cap is reached the search stops and the best arrangement found
                so far is returned (still a valid packing, seeded by the warm
                start). Accepts:

                * an ``int`` -- a fixed node cap;
                * a :class:`TimeBudget` -- a target runtime in seconds, scaled
                  to a node cap for the problem's ``n`` (the default,
                  ``TimeBudget(1.0)``, aims for ~1s);
                * ``None`` -- uncapped, provably-optimal search.
            verbose: When True, log pruning statistics at INFO level after the
                search completes.
        """
        super().__init__(size, alignment)
        self.node_budget = node_budget
        self.verbose = verbose
        # Resolved integer cap for the current run (set in plan_layout once n
        # is known); None means uncapped.
        self._node_cap: Optional[int] = None

        # When True, the upper-bound prune is disabled. Used by the class-count
        # unit test so that every equivalence-class representative is visited.
        self.disable_upper_bound = False

        # Search state / statistics, reset on every plan_layout call.
        self._bufs: list[LifetimeBoundBuffer] = []
        self._interacts_with: list[set[int]] = []
        self._inplace_parent_idx: list[list[int]] = []
        self._inplace_child_idx: list[list[int]] = []
        self._best_wt = 0
        self._best_placement: dict[int, int] = {}
        self._aborted = False
        self.nodes = 0
        # Terminal nodes that placed every buffer: with the upper-bound prune
        # off and ample capacity these are exactly the equivalence-class
        # representatives, prod(1 + d_i) of them.
        self.complete_orders = 0
        # Terminal nodes where buffers remain but none can be placed next (a
        # canonical prefix with no canonical completion, or no remaining fit).
        self.dead_ends = 0
        self.upper_bound_prunes = 0
        self.strong_prunes = 0

    # ------------------------------------------------------------------
    # Weighting
    # ------------------------------------------------------------------

    def _weight(self, buf: LifetimeBoundBuffer) -> int:
        """Weight contributed by pinning ``buf``.

        Currently the buffer size, matching the quantity the first-fit/best-fit
        solvers maximise. Isolated here so a richer cost model (e.g. HBM bytes
        saved) can be substituted without touching the search.
        """
        return buf.size

    # ------------------------------------------------------------------
    # Drop (Tetris-style placement of a single buffer)
    # ------------------------------------------------------------------

    def _drop(self, idx: int, placed: dict[int, int]) -> Optional[int]:
        """Return the lowest legal aligned address for buffer ``idx``, or None.

        ``placed`` maps already-placed buffer indices to their addresses. The
        buffer lands at the lowest aligned address that clears every placed
        buffer over its lifetime, with the in-place exception: it may share the
        exact address of an in-place parent at its first column, or of an
        in-place child at its last column. Returns None if the buffer would not
        fit within the capacity.
        """
        b = self._bufs[idx]
        b_start = b.start_time
        end_col = b.end_time - 1

        # Tops over the start column, the end column, and the interior columns
        # (b_start + 1 .. end_col), tracking the topmost buffer at each end.
        start_top = 0
        start_buf: Optional[int] = None
        end_top = 0
        end_buf: Optional[int] = None
        interior_top = 0
        has_interior = b_start + 1 < end_col

        for p_idx, p_y in placed.items():
            p = self._bufs[p_idx]
            p_top = p_y + p.size
            if p.start_time <= b_start < p.end_time and p_top > start_top:
                start_top = p_top
                start_buf = p_idx
            if p.start_time <= end_col < p.end_time and p_top > end_top:
                end_top = p_top
                end_buf = p_idx
            if (
                has_interior
                and p.start_time < end_col
                and b_start + 1 < p.end_time
                and p_top > interior_top
            ):
                interior_top = p_top

        # Normal drop: above everything across the whole lifetime, then aligned.
        normal_y = round_up_to_alignment(
            max(start_top, interior_top, end_top), self.alignment
        )
        best_y = normal_y

        # In-place at the start column: reuse the parent's address.
        if start_buf is not None and start_buf in self._inplace_parent_idx[idx]:
            y = placed[start_buf]
            if y >= interior_top and y >= end_top:
                best_y = min(best_y, y)

        # In-place at the end column: reuse the (in-place child) buffer's
        # address. Only meaningful for multi-column buffers.
        if (
            end_col != b_start
            and end_buf is not None
            and end_buf in self._inplace_child_idx[idx]
        ):
            y = placed[end_buf]
            if y >= start_top and y >= interior_top:
                best_y = min(best_y, y)

        if best_y + b.size > self.limit:
            return None
        return best_y

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _eligible(self, idx: int, last: Optional[int]) -> bool:
        """Whether placing ``idx`` right after ``last`` keeps the order canonical.

        Canonical (lexicographically-smallest) orders forbid an adjacent descent
        between two non-interacting buffers, so ``idx`` may follow ``last`` iff
        it sorts after ``last`` or interacts with it.
        """
        if last is None:
            return True
        return idx > last or idx in self._interacts_with[last]

    def _all_inplace_partners_placed(self, idx: int, placed: dict[int, int]) -> bool:
        """True iff every in-place partner of ``idx`` is already placed.

        While a partner is unplaced, placing it later could lower ``idx``'s drop
        address (the in-place exception), so a currently-unfitting ``idx`` is not
        yet provably dead.
        """
        for p in self._inplace_parent_idx[idx]:
            if p not in placed:
                return False
        for c in self._inplace_child_idx[idx]:
            if c not in placed:
                return False
        return True

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _search(
        self,
        last: Optional[int],
        remaining: set[int],
        placed: dict[int, int],
        placed_wt: int,
    ) -> None:
        if self._aborted:
            return
        self.nodes += 1
        if self._node_cap is not None and self.nodes > self._node_cap:
            self._aborted = True
            return

        # Classify remaining buffers: drop each on the current skyline. A buffer
        # that does not fit and has no unplaced in-place partner can never be
        # placed in this subtree (the skyline only grows), so it is excluded
        # from both the optimistic bound and the candidate successors.
        drops: dict[int, int] = {}
        optimistic = placed_wt
        for i in remaining:
            y = self._drop(i, placed)
            if y is not None:
                drops[i] = y
                optimistic += self._weight(self._bufs[i])
            elif not self._all_inplace_partners_placed(i, placed):
                optimistic += self._weight(self._bufs[i])
            else:
                self.strong_prunes += 1

        # Upper-bound prune.
        if not self.disable_upper_bound and optimistic <= self._best_wt:
            self.upper_bound_prunes += 1
            return

        # Record the current (partial) placement; unplaced buffers are evicted.
        if placed_wt > self._best_wt:
            self._best_wt = placed_wt
            self._best_placement = dict(placed)

        # Branch on the buffers that fit now and keep the order canonical,
        # heaviest first then by sort index for a strong early incumbent.
        successors = [i for i in drops if self._eligible(i, last)]
        if not successors:
            if remaining:
                self.dead_ends += 1
            else:
                self.complete_orders += 1
            return
        successors.sort(key=lambda i: (-self._weight(self._bufs[i]), i))

        for i in successors:
            placed[i] = drops[i]
            remaining.discard(i)
            self._search(i, remaining, placed, placed_wt + self._weight(self._bufs[i]))
            remaining.add(i)
            del placed[i]

    # ------------------------------------------------------------------
    # Warm start
    # ------------------------------------------------------------------

    def _evaluate_order(self, order: list[int]) -> tuple[int, dict[int, int]]:
        """Drop buffers in ``order`` (placing each that fits) and return the
        achieved (weight, placement). Shared by the warm start and the test
        harness; uses the same drop semantics as the search.
        """
        placed: dict[int, int] = {}
        weight = 0
        for i in order:
            y = self._drop(i, placed)
            if y is not None:
                placed[i] = y
                weight += self._weight(self._bufs[i])
        return weight, placed

    def _warm_start(self) -> tuple[int, dict[int, int]]:
        """Greedy lower bound to seed the incumbent before the exhaustive search.

        Tries a couple of cheap drop orders and keeps the best, so the
        upper-bound prune bites from the first leaves.
        """
        n = len(self._bufs)
        orders = [
            sorted(range(n), key=lambda i: (-self._weight(self._bufs[i]), i)),
            sorted(range(n), key=lambda i: (self._bufs[i].start_time, i)),
        ]
        best_wt = 0
        best_placement: dict[int, int] = {}
        for order in orders:
            wt, placement = self._evaluate_order(order)
            if wt > best_wt:
                best_wt = wt
                best_placement = placement
        return best_wt, best_placement

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def _prepare(self, buffers: list[LifetimeBoundBuffer]) -> None:
        """Build the internal buffer list and interaction / in-place indices."""
        # Only buffers with a positive lifetime can be placed; the rest stay
        # evicted (mirrors the first-fit/best-fit solvers).
        self._bufs = sorted(
            (b for b in buffers if b.end_time > b.start_time),
            key=lambda b: (b.start_time, b.end_time, b.name),
        )
        n = len(self._bufs)
        name_to_idx = {b.name: i for i, b in enumerate(self._bufs)}

        self._interacts_with = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if _interacts(self._bufs[i], self._bufs[j]):
                    self._interacts_with[i].add(j)
                    self._interacts_with[j].add(i)

        # in_place_parents holds parent *names*; resolve to indices and the
        # reverse (child) relation. Partners outside the active set are ignored.
        self._inplace_parent_idx = [[] for _ in range(n)]
        self._inplace_child_idx = [[] for _ in range(n)]
        for i, b in enumerate(self._bufs):
            for parent_name in b.in_place_parents:
                p = name_to_idx.get(parent_name)
                if p is not None:
                    self._inplace_parent_idx[i].append(p)
                    self._inplace_child_idx[p].append(i)

    def plan_layout(
        self, buffers: list[LifetimeBoundBuffer]
    ) -> list[LifetimeBoundBuffer]:
        """Assign addresses to the maximum-weight placeable subset of buffers.

        Args:
            buffers: Candidate buffers, all with ``address is None``.

        Returns:
            The same buffer objects, with ``address`` set on every pinned buffer
            and left as ``None`` on every evicted buffer.
        """
        if not buffers:
            return []
        assert all(buf.address is None for buf in buffers), (
            "Buffers cannot be previously or partially planned"
        )
        _assert_in_place_relationships(buffers)

        self._prepare(buffers)
        self.nodes = 0
        self.complete_orders = 0
        self.dead_ends = 0
        self.upper_bound_prunes = 0
        self.strong_prunes = 0
        self._aborted = False

        self._best_wt, self._best_placement = self._warm_start()

        n = len(self._bufs)
        self._node_cap = (
            self.node_budget.node_cap(n)
            if isinstance(self.node_budget, TimeBudget)
            else self.node_budget
        )
        self._search(None, set(range(n)), {}, 0)

        for i, b in enumerate(self._bufs):
            b.address = self._best_placement.get(i)

        if self.verbose:
            logger.info(
                "ExhaustiveLayoutSolver: %d buffers, weight=%d, nodes=%d/%s, "
                "complete_orders=%d, dead_ends=%d, ub_prunes=%d, "
                "strong_prunes=%d%s",
                n,
                self._best_wt,
                self.nodes,
                "inf" if self._node_cap is None else self._node_cap,
                self.complete_orders,
                self.dead_ends,
                self.upper_bound_prunes,
                self.strong_prunes,
                " (aborted: node budget reached)" if self._aborted else "",
            )

        return buffers

    # ------------------------------------------------------------------
    # Debugging / validation helpers
    # ------------------------------------------------------------------

    def class_count(self, buffers: list[LifetimeBoundBuffer]) -> int:
        """Number of drop-order equivalence classes, ``prod(1 + d_i)``.

        Equivalently the number of acyclic orientations of the interaction
        graph, which is the number of complete orders the search visits with the
        upper-bound prune disabled and ample capacity (so nothing is evicted).

        ``d_i`` counts the interacting neighbours of buffer ``i`` that come
        later in a *perfect elimination order*. For an interval graph that order
        is by end time (its forward neighbours all share the column just before
        ``end_time``, hence form a clique); start time -- the order used to
        canonicalise the search itself -- is not a perfect elimination order and
        would give the wrong product.
        """
        self._prepare(buffers)
        n = len(self._bufs)
        peo_rank = {
            idx: rank
            for rank, idx in enumerate(
                sorted(
                    range(n),
                    key=lambda i: (
                        self._bufs[i].end_time,
                        self._bufs[i].start_time,
                        self._bufs[i].name,
                    ),
                )
            )
        }
        product = 1
        for i in range(n):
            forward = sum(
                1 for j in self._interacts_with[i] if peo_rank[j] > peo_rank[i]
            )
            product *= 1 + forward
        return product
