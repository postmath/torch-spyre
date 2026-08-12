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

"""Warm-start transfer experiment for the joint work-division + LX-layout SA.

WHAT THIS DECIDES
-----------------
The joint annealer keeps *one* layout permutation ``pi`` and re-uses it across
work-division changes as a "warm start" (CO_OPTIMIZING_PLAN.md, "Layout stays
meaningful across a work-division change"). That warm start is the whole reason
the joint loop can be cheaper than the nested baseline (enumerate divisions x
full layout solve each). But it also creates the risk in review point #1: a
division flip is judged by Metropolis against a ``pi`` tuned for the *old*
division, so good flips can be rejected and the search collapses to "layout
search around the seed division" == the lower bound (guess one division, solve
layout once).

The severity of #1 is exactly the validity of the warm-start claim, so we
measure it directly. At the layout solver's interface a division change is just:
  (a) RESIZE a correlated subset of buffers (size = total / prod(splits)), and
  (b) TOGGLE eligibility of some buffers (K-split eviction).
on a STABLE buffer set. This harness models (a) -- resize-only, which is the
CONSERVATIVE case: eligibility toggling (b) only removes buffers and relieves
capacity pressure, making the warm start easier. So resize-only lower-bounds how
well the warm start transfers.

Everything runs against the existing ``SimulatedAnnealingSolverWithBuffers`` on synthetic
buffer universes. No division machinery, memory term, or ``quality()``
reweighting is needed -- this de-risks the core bet using only what exists today.

The HBM-value reweighting of ``quality()`` has landed (commit f012b3d1:
``buffer_quality = (uses + write_bonus) * size``, a value != weight knapsack),
so this harness already scores on the real layout objective. The one piece not
reflected -- access multiplicity driven by the *consuming op's split* (matmul
cohort re-reads) -- belongs to the joint optimizer's memory term, not
``quality()``, so it does not affect this layout-transfer experiment.

To use REAL universes instead of synthetic ones, dump a graph's
scratchpad buffer set (name, size, uses) under the seed division to JSON and
load it in ``make_universe`` -- the rest of the harness is unchanged.

TWO PARTS
---------
Part A -- Transfer curve. For each division-change magnitude, how many warm
    reorder ("burst") steps b does it take for the warm-started layout to reach
    within eps of a cold full-budget solve, vs a cold start from scratch?
      * b*_warm << L and b*_warm << b*_cold  => warm start carries real
        reusable ordering information; the joint loop lives in the target gap and
        b*_warm is the burst size to configure.
      * b*_warm ~ L (~ a full solve)         => ordering does not transfer; the
        joint framing buys nothing over the nested baseline. Reconsider.

Part B -- Decision fidelity (the direct test of the #1 collapse risk). Does
    ranking divisions by their warm-seed score (stale pi, b=0) agree with ranking
    them by their own cold-optimal score? High agreement => Metropolis on the
    warm seed will not systematically mis-accept/reject divisions; collapse-to-
    baseline is unlikely. We also report agreement after a small burst (the
    compound "flip + burst" move), which should be strictly better.

Run from the repository root::

    python docs/source/user_guide/examples/scratchpad/warm_start_transfer.py
"""

import copy
import math
import random as rnd
from itertools import accumulate
from typing import Optional

import torch  # noqa: F401  (import torch first so it autoloads the backend)
import torch_spyre  # noqa: F401  (finish autoload before deep submodule imports)

from torch_spyre._inductor.scratchpad.plan_solver import LifetimeBoundBuffer
from torch_spyre._inductor.scratchpad.permutation_layout import (
    PermutationBasedLayoutSolver,
)
from torch_spyre._inductor.scratchpad.simulated_annealing import (
    SimulatedAnnealingSolverWithBuffers,
)
from torch_spyre._inductor.scratchpad.cooling_schedules import (
    SelfCalibratingReheatingSchedule,
    peak_memory_load,
)

ALIGN = 128


# --------------------------------------------------------------------------- #
# Buffer universe + division-change (resize) model                            #
# --------------------------------------------------------------------------- #
def make_universe(n: int, seed: int) -> list[LifetimeBoundBuffer]:
    """A synthetic buffer universe (biased to large sizes, short lifetimes),
    matching the repo's random_buffers example so results are comparable.

    Replace this body with a loader for a real dumped universe to raise
    fidelity: ``[LifetimeBoundBuffer(name, size, uses) for ... in json]``."""
    random = rnd.Random(seed)
    buffers = []
    for i in range(n):
        duration = random.randrange((n - 1) // 2)
        duration = duration * duration // (n - 1)
        t_start = random.randrange(n - duration)
        t_end = t_start + duration + 1
        size = random.randrange(1_000_000)
        size = max(1, math.isqrt(size * 1_000_000))
        uses = [t_start] if t_end == t_start else [t_start, t_end]
        buffers.append(LifetimeBoundBuffer(f"B{i}", size, uses))
    return buffers


# Per-core size ratios a division flip induces: splitting an op more (k -> m*k)
# shrinks its buffers by 1/m; splitting less grows them. A compatibility region
# re-tiles coherently, so one factor is applied to the whole region.
_REGION_FACTORS = [1.0 / 3.0, 0.5, 2.0, 3.0]


def perturb(
    base: list[LifetimeBoundBuffer], magnitude: float, seed: int
) -> list[LifetimeBoundBuffer]:
    """Model a work-division change: coherently resize a time-clustered region
    covering ``magnitude`` of the universe (a compatibility region re-tiling).
    Buffer set and lifetimes are unchanged (the plan's stable universe); only
    sizes move. ``magnitude`` in (0, 1]."""
    random = rnd.Random(seed)
    n = len(base)
    k = max(1, round(magnitude * n))
    factor = random.choice(_REGION_FACTORS)
    # Region = the k buffers whose lifetimes cluster nearest a random center
    # tick (a proxy for "ops near each other in the schedule share tensors").
    center = random.uniform(0, n)
    order = sorted(range(n), key=lambda i: abs(base[i].start_time - center))
    region = set(order[:k])
    out = copy.deepcopy(base)
    for i in region:
        out[i].size = max(1, round(out[i].size * factor))
    return out


# --------------------------------------------------------------------------- #
# Layout-solver drivers                                                        #
# --------------------------------------------------------------------------- #
def _permutation_quality(
    buffers: list[LifetimeBoundBuffer], perm: list[int], capacity: int
) -> float:
    """Quality of applying ``perm`` to ``buffers`` with no refinement (b = 0)."""
    return PermutationBasedLayoutSolver(
        copy.deepcopy(buffers), list(perm), capacity, ALIGN
    ).quality()


def solve_curve(
    buffers: list[LifetimeBoundBuffer],
    capacity: int,
    budget: int,
    seed: int,
    *,
    initial,
    cycles: int,
) -> tuple[float, list[int], list[float]]:
    """Run the SA layout solver for ``budget`` steps. Returns
    (best_quality, best_permutation, best_so_far_curve) where the curve[b] is the
    best quality found within the first b steps (monotone non-decreasing).

    ``initial`` is a first permutation (list[int]) for a WARM start, or
    "first_fit" for a COLD start. ``cycles`` selects the schedule regime:
    1 == a single self-calibrated cool (a refinement "burst", right for warm);
    >1 == reheating exploration (right for a cold from-scratch solve)."""
    solver = SimulatedAnnealingSolverWithBuffers(
        copy.deepcopy(buffers),
        capacity,
        ALIGN,
        initial=initial,
        schedule=SelfCalibratingReheatingSchedule(total_steps=budget, cycles=cycles),
        random=rnd.Random(seed),
    )
    # The constructor resolves "first_fit"/etc. into a concrete permutation in
    # solver.initial and sets best_quality to that starting layout's quality.
    init_q = _permutation_quality(buffers, solver.initial, capacity)
    solver.solve()
    log = solver.quality_logs[-1] if solver.quality_logs else []
    curve = list(accumulate([init_q, *log], max))
    return solver.best_quality, solver.best_permutation, curve


def cold_reference(
    buffers: list[LifetimeBoundBuffer], capacity: int, budget: int, restarts: int
) -> tuple[float, list[int]]:
    """Best cold solve over several restarts -- the "true optimum" proxy Q*."""
    best_q, best_perm = -math.inf, []
    for r in range(restarts):
        q, perm, _ = solve_curve(
            buffers, capacity, budget, seed=1000 + r, initial="first_fit", cycles=4
        )
        if q > best_q:
            best_q, best_perm = q, perm
    return best_q, best_perm


def steps_to_parity(curve: list[float], target: float, eps: float) -> Optional[int]:
    """First step index at which the curve reaches (1 - eps) * target, or None."""
    threshold = (1.0 - eps) * target
    for b, q in enumerate(curve):
        if q >= threshold:
            return b
    return None


# --------------------------------------------------------------------------- #
# Rank-correlation helpers (Spearman via Pearson-on-ranks; no scipy dep)       #
# --------------------------------------------------------------------------- #
def _ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):  # average ties
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(a: list[float], b: list[float]) -> float:
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    va = math.sqrt(sum((x - ma) ** 2 for x in a))
    vb = math.sqrt(sum((y - mb) ** 2 for y in b))
    return cov / (va * vb) if va > 0 and vb > 0 else 0.0


def spearman(a: list[float], b: list[float]) -> float:
    return _pearson(_ranks(a), _ranks(b))


def pairwise_sign_agreement(pred: list[float], truth: list[float]) -> float:
    """Fraction of pairs (i, j) where pred and truth agree on which is better --
    i.e. how often the warm-seed score picks the better division."""
    n, agree, total = len(pred), 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            ts = truth[i] - truth[j]
            if ts == 0:
                continue
            ps = pred[i] - pred[j]
            total += 1
            if (ps > 0) == (ts > 0):
                agree += 1
    return agree / total if total else 1.0


def weighted_sign_agreement(pred: list[float], truth: list[float]) -> float:
    """Sign-agreement with each pair weighted by the STAKES of the decision,
    ``|truth_i - truth_j|`` (the Q* gap the annealer would misjudge if it ranked
    the pair wrong). Near-tie pairs contribute ~0; large-gap pairs dominate. So a
    high weighted score with a lower unweighted score means the residual errors
    are low-stakes near-ties -- a wrong Metropolis call there costs almost
    nothing. Returns a fraction in [0, 1]."""
    num = den = 0.0
    n = len(pred)
    for i in range(n):
        for j in range(i + 1, n):
            w = abs(truth[i] - truth[j])
            if w == 0.0:
                continue
            den += w
            if (pred[i] - pred[j] > 0) == (truth[i] - truth[j] > 0):
                num += w
    return num / den if den else 1.0


def _median(xs):
    s = sorted(xs)
    return s[len(s) // 2] if s else float("nan")


# --------------------------------------------------------------------------- #
# Experiment                                                                   #
# --------------------------------------------------------------------------- #
def run(
    *,
    n: int = 40,
    capacity_frac: float = 0.6,
    budget_per_buffer: int = 30,
    ref_multiple: int = 6,
    ref_restarts: int = 3,
    eps: float = 0.02,
    magnitudes: tuple[float, ...] = (0.1, 0.25, 0.5),
    trials_per_magnitude: int = 6,
    fidelity_perturbations: int = 24,
    burst_per_buffer: int = 3,
) -> None:
    base = make_universe(n, seed=0)
    # Fixed LX capacity, sized off the BASELINE peak load and held constant
    # across divisions (LX does not grow when you split more). Moderate pressure
    # so the layout actually matters (some spill).
    capacity = int(capacity_frac * peak_memory_load(base))
    L = budget_per_buffer * n  # warm/cold step budget being compared
    ref_budget = ref_multiple * L  # generous budget for the Q* reference
    burst = burst_per_buffer * n  # compound-move "flip + burst" size

    # Baseline-optimal ordering pi0 -- the warm start carried into every flip.
    _, pi0 = cold_reference(base, capacity, ref_budget, ref_restarts)

    print("=" * 78)
    print("WARM-START TRANSFER EXPERIMENT (resize-only == conservative)")
    print(
        f"n={n}  capacity={capacity} ({capacity_frac:.0%} of baseline peak)  "
        f"budget L={L}  ref_budget={ref_budget}  eps={eps:.0%}"
    )
    print("=" * 78)

    # -- Part A: transfer curves ------------------------------------------- #
    print("\n[Part A] Steps-to-parity: warm start (from pi0) vs cold start.")
    print(
        f"{'mag':>5} {'b=0 warm/Q*':>12} {'b*_warm':>9} {'b*_cold':>9} "
        f"{'b*_warm/L':>10} {'warm<cold':>10} {'reach<=L':>9}"
    )
    for mag in magnitudes:
        seed_ratios, bw, bc, warm_wins, reached = [], [], [], 0, 0
        for t in range(trials_per_magnitude):
            u = perturb(base, mag, seed=100 * int(mag * 100) + t)
            qstar, _ = cold_reference(u, capacity, ref_budget, ref_restarts)
            seed_q = _permutation_quality(u, pi0, capacity)
            _, _, warm_curve = solve_curve(
                u, capacity, L, seed=7, initial=pi0, cycles=1
            )
            _, _, cold_curve = solve_curve(
                u, capacity, L, seed=7, initial="first_fit", cycles=4
            )
            # Reference = best quality anyone achieved for this division.
            target = max(qstar, warm_curve[-1], cold_curve[-1])
            seed_ratios.append(seed_q / target if target > 0 else 1.0)
            w = steps_to_parity(warm_curve, target, eps)
            c = steps_to_parity(cold_curve, target, eps)
            bw.append(w if w is not None else L)
            bc.append(c if c is not None else L)
            if w is not None:
                reached += 1
            if (w if w is not None else L) < (c if c is not None else L):
                warm_wins += 1
        print(
            f"{mag:>5.2f} {_median(seed_ratios):>12.3f} {_median(bw):>9} "
            f"{_median(bc):>9} {_median(bw) / L:>10.3f} "
            f"{warm_wins}/{trials_per_magnitude:<8} "
            f"{reached}/{trials_per_magnitude}"
        )
    print(
        "\n  Read: 'b=0 warm/Q*' near 1.0 => the ordering alone (no refinement)\n"
        "  nearly recovers the optimum. b*_warm << b*_cold and << L => the warm\n"
        "  start carries reusable information; use ~b*_warm as the burst size.\n"
        "  b*_warm ~ L => ordering does not transfer; nested baseline is no worse."
    )

    # -- Part B: decision fidelity ----------------------------------------- #
    print("\n[Part B] Do warm-seed scores rank divisions like their own optima?")
    qstars, seed_scores, burst_scores = [], [], []
    rng = rnd.Random(42)
    for k in range(fidelity_perturbations):
        mag = rng.choice(magnitudes)
        u = base if k == 0 else perturb(base, mag, seed=9000 + k)
        qstar, _ = cold_reference(u, capacity, ref_budget, ref_restarts)
        seed_q = _permutation_quality(u, pi0, capacity)
        _, _, burst_curve = solve_curve(
            u, capacity, burst, seed=7, initial=pi0, cycles=1
        )
        qstars.append(qstar)
        seed_scores.append(seed_q)
        burst_scores.append(burst_curve[-1])
    print(
        f"  perturbations={fidelity_perturbations}  burst={burst} steps "
        f"({burst_per_buffer}n)"
    )
    print(
        f"  Spearman(warm-seed b=0, Q*)      = {spearman(seed_scores, qstars):+.3f}"
        f"   sign-agreement = {pairwise_sign_agreement(seed_scores, qstars):.1%}"
    )
    print(
        f"  Spearman(flip+burst, Q*)         = {spearman(burst_scores, qstars):+.3f}"
        f"   sign-agreement = {pairwise_sign_agreement(burst_scores, qstars):.1%}"
    )
    print(
        "\n  Read: high sign-agreement (say >~85%) => Metropolis on the warm state\n"
        "  ranks divisions almost like their own optima, so good flips are NOT\n"
        "  systematically rejected -- the #1 collapse-to-baseline risk is low, and\n"
        "  the compound flip+burst row shows how much a small burst buys back."
    )


def _sweep_once(
    base,
    pi0,
    capacity,
    *,
    magnitudes,
    per_magnitude,
    bursts,
    ref_budget,
    ref_restarts,
    warm_seed,
    pert_seed_start,
):
    """Part-B decision fidelity vs burst size for ONE base instance. For each
    magnitude, build a population of that-magnitude divisions, score each by its
    own cold optimum Q* (once), then -- for each burst size b (steps) -- run a
    DEDICATED b-step warm solve from pi0 and measure how well the refined score
    ranks the divisions vs Q*. Returns (agree, wagree, spear) -- unweighted
    sign-agreement, stakes-weighted sign-agreement, Spearman -- dicts keyed by
    (mag|'ALL', b). ``bursts`` is a list of step counts (0 == raw stale layout)."""
    pops = {}
    seed = pert_seed_start
    for mag in magnitudes:
        divs, qstars = [], []
        for _ in range(per_magnitude):
            u = perturb(base, mag, seed=seed)
            seed += 1
            qstar, _ = cold_reference(u, capacity, ref_budget, ref_restarts)
            divs.append(u)
            qstars.append(qstar)
        pops[mag] = (divs, qstars)

    def scores_at(divs, b):
        if b == 0:
            return [_permutation_quality(u, pi0, capacity) for u in divs]
        return [
            solve_curve(u, capacity, b, seed=warm_seed, initial=pi0, cycles=1)[0]
            for u in divs
        ]

    agree: dict = {}
    wagree: dict = {}
    spear: dict = {}
    pooled_scores: dict[int, list[float]] = {b: [] for b in bursts}
    pooled_q: list[float] = []
    for mag, (divs, qstars) in pops.items():
        pooled_q.extend(qstars)
        for b in bursts:
            s = scores_at(divs, b)
            pooled_scores[b].extend(s)
            agree[(mag, b)] = pairwise_sign_agreement(s, qstars)
            wagree[(mag, b)] = weighted_sign_agreement(s, qstars)
            spear[(mag, b)] = spearman(s, qstars)
    for b in bursts:
        agree[("ALL", b)] = pairwise_sign_agreement(pooled_scores[b], pooled_q)
        wagree[("ALL", b)] = weighted_sign_agreement(pooled_scores[b], pooled_q)
        spear[("ALL", b)] = spearman(pooled_scores[b], pooled_q)
    return agree, wagree, spear


def _print_sweep_table(title, cell, magnitudes, bursts_per_buffer, n):
    print(f"\n{title}")
    cols = "  ".join(
        f"{('b=0' if bpb == 0 else f'{bpb}n'):>9}" for bpb in bursts_per_buffer
    )
    print(f"{'mag':>6}  {cols}")
    for row in [*magnitudes, "ALL"]:
        label = f"{row:>6.2f}" if isinstance(row, float) else f"{row:>6}"
        cells = "  ".join(f"{cell(row, bpb * n):>9}" for bpb in bursts_per_buffer)
        print(f"{label}  {cells}")


def run_burst_sweep(
    *,
    n: int = 40,
    capacity_frac: float = 0.6,
    budget_per_buffer: int = 30,
    ref_multiple: int = 6,
    ref_restarts: int = 3,
    magnitudes: tuple[float, ...] = (0.1, 0.25, 0.5),
    per_magnitude: int = 12,
    bursts_per_buffer: tuple[int, ...] = (0, 1, 2, 3, 5, 8),
    warm_seed: int = 7,
    base_seed: int = 0,
) -> None:
    """Single-instance burst-size sweep (see :func:`_sweep_once`)."""
    base = make_universe(n, seed=base_seed)
    capacity = int(capacity_frac * peak_memory_load(base))
    ref_budget = ref_multiple * budget_per_buffer * n
    _, pi0 = cold_reference(base, capacity, ref_budget, ref_restarts)

    print("=" * 78)
    print("BURST-SIZE SWEEP (Part B decision fidelity vs burst steps)")
    print(
        f"n={n}  capacity={capacity} ({capacity_frac:.0%} of baseline peak)  "
        f"ref_budget={ref_budget}  per_magnitude={per_magnitude}"
    )
    print("=" * 78)

    agree, wagree, spear = _sweep_once(
        base,
        pi0,
        capacity,
        magnitudes=magnitudes,
        per_magnitude=per_magnitude,
        bursts=[bpb * n for bpb in bursts_per_buffer],
        ref_budget=ref_budget,
        ref_restarts=ref_restarts,
        warm_seed=warm_seed,
        pert_seed_start=20000,
    )
    _print_sweep_table(
        "Sign-agreement (fraction of division comparisons pointing the right way):",
        lambda row, b: f"{agree[(row, b)]:.0%}",
        magnitudes,
        bursts_per_buffer,
        n,
    )
    _print_sweep_table(
        "Stakes-weighted sign-agreement (pairs weighted by |Q* gap|):",
        lambda row, b: f"{wagree[(row, b)]:.0%}",
        magnitudes,
        bursts_per_buffer,
        n,
    )
    _print_sweep_table(
        "Spearman(refined score, Q*):",
        lambda row, b: f"{spear[(row, b)]:+.2f}",
        magnitudes,
        bursts_per_buffer,
        n,
    )
    print(
        "\n  Read: find the smallest burst whose row crosses ~85% -- that is the\n"
        "  burst to fold into the compound flip+burst move. Stakes-weighted rising\n"
        "  well above unweighted => residual errors are low-stakes near-ties."
    )


def run_burst_sweep_multi(
    *,
    n: int = 40,
    capacity_frac: float = 0.6,
    budget_per_buffer: int = 30,
    ref_multiple: int = 6,
    ref_restarts: int = 3,
    magnitudes: tuple[float, ...] = (0.1, 0.25, 0.5),
    per_magnitude: int = 32,
    bursts_per_buffer: tuple[int, ...] = (0, 1, 2, 3, 5, 8),
    n_instances: int = 5,
    base_seed_start: int = 100,
    warm_seed: int = 13,
) -> None:
    """Multi-instance burst sweep: repeat :func:`_sweep_once` over ``n_instances``
    independent base universes (fresh seeds) and report, per (magnitude, burst),
    the MEAN sign-agreement and the min-max SPREAD across instances. A large
    ``per_magnitude`` shrinks within-instance sampling noise; multiple instances
    expose instance-to-instance variance -- together these characterize how noisy
    the fidelity estimate really is."""
    ref_budget = ref_multiple * budget_per_buffer * n
    bursts = [bpb * n for bpb in bursts_per_buffer]

    print("=" * 78)
    print("MULTI-INSTANCE BURST-SIZE SWEEP (noise characterization)")
    print(
        f"n={n}  ref_budget={ref_budget}  instances={n_instances} "
        f"(seeds {base_seed_start}..{base_seed_start + n_instances - 1})  "
        f"per_magnitude={per_magnitude}  warm_seed={warm_seed}"
    )
    print("=" * 78)

    per_instance = []  # list of (agree, wagree, spear)
    for inst in range(n_instances):
        base = make_universe(n, seed=base_seed_start + inst)
        capacity = int(capacity_frac * peak_memory_load(base))
        _, pi0 = cold_reference(base, capacity, ref_budget, ref_restarts)
        agree, wagree, spear = _sweep_once(
            base,
            pi0,
            capacity,
            magnitudes=magnitudes,
            per_magnitude=per_magnitude,
            bursts=bursts,
            ref_budget=ref_budget,
            ref_restarts=ref_restarts,
            warm_seed=warm_seed,
            pert_seed_start=1_000_000 + inst * 100_000,
        )
        per_instance.append((agree, wagree, spear))
        # Raw per-instance ALL-row spread, streamed so it is visible mid-run
        # (unweighted / stakes-weighted).
        summary = "  ".join(
            f"{('b=0' if bpb == 0 else f'{bpb}n')}="
            f"{agree[('ALL', bpb * n)]:.0%}/{wagree[('ALL', bpb * n)]:.0%}"
            for bpb in bursts_per_buffer
        )
        print(
            f"  instance {inst} (seed {base_seed_start + inst}) ALL unwt/wtd: "
            f"{summary}",
            flush=True,
        )

    def _stat(idx, row, b, which):
        vals = [inst[idx][(row, b)] for inst in per_instance]
        if which == "mean":
            return sum(vals) / len(vals)
        return (min(vals), max(vals))

    _print_sweep_table(
        f"MEAN sign-agreement (unweighted) across {n_instances} instances:",
        lambda row, b: f"{_stat(0, row, b, 'mean'):.0%}",
        magnitudes,
        bursts_per_buffer,
        n,
    )
    _print_sweep_table(
        "SPREAD across instances (min-max, unweighted):",
        lambda row, b: (lambda lo, hi: f"{lo:.0%}-{hi:.0%}")(
            *_stat(0, row, b, "range")
        ),
        magnitudes,
        bursts_per_buffer,
        n,
    )
    _print_sweep_table(
        f"MEAN STAKES-WEIGHTED sign-agreement across {n_instances} instances:",
        lambda row, b: f"{_stat(1, row, b, 'mean'):.0%}",
        magnitudes,
        bursts_per_buffer,
        n,
    )
    _print_sweep_table(
        "SPREAD across instances (min-max, stakes-weighted):",
        lambda row, b: (lambda lo, hi: f"{lo:.0%}-{hi:.0%}")(
            *_stat(1, row, b, "range")
        ),
        magnitudes,
        bursts_per_buffer,
        n,
    )
    _print_sweep_table(
        f"MEAN Spearman(refined, Q*) across {n_instances} instances:",
        lambda row, b: f"{_stat(2, row, b, 'mean'):+.2f}",
        magnitudes,
        bursts_per_buffer,
        n,
    )
    print(
        "\n  Read: a tight SPREAD (esp. the ALL row) means the estimate is stable.\n"
        "  If STAKES-WEIGHTED sits well above unweighted at a given burst, the\n"
        "  residual misrankings are low-stakes near-ties -- a wrong Metropolis call\n"
        "  there costs little, so a smaller burst is adequate in practice."
    )


def run_burst_scaling(
    *,
    ns: tuple[int, ...] = (10, 20, 40, 80),
    capacity_frac: float = 0.6,
    budget_per_buffer: int = 30,
    ref_multiple: int = 3,
    ref_restarts: int = 2,
    magnitudes: tuple[float, ...] = (0.1, 0.25, 0.5),
    per_magnitude: int = 10,
    n_instances: int = 2,
    base_seed_start: int = 200,
    warm_seed: int = 17,
    abs_bursts: tuple[int, ...] = (0, 5, 10, 20, 40, 80, 160, 240, 320),
    target_wtd: float = 0.90,
) -> None:
    """How does the burst size needed for decision fidelity scale with n?

    The '3n' figure came from a single n=40 run. Here we hold the burst in
    ABSOLUTE step-counts (same sweep-steps as before) and sweep several n, so we
    can see whether b* grows ~linearly (b*/n flat), ~sqrt (b*/sqrt(n) flat), or
    ~constant (b* flat) in n. b*(n) = smallest absolute burst whose pooled
    stakes-weighted ALL sign-agreement reaches ``target_wtd``."""
    print("=" * 78)
    print("BURST-SIZE SCALING vs n (absolute sweep-steps)")
    print(
        f"ref_multiple={ref_multiple} restarts={ref_restarts} "
        f"per_magnitude={per_magnitude} instances={n_instances} "
        f"target_wtd={target_wtd:.0%}"
    )
    print("=" * 78)

    wtd_rows: dict[int, dict[int, float]] = {}
    unwtd_rows: dict[int, dict[int, float]] = {}
    for n in ns:
        ref_budget = ref_multiple * budget_per_buffer * n
        bursts = list(abs_bursts)
        wtd_acc = {b: [] for b in bursts}
        unwtd_acc = {b: [] for b in bursts}
        for inst in range(n_instances):
            base = make_universe(n, seed=base_seed_start + inst)
            capacity = int(capacity_frac * peak_memory_load(base))
            _, pi0 = cold_reference(base, capacity, ref_budget, ref_restarts)
            agree, wagree, _ = _sweep_once(
                base,
                pi0,
                capacity,
                magnitudes=magnitudes,
                per_magnitude=per_magnitude,
                bursts=bursts,
                ref_budget=ref_budget,
                ref_restarts=ref_restarts,
                warm_seed=warm_seed,
                pert_seed_start=2_000_000 + inst * 1_000_000,
            )
            for b in bursts:
                wtd_acc[b].append(wagree[("ALL", b)])
                unwtd_acc[b].append(agree[("ALL", b)])
        wtd_rows[n] = {b: sum(v) / len(v) for b, v in wtd_acc.items()}
        unwtd_rows[n] = {b: sum(v) / len(v) for b, v in unwtd_acc.items()}
        print(
            f"  n={n:>3} done: "
            + "  ".join(f"{b}={wtd_rows[n][b]:.0%}" for b in bursts),
            flush=True,
        )

    def _table(title, rows):
        print(f"\n{title}")
        hdr = "  ".join(f"{b:>5}" for b in abs_bursts)
        print(f"{'n':>4}  {hdr}")
        for n in ns:
            print(f"{n:>4}  " + "  ".join(f"{rows[n][b]:>4.0%}" for b in abs_bursts))

    _table("Stakes-weighted ALL sign-agreement vs absolute burst steps:", wtd_rows)
    _table("Unweighted ALL sign-agreement vs absolute burst steps:", unwtd_rows)

    print(
        f"\nb*(n) = smallest absolute burst reaching stakes-weighted {target_wtd:.0%}:"
    )
    print(f"{'n':>4}  {'b*':>6}  {'b*/n':>7}  {'b*/sqrt(n)':>11}")
    for n in ns:
        bstar = next((b for b in abs_bursts if wtd_rows[n][b] >= target_wtd), None)
        if bstar is None:
            print(f"{n:>4}  {'>max':>6}  {'-':>7}  {'-':>11}")
        else:
            print(
                f"{n:>4}  {bstar:>6}  {bstar / n:>7.2f}  {bstar / math.sqrt(n):>11.2f}"
            )
    print(
        "\n  Read: whichever of b*/n or b*/sqrt(n) is ~flat across n is the scaling."
        "\n  Flat b* (both columns falling) => burst is ~constant in n."
    )


if __name__ == "__main__":
    import sys

    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    if arg == "scaling":
        run_burst_scaling()
    elif arg == "sweepx":
        run_burst_sweep_multi()
    elif arg == "sweep":
        run_burst_sweep()
    else:
        run()
