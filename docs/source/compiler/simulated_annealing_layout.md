
# A   simulated annealing-based memory layout planner

The code base contains a **simulated annealing-based memory-layout planner**.
Like all memory layout planners, when given a set of buffers, each with a size and a
half-open lifetime `[start, end)`, it decides where to place them in a fixed-capacity scratchpad so that
the total size of buffers that fit is maximised. Its code is fairly tricky, which explains the existence
of this document, which started as a comment on the PR that introduced the solver.

The solver's code lives in `torch_spyre/_inductor/scratchpad/`,
which contains the following files.

- **`plan_solver.py`** — the shared `LifetimeBoundBuffer`
  data type, the `MemoryPlanSolver` ABC, and a simple `GreedyLayoutSolver`.
- **`permutation_layout.py`** — the core. A *permutation* is an allocation order;
  `PermutationBasedLayoutSolver` places each buffer on top of the earlier-placed buffers it overlaps
  in time (with in-place reuse), and maintains all addresses **incrementally** under
  `swap`/`rotate`. `ReferencePermutationBasedLayoutSolver` is a slow, obviously-correct O(n²)
  rebuild used as a test oracle, not used in production code. `Profile` is the contact-profile data
  structure the incremental solver is built on.
- **`imanishi_xu.py`** — `ImanishiXuSolverWithBuffers`, a simulated-annealing search over allocation
  orders (following a paper by Imanishi & Xu) that drives the permutation solver by composition, plus a family
  of cooling schedules (default: an auto-calibrated exponential). It is wired in as the opt-in
  `layout_solver = "imanishi_xu"` config option; the default stays `greedy`.
- **`benchmarks/`** and **`examples/scratchpad/`** — profiling scripts, result docs, and runnable
  examples.

**Validation philosophy:** every incremental operation is checked against the
from-scratch reference oracle — randomized *differential* tests, a gated *stress* suite
(`TORCH_SPYRE_STRESS_SCRATCHPAD=1`, tens of thousands of seeds), and in places *exhaustive*
enumeration of all small configurations. This is what makes the subtle in-place edge cases
trustworthy.

**Key invariants that recur:** lifetimes are half-open; per column, address order equals permutation
order (weakly — ties only for in-place reuse); at most two buffers share one address at one tick
(in-place legality caps it); and an in-place pair overlaps at exactly one "transition" tick.

## The work, by theme

Roughly the order it was built.

### 1. Incremental capacity-bounded allocator

*Achieves:* the placement engine and its `swap`. *How:* place in permutation order, each buffer at
`align_up(max top of earlier overlapping)`, with an in-place child allowed to reuse a parent's slot.
`swap` re-places only the affected buffers (min-heap by position; candidates bounded by a
precomputed time-overlap set) rather than rebuilding. The reference O(n²) oracle and full
differential coverage arrive here. (This era uses a set-based *neighbour graph*, later replaced —
see theme 5.)

### 2. Imanishi/Xu annealing solver

*Achieves:* the search that actually optimises the layout. *How:* simulated annealing over
allocation orders. Each step picks a buffer and probes every reinsertion position by bubbling it
across a throwaway `copy()` of the plan, recording `quality()` (total size under capacity,
maintained O(1)) at each, and accepts a move by the Metropolis criterion.

### 3. Cooling schedules and ergonomics

*Achieves:* usable, self-calibrating annealing plus visualisation and examples. *How:* an
acceptance-*responsive* `CoolingSchedule` interface — `reset()` returns the first temperature and
`update(accepted)` the next (or `None` to stop), so a schedule can react to the run. Several
implementations exist: `ExponentialCoolingSchedule` (fixed geometric cool-down),
`CoolingScheduleFromPaper` (kept for comparison), `ReheatingSchedule` (finds the productive
temperature from the acceptance rate, then warm-restarts a band around it), and
`IterableCoolingSchedule`.

The **default is `AutoExponentialCoolingSchedule`** — geometric cooling whose **endpoints are
calibrated to the instance** rather than hard-coded. The solver runs a short warm-up (random
reinsertions on a throwaway copy), measures the typical `|Δquality|` of a move, and sets the start
temperature to accept a mean-magnitude *worsening* move with probability ~0.8 and the end
temperature with ~0.01, then cools geometrically. This transfers across problems of very different
byte scales instead of relying on tuned constants (which would silently degenerate into a random
walk or a greedy search on a differently scaled instance) — the choice favoured by other self-calibrating
SA libraries, and the robust default while we lack representative example models. It is documented
as a *reasonable, non-definitive* default to revisit (e.g. with an acceptance-targeting adaptive
schedule) once we can benchmark on real workloads. Knobs: separate `warmup_steps` and `total_steps`;
the warm-up defaults to a fraction of the budget; the budget is adaptive (`clamp(30·n, 500, 5000)`),
so the n=100 example uses 3000 steps and nothing exceeds 5000. The compile path uses a seeded RNG,
so layout planning is deterministic.

### 4. Composition refactor + copy-vs-swap study

*Achieves:* the architecture where the annealer *owns* a plan and probes on copies, justified by
measurement. *How/why:* the benchmark showed `copy()` is a cheap O(n) and that a second copy in the
sweep wouldn't pay off — so probing on a copy beats sweep-and-restore.

### 5. Contact profiles (replace the neighbour graph)

*Achieves:* a cheaper, sounder representation of who-is-below-whom. *How:* a `Profile` is a step
function over a buffer's lifetime giving its directly below/above neighbour per column. A `swap`
transposes the two buffers only over their shared column range via O(segments) splices instead of a
rebuild. Addresses propagate along *order-above* edges, with an **in-place-status transition rule**
that handles the "poke-through" case (a transparent in-place child sits low while its taller parent
pokes up to carry the buffer above) — this is the subtle bug the differential harness caught and a
follow-up fixed. `contact_at` is a derived, faithful "what does this buffer rest on" view.

### 6. Module split + placement optimisations

*Achieves:* the substrate/policy split (`permutation_layout.py` carved out of `plan_solver.py`) and
a ~38% faster solve. *How:* profiling showed `_placement_decision` dominated — precomputing each
buffer's (static, sparse) in-place-partner set removed ~30M wasted `_in_place_pair` probes, and
inlining `_top` over a flat sizes array removed ~30M method calls.

### 7. Fast rotate for long moves

*Achieves:* a `rotate(i, j)` that is independent of `|i − j|`. *How:* past a
distance threshold, edit the permutation once and recompute addresses (reusing the static overlap
set, never the O(n²) reference scan) plus an incremental single-move contact-profile patch, instead
of `|i − j|` adjacent swaps that re-place the moved element repeatedly. ~50× faster for long dense
rotations; correct in both reuse directions (40k forced-fast-path stress seeds).

### 8. Symmetric `contact_at` + contact-based re-placement

*Achieves:* `_recompute_address` now reads candidates off the contact profile rather than scanning
the whole overlap set. *How:* `contact_at` is made symmetric — it reports the `(parent, child)`
in-place pair in *both* reuse directions, surfacing the buried co-located buffer the in-place
legality test needs. The candidate set built from `contact_at` over a buffer's below-profile
breakpoints is **provably sufficient** for `_placement_decision` (a sufficiency proof, exhaustive
checks to n≤4, and ~1M live per-placement equivalence checks), then inlined to skip a redundant
per-breakpoint bisect.

## Discussion points

Open design questions for this PR, with pros and cons of each alternative.

### Should the O(n²) oracle be in testing or in the production code base?

Today `ReferencePermutationBasedLayoutSolver` lives in `permutation_layout.py` beside the
incremental solver, sharing `PermutationBasedLayoutSolverBase`, and is used only as a differential
test oracle.

**Keep it in the production module (current).**

- Pros: it sits next to the base class and placement primitives (`_placement_decision`,
  `_in_place_pair`) it is built from, so there are no cross-imports of production internals into
  test code; it doubles as a clean, readable reference spec of the placement semantics, usable for
  debugging or validating future solvers; and the shared abstract base genuinely has two concrete
  subclasses, which is what it was designed for.
- Cons: it is dead weight on the compile path — never used in production, yet it implements the
  solver interface, so it reads as a shippable option and could be selected by mistake; it adds API
  surface and an O(n²) implementation with no production purpose.

**Move it to a testing-utilities module.**

- Pros: intent becomes unambiguous — it is a test fixture, lives with the tests, and cannot be
  confused for a production solver; the production module stays focused on the one solver that
  ships.
- Cons: it depends on `PermutationBasedLayoutSolverBase` and the placement primitives, so the move
  either exposes those internals to test code (a fuzzier boundary that must track production
  changes) or duplicates logic; and the base class is then left with a single production subclass,
  weakening the "shared base for two implementations" rationale.

### Should the benchmarks be included in the commit or considered ephemeral?

The branch commits three profiling scripts and three result docs under `benchmarks/`.

**Commit them (current).**

- Pros: anyone can rerun the scripts and check the numbers; the result docs record both the measured
  behaviour and the *rationale* for design choices (the copy-vs-swap decision, the placement
  optimisations, the cooling-schedule budget), which is valuable to future maintainers; and the
  scripts act as living performance documentation that can be re-run after later changes.
- Cons: the numbers bit-rot — they are machine- and run-specific, and a committed doc with stale
  figures can mislead (we already had to refresh them once after renames); nothing enforces that the
  scripts still run or that the figures are current, so they are unguarded dead weight that grows
  over time and arguably belongs in a wiki/notebook rather than the source tree.

**Treat them as ephemeral (do not commit).**

- Pros: the source tree stays lean and product-focused, with no stale-numbers risk and no
  commit-hygiene burden on throwaway profiling scripts.
- Cons: reproducibility and the recorded rationale are lost, so the next person re-derives the
  evidence behind each design decision from scratch.

### Fast-rotate dispatch threshold

`rotate(i, j)` uses the remove/reinsert fast path once `|i − j|` reaches
`max(2, n // 8)`, and the swap chain below it. Correctness is independent of the threshold, so this
is purely a performance choice.

**Fixed `n // 8` (current).**

- Pros: simple, predictable, one comprehensible constant with no per-instance computation; backed by
  the measured crossover (the fast path wins above ~0.04–0.15·n at medium overlap density and
  ~0.13–0.37·n at low), and `n//8` sits below the medium-density crossover so it engages where it
  clearly pays.
- Cons: it is a magic constant that ignores density, so it is mildly pessimistic at low density (it
  engages the fast path a little before its crossover there, costing sub-millisecond time) and is
  not obviously right across the whole range.

**Density-adaptive (key off average overlap degree).**

- Pros: tracks the measured crossover everywhere — capturing medium-density wins down to ~0.05·n and
  avoiding the low-density pessimism — using the average degree, which `overlaps` already makes
  available for free.
- Cons: more complexity for a small payoff (the accepted rotate is a slice of the step; the bubble
  sweep dominates), another heuristic to justify and test, and the crossover model is itself
  average-case, so adaptivity risks chasing noise.

**Expose it / raise it / leave the fast path off by default.**

- Pros: lets callers tune per workload; "off by default" keeps the fully-proven swap chain as the
  only active rotate until the fast path is clearly warranted.
- Cons: forfeits the large dense-regime wins by default (e.g. ~255 ms → ~11 ms for a full rotate at
  n=1000 for a density that is probably unrealistically high) and adds a knob users must understand.

### Fast-rotate profile-update modes

The fast rotate updates the contact profiles in one of two modes (`_rotate_profile_mode`): `"patch"`
(the default incremental single-move splice) or `"rebuild"` (a full `_build_profiles`).

**Keep both modes (current).**

- Pros: `"rebuild"` is a simple, obviously-correct baseline kept as a safety net and as the
  measurement reference Stage 1 was validated against — if a future change breaks the patch,
  flipping the mode is an instant diagnostic and fallback; it is cheap to keep (a one-line branch
  reusing existing code).
- Cons: two code paths to maintain and to cover in the differential suite; the rebuild path is dead
  in production (the patch is always the default and is validated to 40k+ seeds), so it is latent
  code that can rot and invites the question "when would I ever set rebuild?", plus an instance
  attribute that production never sets.

**Drop the rebuild fallback.**

- Pros: one code path, less to maintain and test, and clearer intent; the patch is exhaustively
  validated, so the safety net is arguably unnecessary.
- Cons: loses the easy correctness-diagnostic and the measurement baseline, so a subtle future patch
  bug has no instant fallback to bisect against; and re-introducing the mode later is more work than
  keeping the branch.
