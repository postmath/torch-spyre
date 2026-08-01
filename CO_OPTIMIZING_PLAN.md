# **Joint Work-Division \+ LX Layout via Simulated Annealing: Implementation Plan** (revised)

## **1\. System Goal and Objective Function**

### **1.1 Co-Optimization Target**

The architecture implements a single joint annealing loop over a combined layout-permutation ($\\pi$) and work-division ($W$) state space. This avoids an outer work-division search wrapping an inner layout solver, optimizing per-op core splits and LX scratchpad layouts simultaneously. Joint optimization is necessary because an operation's work division dictates the per-core sizes and availability of the buffers that the layout solver must pack.

This loop is realized as a **third engine on an existing co-optimization substrate** — a colleague's joint core-division + LX-placement framework, a sibling to its CP-SAT and DFS engines. §7 is the authoritative integration story and governs the SA engine wherever the substrate-agnostic design in §§1–6 and the substrate differ; §§2–6 describe the design largely independent of it.

### **1.2 Objective Math**

The optimizer evaluates states using a single soft cost function that approximates total runtime in time units, removing hard feasibility gates. The score for a joint state $(\\pi, W)$ is calculated as follows:  
```math
\text{Score} = \frac{\text{Total HBM Value} - \text{Quality}(\pi, W)}{\text{HBM\_BW}} + \sum_{\text{nodes}} \text{Node Cost}(W)
```

* **Memory (HBM) Term**: Quantifies the time required to transfer every tensor byte not served directly from the LX scratchpad. This represents a single unified traffic pool; a tensor misses LX either due to an incompatible tiling edge or an allocation spill, but both facets are counted exactly once to avoid double-counting. Bytes are converted to time using an HBM bandwidth constant ($\\text{HBM\\\_BW}$).  
* **Node Term**: Accumulates per-op runtime costs that do not have a tensor-residency analog, evaluated via a context-free node oracle. This captures core compute performance, the PSUM ring overhead, and batch penalties.  
* **Quality Function**: Represents the layout solver's kept HBM value (the total bytes $\\times$ access multiplicity saved by remaining LX-resident and compatible). The buffer capacity weight is determined by its per-core footprint, forming a value $\\neq$ weight knapsack problem.

### **1.3 Numerical Execution**

To maintain perfect bit-for-bit consistency and eliminate floating-point non-determinism, the objective is accumulated using integer and fixed-point math. Buffer quality values are scaled by $\\times 2$ to eliminate fractional write bonuses. Node oracle microsecond outputs and HBM bandwidth time translations are mapped to a fixed-point scale via a single deterministic rounding step.

## **2\. Decision Variables and State Management**

### **2.1 The Joint State Space**

The annealing engine directly manipulates two correlated variables:

* **Layout Permutation ($\\pi$)**: Defined over a stable buffer universe containing all intermediate tensors that could ever reside in LX.  
* **Work Division ($W$)**: Maps each operation to a per-dimension split vector (dict\[dim \-\> count\]). The product of the splits must be $\\le \\text{SENCORES}$, each count must cleanly divide its dimension, and reduction dimensions are bound by a maximum of 1 split and a minimum floor defined by span reduction.

Per-tensor residency (LX vs. HBM) is a fully derived attribute, not a search variable. A tensor is placed in LX if and only if it is structurally eligible under the current compatibility constraints and the layout packer fits it without spilling. (On the co-optimization substrate this gains a third, division-invariant gate — a fixed residency **pin mask** (`placement`/`residency_reason`); permanently-pinned buffers are excluded from $\\pi$ entirely and only transient, division-dependent ineligibility "keeps its slot". See §7.4.)

### **2.2 Layout Persistence (Warm-Starting)**

The layout permutation $\\pi$ persists across work-division changes, ensuring that layout optimization progress is preserved. Division updates alter the underlying attributes of the stable buffer universe without changing the buffer set or allocation order:

* size(b): Recalculated as $\\frac{\\text{Total Size}}{\\prod \\text{splits over buffer dims}}$ based on the producer's current division.  
* eligible(b): Updated dynamically based on whether edge compatibility is maintained.

The layout packer processes $\\pi$ by skipping currently ineligible or spilled buffers, which are routed to HBM. The layout solver exposes three incremental, contact-profile-reusing operations to support this:

1. **Reorder**: Swaps or rotates buffer positions within $\\pi$.  
2. **Toggle-Eligibility**: Inserts or removes a buffer at its fixed slot.  
3. **Resize**: Dynamically alters a buffer's footprint in place.

## **3\. Compatibility Regions and Re-tiling Mechanics**

### **3.1 Edge Compatibility Rules**

A tensor's physical tiling across cores is fixed by its producing operation's division vector. A consuming operation can read this tensor directly from LX only if the endpoints' split projections match on every shared dimension:

* **Pointwise Consumers**: Must match the output split vector exactly on all shared dimensions.  
* **Reduction Consumers**: Must match on shared dimensions, and the reduction dimension itself must either be unsplit or handled via the cross-core PSUM ring path ($\\le 1$ split).

Edge compatibility is strictly **all-or-nothing** across a tensor's dimensions; a single mismatched dimension forces the entire tensor through an HBM round-trip.

### **3.2 Compatibility Regions**

Tensors and operations group into maximal subgraphs called **compatibility regions**, across which a single uniform tiling propagates. Regions are bounded by true dimension-changers, such as matmul contractions, reductions, or dim-adding broadcasts.

* **Reshape Traversability**: A split vector successfully propagates through a reshape if and only if it factors through the outer/major dimension factorization. Inner factor splits or cuts across merged strides act as region boundaries.  
* **Region Detection**: On the co-optimization substrate this is **not** separate machinery — a region is defined by **`cd_parent_matches` flood reachability** for a propagated tiling (see §7.2), which subsumes region detection (boundaries emerge as "no menu division carries the tiling across this edge"). The union-find / `propagate_named_dims` / reshape-factoring description here applies only to a standalone (non-substrate) build; membership is per-tiling either way.

### **3.3 Re-tiling Cost Seam**

All edge-miss and re-tiling costs are isolated inside a single extension function:

```python  
retile_cost(producer_tiling, consumer_tiling, tensor) -> int
```

Currently, this function returns 0 for an exact match or the full tensor byte-count for any mismatch. Higher-level "compatibility" states remain derived directly from this function evaluating to 0.

### **3.4 Stickification (STL) is Out of Scope**

Beyond core-division tiling and LX placement, Spyre data also carries a **stick layout** (STL): the inner "stick" dimension must be physically contiguous for the systolic array. A transpose that changes the stick dim is a true data shuffle (a `spyre.restickify` op), priced on a separate axis by the existing `optimize_restickify.py` optimiser, which this effort does **not** touch. So "transpose is region-traversable" means only that the split vector propagates *permuted* on the **tiling** axis — a transpose is not free; it may incur a restickify on the orthogonal STL axis.

This is a deliberate exclusion with a **known consequence**: because the objective prices tiling + residency HBM only and omits restickify, the search can select divisions that shift an *invisible* restickify cost (core division changes per-core shapes, which move restickify costs). That is a **score-incompleteness** inherited by the separate score-fidelity task (see Appendix G), not something this effort corrects; bidirectional co-optimisation would require reordering pipeline passes and is future work.

## **4\. Simulated Annealing Move Types**

The optimization loop selects moves from a unified proposal distribution and evaluates them via standard Metropolis criteria against the single objective function.

### **4.1 Reorder**

Executes standard reinsertion or swap operations directly on the layout permutation $\\pi$ to optimize scratchpad packing.

### **4.2 Atomic Division Flip**

Modifies a single operation's division. This local modification triggers in-place resizing and eligibility toggling for immediate neighbor buffers, rippling outward exclusively via the compatibility rules. (On the co-optimization substrate a division is a **menu-index choice** (`chosen_division`), so a flip changes one op's index — see §7.2.)

### **4.3 Region-Recolor**

Selects a compatibility region and propagates a single tiling across it via deterministic propagation and dimension-remapping. (On the substrate, "region" is defined by **`cd_parent_matches` flood reachability** for the propagated tiling, and "single label" means **mutually-compatible menu indices**, not a literal shared index — see §7.2.) The uniform-flood move is biased (see Appendix C); the fix is a **verify-first escalation**, not a hierarchy that is fully built up front. Start at Tier 0; escalate to Tier 1, then Tier 2, **only if** a run shows heterogeneous optima are actually being missed — Tiers 1 and 2 may never need to be implemented:

* **Tier 0 (Verify)**: Executes plain, reheat-gated uniform recoloring alongside atomic flips to verify if heterogeneous configurations are bypassed in practice.  
* **Tier 1 ($\\beta$-Biased Local Family)**: Samples target region-colorings from a proposal distribution $Q(\\sigma) \\propto \\exp(-\\beta \\cdot m(\\sigma))$, where $m(\\sigma)$ represents a multi-coloredness metric. The neighborhood is restricted to uniform colorings plus small local deviations. The normalizer cancels in the Metropolis ratio, providing a valid, mathematically rigorous diversification channel.  
* **Tier 2 (Exact Region-Block-Gibbs)**: Applied to low-treewidth region topologies (chains and trees). Uses dynamic programming and belief propagation to exactly sample the region's coloring conditional on its local compatibility and node terms. The proposed block state is then evaluated against the global objective (including post-burst layout spill terms) via a standard Metropolis-Hastings acceptance check. High-treewidth regions fall back to Tier 1.

### **4.4 Compound Structural Moves (Layout Burst)**

Atomic division flips and region-recolor moves are structural changes that modify buffer sizes and eligibility. To prevent good structural moves from being rejected due to an unrefined layout order, these moves are executed as **compound moves**.

Proposed Move (Flip or Recolor) ──\> Short Layout Burst (0.25n to 3n steps) ──\> Metropolis Evaluation

The short burst of warm reorder steps allows $\\pi$ to adapt to the new buffer dimensions *before* the Metropolis criteria is calculated. The entire compound sequence is judged as a single unit. To keep compile times bounded, burst sub-steps utilize $O(1)$ single-buffer reinsertions rather than full $O(n)$ optimization sweeps. The burst schedule utilizes either a short fixed cold cooling profile or a fixed low temperature, making the burst budget self-contained and independent of global annealing progress.

## **5\. Acceptance, Schedule, and Proposal Mix**

### **5.1 Schedule Tracking**

The optimization loop utilizes a single instance of SelfCalibratingReheatingSchedule. This schedule processes cyclic hot-to-cold reheating phases on a shared clock, tracking progress via a scale-invariant cycle phase ($\\phi \= \\frac{T}{\\text{band-top}}$) rather than total global steps. All exploration, mix adjustments, and freezing behaviors are tied directly to this cycle phase to ensure they remain synchronized during reheating cycles.

### **5.2 Multi-Move Scale Calibration**

Because moves span highly divergent energy deltas, the single shared clock tracks independent exponential moving averages (EMA, d\_hat\_m) of the move scales for each individual move type.

* **Shared Carrier, Independent Bands**: Each move type maps to an independent acceptance band defined by its own temperature scale ($center\_m \= \\frac{\\hat{d}\_m}{\\sqrt{AB}}$) and a designated cold floor.  
* **Temperature Query Interface**: The schedule exposes a temperature(move\_type) query method alongside an update(move\_type, accepted, move\_scale) feedback hook. This routes metrics directly to the corresponding move type's EMA, allowing rare moves (like region-recolor) to calibrate using horizons scaled to their own sample count.  
* **Acceptance Bands**: Layout reorders target an $\\approx 3\\%$ band, atomic division flips target an $\\approx 0.1\\%$ band, and region-recolor is restricted to an explicit coldest floor ($\\approx 0.01\\%$). This low floor, combined with a decaying reheating envelope as the landscape flattens ($\\hat{d}\_m \\rightarrow 0$), naturally freezes region-recolor moves earliest within each cycle without hard cutoffs.

### **5.3 Variance Control**

Initial deployments utilize per-group feedback exclusively. If instrumented runs reveal a high within-group coefficient of variation ($\\Delta E$ variance caused by small pointwise flips vs. large matmul flips), the affected move type is split into 2 or 3 size-bucketed sub-groups. Each sub-group maintains its own independent EMA feedback stream to ensure stable acceptance rates without distorting the underlying energy distribution.

### **5.4 Proposal Mix Weighting**

Proposal weights are proportional to the explicit decisions available within each move type's neighborhood, tracking the dimensionality of the search space:

* **Region-Recolor Weight**: Proportional to $n\_{\\text{regions}} \\times n\_{\\text{colors}}$.  
* **Atomic Flip Weight**: Proportional to $n\_{\\text{ops}} \\times \\text{degree}$ (available local labels).  
* **Reorder Weight**: Parametrized as a targeted $O(n)$ neighborhood via annealing\_step\_rotate, which selects one buffer and evaluates its valid reinsertion points in a single step. This replaces naive uniform $O(n^2)$ pairs to prevent layout proposals from overwhelming work-division exploration.

Weights are dynamically modulated within each cycle phase:  
```math
\text{Weight}(m) \propto \text{Neighborhood}(m) \times \max\left(w\_{\text{floor}}, \text{hotness}(m, \phi)\right)
```
This ensures structural moves dominate during hot phases, while layout reorders dominate during cold phases within every reheating cycle.

### **5.5 Pruning and Early Termination**

Three SA-loop shortcuts bound compile time and detect optimality (SA-internal — no engine-interface impact):

* **Zero-Spill Reorder Bypassing**: If the layout packer allocates every currently eligible buffer to LX with zero spills, reorder moves are entirely suspended. The loop proposes only division moves until a structural change induces a new allocation spill.  
* **Division Asymmetry**: No inverse pruning rule is applied to division moves when all buffers fit, because reducing work-division splits decreases node-term compute penalties, meaning the system cannot safely lock divisions to maximum split values.  
* **Global Termination Lower-Bound**: The loop terminates immediately if a state hits the theoretical minimum energy profile: the memory term equals 0 (all eligible buffers are fully allocated) and every operation simultaneously sits at its absolute node-cost-minimizing split configuration.

## **6\. Cost Service and Node Oracle**

### **6.1 Unified Scorer Interface**

A single shared scorer calculates the objective function, ensuring perfect alignment between standalone evaluation passes and the simulated annealing engine's incremental delta updates. Time in microseconds is the universal currency.

### **6.2 Traffic Geometry and Conservation**

To prevent double-counting, tensor data movement is governed by a single, decoupled geometry function: bytes\_moved(tensor, consumer\_split). The memory term evaluates this function multiplied by a residency factor (0 if an LX hit, 1 if an HBM miss). For a tensor $T$ consumed by operation $O$, access multiplicity is pinned via the following calculation:  
```math
\text{Multiplicity} = \prod_{d \in \text{dims}(O), , d \notin \text{dims}(T)} S_O[d]
```
This ensures that a matmul splitting an output dimension $M$ while reading a non-split operand correctly scales its input traffic (cohort re-reads) within the memory term.

**Two distinct "multiplicities" — do not conflate them**:

* The **memory-term** multiplicity above is the **division-dependent** cohort formula ($\\prod S\_O\[d\]$ over $O$'s split dims not in $T$). It varies with $W$ and is computed per edge during scoring.  
* The **`quality()`** weighting is a **static, op-access count** — the landed `buffer_quality = (len(uses) + write_bonus) × size` (commit f012b3d1), where `len(uses)` counts consuming ops, *not* the split-projection formula.

These are different numbers. Cohort re-reads belong to the **memory term**, never to `quality()`, precisely because `uses` is a static upstream count that does not vary with $W$. Applying the §6.2 formula to `quality()` would double-count multiplicity.

### **6.3 Node Oracle Contract**

The node oracle evaluates operation costs using a context-free dispatch registry keyed entirely by operation type and its own division vector:

```python  
node_cost(op, own_division) -> int | None
```

The oracle returns None (zero cost) for cheap, memory-bound pointwise operations (add, mul, relu, copy), as their execution overhead is negligible compared to the HBM traffic captured by the memory term. Matmuls, batch matmuls, and convolutions route to native underlying estimators (\_matmul\_split\_cost) with all internal hbm\_us factors stripped out. Cross-core reductions register costs tracking the physical cross-core PSUM reduction ring overhead.

### **6.4 Cost Model Pluggability Seam**

The additive combination ($\\text{Memory} \+ \\text{Node}$) acts as a pluggable service layer. To support future cost models (such as those incorporating compute/HBM overlap), the scorer enforces a strict architectural contract requiring all calculations to remain locally decomposable. The service must expose local contributions such that a single-op division change can be re-scored in $O(\\text{buffers/edges touched})$ time complexity, preventing full graph simulations during annealing steps.

### **6.5 Safety Guards**

The cost service implements three mandatory validation layers:

1. **`hbm_us` Conservation Test**: The strongest tripwire for the `hbm_us` strip. A matmul with all operands forced to HBM (LX disabled/incompatible) must satisfy `memory_term(I/O) == old_hbm_us` on an op-by-op basis — under integer accumulation an *exact* equality. This proves the strip neither double-counts nor drops bytes; any residual mismatch (e.g. `hbm_us` bundling assumptions that do not factor into geometry × residency) is made *visible* rather than silently biasing the split direction.  
2. **Debug Incremental Tripwire**: An active debug assertion executes a full graph objective recompute every $N$ moves, verifying that incremental\_total \== full\_recompute() to catch logic bugs in contact-profile splicing or edge tracking.  
3. **CI Determinism Test**: A continuous integration gate executes the entire optimization pass twice on identical inputs, asserting bit-for-bit identity across both the final op\_it\_space\_splits and the scratchpad layout plan.

## **7\. Integration onto the Co-Optimization Substrate**

This effort does **not** build the joint pipeline from scratch. A colleague's branch (`spectre-ns:refactor-scratchpad-cooptimization`, not yet landed and still subject to change) already provides a joint core-division + LX-placement framework with two engines (CP-SAT/ILP and brute-force DFS). We build the SA optimizer as a **third engine on that substrate** — which is exactly the "pluggable engines (exact/ILP for small graphs)" slot §7 anticipated. Building on it now, even pre-merge, sets up the eventual integration better than building parallel machinery. This section states how the SA design maps onto that substrate; where it and the substrate disagree, this section governs the SA engine.

> **Coupling surface (what we depend on from an unlanded branch — re-check on churn).** `CoOptimizingSolver` ABC (`plan_layout_and_core_divs(buffers) -> buffers`); `CoreDivision` (`output_splits`/`reduction_splits`, `output_partition`, `is_clean`, `signature_key`); `CoreDivisionBuffer` (candidate `core_divisions`, `chosen_division`, `parents`, `cd_parent_matches`, `placement`/`residency_reason`, `boundary_cost`/`spill_write_cost`); the allocator's `cd_parent_matches` / `_per_core_view` slicing gate and its `_enumerate_core_divisions` menu (seed at index 0, dedup by sorted signature). Track these explicitly; when the branch moves, re-verify only against this list.

**Pipeline placement and output contract.** On the substrate the joint allocator (`CoOptimizingAllocator`, invoked by `scratchpad_planning`) already occupies the fused pre-scheduling slot that replaced the legacy Core Division + LX Planning passes; the SA engine plugs into it as a `CoOptimizingSolver`, **not** as a new pass. It consumes the same upstream inputs (dim correspondences via `propagate_named_dims`, committed STLs, `span_reduction` floors), and its output is committed by the allocator (`_commit_divisions` / `_push_allocation`: chosen divisions → `op_it_space_splits`, plus the scratchpad plan) — so downstream codegen is unchanged. Config-gated engine selection lives in `select_allocator` (`layout_solver = cpsat | dfs | SA` → the joint allocator; `greedy | bestfit | firstfit` → placement-only).

### **7.1 Objective — richer shared scorer, engines optimize proxies (conflict `#1`)**

The substrate's objective is **HBM-traffic-only** (`_spill_cost = num_children·size + spill_write_cost`; `boundary_cost`): no node/PSUM/batch term, and it uses consumer *count* (`num_children`) rather than the cohort *multiplicity* of §6.2. Our objective is strictly richer (adds the node term and the split-projection multiplicity). Resolution, following the §6.4 seam:

* The **shared scorer carries the richer objective** (their HBM traffic **+** our node term **+** cohort multiplicity) and is authoritative for SA's Metropolis test and for cross-engine comparison.
* **Engines optimize proxies:** CP-SAT/DFS keep their built-in HBM-only objective **unchanged for now**; SA optimizes the richer shared objective. Comparing all engines with the *shared* scorer is honest — ILP/DFS simply score somewhat worse on the node term they didn't optimize.
* **Two definitions coexist temporarily:** the memory term itself differs (their `num_children` vs our multiplicity), not just the node term. That is acceptable under "engines optimize proxies, scorer is authoritative."
* **Porting the richer objective into CP-SAT/DFS is a separate later change** — the cost model is itself a third team's project (see §6.4); both branches run approximations until it lands. Keep it out of this change.
* The improvement is a **scorer-level add, not a data-model change**: node cost is a function of the op + its chosen `CoreDivision`; multiplicity of the `CoreDivision` splits + tensor dims — both computable from the existing `CoreDivisionBuffer` without modifying the substrate.

### **7.2 Search model — align to the enumerated menu (conflict `#2`)**

SA aligns to the substrate's **enumerate-and-pick** model rather than searching a continuous product space:

* Each op carries a pre-enumerated, deduped candidate menu (`core_divisions`, seed at index 0). **A division state is a choice of menu index per op** (`chosen_division`); SA moves are index changes.
* **Atomic division flip** = change one op's `chosen_division` index; resize/toggle its buffers; ripple compatibility. Unchanged in spirit.
* **Region-recolor** propagates a *tiling*, not a literal shared index: the same index need not mean the same division across ops. It sets the region to **mutually-compatible menu indices** using the substrate's `cd_parent_matches` (the `(parent_idx, child_idx)` pairs inducing the same per-core slicing, already correct across reductions/reshapes) as the compatibility relation — no new dim-remapping code.
* **Regions are defined by `cd_parent_matches` flood reachability, per propagated tiling** — *not* a static partition (a static "non-empty match" test collapses the graph into one region via the trivial whole↔whole pair). A recolor move picks an **(anchor op, non-trivial tiling `T`)**, then floods `cd_parent_matches` restricted to `T` bidirectionally (down via the table, up via its inverse); the reachable set *is* the region for that move. This **deletes the separate region-detector** (union-find / dim-changer classification / reshape-factoring): boundaries emerge for free as "no menu division carries `T` across this edge" — which is *more* accurate than a blanket "reductions/matmuls are boundaries" (a reduction on a surviving axis correctly stays internal). Restrict anchors to **non-trivial (split) divisions** so recolor stays a coordinated splitting move, not a graph-spanning "undivide"; leave undividing to atomic flips.
* **Propagation is a relation, not a function.** A parent index may have several compatible child indices → use a **deterministic tie-break** (sort candidates by index; the flood must not depend on `cd_parent_matches` list order — see §7.5). A **join** (multi-input op) may have an empty intersection → per our "no min-cut" stance, pick by the tie-break and **accept the internal seam** (an internal HBM edge); the flood never fails.
* **Anchor selection is size-proportional (accepted for the first cut).** Uniform-random anchor selection hits a region with probability ∝ its op-count, so a 3×-bigger region gets ~3× the recolor proposals. This is defensible because a bigger region also carries ~proportionally more byte-mass/stakes, so proposal mass ≈ stakes. It under-serves a *small, low-coloring-variety* region (e.g. a short vs long pointwise chain carry similar coloring variety but not similar size) — a rate bias, not a correctness bug, caught by best-seen + atomic flips. Instrument per-region accept/improvement; the fairer alternative (weight ∝ coloring-count) reintroduces the static region enumeration we just deleted, so add it only if measurement demands.
* This **refines but does not change `#4(a)`**: a "coloring" is a menu-index assignment over the flooded set (pairwise-compatible ⇒ homogeneous), and the Tier 0/1/2 staging runs over that space with `cd_parent_matches` as the substrate. The recolor-weight `n_colors` becomes the anchor's non-trivial menu size.

### **7.3 Engine structure — composition, not subclassing (conflict `#3`)**

The reusable machinery is the **packer** (`PermutationBasedLayoutSolver` in `permutation_layout.py` — the contact-profile incremental placer). The layout-only SA annealer (`ImanishiXuSolverWithBuffers`) **remains a standalone `MemoryPlanSolver`** (it is not yet merged and must stay available as a standalone layout solver); we do not replace it.

* **New `SaCoOptimizingSolver(CoOptimizingSolver)`** implements `plan_layout_and_core_divs` and **composes** a packer instance (composition, not inheritance). It adds the division moves, the compound flip+burst, the richer scorer, the per-move-type schedule, and best-seen over `(π, W)`.
* **This supersedes the earlier "SA engine *is a subclass* of `ImanishiXuSolverWithBuffers`" note** (see §7.2, now corrected). Composition is right because the co-opt loop differs from the layout-only loop in three ways that fight subclassing: it mixes move types (not reorder-only), scores a richer objective (not layout `quality()` alone), and its **burst sub-steps are `O(1)` single reinsertions, not the layout-only driver's `O(n)` reinsertion sweep** (§4.4 / `#9.4`). The co-opt engine owns a thin step policy over the shared packer; genuinely common bits (best-seen, schedule wiring, the reinsertion primitive) are factored so both use them without inheritance coupling.
* **Extend the packer additively** with the two ops co-opt needs — `resize` (change a buffer footprint in place, re-place) and `toggle-eligibility` (honor the residency flag) — both harmless to standalone layout-only use.
* The **one-shot interface is satisfied by an internal iterative solve**: SA runs its full anneal inside `plan_layout_and_core_divs` and returns best `(chosen_division, address)` per buffer. No incremental scoring leaks across the interface.
* **Two bonuses:** it cleans up `#9.5` (SA is now a proper `CoOptimizingSolver` sibling that owns its placement machinery, so the `layout_solver` knob cleanly applies to the placement-only path and the "invalid combo" caveat mostly dissolves); and DFS, which "delegates placement to an inner `MemoryPlanSolver`," can use the standalone `ImanishiXuSolverWithBuffers` as its inner placer for free. **Keep the packer SA-engine-private for this change** (lowest coupling to the moving branch); revisit sharing later.

This maps onto the substrate's three layers: the **allocator slot** (config-gated `select_allocator`, above), the **`CoOptimizingSolver` engine** (SA, sibling to CP-SAT/DFS), and the **shared services** (the scorer + node oracle of §7.1 and the `cd_parent_matches` compatibility gate). Cross-engine comparison is `score(engine.solve(problem))` on the one shared scorer — engines are judged on a single yardstick, though (per `#9.5`) not orthogonally interchangeable with arbitrary layout solvers. There is **no separate `LegacyEngine`**: the seed is index-0 + a FirstFit π (§8.2).

### **7.4 Residency — a fixed pin gate above the derived eligibility (conflict `#5`)**

The substrate pins some buffers out of LX up front via `residency_reason` (7 first-fail reasons in `_residency_reason`). These are **division-invariant** graph/op/lifetime properties (the substrate explicitly does *not* pre-filter core-division consistency here — that is the `cd_parent_matches` gate's job). So residency becomes a **three-gate** derivation:

`resident(b) = residency_allowed(b)` [fixed pin — a mask computed once] `AND compatible(b, W)` [moves] `AND packer_keeps(b)` [spill].

* **Six of the seven pins are genuine LX-feasibility constraints** ("op not allowed", "extern kernel user", "mutation target", "graph output (no clone)", "partial/offset read", "lx back gap") — correctness, not value. Honoring them does **not** violate "no hard gate / everything priced": that principle governs the division/layout *search*, not physical residency feasibility.
* **Only "single use" (`len(uses) <= 1`) is a value pin** — the one place the substrate hard-codes a decision our soft objective would rather price. It is largely redundant with our value≠weight `quality()` (a single-use buffer already has the lowest value/weight ratio). **Honor it for the first cut** (parity, interoperability); **earmark it as the sole relaxation candidate** — later drop it from the pins and let the objective decide. Because pins are part of the *shared* residency model (ILP/DFS honor `placement` too), that relaxation is a separate cross-engine change.
* **K-split eviction is the *second* gate, not a pin:** a reduction-split `CoreDivision` can never host a *resident producer* — the `cd_parent_matches` builder excludes a partial-reduction write on the **producer** side, so such a division never appears as a `parent_idx` and its buffer is gated out by compatibility. (It may still appear as a *consumer* index: a K-split reading a clean parent via the PSUM ring — verified in the captures, reduction splits occur only as `child_idx`.) Consistent with our plan. *Correction (verified against the landed substrate):* the eviction is **not** via `is_clean=False → signature_key=None → no matches` — `signature_key`/`is_clean` are defined on `CoreDivision` but have no callers in the substrate; `cd_parent_matches` matches by per-core view (a coeff-keyed signature "would conflate axes" across reductions/reshapes), and the K-split gate is the producer partial-write exclusion above.

**π is built over `placement=True` buffers only.** Permanent pins are excluded from the permutation entirely — they are not residency *decisions*, they can never be resident for any `(π, W)`, and carrying them would be dead slots the packer skips every step (inflating `n`, the reorder-sweep cost, the `~30n` budget, and proposal mass). Only **transient** ineligibility (division-dependent incompatibility / K-split) "keeps its slot and re-enters coherently." Pinned buffers still **participate in the problem** — they gate neighbors via `cd_parent_matches` and contribute always-HBM traffic to the objective (a function of `W`, not `π`) — just not in `π`. (This tightens §2.1's "could ever be LX-resident" to mean exactly `placement=True`.)

### **7.5 Determinism (conflict `#6`)**

The substrate is favorable here: **all costs are pure integers** (every float site is `math.ceil(int/int)`; no float accumulation in the cost path — the invariant §1.3 wanted, already satisfied), and the **candidate menu is canonicalized** (dedup by `tuple(sorted(...))`, seed at index 0), so `chosen_division` indices are stable run-to-run — essential, since SA moves are index changes. Three SA-side obligations:

1. **No set-iteration into ordered results.** CPython string-`set` iteration is `PYTHONHASHSEED`-randomized. Anywhere SA turns a collection into an *ordered* decision (candidate lists, region-flood order, argmax/tie-break, best-seen ties), iterate a **sorted / insertion-ordered** structure, never a bare `set`. (The substrate's sets are membership-only — good; we must match that discipline.)
2. **Seeded RNG** — unchanged; SA draws from a fixed `Random(seed)` over sorted domains.
3. **Sort `cd_parent_matches` candidates in the flood tie-break.** *Verified:* the substrate builds each match list by nested `enumerate` over the two candidate menus (`_cd_parent_matches`, `allocator.py:1200`), so pair order is deterministic `(parent_idx, consumer_idx)` ascending — no sets or string-keyed dicts feed it. So we do **not** inherit a hash-order dependence today. Sort defensively anyway: it costs nothing and makes our flood independent of their build order, so a future refactor of theirs cannot silently break SA determinism.

**The CI determinism test (§6.5) is scoped to the SA engine** (twice → identical `op_it_space_splits` + plan). Pass-level bit-reproducibility is now **per-engine**: CP-SAT (multi-worker/time-limited OR-Tools) may not be bit-reproducible, which is the substrate's concern, not ours; SA is fully deterministic under seeded RNG + integer accumulation.

## **8\. Implementation Order**

### **8.1 Guiding principles**

1. **Substrate-churn isolation.** The colleague's branch is unlanded and moving; all substrate-touching code sits behind one thin **adapter** (the §7 coupling surface), and the rest builds against a local **fake** substrate that never sees their churn.
2. **Vertical slice early, sophistication late.** A trivial end-to-end SA loop, scored and beating baseline, comes first to de-risk the architecture; region-recolor, the per-move-type schedule, and the `#4(a)` tiers are increments on a working core, each independently validated.
3. **The ≥-baseline guarantee makes phasing safe.** Seed-from-baseline + keep-best means every phase returns ≥ baseline regardless of how crude that phase's moves/schedule are — so each increment is validatable in isolation, with no "must be fully tuned before anything works" cliff.

### **8.2 Seed (settled)**

The initial state is **every op at `chosen_division=0`** (the committed / legacy split) with **π taken from the FirstFit layout** (`initial="first_fit"` → `SolverToPermutation` sort-by-address; already supported). No separate `LegacyEngine` is needed: the substrate's index-0 / `_fixed_division` *is* the legacy division, so the plan's earlier "LegacyEngine-as-seed" collapses to "start at index 0 + one FirstFit-derived π".

### **8.3 Phases**

* **Phase 0 — Substrate adapter + fake.** A Protocol/adapter capturing exactly the §7 coupling surface, plus a local fake substrate. *Validation:* type-checks, fake round-trips. *This is the one place their churn lands.*
* **Phase 1 — Packer extensions.** `resize` + `toggle-eligibility` on `PermutationBasedLayoutSolver`, incremental via contact profiles. *Validation:* differential test (incremental == rebuild). *Coupling: none (our file).* Independent of Phase 2 (parallelizable).
* **Phase 2 — Shared scorer + node oracle.** Integer-accumulated memory term (`bytes_moved` × multiplicity × residency) + node term (matmul with `hbm_us` stripped, reductions/PSUM, pointwise→None). *First task:* pin down `_matmul_split_cost` sourcing and confirm `hbm_us` strips cleanly. *Validation:* the `hbm_us` conservation test (exact integer equality) + multiplicity/node units. *Checkpoint:* re-run `warm_start_transfer.py` under the **real** scorer to confirm the burst-size conclusions hold under the true objective (retires the last `#1` caveat).
* **Phase 3 — Minimal end-to-end engine (the big de-risk).** `SaCoOptimizingSolver` composing the packer; moves = **reorder + atomic-flip only**; compound flip+burst; crude single schedule; best-seen; the §8.2 seed. *Validation:* determinism (twice → identical) + **≥ baseline on the shared scorer**. Proves the loop, scorer wiring, interface, and guarantee.
* **Phase 4 — Region-recolor.** The `cd_parent_matches` flood (deterministic tie-break, accept-the-seam, size-proportional anchor) as the third move. *Validation:* Tier-0 instrumentation (does it help; are heterogeneous optima found?); determinism preserved. *Heaviest substrate coupling — deliberately after the core is proven.*
* **Phase 5 — Schedule + proposal mix.** One-clock / N-move-type schedule, cycle-phase mix, acceptance bands, reheat-coupled freeze; within-group-CV instrumentation. *Validation:* per-move-type acceptance traces; beat Phase 3/4's crude-schedule score.
* **Phase 6 — Pipeline integration + full validation.** Register as a selectable engine, config-gate, apply Result; write the **real** adapter impl. *Validation:* flag-off vs flag-on on the shared scorer on real graphs; SA-scoped CI determinism; compile-time budget (`#9.4`). *The part that genuinely needs the landed branch.*

**Confirmed ordering calls:** Phases 0–5 build against the fake and proceed **pre-merge**; only Phase 6 needs the merge. Region-recolor sits **after** the proven core (3 → 4), since ≥baseline holds without it and it is the riskiest/most-coupled move.

**Deferred, gated on instrumentation:** `#4(a)` Tiers 1/2 (only if Phase 4 shows heterogeneous optima missed); variance bucketing (only if Phase 5 CV is high); per-region anchor fairness (only if Phase 4 shows small regions leaving value).

### **8.4 Lineage reconciliation (resolved — Option C)**

The co-opt engine depends on **two unlanded upstreams** — our SA solver (`imanishi_xu`/`permutation_layout`, merges on its own review timeline) and the substrate branch (unlanded, may change in review) — neither on `main`. So the co-opt PR cannot land until both parents do, regardless; the goal is productive development meanwhile with minimal rework when the parents land *possibly modified by review*.

**Resolved — Option C (dump-driven fake).** The substrate is runnable on simple examples via a **provisional merge branch** (both upstreams + substrate), so we can capture graphs. We therefore **develop on our lineage against the adapter + a fake driven by captured real substrate dumps** (`CoreDivisionBuffer` / `cd_parent_matches` serialized from the colleague's allocator). The provisional merge conveniently doubles as (a) the capture source and (b) an occasional live-integration spike target for a real-substrate check mid-development — a little of the "spikes" fallback for free, without its continuous merge tax.

**Why Option C.** Our SA solver is real and owned; **zero exposure to substrate churn** during Phases 0–5; the co-opt engine + adapter is a **clean, self-contained, reviewable diff** that only needs the real adapter impl swapped in at Phase 6; and driving the fake from **real dumps** retires the "fake diverges from real" risk without a live merge. It does **not** block on the slow standalone-solver review (we base off our own lineage, which has it). *Fallback had the branch not been runnable:* keep the adapter/clean-diff approach but add periodic throwaway integration spikes in place of dump-driven validation.

Rejected alternatives: a **standing integration branch merging both upstreams** (continuous merge tax from two moving parents, un-landable as a unit, large diff); **vendor-copying the SA solver onto the substrate branch** (copies diverge from the files that will land; duplicate code; muddies "solver stays standalone").

**Independent recommendation:** push the standalone SA solver toward its own merge as it becomes ready — it shortens the co-opt dependency chain and is independently useful — but its landing is an external dependency to flag, not something this effort controls.

**Residual — capture scope vs corpus.** "Very simple examples" is enough for what the early phases need: real `CoreDivisionBuffer` / `cd_parent_matches` **shapes** (interface fidelity for the adapter/fake), and a real-shaped small problem for Phase 2/3 correctness, determinism, and the ≥-baseline check. It is **not** enough for the *quality* validation in Phases 4–6 — region-recolor's value, heterogeneous optima, and the compile-time budget (`#9.4`) only show up on graphs with multiple regions and non-trivial `n`. So the validation **corpus** (larger/multi-region graphs) is a distinct, still-open input: obtained either by capturing bigger examples once the merge runs them, or by scaling up synthetically. Track it as a Phase-4 prerequisite, not a Phase-0 blocker.

## **Appendix: Historical Decisions, Motivations, and Dead Ends**

### **A. Dissolution of the Completion Sub-Problem**

* **Context**: References Main Plan **Section 1.1** and **Section 4.3**.  
* **History & Dead Ends**: Earlier design iterations attempted to model work division via an outer search loop that relied on an explicit, optimal "completion solver" to derive follower operation divisions. This approach required solving a multiway min-cut graph problem over conjunctive per-tensor dimension costs, which proved to be fundamentally NP-hard. The completion solver incorporated complex three-axis dominance pruning and a dominant-completion mechanism to navigate the search space.  
* **Design Decision**: This entire framework was abandoned. The architecture was shifted to a single joint annealing loop where the combination of cheap atomic flips and region-recolor moves allows the annealer to organically discover optimal cuts guided directly by the unified soft objective function. All that remains of the original "pivoting" concept is the static detection of compatibility region boundaries.

### **B. Evolution of the Temperature Schedule**

* **Context**: References Main Plan **Section 5.1** and **Section 5.2**.  
* **History & Dead Ends**: The initial schedule specification was written for a monotone cooling profile characterized by a globally precomputable $T(p)$ curve. This framework attempted to manage diverse move types by instantiating independent, concurrent schedule instances per move group and enforced a strict lockstep discipline to prevent them from drifting out of sync.  
* **Design Decision**: This approach was discarded because it was highly vulnerable to silent desynchronization errors. The system was refactored around the SelfCalibratingReheatingSchedule, which utilizes cyclic reheating. Because absolute temperatures shift during reheating phases, old control rules linking exploration strictly to global step counts ($p \\approx T \\approx \\text{refinement}$) would actively conflict with late-stage reheats. All controls were therefore rewritten to key off a scale-invariant cycle phase, and the multi-instance design was compressed into a single shared carrier clock utilizing type-specific EMAs and custom acceptance bands.

### **C. Mitigating the Region-Recolor Ratchet**

* **Context**: References Main Plan **Section 4.3**.  
* **History & Dead Ends**: A naive uniform-flood proposal over a compatibility region creates a many-to-one mapping, as multiple heterogeneous states collapse into the same uniform coloring. This asymmetry breaks proposal balance, creating a mathematical "ratchet" that artificially biases the sampler toward homogeneous states, even when a local heterogeneous min-cut would achieve a superior global score. **The bias concerns only heterogeneous-source (hetero→homo) recolors**; a homo→homo recolor (relabeling an already-uniform region between two uniform tilings) has a single-recolor inverse, is symmetric, and is already unbiased — so the tier machinery below applies only to the heterogeneous-source case.  
* **Rejected Ideas**: To balance this, a heuristic penalty term $\\exp(-\\beta \\cdot m(\\sigma))$ punishing uniform colorings was considered. This was rejected because referencing a state metric in the proposal without a corresponding physical reverse proposal path violates detailed balance and invalidates the Metropolis-Hastings stationary distribution. Proposing alternative target states using an inner MCMC chain was also rejected, as calculating the resulting proposal densities online is computationally intractable.  
* **Design Decision**: These failures motivated the rigorous three-tier architecture, using mathematically exact $\\beta$-biased local families (Tier 1\) and true dynamic programming block-Gibbs sampling over tree-like subgraphs (Tier 2\) to preserve detailed balance. It is a verify-first escalation: Tier 0 (plain recolor) is the default, and Tiers 1–2 are added only if a run shows heterogeneous optima are missed.

### **D. Rejection of an Explicit Eviction Move**

* **Context**: References Main Plan **Section 2.1** and **Section 4.2**.  
* **History & Dead Ends**: A fourth core move type was seriously evaluated: an explicit residency-flip/eviction move to manually eject a fitting, compatible buffer from LX, freeing scratchpad capacity or unpinning neighbor divisions.  
* **Design Decision**: This move type was dropped from the final specification. Structural analysis proved that the existing reorder move inherently handles eviction by demoting a buffer's position in $\\pi$, causing it to lose the allocation race and spill. Furthermore, buffers whose neighboring operations introduce severe division incompatibility automatically drop to a near-zero quality value and are naturally excluded by the packer, making a dedicated eviction move unnecessary.

### **E. Postponement of Per-Proposal Normalization**

* **Context**: References Main Plan **Section 5.3**.  
* **History & Dead Ends**: The wide variation in energy scales across different moves (e.g., small pointwise flips vs. region recoloring involving megabytes of traffic) carries a risk of "freeze/boil" pathology. Under a single raw temperature, high-impact structural moves can freeze completely while small moves engage in an unhelpful random walk, even though the aggregate acceptance rate matches targets.  
* **Design Decision**: To address this safely without distorting the Lam-Delosme controller's thermodynamic assumptions, the plan defers instantaneous per-proposal normalization ($\\frac{\\Delta E}{\\text{HBM Value}}$). The implementation plan mandates a simpler size-bucketed sub-group feedback mechanism as the primary defense, to be activated only if initial instrumented profiling runs document an unacceptably high within-group coefficient of variation.

### **F. Pointwise Node Cost Clarification**

* **Context**: References Main Plan **Section 6.3**.  
* **History & Dead Ends**: Historical documentation justified assigning a None node cost to pointwise operations by asserting that their per-core compute cost was entirely split-invariant.  
* **Design Decision**: This reasoning was mathematically incorrect, as per-core compute time ($\\frac{\\text{elements}}{\\text{cores}} \\times \\text{per-element cost}$) drops when an operation is split across more cores. The plan corrects this justification: pointwise operations are assigned a None cost because they are fundamentally memory-bound, meaning their compute variation is completely trivial relative to the HBM transfer costs tracked by the memory term. True compute-bound transcendental operations (exp, gelu, tanh) have been explicitly deferred to a future registry extension task.

### **G. Validation Bounds and the Placeholder Cost Model**

* **Context**: References Main Plan **Section 1.2** and **Section 6.4**.  
* **History & Dead Ends**: The simple additive combination of memory and node terms ($\\text{Memory} \+ \\text{Node}$) acts as an approximation that assumes zero physical concurrency or overlap between compute operations and HBM background transfers. This can misrank certain work divisions that actively trade mass between the two pools.  
* **Design Decision**: This limitation is accepted as a temporary placeholder to insulate the simulated annealing project from a separate, ongoing cost-model development effort. Consequently, the scope of validation for this pass is restricted to showing improvements against the *scorer's own metric* and verifying the correctness of the incremental layout operations. True alignment with real hardware wall-clock execution is explicitly out of scope and deferred to the incoming cost-model integration pass.

### **H. Compile-Time Budget Tuning**

* **Context**: References Main Plan **Section 4.4** and **Section 5.4**.  
* **History & Dead Ends**: Early exploration with warm-starting proposed a rigid layout burst size of $\\sim 3n$ steps following every structural move, derived from conservative worst-case limits in synthetic testing to ensure a $95\\%$ stakes-weighted decision alignment.  
* **Design Decision**: Two *separate* findings relax the rigid $3n$ rule. **(i) Burst-size scaling** — empirical sweeps across buffer counts ($n \= 10$ to $80$) showed the burst needed for a target is much smaller than $3n$ and grows **sub-linearly** for a lenient (90%) target (~$0.25n$–$0.5n$) versus roughly linearly only for a strict (95%) target; "$3n$" was the conservative worst-instance/95%-floor figure. A lenient target is defensible because the remaining misrankings are low-stakes near-ties, so burst size becomes a tunable *fidelity knob*, not a fixed cost. **(ii) Sub-step granularity** — a separate cost analysis (not the sweep) shows that if burst sub-steps are full $O(n)$ layout sweeps, a burst is $O(n^2)$ and the whole run $O(n^3)$; using $O(1)$ single-buffer reinsertions per burst step makes a burst $O(n)$ and the run $O(n^2)$. Both are enforced, and paired with targeted $O(n)$ reorder neighborhoods (annealing\_step\_rotate) so layout choices do not starve work-division exploration. Compile time is treated as a bounded constraint enforced by these knobs plus a **bounded/reactive move mix** (whose safe aggressiveness is set by the $n/n_{\\text{ops}}$ ratio) and a schedule step-budget that degrades via best-seen rather than growing unbounded.

### **I. Sequential Core Execution Assumption**

* **Context**: References Main Plan **Section 2.1** and **Section 6.3**.  
* **History & Dead Ends**: The entire optimization framework relies on the fundamental architectural constraint that operations execute sequentially on hardware, with each individual op fanning out across up to $\\text{SENCORES}$ one at a time.  
* **Design Decision**: This assumption underpins the context-free design of the node oracle (as an operation's performance does not depend on concurrent co-tenants) and guarantees that memory remains the sole coupling mechanism across operations. The plan establishes a formal trigger condition: if the execution backend introduces concurrent operational pipelining, cores will transform into a shared, time-multiplexed budget, invalidating this search framing and requiring a full rewrite of the node cost service.

### **J. Fixed Coarse-Tile Groups**

* **Context**: References Main Plan **Section 5.4** and **Section 7**.  
* **History & Dead Ends**: The optimizer operates under a fixed search-space restriction where coarse-tile groups are pre-committed upstream and treated as immutable structures.  
* **Design Decision**: This design choice assumes that coarse-tiling decisions are largely independent of subsequent work-division splits. It remains a known point of incompleteness: if a different coarse-tiling configuration would unlock significantly better joint layouts, the simulated annealing loop cannot discover it. This constraint is accepted, but it is flagged as a trigger to revisit if profiling data reveals that coarse-tile boundaries are frequently throttling optimal work-division splits.


