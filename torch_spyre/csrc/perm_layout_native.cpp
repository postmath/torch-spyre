/*
 * Copyright 2026 The Torch-Spyre Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Native (C++) accelerator for the permutation-based scratchpad layout packer.
//
// This mirrors the *observable* behaviour of the canonical Python
// ``PermutationBasedLayoutSolver``
// (``torch_spyre/_inductor/scratchpad/permutation_layout.py``): given a
// permutation (allocation order), per-buffer sizes and LX-eligibility, it
// places every buffer on top of the earlier-placed, time-overlapping buffers
// (with in-place reuse and a capacity/eviction gate) and exposes the resulting
// ``addresses`` (None == evicted / HBM), ``quality()`` and
// ``count_allocated()``.
//
// Placement is a pure function of (permutation, sizes, eligibility, lifetimes),
// so after every mutating op the layout is recomputed from scratch in
// permutation order using precomputed time-overlap sets plus the saturation
// early-stop -- i.e. the same decision the Python reference/from-scratch placer
// makes, which the incremental Python packer is differentially proven equal to.
// The internal representation is therefore deliberately simpler than Python's
// below/above contact profiles (the task allows any internal representation
// that reproduces the observable state); the speedup comes from staying in C++
// and never touching a Python object per operation. Buffer fields are read
// exactly once, in the constructor.

#include "perm_layout_native.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace torch_spyre {
namespace scratchpad {

namespace {

// Immutable-during-planning data derived from the buffers, the capacity and the
// alignment. Shared by reference between a solver and its ``copy()`` clones.
struct StaticData {
  int n = 0;
  int64_t capacity = 0;
  int64_t alignment = 128;
  std::vector<int64_t> start;  // start_time == uses.front()
  std::vector<int64_t> end;    // end_time == uses.back() + 1
  std::vector<double> weight;  // len(uses) + (first_use_is_read ? 0.0 : 0.5)
  // Time-overlap members of each buffer (order-independent, lifetimes only).
  std::vector<std::vector<int>> overlap;
  // Possible in-place partners: declared parents + children that declare it.
  std::vector<std::vector<int>> inplace_partners;
  // Parent indices this buffer names in ``in_place_parents`` (for direction).
  std::vector<std::unordered_set<int>> declared_parents;
  // Saturation early-stop interval data (lifetimes only).
  int num_intervals = 0;
  std::vector<int> total_at;                       // alive count per interval
  std::vector<std::pair<int, int>> buf_intervals;  // [lo, hi) interval range
};

}  // namespace

class NativePermutationLayoutSolver {
 public:
  NativePermutationLayoutSolver(const py::list& buffers,
                                std::vector<int> permutation, int64_t capacity,
                                int64_t alignment, const py::object& eligible) {
    auto st = std::make_shared<StaticData>();
    const int n = static_cast<int>(buffers.size());
    st->n = n;
    st->capacity = capacity;
    st->alignment = alignment;

    // Validate the permutation is a permutation of range(n).
    {
      std::vector<int> sorted_perm = permutation;
      std::sort(sorted_perm.begin(), sorted_perm.end());
      bool ok = static_cast<int>(sorted_perm.size()) == n;
      for (int i = 0; ok && i < n; ++i) ok = sorted_perm[i] == i;
      if (!ok) {
        throw std::invalid_argument(
            "permutation must be a permutation of range(len(buffers))");
      }
    }

    // Read every buffer field exactly once.
    st->start.resize(n);
    st->end.resize(n);
    st->weight.resize(n);
    sizes_.resize(n);
    std::vector<std::string> names(n);
    std::vector<std::vector<std::string>> parent_names(n);
    for (int i = 0; i < n; ++i) {
      py::handle b = buffers[i];
      names[i] = b.attr("name").cast<std::string>();
      sizes_[i] = b.attr("size").cast<int64_t>();
      std::vector<int64_t> uses = b.attr("uses").cast<std::vector<int64_t>>();
      bool first_read = b.attr("first_use_is_read").cast<bool>();
      parent_names[i] =
          b.attr("in_place_parents").cast<std::vector<std::string>>();
      st->start[i] = uses.front();
      st->end[i] = uses.back() + 1;
      st->weight[i] =
          static_cast<double>(uses.size()) + (first_read ? 0.0 : 0.5);
    }

    std::unordered_map<std::string, int> name_to_idx;
    name_to_idx.reserve(n * 2);
    for (int i = 0; i < n; ++i) name_to_idx.emplace(names[i], i);

    // Time-overlap sets (half-open [start, end) intervals).
    st->overlap.assign(n, {});
    for (int a = 0; a < n; ++a) {
      for (int c = a + 1; c < n; ++c) {
        if (st->start[a] < st->end[c] && st->start[c] < st->end[a]) {
          st->overlap[a].push_back(c);
          st->overlap[c].push_back(a);
        }
      }
    }

    // In-place partners and declared-parent direction.
    st->declared_parents.assign(n, {});
    std::vector<std::unordered_set<int>> partner_sets(n);
    for (int child = 0; child < n; ++child) {
      for (const std::string& pname : parent_names[child]) {
        auto it = name_to_idx.find(pname);
        if (it == name_to_idx.end()) continue;
        int parent = it->second;
        st->declared_parents[child].insert(parent);
        partner_sets[child].insert(parent);
        partner_sets[parent].insert(child);
      }
    }
    st->inplace_partners.assign(n, {});
    for (int i = 0; i < n; ++i) {
      st->inplace_partners[i].assign(partner_sets[i].begin(),
                                     partner_sets[i].end());
    }

    BuildIntervalData(st.get());

    st_ = std::move(st);

    // Dynamic state.
    permutation_ = std::move(permutation);
    position_.assign(n, 0);
    for (int pos = 0; pos < n; ++pos) position_[permutation_[pos]] = pos;
    if (eligible.is_none()) {
      eligible_.assign(n, 1);
    } else {
      std::vector<bool> flags = eligible.cast<std::vector<bool>>();
      if (static_cast<int>(flags.size()) != n) {
        throw std::invalid_argument("eligible must have one flag per buffer");
      }
      eligible_.resize(n);
      for (int i = 0; i < n; ++i) eligible_[i] = flags[i] ? 1 : 0;
    }
    addr_.assign(n, 0);
    allocated_.assign(n, 0);
    cand_mark_.assign(n, 0);
    RecomputeAll();
  }

  double swap(int i) {
    const int n = st_->n;
    if (i < 0 || i >= n - 1) {
      throw std::invalid_argument("swap index out of range");
    }
    const int x = permutation_[i];
    const int y = permutation_[i + 1];
    permutation_[i] = y;
    permutation_[i + 1] = x;
    position_[x] = i + 1;
    position_[y] = i;
    // No-op cases (identical to the Python packer): a transparent (HBM) member
    // is outside the eligible stacking order, and independent buffers do not
    // affect any address -- only the positions moved.
    if (!(eligible_[x] && eligible_[y])) return 0.0;
    if (!Overlaps(x, y)) return 0.0;
    const double old_total = total_quality_;
    RecomputeAll();
    return total_quality_ - old_total;
  }

  double rotate(int i, int j) {
    if (i == j) return 0.0;
    if (!eligible_[permutation_[i]]) {
      // Moving a transparent (HBM) buffer changes no eligible buffer's relative
      // order, so no address moves; just relocate its slot.
      MoveInPermutation(i, j);
      return 0.0;
    }
    const double old_total = total_quality_;
    MoveInPermutation(i, j);
    RecomputeAll();
    return total_quality_ - old_total;
  }

  double resize(int idx, int64_t new_size) {
    const double old_total = total_quality_;
    sizes_[idx] = new_size;
    // An ineligible buffer is in HBM: its size affects nothing observable, but
    // the new size is recorded so a later set_eligible(True) uses it.
    if (!eligible_[idx]) return 0.0;
    RecomputeAll();
    return total_quality_ - old_total;
  }

  double set_eligible(int idx, bool flag) {
    if (static_cast<bool>(eligible_[idx]) == flag) return 0.0;
    const double old_total = total_quality_;
    eligible_[idx] = flag ? 1 : 0;
    RecomputeAll();
    return total_quality_ - old_total;
  }

  NativePermutationLayoutSolver copy() const {
    return *this;
  }

  double quality() const {
    return total_quality_;
  }
  int count_allocated() const {
    return total_allocated_count_;
  }

  py::list addresses() const {
    py::list out;
    for (int i = 0; i < st_->n; ++i) {
      if (allocated_[i]) {
        out.append(py::cast(addr_[i]));
      } else {
        out.append(py::none());
      }
    }
    return out;
  }

 private:
  static void BuildIntervalData(StaticData* st) {
    const int n = st->n;
    std::vector<int64_t> pts;
    pts.reserve(2 * n);
    for (int i = 0; i < n; ++i) {
      pts.push_back(st->start[i]);
      pts.push_back(st->end[i]);
    }
    std::sort(pts.begin(), pts.end());
    pts.erase(std::unique(pts.begin(), pts.end()), pts.end());
    const int k = std::max(0, static_cast<int>(pts.size()) - 1);
    st->num_intervals = k;
    st->total_at.assign(k, 0);
    st->buf_intervals.assign(n, {0, 0});
    if (k == 0) return;
    // Delta sweep: +1 at each start interval, -1 at each end interval.
    std::vector<int> deltas(k + 1, 0);
    auto bisect_left = [&pts](int64_t v) {
      return static_cast<int>(std::lower_bound(pts.begin(), pts.end(), v) -
                              pts.begin());
    };
    for (int i = 0; i < n; ++i) {
      int lo = bisect_left(st->start[i]);
      int hi = bisect_left(st->end[i]);
      deltas[lo] += 1;
      deltas[hi] -= 1;
      st->buf_intervals[i] = {lo, hi};
    }
    int running = 0;
    for (int i = 0; i < k; ++i) {
      running += deltas[i];
      st->total_at[i] = running;
    }
  }

  bool Overlaps(int x, int y) const {
    return st_->start[x] < st_->end[y] && st_->start[y] < st_->end[x];
  }

  int64_t AlignUp(int64_t addr) const {
    const int64_t a = st_->alignment;
    return ((addr + a - 1) / a) * a;
  }

  // Returns (parent, child) for an in-place pair. One of the two members must
  // declare the other as an in-place parent.
  std::pair<int, int> InPlacePair(int i, int j) const {
    if (st_->declared_parents[i].count(j)) return {j, i};  // j parents i
    return {i, j};                                         // i parents j
  }

  // Mirrors PermutationBasedLayoutSolverBase._placement_decision. ``cand`` are
  // the already-placed, time-overlapping, eligible candidates for ``idx``.
  // Returns true (with ``*out_addr`` set) if placed, false if evicted (None).
  bool PlaceDecision(int idx, const std::vector<int>& cand, int64_t* out_addr) {
    const int64_t cap = st_->capacity;
    if (cand.empty()) {
      // Lone buffer sits on the floor unless it alone exceeds capacity.
      if (sizes_[idx] > cap) return false;
      *out_addr = 0;
      return true;
    }
    // A None (evicted) candidate dominates: idx would rest on it.
    for (int p : cand) {
      if (!allocated_[p]) return false;
    }
    int64_t max_top = std::numeric_limits<int64_t>::min();
    for (int p : cand) {
      const int64_t top = addr_[p] + sizes_[p];
      if (top > max_top) max_top = top;
    }
    // Try to drop into an in-place partner's slot.
    const std::vector<int>& partners = st_->inplace_partners[idx];
    if (!partners.empty()) {
      ++cand_gen_;
      for (int p : cand) cand_mark_[p] = cand_gen_;
      for (int partner : partners) {
        if (cand_mark_[partner] != cand_gen_) continue;
        std::pair<int, int> pr = InPlacePair(idx, partner);
        const int parent = pr.first;
        const int child = pr.second;
        if (sizes_[child] > sizes_[parent]) continue;  // _can_inplace
        const int64_t partner_addr = addr_[partner];
        int64_t others_top = 0;
        for (int q : cand) {
          if (q == partner) continue;
          const int64_t top = addr_[q] + sizes_[q];
          if (top > others_top) others_top = top;
        }
        if (others_top <= partner_addr) {
          if (partner_addr + sizes_[idx] > cap) return false;
          *out_addr = partner_addr;
          return true;
        }
      }
    }
    const int64_t aligned = AlignUp(max_top);
    if (aligned + sizes_[idx] > cap) return false;
    *out_addr = aligned;
    return true;
  }

  // Places every buffer in permutation order with the saturation early-stop,
  // rebuilding addresses, quality and the allocated count from scratch.
  void RecomputeAll() {
    const int n = st_->n;
    for (int pos = 0; pos < n; ++pos) position_[permutation_[pos]] = pos;
    total_quality_ = 0.0;
    total_allocated_count_ = 0;

    const int k = st_->num_intervals;
    placed_at_.assign(k, 0);
    has_none_at_.assign(k, 0);
    done_at_.assign(k, 0);
    int not_done = 0;
    for (int t = 0; t < k; ++t) {
      if (st_->total_at[t] == 0) {
        done_at_[t] = 1;
      } else {
        ++not_done;
      }
    }

    int stop = n;
    for (int pos = 0; pos < n; ++pos) {
      if (not_done == 0) {
        stop = pos;
        break;
      }
      const int idx = permutation_[pos];
      if (!eligible_[idx]) {
        // Ineligible: routed to HBM, transparent to the stack. No address / no
        // quality, and it does not mark its intervals has_none (nothing rests
        // on it), but it is still counted into placed_at so an interval whose
        // remaining occupants are all ineligible can still saturate.
        allocated_[idx] = 0;
        const int lo = st_->buf_intervals[idx].first;
        const int hi = st_->buf_intervals[idx].second;
        for (int t = lo; t < hi; ++t) {
          ++placed_at_[t];
          if (!done_at_[t] && placed_at_[t] == st_->total_at[t]) {
            done_at_[t] = 1;
            --not_done;
          }
        }
        continue;
      }
      cand_.clear();
      for (int w : st_->overlap[idx]) {
        if (position_[w] < pos && eligible_[w]) cand_.push_back(w);
      }
      int64_t a = 0;
      const bool placed = PlaceDecision(idx, cand_, &a);
      allocated_[idx] = placed ? 1 : 0;
      addr_[idx] = a;
      const bool evicted = !placed;
      if (!evicted) {
        total_quality_ += st_->weight[idx] * static_cast<double>(sizes_[idx]);
        ++total_allocated_count_;
      }
      const int lo = st_->buf_intervals[idx].first;
      const int hi = st_->buf_intervals[idx].second;
      for (int t = lo; t < hi; ++t) {
        ++placed_at_[t];
        if (evicted) has_none_at_[t] = 1;
        if (!done_at_[t] &&
            (has_none_at_[t] || placed_at_[t] == st_->total_at[t])) {
          done_at_[t] = 1;
          --not_done;
        }
      }
    }
    for (int pos = stop; pos < n; ++pos) allocated_[permutation_[pos]] = 0;
  }

  void MoveInPermutation(int i, int j) {
    const int x = permutation_[i];
    permutation_.erase(permutation_.begin() + i);
    permutation_.insert(permutation_.begin() + j, x);
    const int lo = std::min(i, j);
    const int hi = std::max(i, j);
    for (int p = lo; p <= hi; ++p) position_[permutation_[p]] = p;
  }

  std::shared_ptr<const StaticData> st_;

  // Dynamic per-plan state (deep-copied by copy()).
  std::vector<int> permutation_;
  std::vector<int> position_;   // inverse of permutation_
  std::vector<int64_t> sizes_;  // mutable footprint (resize)
  std::vector<char> eligible_;  // LX-eligibility (set_eligible)
  std::vector<int64_t> addr_;   // address; valid iff allocated_[i]
  std::vector<char> allocated_;
  double total_quality_ = 0.0;
  int total_allocated_count_ = 0;

  // Scratch reused across RecomputeAll / PlaceDecision calls.
  std::vector<int> cand_;
  std::vector<int> cand_mark_;
  int cand_gen_ = 0;
  std::vector<int> placed_at_;
  std::vector<char> has_none_at_;
  std::vector<char> done_at_;
};

void register_perm_layout_native(py::module_& m) {
  py::class_<NativePermutationLayoutSolver>(m, "NativePermutationLayoutSolver")
      .def(py::init<const py::list&, std::vector<int>, int64_t, int64_t,
                    const py::object&>(),
           py::arg("buffers"), py::arg("permutation"), py::arg("size"),
           py::arg("alignment") = 128, py::arg("eligible") = py::none())
      .def("swap", &NativePermutationLayoutSolver::swap, py::arg("i"))
      .def("rotate", &NativePermutationLayoutSolver::rotate, py::arg("i"),
           py::arg("j"))
      .def("resize", &NativePermutationLayoutSolver::resize, py::arg("idx"),
           py::arg("new_size"))
      .def("set_eligible", &NativePermutationLayoutSolver::set_eligible,
           py::arg("idx"), py::arg("flag"))
      .def("copy", &NativePermutationLayoutSolver::copy)
      .def("quality", &NativePermutationLayoutSolver::quality)
      .def("count_allocated", &NativePermutationLayoutSolver::count_allocated)
      .def_property_readonly("addresses",
                             &NativePermutationLayoutSolver::addresses);
}

}  // namespace scratchpad
}  // namespace torch_spyre
