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

// Native (C++/pybind11) reimplementation of the incremental
// PermutationBasedLayoutSolver from
// torch_spyre/_inductor/scratchpad/permutation_layout.py, folded into the _C
// extension as an opt-in accelerator. The Python implementation remains the
// canonical correctness oracle; this class mirrors its incremental data
// structures (below/above contact profiles + inplace_reuse) so that per-op
// cost stays local and the asymptotic scaling of the Python fast packer is
// preserved with C++ constant factors.
//
// The internal Profile step function ports
// torch_spyre/_inductor/scratchpad/contact_profile.py.

#include "permutation_layout_native.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <optional>
#include <queue>
#include <set>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace torch_spyre {
namespace scratchpad {

namespace {

// A profile label is a neighbouring buffer index, or "no neighbour" (floor /
// ceiling). kNone encodes the latter (Python's None); real labels are >= 0.
constexpr int kNone = -1;

// ---------------------------------------------------------------------------
// Profile: a step function from a half-open span to labels (Optional[int]).
// Faithful port of contact_profile.Profile. Stored as parallel vectors:
// `starts` of length n+1 and `labels` of length n; segment i covers
// [starts[i], starts[i+1]) carrying labels[i]. Note that `starts` can
// therefore not be empty. Canonical form: `starts` strictly increasing and no
// two adjacent segments carry equal labels.
// ---------------------------------------------------------------------------
struct Profile {
  std::vector<int64_t> starts;
  std::vector<int> labels;

  // Merge adjacent segments carrying equal labels, in place: compact the
  // segments that survive down to the front and drop the tail. Only shrinking
  // resizes are used, and those never reallocate, so this allocates nothing --
  // on any input, not just the already-canonical one. It is idempotent, which
  // is what lets a caller that only sometimes produces a duplicate pair (see
  // relabel) run it unconditionally rather than testing first.
  //
  // A surviving run keeps the start of its *first* segment and the span end is
  // restored at the end, which is how the absorbed segments' starts disappear.
  // Requires a non-empty `starts`, as the Profile invariant already does.
  static void coalesce(std::vector<int64_t>& starts, std::vector<int>& labels) {
    const int64_t span_end = starts.back();
    size_t kept = 0;
    for (size_t i = 0; i < labels.size(); ++i) {
      if (kept > 0 && labels[kept - 1] == labels[i]) {
        continue;  // absorbed into the run at kept - 1, which keeps its start
      }
      labels[kept] = labels[i];
      starts[kept] = starts[i];
      ++kept;
    }
    labels.resize(kept);
    starts.resize(kept + 1);
    starts.back() = span_end;
  }

  static Profile uniform(int64_t span_start, int64_t span_end, int label) {
    return Profile{{span_start, span_end}, {label}};
  }

  // Takes the segments by value so a caller holding the only copy can move
  // them in and the coalescing happens in the Profile's own storage.
  static Profile from_segments(std::vector<int64_t> starts,
                               std::vector<int> labels) {
    coalesce(starts, labels);
    return Profile{std::move(starts), std::move(labels)};
  }

  int64_t span_start() const {
    return starts.front();
  }
  int64_t span_end() const {
    return starts.back();
  }

  // The greatest i such that starts[i] <= a.
  size_t starts_index_at_most(int64_t a) const {
    return std::upper_bound(starts.begin(), starts.end(), a) - starts.begin() -
           1;
  }

  // The smallest i such that starts[i] >= a.
  size_t starts_index_at_least(int64_t a) const {
    return std::lower_bound(starts.begin(), starts.end(), a) - starts.begin();
  }

  // Label of the segment containing column t (t in span).
  int label_at(int64_t t) const {
    // t < starts.front() underflows to a huge index, so one bound catches both
    // ends; the limit is labels.size(), since t == span_end has no segment.
    size_t i = starts_index_at_most(t);
    if (i >= labels.size()) {
      throw std::out_of_range("t out of range for label_at");
    }
    return labels[i];
  }

  // The segments clipped to [a, b) as a (starts, labels) pair. An empty range
  // yields ([a], []).
  std::pair<std::vector<int64_t>, std::vector<int>> segments(int64_t a,
                                                             int64_t b) const {
    std::vector<int64_t> out_starts;
    std::vector<int> out_labels;
    out_starts.push_back(a);
    if (a != b) {
      // i = bisect_right(starts, a) - 1
      size_t i = starts_index_at_most(a);
      while (starts[i] < b) {
        out_labels.push_back(labels[i]);
        out_starts.push_back(std::min(starts[i + 1], b));
        ++i;
      }
    }
    return {std::move(out_starts), std::move(out_labels)};
  }

  // Replace the function on [a, b) with the given segments, coalescing at both
  // seams. No-op if a == b.
  //
  // The tiling is taken as spans so that a caller holding it in anything
  // contiguous -- a vector, or the stack arrays splice_uniform uses -- can pass
  // it without materialising a vector for the call.
  //
  // Preconditions (both are the caller's responsibility; unlike the Python
  // implementation, which rebuilds through a full coalescing pass, this edits
  // in place and so only reconciles the two seams):
  //   * `*this` is canonical, and
  //   * seg_starts/seg_labels exactly tile [a, b) and are canonical too --
  //     strictly increasing starts and no two adjacent labels equal.
  // A caller that can violate the second (see relabel) must coalesce first.
  void splice(int64_t a, int64_t b, std::span<const int64_t> seg_starts,
              std::span<const int> seg_labels) {
    if (a == b) {
      return;
    }
    if (a > b) {
      throw std::runtime_error("a > b in splice");
    }
    if (a < starts.front() || b > starts.back()) {
      throw std::runtime_error(
          "[a, b) not contained in bounds of profile in splice");
    }
    if (seg_labels.empty() || seg_starts.size() != seg_labels.size() + 1 ||
        seg_starts.front() != a || seg_starts.back() != b) {
      throw std::runtime_error("segments do not tile [a, b) in splice");
    }

    int64_t i_a = static_cast<int64_t>(starts_index_at_least(a)) - 1;
    int64_t i_b = static_cast<int64_t>(starts_index_at_most(b));
    // Now i_a is the index of the label we need to check for coalescing at
    // the start seam, or -1, and i_b is the index of the label we need to
    // check for coalescing at the end seam, or starts.size(). Using int64_t for
    // i_a to allow -1, and for i_b for symmetry.

    bool coalesced_start = true;
    bool coalesced_end = false;
    if (i_a < 0 || labels[i_a] != seg_labels.front()) {
      ++i_a;
      coalesced_start = false;
    }
    if (i_b < static_cast<int64_t>(labels.size()) &&
        labels[i_b] == seg_labels.back()) {
      ++i_b;
      coalesced_end = true;
    }
    // Now we need to replace labels entries [i_a, i_b) with seg_labels, and
    // starts entries [i_a, i_b] with seg_starts followed by the start of the
    // segment that ends up after the replaced range. In Python notation, labels
    // becomes labels[:i_a] + seg_labels + labels[i_b:], and starts becomes
    // starts[:i_a] + seg_starts + starts[i_b + 1:].
    //
    // Two degenerate index relations are worth calling out: i_b == i_a is a
    // plain insert (no existing segment is consumed), and i_b == i_a - 1 --
    // which happens when a and b both fall strictly inside one segment and
    // neither seam coalesces -- means that segment is *split*, so entry i_b of
    // both vectors is duplicated: it stays as the left remainder and reappears
    // after the replaced range as the right remainder.

    const int64_t n_seg = static_cast<int64_t>(seg_labels.size());
    const int64_t extra_segments = n_seg - (i_b - i_a);

    if (extra_segments > 0) {
      // Inserting at i_b keeps entries [0, i_b) in place and shifts the tail to
      // exactly where it belongs. Every inserted slot is overwritten below
      // except entry i_b in the split case above, which must keep its own old
      // value -- so seed the new slots with it rather than with a placeholder.
      const int filler =
          i_b < static_cast<int64_t>(labels.size()) ? labels[i_b] : kNone;
      labels.insert(labels.begin() + i_b, extra_segments, filler);
      starts.insert(starts.begin() + i_b, extra_segments, starts[i_b]);
    } else if (extra_segments < 0) {
      labels.erase(labels.begin() + i_b + extra_segments, labels.begin() + i_b);
      starts.erase(starts.begin() + i_b + extra_segments, starts.begin() + i_b);
    }

    // Now, overwrite labels[i_a] and further, and starts[i_a] and further --
    // except that starts[i_a] keeps the value it has when the start seam
    // coalesced, that being the absorbed neighbouring segment's own start.
    for (int64_t i = 0; i < n_seg; ++i) {
      labels[i_a + i] = seg_labels[i];
      if (i || !coalesced_start) {
        starts[i_a + i] = seg_starts[i];
      }
    }
    // Likewise, the segment following the replaced range now starts at b,
    // unless the end seam coalesced, in which case it keeps its own start.
    if (!coalesced_end) {
      starts[i_a + n_seg] = b;
    }
  }

  // Replace the function on [a, b) with a single run carrying `label`. This is
  // the shape every splice outside relabel has: the caller knows the whole
  // range takes one label and would otherwise heap-allocate a two-element and
  // a one-element vector to say so. A one-segment tiling is canonical by
  // construction, so splice's second precondition is free here.
  void splice_uniform(int64_t a, int64_t b, int label) {
    const std::array<int64_t, 2> seg_starts{a, b};
    const std::array<int, 1> seg_labels{label};
    splice(a, b, seg_starts, seg_labels);
  }

  // For every segment within [a, b) whose label == from_label, replace it with
  // to_label; coalesce afterwards. (Python's relabel takes a dict; the packer
  // only ever passes a single {from: to} mapping.)
  void relabel(int64_t a, int64_t b, int from_label, int to_label) {
    if (a == b) {
      return;
    }
    auto [seg_s, seg_l] = segments(a, b);
    for (int& lab : seg_l) {
      if (lab == from_label) {
        lab = to_label;
      }
    }
    // Mapping can leave two adjacent segments carrying the same label -- when a
    // relabelled segment abuts one that already carried to_label -- which
    // splice does not accept, so coalesce before handing the tiling over.
    // Whether it actually has to merge anything depends on to_label and on
    // what the profile already carries, so this runs unconditionally; being
    // in place and idempotent, the case that needs no merge costs one pass.
    coalesce(seg_s, seg_l);
    splice(a, b, seg_s, seg_l);
  }

  // Throws if the canonical-form invariants are broken. Mirrors Python's
  // Profile.validate; used by the differential tests.
  void validate() const {
    if (starts.size() != labels.size() + 1) {
      throw std::runtime_error("length mismatch");
    }
    if (labels.empty()) {
      throw std::runtime_error("profile must have at least one segment");
    }
    for (size_t i = 0; i + 1 < starts.size(); ++i) {
      if (starts[i] >= starts[i + 1]) {
        throw std::runtime_error("starts not strictly increasing");
      }
    }
    for (size_t i = 0; i + 1 < labels.size(); ++i) {
      if (labels[i] == labels[i + 1]) {
        throw std::runtime_error("adjacent labels equal");
      }
    }
  }

  bool operator==(const Profile& o) const {
    return starts == o.starts && labels == o.labels;
  }
};

// ---------------------------------------------------------------------------
// NativePermutationLayoutSolver
// ---------------------------------------------------------------------------
class NativePermutationLayoutSolver {
 public:
  NativePermutationLayoutSolver(const py::list& buffers,
                                const std::vector<int>& permutation,
                                int64_t capacity, int64_t alignment,
                                const py::object& eligible) {
    n_ = static_cast<int>(buffers.size());
    capacity_ = capacity;
    alignment_ = alignment;

    // Validate the permutation is a permutation of range(n).
    {
      std::vector<int> sorted_perm = permutation;
      std::sort(sorted_perm.begin(), sorted_perm.end());
      bool ok = static_cast<int>(sorted_perm.size()) == n_;
      for (int i = 0; ok && i < n_; ++i) {
        if (sorted_perm[i] != i) {
          ok = false;
        }
      }
      if (!ok) {
        throw py::value_error(
            "permutation must be a permutation of range(len(buffers))");
      }
    }
    permutation_ = permutation;

    // Read every buffer attribute ONCE here; never call back into Python later.
    start_.resize(n_);
    end_.resize(n_);
    sizes_.resize(n_);
    weight_.resize(n_);
    qualities_.resize(n_);
    std::vector<std::vector<std::string>> parent_names(n_);
    std::vector<std::string> names(n_);
    for (int i = 0; i < n_; ++i) {
      py::object buf = buffers[i];
      names[i] = buf.attr("name").cast<std::string>();
      sizes_[i] = buf.attr("size").cast<int64_t>();
      const std::vector<int64_t> uses =
          buf.attr("uses").cast<std::vector<int64_t>>();
      start_[i] = uses.front();
      end_[i] = uses.back() + 1;
      const bool first_use_is_read = buf.attr("first_use_is_read").cast<bool>();
      // quality = (len(uses) + (0.0 if first_use_is_read else 0.5)) * size.
      weight_[i] =
          static_cast<double>(uses.size()) + (first_use_is_read ? 0.0 : 0.5);
      qualities_[i] = weight_[i] * static_cast<double>(sizes_[i]);
      parent_names[i] =
          buf.attr("in_place_parents").cast<std::vector<std::string>>();
    }

    name_to_idx_.reserve(n_);
    for (int i = 0; i < n_; ++i) {
      name_to_idx_[names[i]] = i;
    }

    // Eligibility (default all-True).
    eligible_.assign(n_, true);
    if (!eligible.is_none()) {
      const std::vector<bool> e = eligible.cast<std::vector<bool>>();
      if (static_cast<int>(e.size()) != n_) {
        throw py::value_error("eligible must have one flag per buffer");
      }
      for (int i = 0; i < n_; ++i) {
        eligible_[i] = e[i];
      }
    }

    // Static in-place data: declared parents (as indices) and the symmetric
    // partner sets.
    parent_set_.assign(n_, {});
    inplace_partners_.assign(n_, {});
    for (int child = 0; child < n_; ++child) {
      for (const std::string& pname : parent_names[child]) {
        auto it = name_to_idx_.find(pname);
        if (it == name_to_idx_.end()) {
          continue;
        }
        const int parent = it->second;
        parent_set_[child].insert(parent);
        inplace_partners_[child].insert(parent);
        inplace_partners_[parent].insert(child);
      }
    }

    build_interval_data();

    addresses_.assign(n_, std::optional<int64_t>(0));
    total_quality_ = 0.0;
    total_allocated_count_ = 0;

    build();
  }

  // --- observable accessors ------------------------------------------------

  double quality() const {
    return total_quality_;
  }
  int count_allocated() const {
    return total_allocated_count_;
  }

  py::list addresses() const {
    py::list out;
    for (int i = 0; i < n_; ++i) {
      if (addresses_[i].has_value()) {
        out.append(py::int_(*addresses_[i]));
      } else {
        out.append(py::none());
      }
    }
    return out;
  }

  std::vector<int> permutation() const {
    return permutation_;
  }

  // Below/above profile as (starts, labels) with None for floor/ceiling; used
  // by the differential test to compare the contact relation exactly.
  py::tuple below_profile(int idx) const {
    return profile_tuple(below_[idx]);
  }
  py::tuple above_profile(int idx) const {
    return profile_tuple(above_[idx]);
  }

  py::dict inplace_reuse() const {
    py::dict out;
    for (int i = 0; i < n_; ++i) {
      if (reuse_[i] != kNone) {
        out[py::int_(i)] = py::int_(reuse_[i]);
      }
    }
    return out;
  }

  // --- public mutating API (mirrors the Python packer) ---------------------

  double swap(int i) {
    if (i < 0 || i >= n_ - 1) {
      throw py::index_error("swap position out of range");
    }
    const int x = permutation_[i];
    const int y = permutation_[i + 1];
    permutation_[i] = y;
    permutation_[i + 1] = x;
    position_[x] = i + 1;
    position_[y] = i;
    if (!(eligible_[x] && eligible_[y])) {
      return 0.0;  // at least one is transparent (HBM): only positions moved.
    }
    if (!overlaps(x, y)) {
      return 0.0;  // independent buffers: order does not affect any address.
    }
    const int64_t a = std::max(start_[x], start_[y]);
    const int64_t b = std::min(end_[x], end_[y]);
    update_profiles_for_swap(x, y, a, b);
    const double old_total = total_quality_;
    std::set<int> seed{x, y};
    add_labels(seed, above_[x]);
    add_labels(seed, above_[y]);
    propagate_addresses(seed);
    return total_quality_ - old_total;
  }

  double rotate(int i, int j) {
    if (i == j) {
      return 0.0;
    }
    if (!eligible_[permutation_[i]]) {
      // Moving a transparent (HBM) buffer changes no eligible buffer's relative
      // order, so no address or profile moves; just relocate its slot.
      move_in_permutation(i, j);
      return 0.0;
    }
    if (std::abs(i - j) < rotate_remove_insert_threshold_) {
      return swap_chain_rotate(i, j);
    }
    return fast_rotate(i, j);
  }

  double resize(int idx, int64_t new_size) {
    const double old_total = total_quality_;
    const double old_q = qualities_[idx];
    sizes_[idx] = new_size;
    qualities_[idx] = weight_[idx] * static_cast<double>(new_size);
    // Reconcile the running total to the new quality *before* reflow (the
    // incremental reflow re-scores idx via a remove/re-add of qualities_[idx]).
    if (is_fully_allocated(idx)) {
      total_quality_ += qualities_[idx] - old_q;
    }
    reflow_resized(idx);
    return total_quality_ - old_total;
  }

  double set_eligible(int idx, bool flag) {
    if (eligible_[idx] == flag) {
      return 0.0;
    }
    const double old_total = total_quality_;
    reflow_eligibility(idx, flag);
    return total_quality_ - old_total;
  }

  NativePermutationLayoutSolver copy() const {
    return *this;
  }

  int rotate_remove_insert_threshold() const {
    return rotate_remove_insert_threshold_;
  }
  void set_rotate_remove_insert_threshold(int v) {
    rotate_remove_insert_threshold_ = v;
  }

 private:
  // --- geometry / placement helpers ---------------------------------------

  bool overlaps(int i, int j) const {
    return start_[i] < end_[j] && start_[j] < end_[i];
  }

  bool is_fully_allocated(int idx) const {
    return addresses_[idx].has_value();
  }

  int64_t align_up(int64_t addr) const {
    return ((addr + alignment_ - 1) / alignment_) * alignment_;
  }

  // Returns (parent, child) if i and j form an in-place pair, else (-1, -1).
  std::pair<int, int> in_place_pair(int i, int j) const {
    if (parent_set_[i].count(j)) {
      return {j, i};  // j is the parent of i
    }
    if (parent_set_[j].count(i)) {
      return {i, j};  // i is the parent of j
    }
    return {kNone, kNone};
  }

  bool can_inplace(int parent, int child) const {
    return sizes_[child] <= sizes_[parent];
  }

  // True if idx may reuse partner's slot: they have to be a declared in-place
  // pair, and the child has to fit in the parent's footprint. Mirrors the
  // reference's _in_place_pair + _can_inplace gate. Reads the plan-local
  // sizes_, so a resize that crosses the fit boundary flips legality.
  bool inplace_legal(int idx, int partner) const {
    if (!inplace_partners_[idx].contains(partner)) {
      return false;
    }
    const std::pair<int, int> pr = in_place_pair(idx, partner);
    return can_inplace(pr.first, pr.second);
  }

  // m dies exactly as idx starts (resp. starts exactly as idx ends), so idx may
  // be able to take over its slot. Deliberately address-free: the caller
  // settles these before knowing whether anything has been evicted.
  bool is_inplace_parent_candidate(int idx, int m) const {
    return m != kNone && eligible_[m] && end_[m] == start_[idx] + 1 &&
           inplace_legal(idx, m);
  }

  bool is_inplace_child_candidate(int idx, int m) const {
    return m != kNone && eligible_[m] && start_[m] + 1 == end_[idx] &&
           inplace_legal(idx, m);
  }

  // Decide idx's address from the buffers it rests on, read off its
  // below-profile. Returns (address, partner): address == nullopt means
  // evicted; partner is the neighbour whose address was reused in-place, or
  // kNone.
  //
  // This is the single eviction authority, mirroring the Python reference:
  // nullopt is returned whenever idx would not fit entirely below capacity_,
  // and eviction is upward-closed (resting on an evicted buffer evicts idx).
  std::pair<std::optional<int64_t>, int> placement_decision(int idx) const {
    const Profile& below = below_[idx];
    const std::size_t n_below = below.labels.size();
    if (n_below == 0) {
      // This would mean buffer idx has lifetime 0, which is unlikely. But by
      // guarding against this here, we can assume there is >=1 below-neighbour
      // for the rest of this method.
      if (sizes_[idx] > capacity_) {
        return {std::nullopt, kNone};
      }
      return {std::optional<int64_t>(0), kNone};
    }

    // See if we could potentially in-place with a parent, and likewise with a
    // child. This is determined by in_place_*_candidate == kNone. Neither test
    // reads an address, so both can be settled before the eviction scan below;
    // a qualifying candidate is a real below-profile label, so that scan then
    // guarantees it has an address by the time one is read off it.
    const int in_place_parent_candidate =
        is_inplace_parent_candidate(idx, below.labels[0]) ? below.labels[0]
                                                          : kNone;
    const int in_place_child_candidate =
        is_inplace_child_candidate(idx, below.labels[n_below - 1])
            ? below.labels[n_below - 1]
            : kNone;

    // Highest top among the buffers idx rests on: over all of them (the height
    // idx stacks at), and over all but the parent / all but the child (what
    // each in-place drop has to clear). A label can recur non-adjacently in the
    // profile, so exclude a candidate by identity rather than by position.
    //
    // The below-profile's labels alone are not the full set. A neighbour m that
    // dropped into a partner's slot shares its address with that partner, and
    // the partner still occupies those columns while never being a
    // below-neighbour of idx -- m hides it. Its top can even exceed m's, since
    // m may be the smaller child of the pair. So every neighbour contributes
    // its reuse partner too, as the candidate scan this replaced used to.
    //
    // Eviction is upward-closed: idx resting on a buffer with no address has no
    // address either. A kNone label is the floor, not an evicted neighbour.
    int64_t max_top = 0;
    int64_t top_without_parent = 0;
    int64_t top_without_child = 0;
    bool evicted = false;
    const auto contribute = [&](int buffer) {
      if (!addresses_[buffer].has_value()) {
        evicted = true;
        return;
      }
      const int64_t top = *addresses_[buffer] + sizes_[buffer];
      max_top = std::max(max_top, top);
      if (buffer != in_place_parent_candidate) {
        top_without_parent = std::max(top_without_parent, top);
      }
      if (buffer != in_place_child_candidate) {
        top_without_child = std::max(top_without_child, top);
      }
    };
    for (std::size_t i = 0; i < n_below; ++i) {
      const int m = below.labels[i];
      if (m == kNone) {
        continue;  // floor: contributes no top
      }
      contribute(m);
      const int reused = reuse_[m];
      if (reused != kNone && start_[reused] <= below.starts[i] &&
          below.starts[i] < end_[reused]) {
        contribute(reused);
      }
    }
    if (evicted) {
      return {std::nullopt, kNone};
    }

    // Drop into an in-place partner's slot, but only when every other
    // neighbour already tops out at or below the partner's address -- else idx
    // would land partway into occupied space. At most one of the two can
    // qualify - only because the reference implementation does it that way:
    // if there are both a parent and child candidate at the same base address,
    // then it would be perfectly legal for all three to be at the same address,
    // but the reference implementation can only find this if they occur in the
    // permutation in order or in reverse order, which means we have to reject
    // that case here, too (or change the reference implementation). Since this
    // is a rare case (except maybe for base address 0), and we can get the
    // better result anyway by finding a different order for the buffers, we
    // don't make this change now.
    if (in_place_parent_candidate != kNone) {
      const int64_t addr = *addresses_[in_place_parent_candidate];
      if (top_without_parent <= addr) {
        if (addr + sizes_[idx] > capacity_) {
          return {std::nullopt, kNone};
        }
        return {std::optional<int64_t>(addr), in_place_parent_candidate};
      }
    }

    if (in_place_child_candidate != kNone) {
      const int64_t addr = *addresses_[in_place_child_candidate];
      if (top_without_child <= addr) {
        if (addr + sizes_[idx] > capacity_) {
          return {std::nullopt, kNone};
        }
        return {std::optional<int64_t>(addr), in_place_child_candidate};
      }
    }

    const int64_t aligned = align_up(max_top);
    if (aligned + sizes_[idx] > capacity_) {
      return {std::nullopt, kNone};
    }
    return {std::optional<int64_t>(aligned), kNone};
  }

  // --- saturation early-stop interval data (static) ------------------------

  void build_interval_data() {
    if (n_ == 0) {
      return;
    }

    std::set<int64_t> pts;
    for (int i = 0; i < n_; ++i) {
      pts.insert(start_[i]);
      pts.insert(end_[i]);
    }
    interval_starts_.assign(pts.begin(), pts.end());

    // interval_index[j] will be assigned for j = start_[i] and j = end_[i] but
    // not for other values. The largest such j is the last breakpoint, so the
    // vector has to be one longer than that to be indexable there.
    std::vector<int> interval_index(interval_starts_.back() + 1, 0);
    for (std::size_t i = 0; i < interval_starts_.size(); ++i) {
      interval_index[interval_starts_[i]] = static_cast<int>(i);
    }

    const int k = std::max(0, static_cast<int>(interval_starts_.size()) - 1);
    num_intervals_ = k;
    total_at_.assign(k, 0);
    std::vector<int> deltas(k + 1, 0);
    for (int i = 0; i < n_; ++i) {
      deltas[interval_index[start_[i]]] += 1;
      deltas[interval_index[end_[i]]] -= 1;
    }
    int running = 0;
    for (int t = 0; t < k; ++t) {
      running += deltas[t];
      total_at_[t] = running;
    }
    buf_intervals_.assign(n_, {0, 0});
    for (int i = 0; i < n_; ++i) {
      buf_intervals_[i] = {interval_index[start_[i]], interval_index[end_[i]]};
    }
  }

  // Place every buffer in permutation order with the saturation early-stop.
  // get_candidates(pos, idx) returns the already-placed candidate list.
  void sequential_place() {
    std::fill(reuse_.begin(), reuse_.end(), kNone);
    total_quality_ = 0.0;
    total_allocated_count_ = 0;

    const int k = num_intervals_;
    std::vector<int> placed_at(k, 0);
    std::vector<char> has_none_at(k, 0);
    std::vector<char> done_at(k, 0);
    int not_done = 0;
    for (int t = 0; t < k; ++t) {
      done_at[t] = (total_at_[t] == 0) ? 1 : 0;
      if (!done_at[t]) {
        ++not_done;
      }
    }

    int stop = n_;
    for (int pos = 0; pos < n_; ++pos) {
      if (not_done == 0) {
        stop = pos;
        break;
      }
      const int idx = permutation_[pos];
      if (!eligible_[idx]) {
        addresses_[idx] = std::nullopt;
        const auto [lo, hi] = buf_intervals_[idx];
        for (int t = lo; t < hi; ++t) {
          placed_at[t] += 1;
          if (!done_at[t] && placed_at[t] == total_at_[t]) {
            done_at[t] = 1;
            --not_done;
          }
        }
        continue;
      }
      const auto [addr, partner] = placement_decision(idx);
      addresses_[idx] = addr;
      reuse_[idx] = partner;
      const bool evicted = !addr.has_value();
      if (!evicted) {
        total_quality_ += qualities_[idx];
        total_allocated_count_ += 1;
      }
      const auto [lo, hi] = buf_intervals_[idx];
      for (int t = lo; t < hi; ++t) {
        placed_at[t] += 1;
        if (evicted) {
          has_none_at[t] = 1;
        }
        if (!done_at[t] && (has_none_at[t] || placed_at[t] == total_at_[t])) {
          done_at[t] = 1;
          --not_done;
        }
      }
    }
    for (int pos = stop; pos < n_; ++pos) {
      addresses_[permutation_[pos]] = std::nullopt;
    }
  }

  // --- build ---------------------------------------------------------------

  void build() {
    reuse_.assign(n_, kNone);
    position_.assign(n_, 0);
    for (int p = 0; p < n_; ++p) {
      position_[permutation_[p]] = p;
    }
    build_profiles();
    // Candidates: earlier, time-overlapping, eligible buffers (a prior-scan).
    sequential_place();
    // Static time-overlap sets.
    overlap_.assign(n_, {});
    for (int a = 0; a < n_; ++a) {
      for (int b = a + 1; b < n_; ++b) {
        if (overlaps(a, b)) {
          overlap_[a].push_back(b);
          overlap_[b].push_back(a);
        }
      }
    }
    // TODO(postmath): This formula was translated from the python version of
    // this code, so it should probably be re-derived - but a very brief
    // profiling session suggests that it is not far off. So let's re-derive it
    // once things are stable.
    rotate_remove_insert_threshold_ = std::max(2, n_ / 8);
  }

  void build_profiles() {
    below_.assign(n_, Profile());
    above_.assign(n_, Profile());
    if (n_ == 0) {
      return;
    }
    std::vector<std::vector<int64_t>> below_s(n_), above_s(n_);
    std::vector<std::vector<int>> below_l(n_), above_l(n_);
    std::set<int64_t> bp;
    for (int i = 0; i < n_; ++i) {
      bp.insert(start_[i]);
      bp.insert(end_[i]);
    }
    std::vector<int64_t> breakpoints(bp.begin(), bp.end());
    for (size_t bi = 0; bi + 1 < breakpoints.size(); ++bi) {
      const int64_t t0 = breakpoints[bi];
      std::vector<int> alive;
      for (int i = 0; i < n_; ++i) {
        if (eligible_[i] && start_[i] <= t0 && t0 < end_[i]) {
          alive.push_back(i);
        }
      }
      std::sort(alive.begin(), alive.end(),
                [this](int a, int b) { return position_[a] < position_[b]; });
      for (size_t k = 0; k < alive.size(); ++k) {
        const int c = alive[k];
        const int below = k > 0 ? alive[k - 1] : kNone;
        const int above = k + 1 < alive.size() ? alive[k + 1] : kNone;
        below_s[c].push_back(t0);
        below_l[c].push_back(below);
        above_s[c].push_back(t0);
        above_l[c].push_back(above);
      }
    }
    for (int i = 0; i < n_; ++i) {
      if (!eligible_[i]) {
        below_[i] = Profile::uniform(start_[i], end_[i], kNone);
        above_[i] = Profile::uniform(start_[i], end_[i], kNone);
        continue;
      }
      below_s[i].push_back(end_[i]);
      below_[i] =
          Profile::from_segments(std::move(below_s[i]), std::move(below_l[i]));
      above_s[i].push_back(end_[i]);
      above_[i] =
          Profile::from_segments(std::move(above_s[i]), std::move(above_l[i]));
    }
  }

  // --- swap incremental machinery ------------------------------------------

  void update_profiles_for_swap(int x, int y, int64_t a, int64_t b) {
    auto [old_x_below_s, old_x_below_l] = below_[x].segments(a, b);
    auto [old_y_above_s, old_y_above_l] = above_[y].segments(a, b);
    // Downward and upward are exact mirrors.
    splice_half(below_, above_, x, y, a, b, old_x_below_s, old_x_below_l);
    splice_half(above_, below_, y, x, a, b, old_y_above_s, old_y_above_l);
  }

  // One side of the transposition: `lo` was directly below `hi` in the
  // `primary` direction over [a, b); after the swap `hi` is.
  void splice_half(std::vector<Profile>& primary, std::vector<Profile>& reverse,
                   int lo, int hi, int64_t a, int64_t b,
                   const std::vector<int64_t>& old_lo_s,
                   const std::vector<int>& old_lo_l) {
    primary[lo].splice_uniform(a, b, hi);
    primary[hi].splice(a, b, old_lo_s, old_lo_l);
    for (size_t k = 0; k < old_lo_l.size(); ++k) {
      const int label = old_lo_l[k];
      if (label != kNone) {
        reverse[label].relabel(old_lo_s[k], old_lo_s[k + 1], lo, hi);
      }
    }
  }

  // Re-place z's address from the buffers it actually rests on, read off the
  // (already-spliced) below-profile.
  void recompute_address(int z) {
    const auto [addr, partner] = placement_decision(z);
    addresses_[z] = addr;
    reuse_[z] = partner;  // kNone when not reused / evicted
  }

  // Re-place the buffers in `seed` and everything transitively resting on them.
  void propagate_addresses(const std::set<int>& seed) {
    using Node = std::pair<int, int>;  // (position, idx)
    std::priority_queue<Node, std::vector<Node>, std::greater<Node>> heap;
    std::vector<char> queued(n_, 0);
    std::vector<char> flipped(n_, 0);
    for (const int idx : seed) {
      heap.push({position_[idx], idx});
      queued[idx] = 1;
    }

    auto dirty = [&](int w, int pos_z) {
      if (w != kNone && !queued[w] && position_[w] > pos_z) {
        queued[w] = 1;
        heap.push({position_[w], w});
      }
    };

    while (!heap.empty()) {
      const int z = heap.top().second;
      heap.pop();
      queued[z] = 0;
      if (flipped[z]) {
        continue;
      }
      const int pos_z = position_[z];
      const std::optional<int64_t> old_addr = addresses_[z];
      const int old_partner = reuse_[z];
      if (is_fully_allocated(z)) {
        total_quality_ -= qualities_[z];
        total_allocated_count_ -= 1;
      }
      recompute_address(z);
      if (!addresses_[z].has_value()) {
        // z is evicted and final; everything resting on it is evicted too.
        flip_evicted_closure(z, flipped);
        continue;
      }
      total_quality_ += qualities_[z];
      total_allocated_count_ += 1;
      const int new_partner = reuse_[z];
      if (addresses_[z] != old_addr) {
        for (const int w : label_set(above_[z])) {
          dirty(w, pos_z);
        }
      }
      if (new_partner != old_partner) {
        for (const int partner : {old_partner, new_partner}) {
          if (partner == kNone) {
            continue;
          }
          const std::pair<int, int> pr = in_place_pair(z, partner);
          const int parent = pr.first;
          const int child = pr.second;
          const int64_t t = start_[child];
          dirty(above_[child].label_at(t), pos_z);
          dirty(above_[parent].label_at(t), pos_z);
        }
      }
    }
  }

  void flip_evicted_closure(int z, std::vector<char>& flipped) {
    std::vector<int> stack;
    for (const int w : label_set(above_[z])) {
      if (w != kNone) {
        stack.push_back(w);
      }
    }
    while (!stack.empty()) {
      const int w = stack.back();
      stack.pop_back();
      if (flipped[w]) {
        continue;
      }
      flipped[w] = 1;
      if (is_fully_allocated(w)) {
        total_quality_ -= qualities_[w];
        total_allocated_count_ -= 1;
      }
      addresses_[w] = std::nullopt;
      reuse_[w] = kNone;
      for (const int u : label_set(above_[w])) {
        if (u != kNone && !flipped[u]) {
          stack.push_back(u);
        }
      }
    }
  }

  // --- fast rotate ---------------------------------------------------------

  double swap_chain_rotate(int i, int j) {
    double delta = 0.0;
    if (i < j) {
      for (int k = i; k < j; ++k) {
        delta += swap(k);
      }
    } else {  // j < i
      for (int k = i - 1; k >= j; --k) {
        delta += swap(k);
      }
    }
    return delta;
  }

  double fast_rotate(int i, int j) {
    const double old_total = total_quality_;
    const int x = permutation_[i];
    const Profile old_below = below_[x];
    const Profile old_above = above_[x];
    move_in_permutation(i, j);
    // The profiles have to be patched before the addresses are recomputed:
    // placement_decision reads below_, so a stale profile would just replay the
    // pre-move addresses. Patching only needs the permutation (already moved),
    // never addresses_, so this order is the safe one.
    patch_profiles_for_move(x, old_below, old_above);
    recompute_all_addresses();
    return total_quality_ - old_total;
  }

  void move_in_permutation(int i, int j) {
    const int x = permutation_[i];
    permutation_.erase(permutation_.begin() + i);
    permutation_.insert(permutation_.begin() + j, x);
    const int lo = std::min(i, j);
    const int hi = std::max(i, j);
    for (int p = lo; p <= hi; ++p) {
      position_[permutation_[p]] = p;
    }
  }

  void recompute_all_addresses() {
    sequential_place();
  }

  void patch_profiles_for_move(int x, const Profile& old_below,
                               const Profile& old_above) {
    stitch_around_removed(x, old_below, old_above);
    insert_into_profiles(x);
  }

  // Splice x out of the profiles: over each column x occupied, its old below
  // neighbour a and above neighbour b become directly adjacent.
  void stitch_around_removed(int x, const Profile& old_below,
                             const Profile& old_above) {
    // iter_common over the two profiles' shared span.
    std::vector<int64_t> cuts;
    cuts.insert(cuts.end(), old_below.starts.begin(), old_below.starts.end());
    cuts.insert(cuts.end(), old_above.starts.begin(), old_above.starts.end());
    std::sort(cuts.begin(), cuts.end());
    cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());
    for (size_t c = 0; c + 1 < cuts.size(); ++c) {
      const int64_t lo = cuts[c];
      const int64_t hi = cuts[c + 1];
      const int a = old_below.label_at(lo);
      const int b = old_above.label_at(lo);
      if (a != kNone) {
        above_[a].splice_uniform(lo, hi, b);
      }
      if (b != kNone) {
        below_[b].splice_uniform(lo, hi, a);
      }
    }
  }

  // Build x's own profiles at its current position and splice x into each new
  // neighbour's profile.
  void insert_into_profiles(int x) {
    const int64_t s_x = start_[x];
    const int64_t e_x = end_[x];
    const int pos_x = position_[x];
    const std::vector<int>& members = overlap_[x];

    std::set<int64_t> cut_set{s_x, e_x};
    for (const int w : members) {
      if (!eligible_[w]) {
        continue;
      }
      if (start_[w] > s_x) {
        cut_set.insert(start_[w]);
      }
      if (end_[w] < e_x) {
        cut_set.insert(end_[w]);
      }
    }
    std::vector<int64_t> cut_list;
    for (const int64_t c : cut_set) {
      if (s_x <= c && c <= e_x) {
        cut_list.push_back(c);
      }
    }
    // cut_set is ordered; cut_list inherits the order.

    std::vector<int64_t> below_starts, above_starts;
    std::vector<int> below_labels, above_labels;
    for (size_t ci = 0; ci + 1 < cut_list.size(); ++ci) {
      const int64_t lo = cut_list[ci];
      const int64_t hi = cut_list[ci + 1];
      int below = kNone;
      int below_pos = -1;
      int above = kNone;
      int above_pos = n_;  // len(permutation)
      for (const int w : members) {
        if (!eligible_[w]) {
          continue;
        }
        if (start_[w] <= lo && lo < end_[w]) {
          const int pw = position_[w];
          if (pw < pos_x) {
            if (pw > below_pos) {
              below_pos = pw;
              below = w;
            }
          } else if (above == kNone || pw < above_pos) {
            above_pos = pw;
            above = w;
          }
        }
      }
      below_starts.push_back(lo);
      below_labels.push_back(below);
      above_starts.push_back(lo);
      above_labels.push_back(above);
      if (below != kNone) {
        above_[below].splice_uniform(lo, hi, x);
      }
      if (above != kNone) {
        below_[above].splice_uniform(lo, hi, x);
      }
    }
    below_starts.push_back(e_x);
    above_starts.push_back(e_x);
    below_[x] = Profile::from_segments(std::move(below_starts),
                                       std::move(below_labels));
    above_[x] = Profile::from_segments(std::move(above_starts),
                                       std::move(above_labels));
  }

  // --- resize / eligibility reflow -----------------------------------------

  std::set<int> above_seed(int idx) const {
    std::set<int> s;
    add_labels(s, above_[idx]);
    return s;
  }

  std::set<int> inplace_pokethrough_seed(int idx) const {
    std::set<int> extra;
    for (const int partner : inplace_partners_[idx]) {
      if (reuse_[idx] == partner || reuse_[partner] == idx) {
        const std::pair<int, int> pr = in_place_pair(idx, partner);
        const int parent = pr.first;
        const int child = pr.second;
        const int64_t t = start_[child];
        for (const int w :
             {above_[child].label_at(t), above_[parent].label_at(t)}) {
          if (w != kNone) {
            extra.insert(w);
          }
        }
      }
    }
    return extra;
  }

  void reflow_resized(int idx) {
    if (!eligible_[idx]) {
      return;
    }
    std::set<int> seed{idx};
    for (const int w : above_seed(idx)) {
      seed.insert(w);
    }
    for (const int w : inplace_pokethrough_seed(idx)) {
      seed.insert(w);
    }
    propagate_addresses(seed);
  }

  void reflow_eligibility(int idx, bool flag) {
    std::set<int> seed;
    if (flag) {
      // HBM -> LX: reinsert idx into the order at its retained slot.
      eligible_[idx] = true;
      insert_into_profiles(idx);
      seed.insert(idx);
      for (const int w : above_seed(idx)) {
        seed.insert(w);
      }
    } else {
      // LX -> HBM: capture idx's adjacency, stitch its neighbours together,
      // drop its quality/address, and give it a transparent profile.
      const Profile old_below = below_[idx];
      const Profile old_above = above_[idx];
      add_labels(seed, old_above);
      if (is_fully_allocated(idx)) {
        total_quality_ -= qualities_[idx];
        total_allocated_count_ -= 1;
      }
      addresses_[idx] = std::nullopt;
      reuse_[idx] = kNone;
      eligible_[idx] = false;
      stitch_around_removed(idx, old_below, old_above);
      below_[idx] = Profile::uniform(start_[idx], end_[idx], kNone);
      above_[idx] = Profile::uniform(start_[idx], end_[idx], kNone);
    }
    propagate_addresses(seed);
  }

  // --- small utilities -----------------------------------------------------

  // Insert every non-None label of `prof` into `s`.
  static void add_labels(std::set<int>& s, const Profile& prof) {
    for (const int lab : prof.labels) {
      if (lab != kNone) {
        s.insert(lab);
      }
    }
  }

  // Distinct labels of `prof` (including kNone).
  static std::vector<int> label_set(const Profile& prof) {
    std::vector<int> out;
    for (const int lab : prof.labels) {
      if (std::find(out.begin(), out.end(), lab) == out.end()) {
        out.push_back(lab);
      }
    }
    return out;
  }

  static py::tuple profile_tuple(const Profile& prof) {
    py::list starts;
    for (const int64_t s : prof.starts) {
      starts.append(py::int_(s));
    }
    py::list labels;
    for (const int lab : prof.labels) {
      if (lab == kNone) {
        labels.append(py::none());
      } else {
        labels.append(py::int_(lab));
      }
    }
    return py::make_tuple(starts, labels);
  }

  // --- state ---------------------------------------------------------------

  int n_ = 0;
  int64_t capacity_ = 0;
  int64_t alignment_ = 128;
  std::vector<int> permutation_;
  std::unordered_map<std::string, int> name_to_idx_;

  // Per-buffer static data.
  std::vector<int64_t> start_, end_, sizes_;
  std::vector<double> weight_, qualities_;
  std::vector<char> eligible_;
  std::vector<std::unordered_set<int>> parent_set_;  // declared parents
  std::vector<std::set<int>> inplace_partners_;      // symmetric partners

  // Saturation early-stop interval data.
  std::vector<int64_t> interval_starts_;
  int num_intervals_ = 0;
  std::vector<int> total_at_;
  std::vector<std::pair<int, int>> buf_intervals_;

  // Dynamic layout state.
  std::vector<std::optional<int64_t>> addresses_;
  double total_quality_ = 0.0;
  int total_allocated_count_ = 0;
  std::vector<int> position_;
  std::vector<std::vector<int>> overlap_;  // static time-overlap sets
  std::vector<int> reuse_;  // inplace_reuse: idx -> partner/kNone
  std::vector<Profile> below_, above_;
  int rotate_remove_insert_threshold_ = 2;
};

// --- Optional[int] <-> kNone at the Python boundary -----------------------

std::vector<int> labels_from_python(const std::vector<std::optional<int>>& in) {
  std::vector<int> out;
  out.reserve(in.size());
  for (const std::optional<int>& v : in) {
    out.push_back(v.has_value() ? *v : kNone);
  }
  return out;
}

std::vector<std::optional<int>> labels_to_python(const std::vector<int>& in) {
  std::vector<std::optional<int>> out;
  out.reserve(in.size());
  for (const int v : in) {
    out.push_back(v == kNone ? std::nullopt : std::optional<int>(v));
  }
  return out;
}

std::optional<int> label_to_python(int label) {
  return label == kNone ? std::nullopt : std::optional<int>(label);
}

// Exposes the internal Profile to Python so the differential tests can compare
// it against contact_profile.Profile method by method, rather than only
// indirectly through the solver's below/above profiles. Test-only: nothing in
// torch_spyre itself constructs one of these.
void register_profile(py::module_& m) {
  py::class_<Profile>(m, "NativeProfile")
      .def(py::init([](const std::vector<int64_t>& starts,
                       const std::vector<std::optional<int>>& labels) {
             Profile p;
             p.starts = starts;
             p.labels = labels_from_python(labels);
             return p;
           }),
           py::arg("starts"), py::arg("labels"))
      .def_static(
          "uniform",
          [](int64_t span_start, int64_t span_end, std::optional<int> label) {
            return Profile::uniform(span_start, span_end,
                                    label.value_or(kNone));
          },
          py::arg("span_start"), py::arg("span_end"), py::arg("label"))
      .def_static(
          "from_segments",
          [](std::vector<int64_t> starts,
             const std::vector<std::optional<int>>& labels) {
            return Profile::from_segments(std::move(starts),
                                          labels_from_python(labels));
          },
          py::arg("starts"), py::arg("labels"))
      .def_property_readonly("starts",
                             [](const Profile& p) { return p.starts; })
      .def_property_readonly(
          "labels", [](const Profile& p) { return labels_to_python(p.labels); })
      .def_property_readonly("span_start", &Profile::span_start)
      .def_property_readonly("span_end", &Profile::span_end)
      .def(
          "label_at",
          [](const Profile& p, int64_t t) {
            return label_to_python(p.label_at(t));
          },
          py::arg("t"))
      .def(
          "segments",
          [](const Profile& p, int64_t a, int64_t b) {
            auto [s, l] = p.segments(a, b);
            return py::make_tuple(s, labels_to_python(l));
          },
          py::arg("a"), py::arg("b"))
      .def(
          "splice",
          [](Profile& p, int64_t a, int64_t b,
             const std::vector<int64_t>& seg_starts,
             const std::vector<std::optional<int>>& seg_labels) {
            const std::vector<int> labels = labels_from_python(seg_labels);
            p.splice(a, b, seg_starts, labels);
          },
          py::arg("a"), py::arg("b"), py::arg("seg_starts"),
          py::arg("seg_labels"))
      .def(
          "splice_uniform",
          [](Profile& p, int64_t a, int64_t b, std::optional<int> label) {
            p.splice_uniform(a, b, label.value_or(kNone));
          },
          py::arg("a"), py::arg("b"), py::arg("label"))
      .def(
          "relabel",
          [](Profile& p, int64_t a, int64_t b, std::optional<int> from_label,
             std::optional<int> to_label) {
            p.relabel(a, b, from_label.value_or(kNone),
                      to_label.value_or(kNone));
          },
          py::arg("a"), py::arg("b"), py::arg("from_label"),
          py::arg("to_label"))
      .def("validate", &Profile::validate)
      .def("__eq__", [](const Profile& a, const Profile& b) { return a == b; })
      .def("__repr__", [](const Profile& p) {
        std::string out = "NativeProfile(";
        for (size_t i = 0; i < p.labels.size(); ++i) {
          out += (i ? ", " : "");
          out += "[" + std::to_string(p.starts[i]) + "," +
                 std::to_string(p.starts[i + 1]) + ")=" +
                 (p.labels[i] == kNone ? "None" : std::to_string(p.labels[i]));
        }
        return out + ")";
      });
}

}  // namespace

void register_native_permutation_layout(py::module_& m) {
  register_profile(m);
  py::class_<NativePermutationLayoutSolver>(m, "NativePermutationLayoutSolver")
      .def(py::init<const py::list&, const std::vector<int>&, int64_t, int64_t,
                    const py::object&>(),
           py::arg("buffers"), py::arg("permutation"), py::arg("capacity"),
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
      .def("below_profile", &NativePermutationLayoutSolver::below_profile,
           py::arg("idx"))
      .def("above_profile", &NativePermutationLayoutSolver::above_profile,
           py::arg("idx"))
      .def("inplace_reuse", &NativePermutationLayoutSolver::inplace_reuse)
      .def_property_readonly("addresses",
                             &NativePermutationLayoutSolver::addresses)
      .def_property_readonly("permutation",
                             &NativePermutationLayoutSolver::permutation)
      .def_property(
          "_rotate_remove_insert_threshold",
          &NativePermutationLayoutSolver::rotate_remove_insert_threshold,
          &NativePermutationLayoutSolver::set_rotate_remove_insert_threshold);
}

}  // namespace scratchpad
}  // namespace torch_spyre
