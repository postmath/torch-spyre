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

"""
Relative-error minimax piecewise-linear approximation of g(x) = x**r,
0 < r < 1, on a general interval [a, b] with 0 < a < b.

Minimises  E = max |log(p(x)/x**r)|, i.e. p stays inside the multiplicative
band  exp(-E) <= p/g <= exp(+E).

Everything follows from scale covariance: x -> lam*x, y -> lam**r * y maps
lines to lines and leaves log(p/g) alone.  So a segment's optimality
condition depends only on the RATIO of its endpoints, never on where it
sits.  Since the alternation forces log(p/g) = +E at every free knot, each
segment type has a single characteristic ratio fixed by E alone:

    rho(E)     both endpoints lifted to +E   (a generic segment)
    sig_L(E)   left endpoint exact, right at +E
    sig_R(E)   left at +E, right endpoint exact

and the whole problem collapses to one span equation

    sig_L^[pin_left] * rho^(N - pin_left - pin_right) * sig_R^[pin_right]
        =  b / a

solved for E by bisection.  Both endpoints are pinned by default.  With
neither pinned there is nothing to solve: rho = (b/a)**(1/N) and E
follows in closed form.

Consequences worth knowing:
  * knots are geometric apart from the pinned end segments, and exactly
    x_j = a * (b/a)**(j/N) when nothing is pinned,
  * cost depends on b/a only through log(b/a)/N -- decades of dynamic
    range, not absolute width,
  * the answer is invariant under r <-> 1-r.
"""

import numpy as np


def _make(r):
    C = (1.0 - r) ** (1.0 - r) * r**r

    def min_log_ratio(lam_u, t, lam_v):
        """
        log of  min_{x in [1,t]} line(x)/x**r  for the segment joining
        (1, exp(lam_u)) to (t, exp(lam_v) * t**r).

        min_{x>0} (a0 + s*x)/x**r = a0**(1-r) * s**r / C when a0, s > 0,
        attained at x* = r*a0/((1-r)*s); otherwise the min is at an endpoint,
        so clamp rather than trusting the closed form.
        """
        yu, yv = np.exp(lam_u), np.exp(lam_v) * t**r
        s = (yv - yu) / (t - 1.0)
        a0 = yu - s
        best = min(lam_u, lam_v)
        if a0 > 0.0 and s > 0.0:
            xstar = r * a0 / ((1.0 - r) * s)
            if 1.0 < xstar < t:
                best = min(best, (1.0 - r) * np.log(a0) + r * np.log(s) - np.log(C))
        return best

    def seg_ratio(E, lam_u, lam_v):
        """Endpoint ratio t > 1 of a segment whose dip is exactly -E."""
        g = lambda v: min_log_ratio(lam_u, np.exp(v), lam_v) + E
        hi = 1.0
        while g(hi) > 0.0:
            hi *= 2.0
            if hi > 700.0:
                raise RuntimeError("E too large to realise")
        lo = 0.0
        for _ in range(300):
            mid = 0.5 * (lo + hi)
            if mid <= lo or mid >= hi:
                break
            if g(mid) > 0.0:
                lo = mid
            else:
                hi = mid
        return np.exp(0.5 * (lo + hi))

    return C, min_log_ratio, seg_ratio


def minimax_ratio_pwl(r, N, a, b, pin_left=True, pin_right=True, iters=300):
    """
    Return (xs, ys, E) for the relative-error minimax approximant on [a, b].

    pin_left / pin_right force p to be exact at that endpoint (log-error 0
    there instead of +E).  Both default to True, so p(a) = a**r and
    p(b) = b**r unless you ask otherwise; this needs N >= 2.  Pass
    pin_left=pin_right=False for the unconstrained optimum, which is
    closed form and somewhat more accurate at the same N.

    xs, ys have length N+1.  E is the sup of |log(p/g)|; the multiplicative
    band is exp(-E) .. exp(+E).
    """
    if not 0.0 < r < 1.0:
        raise ValueError("need 0 < r < 1")
    if not 0.0 < a < b:
        raise ValueError("need 0 < a < b")
    if N < 1:
        raise ValueError("need N >= 1")
    n_pin = int(pin_left) + int(pin_right)
    if N < n_pin:
        raise ValueError("need N >= number of pinned endpoints")

    C, min_log_ratio, seg_ratio = _make(r)
    R = b / a
    logR = np.log(R)
    n_free = N - n_pin

    # --- fully unpinned: closed form, no root finding ---------------------
    if n_pin == 0:
        rho = R ** (1.0 / N)
        # both ends lifted by k=exp(E) scales the line by k, so the dip is
        # k*Psi(rho) and the condition k*Psi = 1/k gives E = -log(Psi)/2
        E = -0.5 * min_log_ratio(0.0, rho, 0.0)
        xs = a * rho ** np.arange(N + 1)
        xs[-1] = b
        return xs, np.exp(E) * xs**r, E

    # --- N segments, at least one pinned end: shoot on E ------------------
    def log_span(E):
        tot = n_free * np.log(seg_ratio(E, E, E)) if n_free else 0.0
        if pin_left:
            tot += np.log(seg_ratio(E, 0.0, E))
        if pin_right:
            tot += np.log(seg_ratio(E, E, 0.0))
        return tot

    lo, hi = 0.0, 1.0
    while log_span(hi) < logR:  # span grows with E
        hi *= 2.0
        if hi > 500.0:
            raise RuntimeError("failed to bracket E")
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if mid <= lo or mid >= hi:
            break
        if log_span(mid) < logR:
            lo = mid
        else:
            hi = mid
    E = 0.5 * (lo + hi)

    ratios = []
    if pin_left:
        ratios.append(seg_ratio(E, 0.0, E))
    ratios += [seg_ratio(E, E, E)] * n_free
    if pin_right:
        ratios.append(seg_ratio(E, E, 0.0))

    xs = a * np.concatenate(([1.0], np.cumprod(ratios)))
    xs[-1] = b
    ys = np.exp(E) * xs**r
    if pin_left:
        ys[0] = a**r
    if pin_right:
        ys[-1] = b**r
    return xs, ys, E


def segments_for_tolerance(
    r, tol, a, b, pin_left=True, pin_right=True, relative="log", nmax=100000
):
    """
    Smallest N meeting a tolerance.  relative='log' reads tol as a bound on
    |log(p/g)|; 'band' reads it as a bound on |p/g - 1| (so E = log1p(tol)).
    """
    target = tol if relative == "log" else np.log1p(tol)
    n = max(1, int(pin_left) + int(pin_right))
    while n <= nmax:
        if minimax_ratio_pwl(r, n, a, b, pin_left, pin_right)[2] <= target:
            return n
        n += 1
    raise RuntimeError("tolerance not reachable within nmax")


def check(xs, ys, r, per_segment=200001):
    """Dense-sampled sup of |log(p/g)|, plus the signed extrema."""
    peaks = np.log(ys) - r * np.log(xs)
    dips = []
    for u, v, yu, yv in zip(xs[:-1], xs[1:], ys[:-1], ys[1:]):
        x = np.exp(np.linspace(np.log(u), np.log(v), per_segment))
        line = yu + (yv - yu) * (x - u) / (v - u)
        dips.append((np.log(line) - r * np.log(x)).min())
    dips = np.array(dips)
    return max(peaks.max(), -dips.min()), peaks, dips


# ---------------------------------------------------------------------------
# sympy output
# ---------------------------------------------------------------------------


def sympy_piecewise(r, N, a, b, pin_left=True, pin_right=True, x=None, outside="nan"):
    """
    Build a sympy Piecewise for the minimax approximant.  Coefficients and
    breakpoints are plain double-precision Floats.

    x        symbol to use (default Symbol('x', real=True))
    outside  'nan'    -> nan outside [a, b]
             'clamp'  -> held at the endpoint values
             'extend' -> end segments extrapolated

    Each piece is written a0 + s*x rather than y0 + s*(x - x0).  That is the
    better form to evaluate, because a0 and s are both positive on [a, b], so
    the sum never cancels.  Forming a0 is the delicate part:

        a0 = (y0*x1 - y1*x0) / (x1 - x0)

    is a difference of nearly equal products, and a0 works out to roughly
    (1-r)*y0, so in doubles it sheds about log10(1/(1-r)) digits -- two of
    them at r = 0.99.  So both coefficients are formed in mpmath at extended
    precision and rounded once, giving correctly rounded doubles.  That is
    the only place extended precision is used; the solve itself is double.
    """
    import mpmath as mp
    import sympy as sp

    if x is None:
        x = sp.Symbol("x", real=True)
    if outside not in ("nan", "clamp", "extend"):
        raise ValueError("outside must be 'nan', 'clamp' or 'extend'")

    xs, ys, _ = minimax_ratio_pwl(r, N, a, b, pin_left, pin_right)

    coeffs = []
    with mp.workdps(40):
        for x0, x1, y0, y1 in zip(xs[:-1], xs[1:], ys[:-1], ys[1:]):
            X0, X1 = mp.mpf(float(x0)), mp.mpf(float(x1))
            Y0, Y1 = mp.mpf(float(y0)), mp.mpf(float(y1))
            s = (Y1 - Y0) / (X1 - X0)
            a0 = (Y0 * X1 - Y1 * X0) / (X1 - X0)
            coeffs.append((float(a0), float(s)))

    F = sp.Float
    knots = [F(float(v)) for v in xs]
    piece = lambda k: F(coeffs[k][0]) + F(coeffs[k][1]) * x

    pieces = []
    if outside == "extend":
        pieces += [(piece(k), x < knots[k + 1]) for k in range(N - 1)]
        pieces.append((piece(N - 1), sp.true))
    else:
        out_lo = sp.nan if outside == "nan" else F(float(ys[0]))
        out_hi = sp.nan if outside == "nan" else F(float(ys[-1]))
        pieces.append((out_lo, x < knots[0]))
        pieces += [(piece(k), x < knots[k + 1]) for k in range(N - 1)]
        pieces.append((piece(N - 1), x <= knots[N]))
        pieces.append((out_hi, sp.true))

    return sp.Piecewise(*pieces)


def verify_expr(expr, x, r, dps=40):
    """
    Check a built expression at its exact extrema rather than on a grid.

    For a piece a0 + s*x the log-ratio log(p/g) has a single interior
    stationary point, at x* = r*a0/((1-r)*s), so every peak and dip can be
    hit exactly.  Grid sampling would need absurdly many points to resolve
    the dips, and quietly reports the sup too low.

    Returns (sup, peaks, dips) as floats.  Needs outside='nan' or 'clamp' so
    the domain edges can be recovered from the expression.
    """
    import mpmath as mp
    import sympy as sp

    lin = [(e, c) for e, c in expr.args if e.has(x)]
    bnds = [(float(c.rhs) if c is not sp.true else None) for e, c in lin]
    edge = [c.rhs for e, c in expr.args if not e.has(x) and c is not sp.true]
    if not edge:
        raise ValueError("need outside='nan' or 'clamp' to locate the domain")

    with mp.workdps(dps):
        rm = mp.mpf(float(r))
        left = mp.mpf(float(edge[0]))
        peaks, dips = [], []
        for k, (e, _) in enumerate(lin):
            s, a0 = (mp.mpf(float(c)) for c in sp.Poly(e, x).all_coeffs())
            u = left if k == 0 else mp.mpf(bnds[k - 1])
            v = mp.mpf(bnds[k]) if bnds[k] is not None else mp.inf
            for xv in (u, v):
                if xv != mp.inf:
                    peaks.append(mp.log(a0 + s * xv) - rm * mp.log(xv))
            xstar = rm * a0 / ((1 - rm) * s)
            if u < xstar < v:
                dips.append(mp.log(a0 + s * xstar) - rm * mp.log(xstar))
        sup = max(max(abs(p) for p in peaks), max(abs(d) for d in dips))
    return float(sup), [float(p) for p in peaks], [float(d) for d in dips]


if __name__ == "__main__":
    import sympy as sp

    r, a, b = 0.4, 3.0, 4096.0
    print(f"g(x) = x**{r} on [{a}, {b}]   ({np.log10(b / a):.2f} decades)\n")
    print(
        f"{'N':>3} {'default':>12} {'pin right':>12} {'pin left':>12}"
        f" {'unpinned':>12} {'band':>10}"
    )
    for N in (2, 4, 8, 16, 32):
        E = minimax_ratio_pwl(r, N, a, b)[2]
        Er = minimax_ratio_pwl(r, N, a, b, pin_left=False)[2]
        El = minimax_ratio_pwl(r, N, a, b, pin_right=False)[2]
        Ef = minimax_ratio_pwl(r, N, a, b, pin_left=False, pin_right=False)[2]
        print(
            f"{N:>3} {E:>12.5e} {Er:>12.5e} {El:>12.5e} {Ef:>12.5e} {np.expm1(E):>9.3%}"
        )

    n = segments_for_tolerance(r, 1e-3, a, b, relative="band")
    print(
        f"\nsegments for a 0.1% band: N = {n}  ({n / np.log10(b / a):.2f} per decade)"
    )

    x = sp.Symbol("x", real=True)
    expr = sympy_piecewise(r, 3, a, b, x=x)
    print("\nsympy_piecewise(0.4, 3, 3, 4096):")
    sp.pprint(expr)

    sup, peaks, dips = verify_expr(expr, x, r)
    print(f"\nexact extrema:  sup = {sup:.17g}")
    print(f"                E   = {minimax_ratio_pwl(r, 3, a, b)[2]:.17g}")
    print(f"                dip spread = {max(dips) - min(dips):.2e}")
