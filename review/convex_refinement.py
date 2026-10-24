"""Training-only ADMM refinement of a *fixed-feature* weighted pinball model.

The objective, omitting any label-only constant, is

    P(w,b) = .5 w.T R w - c.T w + sum_i C_i rho_tau(y_i-Z_i w-b).

R must be symmetric positive definite.  Landmarks, preprocessing, R, c, C and
tau are inputs and are never selected here.  With A=[Z,1], ADMM splits
A theta + e = y.  Its weight/intercept step uses one m-by-m factorization;
its residual step is asymmetric soft thresholding.  For any feasible v,

    D(v) = y.T v - .5 (c+Z.T v).T R^{-1} (c+Z.T v),
    C*(tau-1) <= v <= C*tau,  sum(v)=0,

is a lower bound.  We project the ADMM multiplier into this feasible set and
stop on a primal-dual objective gap, not on validation/test predictions.
Certificates are numerical, with feasibility residuals reported explicitly;
they are not interval-arithmetic proofs.  They concern the given approximate
feature objective, not an exact-kernel or generalization error bound.

ADMM reference: Boyd et al. (2011), Foundations and Trends in Machine Learning
3(1), 1--122, https://web.stanford.edu/~boyd/papers/admm_distr_stats.html .
"""
from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Callable

import numpy as np
from scipy.linalg import cho_factor, cho_solve


def primal_objective(Z, y, R, c, C, tau, w, b):
    """Objective without the label-only constant, in the supplied label units."""
    residual = np.asarray(y) - np.asarray(Z) @ np.asarray(w) - float(b)
    return float(.5 * w @ R @ w - c @ w
                 + np.asarray(C) @ np.maximum(tau * residual, (tau - 1) * residual))


def _weights(C, tau):
    C = np.asarray(C, dtype=np.float64)
    if C.ndim != 1 or not len(C) or not np.isfinite(C).all() or np.any(C < 0):
        raise ValueError("C must be a nonempty, finite, nonnegative vector.")
    if not np.any(C > 0):
        raise ValueError("At least one loss coefficient must be positive.")
    if not np.isfinite(tau) or not 0 < tau < 1:
        raise ValueError("tau must be strictly between zero and one.")
    return C


def project_dual(v, C, tau):
    """Euclidean projection onto the dual box intersected with sum(v)=0.

    clip(v-nu, lower, upper) has a monotone sum, so scalar bisection suffices.
    A final within-box correction removes the remaining summation roundoff.
    Zero coefficients and very unequal weights are supported.
    """
    C = _weights(C, tau)
    v = np.asarray(v, dtype=np.float64)
    if v.shape != C.shape or not np.isfinite(v).all():
        raise ValueError("v must be finite and have the same shape as C.")
    lower, upper = (tau - 1) * C, tau * C
    left, right = float(np.min(v - upper)), float(np.max(v - lower))
    for _ in range(100):
        middle = left + .5 * (right - left)
        projected = np.clip(v - middle, lower, upper)
        total = float(projected.sum())
        if total > 0:
            left = middle
        else:
            right = middle
        if right - left <= 4 * np.finfo(float).eps * max(1., abs(left), abs(right)):
            break
    projected = np.clip(v - (left + .5 * (right - left)), lower, upper)
    # Bisection's final error is normally roundoff-sized.  All boxes contain
    # zero, so there is always enough room to correct a nonzero total.
    for _ in range(4):
        total = float(projected.sum())
        if total == 0:
            break
        room = projected - lower if total > 0 else upper - projected
        candidates = np.flatnonzero(room > 0)
        for index in candidates[np.argsort(room[candidates])[::-1]]:
            adjustment = min(abs(total), float(room[index]))
            projected[index] -= np.copysign(adjustment, total)
            total -= np.copysign(adjustment, total)
            if abs(total) <= np.finfo(float).eps:
                break
    return projected


def _feasibility(v, C, tau):
    return {
        "dual_sum_residual": float(abs(v.sum())),
        "dual_box_violation": float(max(0., np.max((tau - 1) * C - v),
                                         np.max(v - tau * C))),
    }


def dual_bound(Z, y, R, c, C, tau, v):
    """Evaluate a numerically feasible dual lower bound; reject infeasible v."""
    C = _weights(C, tau)
    v = np.asarray(v, dtype=np.float64)
    if v.shape != C.shape or not np.isfinite(v).all():
        raise ValueError("v must be finite and have the same shape as C.")
    feasibility = _feasibility(v, C, tau)
    tolerance = 100 * np.finfo(float).eps * max(1., float(C.sum()))
    if max(feasibility.values()) > tolerance:
        raise ValueError("The proposed dual vector violates the box or zero-sum constraint.")
    h = np.asarray(c) + np.asarray(Z).T @ v
    return float(np.asarray(y) @ v - .5 * h @ cho_solve(cho_factor(R, lower=True), h))


def _weighted_quantile(values, C, tau):
    positive = C > 0
    values, weights = values[positive], C[positive]
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    index = min(np.searchsorted(cumulative, tau * cumulative[-1], side="left"), len(order) - 1)
    return float(values[order[index]])


@dataclass
class RefinementResult:
    w: np.ndarray
    b: float
    dual_variable: np.ndarray
    initial_primal: float
    primal: float
    dual_bound: float
    gap: float
    relative_gap: float
    converged: bool
    iterations: int
    elapsed_seconds: float
    stop_reason: str
    dual_sum_residual: float
    dual_box_violation: float
    gap_atol: float
    gap_rtol: float
    history: list[dict] = field(default_factory=list)

    def report(self):
        """JSON-ready diagnostics; arrays should be saved separately as NPZ."""
        return {key: value for key, value in vars(self).items()
                if key not in {"w", "b", "dual_variable"}}


def refine_pinball(Z, y, R, c, C, tau, *, w0=None, b0=0., max_iter=5000,
                   gap_rtol=1e-4, gap_atol=1e-7, rho=None, check_every=25,
                   adaptive_rho=True, max_seconds=None,
                   callback: Callable[[dict], None] | None = None):
    """Return the best feasible primal iterate and best feasible dual bound.

    Success means P-D <= gap_atol + gap_rtol*max(1,abs(P),abs(D)).  P and D
    omit the same label-only constant.  The gap is unchanged by adding that
    constant; the stated relative normalization is not invariant to it.

    ADMM residuals are diagnostics for its latest split state, whereas the
    returned primal/dual can be better incumbents from earlier iterations.
    Intercepts are minimized exactly at diagnostic checks.  Adaptive rho uses
    only training residuals, at check boundaries, and preserves the multiplier.
    No finite-iteration rate or predictive-performance guarantee is asserted.
    """
    start = time.perf_counter()
    Z, y = np.asarray(Z, dtype=np.float64), np.asarray(y, dtype=np.float64)
    R, c = np.asarray(R, dtype=np.float64), np.asarray(c, dtype=np.float64)
    C = _weights(C, tau)
    if Z.ndim != 2 or min(Z.shape) <= 0:
        raise ValueError("Z must be a nonempty two-dimensional feature matrix.")
    n, m = Z.shape
    if y.shape != (n,) or C.shape != (n,) or c.shape != (m,) or R.shape != (m, m):
        raise ValueError("Incompatible feature, label, weight, R or c dimensions.")
    if not all(np.isfinite(a).all() for a in (Z, y, R, c)):
        raise ValueError("Inputs must all be finite.")
    if not np.allclose(R, R.T, rtol=1e-12, atol=1e-12):
        raise ValueError("R must be symmetric positive definite.")
    R = (R + R.T) * .5
    try:
        r_factor = cho_factor(R, lower=True, check_finite=False)
    except np.linalg.LinAlgError as error:
        raise ValueError("R must be symmetric positive definite.") from error
    for name, value in (("max_iter", max_iter), ("check_every", check_every)):
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    if not all(np.isfinite(v) and v >= 0 for v in (gap_atol, gap_rtol)):
        raise ValueError("Gap tolerances must be finite and nonnegative.")
    if gap_atol == 0 and gap_rtol == 0:
        raise ValueError("At least one gap tolerance must be positive.")
    if max_seconds is not None and (not np.isfinite(max_seconds) or max_seconds <= 0):
        raise ValueError("max_seconds must be finite and positive if supplied.")
    rho = float(np.median(C[C > 0])) if rho is None else float(rho)
    if not np.isfinite(rho) or rho <= 0:
        raise ValueError("rho must be finite and positive.")
    w = np.zeros(m) if w0 is None else np.asarray(w0, dtype=np.float64).copy()
    b = float(b0)
    if w.shape != (m,) or not np.isfinite(w).all() or not np.isfinite(b):
        raise ValueError("The initial weight and intercept must be finite and correctly shaped.")
    initial_primal = primal_objective(Z, y, R, c, C, tau, w, b)
    best_w, best_b, best_primal = w.copy(), b, initial_primal
    best_v = np.zeros(n)
    best_dual = float(-.5 * c @ cho_solve(r_factor, c, check_finite=False))
    history = []
    converged, stop_reason = False, "max_iter"

    mean_Z = Z.mean(axis=0)
    centered_Z = Z - mean_Z
    gram = centered_Z.T @ centered_Z
    factor = cho_factor(R + rho * gram, lower=True, check_finite=False)
    e = y - Z @ w - b
    u = np.zeros(n)

    def evaluate(iteration, primal_residual, dual_residual):
        nonlocal best_w, best_b, best_primal, best_v, best_dual
        # Every (w,b) is primal-feasible by assigning e=y-Zw-b.  Minimizing b
        # can improve an incumbent but does not alter ADMM's internal state.
        offset = y - Z @ w
        candidate_b = _weighted_quantile(offset, C, tau)
        value = primal_objective(Z, y, R, c, C, tau, w, candidate_b)
        if value < best_primal:
            best_w, best_b, best_primal = w.copy(), candidate_b, value
        candidate_v = project_dual(-rho * u, C, tau)
        h = c + Z.T @ candidate_v
        bound = float(y @ candidate_v - .5 * h @ cho_solve(r_factor, h, check_finite=False))
        if bound > best_dual:
            best_v, best_dual = candidate_v.copy(), bound
        scale = max(1., abs(best_primal), abs(best_dual))
        raw_gap = best_primal - best_dual
        if raw_gap < -1e-9 * scale:
            raise FloatingPointError("A dual bound exceeds the primal objective beyond roundoff.")
        gap = max(0., raw_gap)
        row = {"iteration": iteration, "primal": best_primal, "dual_bound": best_dual,
               "gap": gap, "relative_gap": gap / scale, "raw_gap": raw_gap,
               "primal_residual": float(primal_residual), "dual_residual": float(dual_residual),
               "rho": rho, "elapsed_seconds": time.perf_counter() - start,
               **_feasibility(best_v, C, tau)}
        history.append(row)
        if callback is not None:
            callback(dict(row))
        return gap <= gap_atol + gap_rtol * scale

    converged = evaluate(0, 0., 0.)
    iteration = 0
    if converged:
        stop_reason = "duality_gap"
    else:
        for iteration in range(1, max_iter + 1):
            previous_e = e
            rhs = y - e - u
            w = cho_solve(factor, c + rho * (centered_Z.T @ rhs), check_finite=False)
            b = float(rhs.mean() - mean_Z @ w)
            prediction = Z @ w + b
            prox_input = y - prediction - u
            e = (np.maximum(prox_input - C * tau / rho, 0.)
                 + np.minimum(prox_input + C * (1. - tau) / rho, 0.))
            residual = prediction + e - y
            u += residual
            timed_out = max_seconds is not None and time.perf_counter() - start >= max_seconds
            if iteration % check_every == 0 or iteration == max_iter or timed_out:
                primal_residual = np.linalg.norm(residual)
                e_difference = e - previous_e
                dual_residual = rho * np.linalg.norm(np.r_[Z.T @ e_difference, e_difference.sum()])
                converged = evaluate(iteration, primal_residual, dual_residual)
                if converged:
                    stop_reason = "duality_gap"
                    break
                if timed_out:
                    stop_reason = "time_limit"
                    break
                if adaptive_rho:
                    # Normalize differently dimensioned residuals before the
                    # usual residual-balancing heuristic.  This changes only
                    # optimization, and never the target objective.
                    # Center location terms because the intercept is free;
                    # an additive label shift should not change rho updates.
                    primal_scale = max(1., np.linalg.norm(prediction - prediction.mean()),
                                       np.linalg.norm(e), np.linalg.norm(y - y.mean()))
                    multiplier = -rho * u
                    dual_scale = max(1., np.linalg.norm(np.r_[Z.T @ multiplier, multiplier.sum()]),
                                     np.linalg.norm(R @ w - c))
                    r_normalized, s_normalized = primal_residual / primal_scale, dual_residual / dual_scale
                    next_rho = rho
                    if r_normalized > 10 * s_normalized and rho < 1e8:
                        next_rho = rho * 2
                    elif s_normalized > 10 * r_normalized and rho > 1e-12:
                        next_rho = rho / 2
                    if next_rho != rho:
                        u *= rho / next_rho
                        rho = next_rho
                        factor = cho_factor(R + rho * gram, lower=True, check_finite=False)
    last = history[-1]
    return RefinementResult(
        w=best_w, b=best_b, dual_variable=best_v, initial_primal=initial_primal,
        primal=best_primal, dual_bound=best_dual, gap=last["gap"],
        relative_gap=last["relative_gap"], converged=converged, iterations=iteration,
        elapsed_seconds=time.perf_counter() - start, stop_reason=stop_reason,
        **_feasibility(best_v, C, tau), gap_atol=gap_atol, gap_rtol=gap_rtol, history=history,
    )
