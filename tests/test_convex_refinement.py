"""Independent small-QP checks of the fixed-feature ADMM objective and bounds."""
import numpy as np
import pytest
import cvxopt

from review.convex_refinement import (
    dual_bound, primal_objective, project_dual, refine_pinball,
)


def independent_primal_qp(Z, y, R, c, C, tau):
    """Slack-variable primal with equality constraints; no ADMM operations."""
    n, m = Z.shape
    dimension = m + 1 + 2 * n
    P = np.zeros((dimension, dimension))
    P[:m, :m] = R
    q = np.r_[-c, 0., C * tau, C * (1 - tau)]
    A = np.column_stack((Z, np.ones(n), np.eye(n), -np.eye(n)))
    G = np.column_stack((np.zeros((2 * n, m + 1)), -np.eye(2 * n)))
    result = cvxopt.solvers.qp(
        cvxopt.matrix(P), cvxopt.matrix(q), cvxopt.matrix(G),
        cvxopt.matrix(np.zeros(2 * n)), cvxopt.matrix(A), cvxopt.matrix(y),
        options={"show_progress": False, "abstol": 1e-11,
                 "reltol": 1e-11, "feastol": 1e-11},
    )
    assert result["status"] == "optimal"
    theta = np.asarray(result["x"]).ravel()
    return float(result["primal objective"]), theta[:m], float(theta[m])


@pytest.mark.parametrize("tau", [.025, .5, .975])
@pytest.mark.parametrize("rank_deficient_features", [False, True])
def test_admm_primal_and_feasible_dual_bracket_independent_qp(tau, rank_deficient_features):
    rng = np.random.default_rng(71)
    Z, y = rng.normal(size=(18, 4)), rng.normal(size=18)
    if rank_deficient_features:
        Z[:, 2] = Z[:, 0]
        Z[7] = Z[3]
    T = rng.normal(size=(4, 4))
    R, c = np.eye(4) + T.T @ T, np.array([.8, -.4, .3, 1.1])
    C = np.r_[np.full(12, .15), np.full(5, 1.7), 0.]
    exact, exact_w, _ = independent_primal_qp(Z, y, R, c, C, tau)
    fitted = refine_pinball(Z, y, R, c, C, tau, max_iter=15000,
                           gap_rtol=1e-7, gap_atol=1e-9, check_every=20)
    assert fitted.converged
    assert fitted.stop_reason == "duality_gap"
    assert fitted.dual_bound <= exact + 2e-8
    assert fitted.primal >= exact - 2e-8
    assert fitted.gap <= 1e-9 + 1e-7 * max(1., abs(fitted.primal), abs(fitted.dual_bound))
    np.testing.assert_allclose(fitted.primal, exact, rtol=2e-7, atol=2e-7)
    np.testing.assert_allclose(fitted.w, exact_w, rtol=2e-3, atol=2e-3)
    assert fitted.dual_variable[-1] == 0
    assert fitted.dual_box_violation == 0
    assert fitted.dual_sum_residual < 1e-12
    np.testing.assert_allclose(
        dual_bound(Z, y, R, c, C, tau, fitted.dual_variable), fitted.dual_bound,
        rtol=1e-12, atol=1e-12,
    )


def test_dual_projection_is_box_constrained_zero_sum_and_nearest_point():
    v = np.array([12., -7., .03, 1.2, 0., -3.])
    C, tau = np.array([.01, 2., 0., .8, 5., .2]), .91
    projected = project_dual(v, C, tau)
    assert np.max(projected - C * tau) <= 0
    assert np.max(C * (tau - 1) - projected) <= 0
    assert abs(projected.sum()) < 1e-14
    # Independent convex projection via a dense QP.
    result = cvxopt.solvers.qp(
        cvxopt.matrix(np.eye(len(v))), cvxopt.matrix(-v),
        cvxopt.matrix(np.vstack((np.eye(len(v)), -np.eye(len(v))))),
        cvxopt.matrix(np.r_[C * tau, C * (1 - tau)]),
        cvxopt.matrix(np.ones((1, len(v)))), cvxopt.matrix(np.zeros(1)),
        options={"show_progress": False, "abstol": 1e-10,
                 "reltol": 1e-10, "feastol": 1e-10},
    )
    assert result["status"] == "optimal"
    np.testing.assert_allclose(projected, np.asarray(result["x"]).ravel(), atol=2e-7)


def test_gap_zero_with_free_intercept_and_translation_equivariance():
    rng = np.random.default_rng(8)
    Z = rng.normal(size=(15, 3))
    R, c, C = np.eye(3), np.zeros(3), np.linspace(.1, .7, 15)
    fitted = refine_pinball(Z, np.full(15, 4.2), R, c, C, .7)
    assert fitted.converged and fitted.iterations == 0
    assert fitted.primal == 0 and fitted.dual_bound == 0
    assert fitted.b == 4.2
    y = rng.normal(size=15)
    first = refine_pinball(Z, y, R, c, C, .8, gap_rtol=1e-7)
    second = refine_pinball(Z, y + 13.4, R, c, C, .8, b0=13.4, gap_rtol=1e-7)
    np.testing.assert_allclose(first.w, second.w, atol=1e-10)
    np.testing.assert_allclose(first.b + 13.4, second.b, atol=1e-10)
    np.testing.assert_allclose(first.primal, second.primal, atol=1e-10)


def test_iteration_limit_is_not_called_convergence_and_incumbent_cannot_worsen():
    rng = np.random.default_rng(4)
    Z, y = rng.normal(size=(30, 5)), rng.normal(size=30)
    R, c, C = np.eye(5), np.linspace(-.5, .5, 5), np.ones(30)
    w0, b0 = rng.normal(size=5), .7
    fitted = refine_pinball(Z, y, R, c, C, .31, w0=w0, b0=b0,
                           max_iter=1, gap_rtol=1e-12, gap_atol=1e-12)
    assert not fitted.converged and fitted.stop_reason == "max_iter"
    assert fitted.primal <= fitted.initial_primal
    np.testing.assert_allclose(fitted.initial_primal,
                              primal_objective(Z, y, R, c, C, .31, w0, b0))
    np.testing.assert_allclose(fitted.primal,
                              primal_objective(Z, y, R, c, C, .31, fitted.w, fitted.b))
    exact, _, _ = independent_primal_qp(Z, y, R, c, C, .31)
    assert fitted.dual_bound <= exact + 1e-8 <= fitted.primal + 1e-8


def test_invalid_parameters_and_infeasible_dual_are_rejected():
    Z, y, R, c, C = np.eye(3), np.arange(3.), np.eye(3), np.zeros(3), np.ones(3)
    with pytest.raises(ValueError, match="dual vector"):
        dual_bound(Z, y, R, c, C, .5, np.array([.2, 0., 0.]))
    for changes in ({"tau": 1.}, {"C": np.zeros(3)}, {"C": -C},
                    {"R": np.zeros((3, 3))}, {"R": np.array([[1., 1., 0.], [0., 1., 0.], [0., 0., 1.]])},
                    {"rho": 0.}, {"max_iter": 0}, {"gap_rtol": -1.}, {"max_seconds": 0.}):
        args = dict(Z=Z, y=y, R=R, c=c, C=C, tau=.5)
        args.update(changes)
        with pytest.raises(ValueError):
            refine_pinball(**args)
