"""Small numerical checks of the revised objective, not benchmark evidence.

Run from the repository root with:
    python -m pytest -q tests/test_centered_residual_objective.py
"""

import itertools

import cvxopt
import numpy as np
import pytest

from Kernel_Function import rbf_kernel
from Train_TL_QLDMR import (
    Fast_TL_QLDMR,
    NystromApproximation,
    TL_QLDMR,
)


def test_exact_kernel_solver_defaults_to_certified_cvxopt_path():
    model = TL_QLDMR(use_gpu=False)
    assert model.solver == "cvxopt"


def pinball(residual, tau):
    return np.maximum(tau * residual, (tau - 1) * residual)


def direct_objective(w, b, Z_S, y_S, Z_T, y_T, model):
    residual_s = y_S - Z_S @ w - b
    centered_residual = residual_s - residual_s.mean()
    domain_difference = (Z_S.mean(axis=0) - Z_T.mean(axis=0)) @ w
    return (
        0.5 * (w @ w)
        + model.lambda1 * np.mean(centered_residual**2) / 2
        + model.lambda2 * domain_difference**2 / 2
        + model.C_S * pinball(residual_s, model.tau).sum()
        + model.C_T * pinball(y_T - Z_T @ w - b, model.tau).sum()
    )


def test_centering_expansion_gradient_and_batch_scaling():
    rng = np.random.default_rng(17)
    Z_S, Z_T = rng.normal(size=(4, 3)), rng.normal(size=(3, 3))
    y_S, y_T = rng.normal(size=4), rng.normal(size=3)
    model = Fast_TL_QLDMR(lambda1=2.1, lambda2=0.7, C_S=0.4, C_T=1.3, tau=0.8)
    model.w, model.b = rng.normal(size=3), 0.37
    R, c = model._compute_regularizer_terms(Z_S, Z_T, y_S)
    Z, y = np.vstack((Z_S, Z_T)), np.concatenate((y_S, y_T))
    weights = np.r_[np.full(len(y_S), model.C_S), np.full(len(y_T), model.C_T)]
    constant = model.lambda1 * np.mean((y_S - y_S.mean()) ** 2) / 2
    expanded = (0.5 * model.w @ R @ model.w - c @ model.w + constant
                + weights @ pinball(y - Z @ model.w - model.b, model.tau))
    direct = direct_objective(model.w, model.b, Z_S, y_S, Z_T, y_T, model)
    np.testing.assert_allclose(expanded, direct, rtol=1e-13, atol=1e-13)
    assert np.linalg.norm(c) > 0.1  # exercise the formerly missing linear term

    analytic_w, analytic_b = model._batch_gradients(Z, y, weights, R, c, len(y))
    theta = np.r_[model.w, model.b]
    finite_difference = []
    for direction in np.eye(len(theta)):
        plus, minus = theta + 1e-6 * direction, theta - 1e-6 * direction
        finite_difference.append((
            direct_objective(plus[:-1], plus[-1], Z_S, y_S, Z_T, y_T, model)
            - direct_objective(minus[:-1], minus[-1], Z_S, y_S, Z_T, y_T, model)
        ) / 2e-6)
    np.testing.assert_allclose(np.r_[analytic_w, analytic_b], finite_difference,
                               rtol=2e-8, atol=2e-8)

    # Average over every size-2 sample: E[g_B] = B/N times the full gradient.
    gradients = []
    for indices in itertools.combinations(range(len(y)), 2):
        idx = list(indices)
        gw, gb = model._batch_gradients(Z[idx], y[idx], weights[idx], R, c, len(y))
        gradients.append(np.r_[gw, gb])
    np.testing.assert_allclose(np.mean(gradients, axis=0),
                               2 / len(y) * np.r_[analytic_w, analytic_b], atol=1e-13)
    # The linear term is unchanged by an additive shift in source labels.
    _, shifted_c = model._compute_regularizer_terms(Z_S, Z_T, y_S + 11)
    np.testing.assert_allclose(shifted_c, c, atol=1e-14)


def solve_independent_feature_qp(Z, y, n_source, model):
    """Direct slack-variable primal, independently assembled from the objective."""
    n, m = Z.shape
    centered_Z = Z[:n_source] - Z[:n_source].mean(axis=0)
    centered_y = y[:n_source] - y[:n_source].mean()
    delta = Z[:n_source].mean(axis=0) - Z[n_source:].mean(axis=0)
    R = np.eye(m) + model.lambda1 / n_source * centered_Z.T @ centered_Z
    R += model.lambda2 * np.outer(delta, delta)
    c = model.lambda1 / n_source * centered_Z.T @ centered_y
    weights = np.r_[np.full(n_source, model.C_S), np.full(n - n_source, model.C_T)]
    P = np.zeros((m + 1 + 2*n, m + 1 + 2*n))
    P[:m, :m] = R
    q = np.r_[-c, 0.0, weights * model.tau, weights * (1 - model.tau)]
    G = np.vstack((
        np.column_stack((-Z, -np.ones(n), -np.eye(n), np.zeros((n, n)))),
        np.column_stack((Z, np.ones(n), np.zeros((n, n)), -np.eye(n))),
        np.column_stack((np.zeros((2*n, m + 1)), -np.eye(2*n))),
    ))
    h = np.r_[-y, y, np.zeros(2*n)]
    result = cvxopt.solvers.qp(
        cvxopt.matrix(P), cvxopt.matrix(q), cvxopt.matrix(G), cvxopt.matrix(h),
        options={"show_progress": False, "abstol": 1e-10, "reltol": 1e-10,
                 "feastol": 1e-10},
    )
    assert result["status"] == "optimal"
    solution = np.array(result["x"]).ravel()
    return solution[:m], solution[m]


@pytest.mark.parametrize("duplicate_samples", [False, True])
def test_exact_kernel_matches_independent_primal_including_singular_kernel(duplicate_samples):
    X = np.array([[-1.4, 0.2], [-0.3, 0.4], [0.7, -0.5], [1.4, 0.7],
                  [-0.8, -0.4], [0.9, 1.0]])
    if duplicate_samples:
        X[4] = X[0]
    y = np.array([-1.2, 0.9, 1.3, -0.4, -0.7, 1.8])
    n_source = 4
    model = TL_QLDMR(lambda1=2.3, lambda2=0.8, C_S=0.7, C_T=1.5,
                     tau=0.7, kernel_gamma=0.8, solver="cvxopt", use_gpu=False)
    assert model.fit(X[:n_source], y[:n_source], X[n_source:], y[n_source:])
    K = rbf_kernel(X, X, model.gamma)
    values, vectors = np.linalg.eigh(K)
    keep = values > 1e-12
    Z = vectors[:, keep] * np.sqrt(values[keep])
    # Exact RKHS feature representation, independent of the dual derivation.
    np.testing.assert_allclose(Z @ Z.T, K, atol=2e-14)
    feature_w, feature_b = solve_independent_feature_qp(Z, y, n_source, model)
    kernel_w = Z.T @ model.beta_primal
    exact_value = direct_objective(kernel_w, model.b, Z[:n_source], y[:n_source],
                                   Z[n_source:], y[n_source:], model)
    primal_value = direct_objective(feature_w, feature_b, Z[:n_source], y[:n_source],
                                    Z[n_source:], y[n_source:], model)
    np.testing.assert_allclose(exact_value, primal_value, rtol=2e-7, atol=2e-7)
    expected_rank = len(X) - int(duplicate_samples)
    assert model.qp_stats["kernel_rank"] == expected_rank

    # Full-landmark Nyström includes the existing 1e-6 spectral jitter,
    # so compare within that declared numerical approximation, not equality.
    mapper = NystromApproximation(n_components=len(X), gamma=model.gamma).fit(X)
    mapped = mapper.transform(X)
    assert np.linalg.norm(mapped @ mapped.T - K, ord=2) < 1.01e-6
    normalization = mapper.normalization_
    np.testing.assert_allclose(normalization, normalization.T, atol=2e-13)
    np.testing.assert_allclose(
        normalization @ (K + 1e-6 * np.eye(len(X))) @ normalization,
        np.eye(len(X)), atol=5e-9,
    )


def test_nystrom_svd_failure_does_not_substitute_or_reuse_a_feature_map(monkeypatch):
    X = np.array([[-1.0], [0.0], [1.0]])
    mapper = NystromApproximation(n_components=3).fit(X)

    def fail_svd(*args, **kwargs):
        raise np.linalg.LinAlgError("injected SVD failure")

    monkeypatch.setattr(np.linalg, "svd", fail_svd)
    with pytest.raises(np.linalg.LinAlgError, match="No alternative feature map"):
        mapper.fit(X + 1)
    assert mapper.normalization_ is None
    with pytest.raises(ValueError, match="not fitted"):
        mapper.transform(X)


def test_fast_adam_update_includes_linear_term():
    X_S, y_S = np.array([[-1.0], [0.0], [1.0]]), np.array([-2.0, 0.3, 1.7])
    X_T, y_T = np.array([[-0.5], [0.7]]), np.array([-0.8, 1.1])
    model = Fast_TL_QLDMR(lambda1=5.0, lambda2=0.2, C_S=0.03, C_T=0.05,
                          tau=0.7, n_components=3, epochs=1, batch_size=16)
    np.random.seed(9)
    model.fit(X_S, y_S, X_T, y_T)
    Z = model.feature_map.transform(np.vstack((X_S, X_T)).astype(np.float32))
    y = np.r_[y_S, y_T]
    weights = np.r_[np.full(3, 0.03, dtype=np.float32),
                    np.full(2, 0.05, dtype=np.float32)]
    pinball_gradient = np.where(y >= 0, -model.tau, 1 - model.tau) * weights
    gradient = Z.T @ pinball_gradient - model.regularizer_linear_
    expected_w = -model.lr * gradient / (np.abs(gradient) + 1e-8)
    np.testing.assert_allclose(model.w, expected_w, rtol=2e-7, atol=1e-9)


def test_legacy_penalties_are_explicit_and_have_no_residual_linear_term():
    Z_S = np.array([[-1.0, 0.4], [0.3, -0.7], [2.0, 0.8]])
    Z_T, y = np.array([[0.5, -0.3]]), np.array([-0.2, 0.5, 1.3])
    for mode, amplitude in (("symmetric", 0.0), ("asymmetric", 0.25)):
        model = Fast_TL_QLDMR(variance_mode=mode, asym_scale=amplitude, tau=0.7)
        R, c = model._compute_regularizer_terms(Z_S, Z_T, y)
        centered = Z_S - Z_S.mean(axis=0)
        weights = np.ones(len(y)) if mode == "symmetric" else (
            1 + amplitude * np.sign(y - np.quantile(y, model.tau)))
        delta = Z_S.mean(axis=0) - Z_T.mean(axis=0)
        expected = (np.eye(2) + centered.T @ (centered * weights[:, None]) / weights.sum()
                    + np.outer(delta, delta))
        np.testing.assert_allclose(R, expected)
        np.testing.assert_array_equal(c, np.zeros(2))
        with pytest.raises(ValueError, match="legacy prediction-variance"):
            TL_QLDMR(solver="cvxopt", variance_mode=mode, asym_scale=amplitude).fit(
                Z_S, y, Z_T, np.array([0.1]))


@pytest.mark.parametrize("kwargs", [
    {"lambda1": -1}, {"lambda2": np.nan}, {"C_S": 0, "C_T": 0},
    {"tau": 0}, {"tau": 1}, {"kernel_gamma": 0},
    {"variance_mode": "unknown"}, {"asym_scale": 0.25},
])
def test_invalid_objective_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        Fast_TL_QLDMR(**kwargs)


def test_historical_batch_solver_cannot_silently_train_another_objective():
    with pytest.raises(ValueError, match="historical batch_sgd"):
        TL_QLDMR(solver="batch_sgd").fit(
            np.array([[0.0], [1.0]]), np.array([0.0, 1.0]),
            np.array([[0.5]]), np.array([0.5]))
