import numpy as np
import pytest

from uec.markov import entropy_production_rate_bits, random_markov_biased, ring_chain_3, sample_markov
from uec.sequential import (
    eprocess_reversibility,
    loop_eprocess_reversibility,
    reversible_mle,
    reversible_projection_rate,
    transition_counts,
    unconstrained_loglik,
)


def _reversible_chain(k, rng):
    W = rng.gamma(1.0, 1.0, size=(k, k))
    W = W + W.T
    return W / W.sum(axis=1, keepdims=True)


def test_reversible_mle_is_reversible_and_bounded():
    rng = np.random.default_rng(0)
    T = random_markov_biased(4, delta=0.7, rng=rng)
    C = transition_counts(sample_markov(T, n=3000, rng=rng), 4)
    That, ll = reversible_mle(C)
    np.testing.assert_allclose(That.sum(axis=1), 1.0)
    assert entropy_production_rate_bits(That) < 1e-8
    assert ll <= unconstrained_loglik(C) + 1e-9


def test_reversible_mle_matches_generic_optimizer():
    optimize = pytest.importorskip("scipy.optimize")
    rng = np.random.default_rng(7)
    for _ in range(5):
        k = int(rng.integers(3, 6))
        C = transition_counts(sample_markov(random_markov_biased(k, 0.6, rng=rng), n=400, rng=rng), k)
        iu = np.triu_indices(k)

        def negll(z):
            X = np.zeros((k, k))
            X[iu] = np.exp(z)
            X = X + X.T - np.diag(np.diag(X))
            T = X / X.sum(axis=1, keepdims=True)
            return -np.sum(np.where(C > 0, C * np.log(np.where(C > 0, T, 1.0)), 0.0))

        best = max(-optimize.minimize(negll, rng.normal(size=len(iu[0])), method="L-BFGS-B").fun
                   for _ in range(4))
        assert reversible_mle(C)[1] >= best - 1e-3


def test_reversible_mle_symmetric_counts_match_unconstrained():
    # Symmetric counts are already reversible, so the constrained optimum is the plain MLE.
    C = np.array([[5.0, 3.0, 1.0], [3.0, 2.0, 4.0], [1.0, 4.0, 6.0]])
    _, ll = reversible_mle(C)
    assert ll == pytest.approx(unconstrained_loglik(C), abs=1e-8)


def test_reversible_mle_batch_matches_single():
    rng = np.random.default_rng(1)
    Cs = np.stack([transition_counts(sample_markov(random_markov_biased(3, 0.5, rng=rng), n=500, rng=rng), 3)
                   for _ in range(4)])
    _, ll_batch = reversible_mle(Cs)
    for C, llb in zip(Cs, ll_batch):
        assert reversible_mle(C)[1] == pytest.approx(llb, abs=1e-8)


def test_projection_rate_properties():
    rng = np.random.default_rng(2)
    assert reversible_projection_rate(_reversible_chain(4, rng)) == pytest.approx(0.0, abs=1e-12)
    for _ in range(20):
        T = random_markov_biased(int(rng.integers(3, 6)), delta=float(rng.uniform(0, 0.9)), rng=rng)
        rho, sigma = reversible_projection_rate(T), entropy_production_rate_bits(T)
        assert 0.0 <= rho <= sigma / 2 + 1e-12


def test_projection_rate_near_equilibrium_is_quarter_sigma():
    T = ring_chain_3(0.2, 0.19)
    assert reversible_projection_rate(T) / entropy_production_rate_bits(T) == pytest.approx(0.25, rel=0.01)


def test_projection_rate_finite_for_one_way_transitions():
    T = np.array([[0.7, 0.3, 0.0], [0.0, 0.7, 0.3], [0.3, 0.0, 0.7]])
    assert reversible_projection_rate(T) == pytest.approx(0.3)


def test_eprocess_detects_irreversible_chain():
    rng = np.random.default_rng(3)
    x = sample_markov(ring_chain_3(0.3, 0.05), n=3000, rng=rng)
    _, log2E = eprocess_reversibility(x, 3, checkpoints=[100, 1000, 2999])
    assert log2E[-1] > np.log2(20)


def test_eprocess_false_alarms_are_rare_on_reversible_chains():
    rng = np.random.default_rng(4)
    T = _reversible_chain(3, rng)
    hits = 0
    for _ in range(40):
        _, log2E = eprocess_reversibility(sample_markov(T, n=800, rng=rng), 3,
                                          checkpoints=range(10, 800, 10))
        hits += int(log2E.max() >= np.log2(20))
    assert hits <= 6  # alpha = 0.05; generous Monte Carlo slack


def test_loop_eprocess_palindromes_carry_no_evidence():
    _, log2E = loop_eprocess_reversibility([0, 1, 0, 1, 0, 1, 0], 2, burn_in=0, ref_state=0)
    np.testing.assert_allclose(log2E, 0.0)


def test_loop_eprocess_detects_irreversible_chain():
    rng = np.random.default_rng(5)
    x = sample_markov(ring_chain_3(0.3, 0.05), n=1000, rng=rng)
    _, log2E = loop_eprocess_reversibility(x, 3, checkpoints=[999])
    assert log2E[-1] > np.log2(20)


def test_loop_eprocess_false_alarms_are_rare_on_reversible_chains():
    rng = np.random.default_rng(6)
    T = _reversible_chain(4, rng)
    hits = 0
    for _ in range(60):
        _, log2E = loop_eprocess_reversibility(sample_markov(T, n=1000, rng=rng), 4)
        hits += int(log2E.max() >= np.log2(20))
    assert hits <= 8  # alpha = 0.05; generous Monte Carlo slack
