"""Unit checks: environments, policies, data, strategies and algorithms."""
from dataclasses import replace

import cvxpy as cp
import numpy as np
import pytest
from scipy.special import expit, logit

from sprlhf import algorithms
from sprlhf.algorithms import Problem
from sprlhf.run import expand_sweep, from_dict, run_trial, with_overrides
from sprlhf.simulation import (EnvConfig, EstimationConfig, LabelerConfig, Streams, bury,
                               fit_bradley_terry, generate_data, greedy_policy, make_env, make_explicit,
                               maximize_concave, n_strategic, occupancy, optimal_value, sample_population,
                               strategic_order, uniform_policy, utilities)


def _env(seed=0, **kw):
    return make_env(EnvConfig(**kw), Streams(seed))


# --- environments --------------------------------------------------------------------

@pytest.mark.parametrize("etype", ["box", "random_features"])
def test_features_bounded_and_policies_valid(etype):
    env = _env(type=etype, d=5, n_contexts=7, n_actions=6, feature_norm=1.3)
    assert np.linalg.norm(env.phi, axis=-1).max() <= 1.3 + 1e-9
    theta = np.random.default_rng(1).standard_normal(env.d)
    for pi in (uniform_policy(env), greedy_policy(env, theta)):
        assert np.allclose(pi.sum(axis=1), 1) and (pi >= 0).all()


def test_box_env_reachable_set_is_axis_aligned_box():
    """Assumption 2: every corner of prod_j [-m_j, m_j] is reachable at once."""
    env = _env(type="box", d=4, n_contexts=9)
    m = np.array([occupancy(env, greedy_policy(env, e))[j] for j, e in enumerate(np.eye(env.d))])
    rng = np.random.default_rng(0)
    for _ in range(20):
        s = rng.choice([-1.0, 1.0], size=env.d)
        z = occupancy(env, greedy_policy(env, s))
        assert np.allclose(z, s * m)  # each coordinate hits its own extreme independently


def test_rotation_preserves_every_reward():
    cfg = EnvConfig(type="box", d=4, n_contexts=6)
    plain = make_env(cfg, Streams(3))
    rotated = make_env(EnvConfig(type="box", d=4, n_contexts=6, rotate=True), Streams(3))
    latent = np.random.default_rng(5).standard_normal((7, 4))
    assert np.allclose(plain.rewards(plain.to_feature_basis(latent)),
                       rotated.rewards(rotated.to_feature_basis(latent)))
    assert not np.allclose(plain.phi, rotated.phi)  # but the coordinates did change


# --- optimization --------------------------------------------------------------------

def test_greedy_is_optimal_for_linear_objectives():
    env = _env(type="random_features", d=4, n_contexts=5, n_actions=6)
    theta = np.random.default_rng(2).standard_normal(env.d)
    pi = maximize_concave(env, lambda z: theta @ z)
    assert np.isclose(utilities(env, theta, pi), optimal_value(env, theta), atol=1e-6)
    assert np.isclose(utilities(env, theta, greedy_policy(env, theta)), optimal_value(env, theta))


def test_maxmin_beats_or_ties_every_greedy_policy_on_worst_case():
    env = _env(type="random_features", d=4, n_contexts=5, n_actions=6)
    thetas = np.random.default_rng(4).standard_normal((5, env.d))
    pi = maximize_concave(env, lambda z: cp.min(thetas @ z))
    best = utilities(env, thetas, pi).min()
    for th in thetas:
        assert best >= utilities(env, thetas, greedy_policy(env, th)).min() - 1e-6


# --- the paper's counterexamples (Prop. 3.3), direct-access versions --------------------

PAPER_ENV = EnvConfig(type="explicit", d=2, features=[[0.5, 0.5], [0.75, 0.0]])  # actions a, b


def _gain(rule, true_thetas, lie, liar=0):
    env = make_explicit(PAPER_ENV)
    fit = algorithms.get(rule)
    honest = fit(Problem(env, reported_thetas=true_thetas))
    reported = true_thetas.copy()
    reported[liar] = lie
    lying = fit(Problem(env, reported_thetas=reported))
    return utilities(env, true_thetas[liar], lying) - utilities(env, true_thetas[liar], honest)


def test_social_welfare_rule_is_manipulable():
    # theta_1 = (1, 0), theta_2 = (0, 1): truthful -> a (utility 1/2); lie (1, -1) -> b (utility 3/4)
    gain = _gain("mean_rule", np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([1.0, -1.0]))
    assert np.isclose(gain, 0.25)


def test_maxmin_rule_is_manipulable():
    # The paper's deterministic example picks b after the lie. With stochastic policies
    # MaxMin mixes a and b instead, but lying still pays: 1/2 -> 1/2 + 1/4 * 4/7.
    gain = _gain("maxmin_rule", np.array([[1.0, 0.0], [0.5, 0.5]]), np.array([1.0, -1.0]))
    assert np.isclose(gain, 0.25 * 4 / 7, atol=1e-5)


def test_median_rule_is_exactly_strategyproof_on_box_env():
    """Paper's Case 1: with direct access and Assumption 2, no misreport ever helps."""
    rng = np.random.default_rng(0)
    for seed in range(15):
        env = _env(seed=seed, type="box", d=4, n_contexts=5)
        thetas = sample_population(LabelerConfig(k=7), env, Streams(seed)).thetas
        honest = algorithms.get("median_rule")(Problem(env, reported_thetas=thetas))
        for _ in range(40):
            reported = thetas.copy()
            reported[0] = rng.standard_normal(env.d) * rng.choice([0.1, 1, 10])
            lying = algorithms.get("median_rule")(Problem(env, reported_thetas=reported))
            assert utilities(env, thetas[0], lying) <= utilities(env, thetas[0], honest) + 1e-12


# --- labelers, streams, config ---------------------------------------------------------

def test_labeler_draws_do_not_depend_on_k():
    env = _env(type="box", d=4)
    small = sample_population(LabelerConfig(k=5), env, Streams(9)).thetas
    large = sample_population(LabelerConfig(k=21), env, Streams(9)).thetas
    assert np.allclose(small, large[:5])


def test_config_sweep_and_overrides():
    cfg = from_dict({"name": "t", "sweep": {"labelers.k": [5, 11], "env.rotate": [False, True]}})
    points = expand_sweep(cfg)
    assert len(points) == 4 and {p[1].labelers.k for p in points} == {5, 11}
    assert with_overrides(cfg, {"algo_params.maxmin_rule.solver": "SCS"}).algo_params == {"maxmin_rule": {"solver": "SCS"}}
    with pytest.raises(KeyError):
        from_dict({"labelers": {"kk": 3}})


# --- preference data, strategies and RLHF algorithms ----------------------------

def _setup(seed=0, n=400, **env_kw):
    st = Streams(seed)
    env = make_env(EnvConfig(**env_kw), st)
    pop = sample_population(LabelerConfig(k=7), env, st)
    return st, env, pop, generate_data(env, pop.thetas, n, st)


def test_data_is_prefix_stable_and_lies_are_local():
    st, env, pop, data = _setup(n=300)
    small = generate_data(env, pop.thetas, 50, st)
    assert np.array_equal(small.X, data.X[:, :50]) and np.array_equal(small.y, data.y[:, :50])
    lied = data.with_labels(2, -pop.thetas[2])
    assert np.array_equal(np.delete(lied.y, 2, 0), np.delete(data.y, 2, 0))
    assert not np.array_equal(lied.y[2], data.y[2])


def test_mle_recovers_parameter_with_lots_of_data():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20000, 3))
    theta = np.array([1.0, -0.5, 2.0])
    y = (rng.random(20000) < expit(X @ theta)).astype(float)
    assert np.allclose(fit_bradley_terry(X, y, 1e-6), theta, atol=0.08)


def test_mle_is_finite_on_separable_data():
    X = np.array([[1.0], [-1.0]] * 50)
    y = np.array([1.0, 0.0] * 50)       # perfectly separable
    theta = fit_bradley_terry(X, y, 1e-3)
    assert np.isfinite(theta).all() and theta[0] > 3


def test_alice_bob_pooled_btl_matches_closed_form():
    env = make_explicit(EnvConfig(type="explicit", d=1, features=[[1.0], [0.0]]))
    data = generate_data(env, np.array([[0.4], [-0.85]]), 60_000, Streams(1))
    assert abs(data.pooled_mle(1e-4)[0] - logit((expit(0.4) + expit(-0.85)) / 2)) < 0.02
    lied = data.with_labels(0, np.array([0.95])).pooled_mle(1e-4)[0]
    assert lied > 0   # C now wins


def test_paper_counterexample_with_data_psw():
    env = make_explicit(EnvConfig(type="explicit", d=2, features=[[0.5, 0.5], [0.75, 0.0], [0.0, 0.0]]))
    truth = np.array([[1.0, 0.0], [0.0, 1.0]])
    lie = np.array([1.0, -1.0])
    data = generate_data(env, truth, 30_000, Streams(0))
    fit, est = algorithms.get("psw"), EstimationConfig()
    honest = fit(Problem(env, truth, data, est))
    lying = fit(Problem(env, truth, data.with_labels(0, lie), est))
    assert np.isclose(utilities(env, truth[0], lying) - utilities(env, truth[0], honest), 0.25, atol=0.02)


@pytest.mark.parametrize("algo", ["pooled_btl", "maxmin", "psw", "mean_of_mles"])
def test_basis_independent_algorithms_ignore_rotation(algo):
    cfg = from_dict({"name": "t", "algorithms": [algo], "data": {"n_per_labeler": 300}})
    rot = replace(cfg, env=replace(cfg.env, rotate=True))
    w = lambda c: [r["value"] for r in run_trial(c, {}, 4) if r["metric"] == "welfare"][0]
    assert abs(w(cfg) - w(rot)) < 1e-4


def test_pmomle_runs_and_is_more_cautious_with_less_data():
    _, env, pop, data = _setup(n=2000, type="box")
    p = Problem(env, pop.thetas, data, EstimationConfig())
    lo, hi = algorithms.median_interval(p)
    assert (hi >= lo).all()
    lo_small, hi_small = algorithms.median_interval(p, conf_scale=6.0)
    assert ((hi_small - lo_small) >= (hi - lo) - 1e-12).all()
    pi = algorithms.get("pmomle")(p)
    assert np.allclose(pi.sum(axis=1), 1)


def test_strategic_sets_are_nested_and_bury_matches_paper():
    order = strategic_order(11, Streams(3))
    sets = [set(order[: n_strategic(f, 11)]) for f in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)]
    assert [len(s) for s in sets] == [0, 1, 2, 3, 4, 5]
    assert all(a <= b for a, b in zip(sets, sets[1:]))
    # paper Prop. 3.3: theta_1 = (1, 0) facing theta_2 = (0, 1) reports (1, -1)
    assert np.allclose(bury(np.array([1.0, 0.0]), np.array([0.0, 1.0]), 1.0), [1.0, -1.0])
