"""The RLHF algorithms.

Every algorithm is a function  fit(problem, **params) -> policy (S, K)  registered
under a name that configs refer to. Per-algorithm parameters come from the config's
``algo_params: {name: {...}}``.

Two families:

* Data-based RLHF algorithms: see only preference data (the four methods compared,
  plus two ablations used by the sanity checks).
* Direct-access rules: see the labelers' reported preference vectors exactly. They are
  the infinite-data limits of the RLHF algorithms, and isolate the effect of the
  aggregation rule from the effect of estimation noise.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import cvxpy as cp
import numpy as np

from .simulation import (ContextualBandit, EstimationConfig, confidence_radius, greedy_policy,
                         maximize_concave)


# ============================================================================
# Registry
# ============================================================================

@dataclass
class Problem:
    """Everything an algorithm is allowed to see. Never contains true parameters."""

    env: ContextualBandit
    reported_thetas: np.ndarray | None = None   # (k, d), direct-access setting
    dataset: Any = None                         # PreferenceData (data-based algorithms)
    estimation: EstimationConfig = field(default_factory=EstimationConfig)


ALGORITHMS: dict[str, Callable[..., np.ndarray]] = {}


def register(name: str):
    def deco(fn):
        if name in ALGORITHMS:
            raise ValueError(f"algorithm {name!r} registered twice")
        ALGORITHMS[name] = fn
        return fn
    return deco


def get(name: str) -> Callable[..., np.ndarray]:
    if name not in ALGORITHMS:
        raise KeyError(f"unknown algorithm {name!r}; registered: {sorted(ALGORITHMS)}")
    return ALGORITHMS[name]


# ============================================================================
# Data-based RLHF algorithms
# ============================================================================
#
# Data-based RLHF algorithms: they see only the preference data.
#
# All share the estimation settings in ``problem.estimation`` (ridge, confidence scale c,
# delta); any of them can be overridden per algorithm via ``algo_params``.
#
# pooled_btl      one Bradley-Terry MLE on everyone's comparisons pooled, then the greedy
#                 policy (the InstructGPT recipe, contextual-bandit version).
# maxmin          MaxMin-RLHF with known labeler identities and no KL term:
#                 maximize min_i <mle_i, z>.
# psw             Pessimistic Social Welfare (utilitarian version, as analyzed by Kleine
#                 Buening et al.): maximize (1/k) sum_i min_{theta_i in C_i} <theta_i, z>
#                 = <mean mle, z> - (r/k) sum_i ||z||_{Sigma_i^-1}.
# pmomle          Pessimistic Median of MLEs: maximize the worst case of <median, z> over
#                 medians of parameters drawn from the confidence sets. Exact worst case is
#                 intractable, so we use a conservative relaxation: each C_i is replaced by
#                 its bounding box, which turns the set of medians into a box
#                 [med_i(lo_i), med_i(hi_i)] per feature (the median is monotone in each
#                 argument). Objective: <mid, z> - sum_j half_j |z_j|. The relaxation is at
#                 least as pessimistic as the exact algorithm.
# median_of_mles  ablation: median of MLEs without pessimism (greedy).
# mean_of_mles    ablation: mean of MLEs without pessimism (greedy).

def _settings(problem: Problem, overrides: dict):
    est = problem.estimation
    ridge = overrides.get("ridge", est.ridge)
    c = overrides.get("conf_scale", est.conf_scale)
    delta = overrides.get("delta", est.delta)
    data = problem.dataset
    if data is None:
        raise ValueError("data-based algorithms need problem.dataset")
    r = confidence_radius(c, data.d, data.n, data.k, delta)
    return data, ridge, r


@register("pooled_btl")
def pooled_btl(problem: Problem, **kw) -> np.ndarray:
    data, ridge, _ = _settings(problem, kw)
    return greedy_policy(problem.env, data.pooled_mle(ridge))


@register("maxmin")
def maxmin(problem: Problem, solver: str | None = None, **kw) -> np.ndarray:
    data, ridge, _ = _settings(problem, kw)
    thetas = data.mles(ridge)
    return maximize_concave(problem.env, lambda z: cp.min(thetas @ z), solver=solver)


@register("psw")
def psw(problem: Problem, solver: str | None = None, **kw) -> np.ndarray:
    data, ridge, r = _settings(problem, kw)
    thetas = data.mles(ridge)
    # ||z||_{Sigma^-1} = ||L^T z|| with L L^T = Sigma^-1
    chols = [np.linalg.cholesky(np.linalg.inv(data.covariance(i, ridge))) for i in range(data.k)]
    mean = thetas.mean(axis=0)
    return maximize_concave(
        problem.env,
        lambda z: mean @ z - (r / data.k) * cp.sum(cp.hstack([cp.norm(L.T @ z) for L in chols])),
        solver=solver)


def median_interval(problem: Problem, **kw) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature [lower, upper] range of the median over the (boxed) confidence sets."""
    data, ridge, r = _settings(problem, kw)
    thetas = data.mles(ridge)
    half = np.stack([r * np.sqrt(np.diag(np.linalg.inv(data.covariance(i, ridge))))
                     for i in range(data.k)])
    return np.median(thetas - half, axis=0), np.median(thetas + half, axis=0)


@register("pmomle")
def pmomle(problem: Problem, solver: str | None = None, **kw) -> np.ndarray:
    lo, hi = median_interval(problem, **kw)
    mid, half = (lo + hi) / 2, (hi - lo) / 2
    return maximize_concave(problem.env, lambda z: mid @ z - cp.sum(cp.multiply(half, cp.abs(z))),
                            solver=solver)


@register("median_of_mles")
def median_of_mles(problem: Problem, **kw) -> np.ndarray:
    data, ridge, _ = _settings(problem, kw)
    return greedy_policy(problem.env, np.median(data.mles(ridge), axis=0))


@register("mean_of_mles")
def mean_of_mles(problem: Problem, **kw) -> np.ndarray:
    data, ridge, _ = _settings(problem, kw)
    return greedy_policy(problem.env, data.mles(ridge).mean(axis=0))


# ============================================================================
# Direct-access rules
# ============================================================================
#
# Direct-access aggregation rules: the infinite-data limits of the RLHF algorithms.
#
# Each rule sees the reported parameters theta_1..theta_k exactly (no estimation).
#
# mean_rule    greedy for the mean parameter: maximizes reported social welfare.
#              Limit of Pessimistic Social Welfare as n -> infinity.
# median_rule  greedy for the coordinate-wise median.
#              Limit of Pessimistic Median of MLEs (the paper's Case 1, where it is
#              exactly strategyproof under the box assumption).
# maxmin_rule  maximizes the minimum reported utility (stochastic policy, no KL term).
#              Limit of MaxMin-RLHF.

def _thetas(problem: Problem) -> np.ndarray:
    if problem.reported_thetas is None:
        raise ValueError("direct-access rules need problem.reported_thetas")
    return np.asarray(problem.reported_thetas, dtype=float)


@register("mean_rule")
def mean_rule(problem: Problem) -> np.ndarray:
    return greedy_policy(problem.env, _thetas(problem).mean(axis=0))


@register("median_rule")
def median_rule(problem: Problem) -> np.ndarray:
    return greedy_policy(problem.env, np.median(_thetas(problem), axis=0))


@register("maxmin_rule")
def maxmin_rule(problem: Problem, solver: str | None = None) -> np.ndarray:
    thetas = _thetas(problem)
    return maximize_concave(problem.env, lambda z: cp.min(thetas @ z), solver=solver)


DATA_BASED = {"pooled_btl", "maxmin", "psw", "pmomle", "median_of_mles", "mean_of_mles"}
