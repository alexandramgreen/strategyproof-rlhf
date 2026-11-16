"""The simulated world: environments, policies, labelers, strategies, data and metrics.

Sections, in dependency order:

  Random streams      named generators that keep every comparison paired
  Settings            dataclasses holding every simulation setting and its default
  Environments        prompts, responses and their feature vectors phi(s, a)
  Policies            stochastic policies, exact utilities, greedy and convex optimization
  Labelers            each labeler's true preference vector theta_i
  Strategic reporting who lies and what they report
  Preference data     Bradley-Terry comparisons, MLEs and confidence sets
  Metrics             scoring a policy against the labelers' true preferences
"""
from __future__ import annotations

import copy
import warnings
import zlib
from dataclasses import dataclass, field
from itertools import product
from typing import Callable

import cvxpy as cp
import numpy as np
from scipy.special import expit


# ============================================================================
# Random streams
# ============================================================================
#
# Named random streams for paired (common-random-number) comparisons.
#
# Every source of randomness gets its own generator keyed by (seed, name, index),
# never by position in the code or by config values. Consequences:
#
# * Within a seed, every algorithm sees identical data.
# * Sweeping one parameter (e.g. the strategic fraction) leaves everything else
#   identical: same env, same labelers, same comparison pairs.
# * Labeler i's randomness is the same whether k = 5 or k = 41, so changing k only
#   adds or removes labelers.

class Streams:
    def __init__(self, seed: int):
        self.seed = int(seed)

    def rng(self, name: str, *index: int) -> np.random.Generator:
        """Independent generator for stream ``name`` (and optional integer indices)."""
        key = [self.seed, zlib.crc32(name.encode())] + [int(i) for i in index]
        return np.random.default_rng(key)


# ============================================================================
# Settings
# ============================================================================
#
# Defaults for every part of the simulation. YAML configs override them (see run.py).

@dataclass
class EnvConfig:
    type: str = "box"              # box | random_features | explicit
    d: int = 6                     # feature dimension
    n_contexts: int = 20           # S
    n_actions: int = 8             # K (random_features only; box uses 2^d)
    feature_norm: float = 1.0      # L: ||phi(s, a)|| <= L
    scale_min: float = 0.5         # box: per-context feature scales drawn from [scale_min, 1]
    rotate: bool = False           # apply a random rotation to features and parameters
    features: list | None = None   # explicit: (K, d) or (S, K, d)
    context_weights: list | None = None  # explicit: rho, defaults to uniform


@dataclass
class LabelerConfig:
    k: int = 11                    # number of labelers (keep odd: even-k medians average two values)
    theta_norm: float = 3.0        # B: ||theta_i|| <= B (3.0: labels ~65% predictable, like real annotators)
    population: str = "symmetric"  # symmetric | explicit
    # symmetric: theta_i = mu + spread * B * u_i,  u_i uniform in the unit ball, ||mu|| = center_norm * B
    center_norm: float = 0.4
    spread: float = 0.6
    thetas: list | None = None                   # explicit: (k, d) true parameters
    # strategic reporting: floor(strategic_fraction * k) labelers, chosen at
    # random per seed (nested: the strategic set at m is contained in the set at m + 1)
    strategic_fraction: float = 0.0
    strategy: str = "exaggerate"                 # exaggerate | bury | truthful
    strategy_strength: float = 5.0               # exaggerate: scale factor; bury: opposition strength


@dataclass
class DataConfig:
    n_per_labeler: int = 500       # comparisons per labeler (context and action pair uniform)


@dataclass
class EstimationConfig:
    """Shared by every data-based algorithm (MLEs, confidence sets)."""
    ridge: float = 1e-3            # lambda in (1/n) sum loss + (lambda/2)||theta||^2, and Sigma + lambda I
    conf_scale: float = 3.0        # c in radius = c * sqrt((d + log(k/delta)) / n); 3.0 is calibrated in sanity.py
    delta: float = 0.1



# ============================================================================
# Environments
# ============================================================================
#
# Contextual bandit environments.
#
# An environment is a finite set of contexts (prompts) s with weights rho(s),
# K actions (responses) per context, and a known feature map phi(s, a) in R^d.
# A labeler with parameter theta gets reward  r(s, a) = <theta, phi(s, a)>.
#
# Every quantity in the paper depends on a policy only through its
# *feature occupancy*  z_pi = E_{s ~ rho} E_{a ~ pi(.|s)} phi(s, a)  (see Policies).
#
# Environments
# ------------
# box              Actions are all 2^d sign patterns, scaled per context.
#                  The set of achievable z_pi is an axis-aligned box, so the
#                  paper's Assumption 2 holds exactly.
# random_features  K random unit-norm actions per context. Assumption 2 fails.
# explicit         Features written by hand (e.g. the paper's counterexamples).
#
# Any environment can be rotated (``rotate: true``): features become R @ phi for a
# random orthogonal R. Labeler parameters are rotated the same way (see
# Labelers), so every reward is unchanged; only the coordinate system moves.
# For the box env this keeps the box shape but makes it no longer axis-aligned.

@dataclass
class ContextualBandit:
    """A finite contextual bandit.

    Attributes
    ----------
    phi : (S, K, d) array   features phi(s, a)
    rho : (S,) array        context distribution (sums to 1)
    R   : (d, d) array      rotation applied to the latent basis (identity if none).
                            Labeler parameters drawn in latent coordinates must be
                            mapped with ``to_feature_basis`` to stay reward-consistent.
    name : str
    """

    phi: np.ndarray
    rho: np.ndarray
    R: np.ndarray = field(default=None)
    name: str = "bandit"

    def __post_init__(self):
        self.phi = np.asarray(self.phi, dtype=float)
        self.rho = np.asarray(self.rho, dtype=float)
        if self.phi.ndim != 3:
            raise ValueError(f"phi must be (S, K, d); got shape {self.phi.shape}")
        if self.rho.shape != (self.phi.shape[0],):
            raise ValueError("rho must have one weight per context")
        if not np.isclose(self.rho.sum(), 1.0):
            raise ValueError("rho must sum to 1")
        if self.R is None:
            self.R = np.eye(self.d)

    @property
    def n_contexts(self) -> int:
        return self.phi.shape[0]

    @property
    def n_actions(self) -> int:
        return self.phi.shape[1]

    @property
    def d(self) -> int:
        return self.phi.shape[2]

    def to_feature_basis(self, theta_latent: np.ndarray) -> np.ndarray:
        """Map parameters drawn in latent coordinates into this env's feature basis.

        Works on a single vector (d,) or a stack (k, d). With R orthogonal,
        <R theta, R phi> = <theta, phi>, so rewards are unchanged.
        """
        return np.asarray(theta_latent) @ self.R.T

    def rewards(self, theta: np.ndarray) -> np.ndarray:
        """Reward table r(s, a) for one theta (d,) -> (S, K), or a stack (k, d) -> (k, S, K)."""
        return np.einsum("skd,...d->...sk", self.phi, np.asarray(theta, dtype=float))


# ----------------------------------------------------------------------------
# Factories
# ----------------------------------------------------------------------------

def _random_rotation(d: int, rng: np.random.Generator) -> np.ndarray:
    """Haar-random orthogonal matrix (QR of a Gaussian, with sign correction)."""
    Q, Rm = np.linalg.qr(rng.standard_normal((d, d)))
    return Q * np.sign(np.diag(Rm))


def make_box(cfg: EnvConfig, rng: np.random.Generator) -> ContextualBandit:
    """Actions = all sign patterns a in {-1, +1}^d.

    phi(s, a) = (L / sqrt(d)) * c_s * a, with per-context positive scales
    c_s in [scale_min, 1]^d, so ||phi|| <= L. For each context the achievable mean
    feature is the box prod_j [-c_sj, c_sj] (scaled), and a weighted sum of boxes
    is a box, so Assumption 2 holds. Because c_s > 0, the optimal sign pattern is
    the same in every context: contexts change magnitudes, not optimal actions.
    """
    d, S = cfg.d, cfg.n_contexts
    if d > 10:
        raise ValueError("box env enumerates 2^d actions; keep d <= 10")
    signs = np.array(list(product([-1.0, 1.0], repeat=d)))          # (2^d, d)
    scales = rng.uniform(cfg.scale_min, 1.0, size=(S, d))            # (S, d)
    phi = (cfg.feature_norm / np.sqrt(d)) * scales[:, None, :] * signs[None, :, :]
    rho = np.full(S, 1.0 / S)
    return ContextualBandit(phi=phi, rho=rho, name="box")


def make_random_features(cfg: EnvConfig, rng: np.random.Generator) -> ContextualBandit:
    """K actions per context with features uniform on the sphere of radius L."""
    S, K, d = cfg.n_contexts, cfg.n_actions, cfg.d
    g = rng.standard_normal((S, K, d))
    phi = cfg.feature_norm * g / np.linalg.norm(g, axis=-1, keepdims=True)
    rho = np.full(S, 1.0 / S)
    return ContextualBandit(phi=phi, rho=rho, name="random_features")


def make_explicit(cfg: EnvConfig, rng: np.random.Generator | None = None) -> ContextualBandit:
    """Features from the config: (K, d) for a single context, or (S, K, d)."""
    if cfg.features is None:
        raise ValueError("explicit env needs env.features")
    phi = np.asarray(cfg.features, dtype=float)
    if phi.ndim == 2:
        phi = phi[None]
    S = phi.shape[0]
    rho = np.full(S, 1.0 / S) if cfg.context_weights is None else np.asarray(cfg.context_weights, float)
    return ContextualBandit(phi=phi, rho=rho / rho.sum(), name="explicit")


ENV_FACTORIES = {
    "box": make_box,
    "random_features": make_random_features,
    "explicit": make_explicit,
}


def make_env(cfg: EnvConfig, streams: Streams) -> ContextualBandit:
    """Build the environment for one trial. Depends only on (seed, env config)."""
    if cfg.type not in ENV_FACTORIES:
        raise ValueError(f"unknown env.type {cfg.type!r}; options: {sorted(ENV_FACTORIES)}")
    env = ENV_FACTORIES[cfg.type](cfg, streams.rng("env"))
    if cfg.rotate:
        R = _random_rotation(env.d, streams.rng("env_rotation"))
        env = ContextualBandit(phi=env.phi @ R.T, rho=env.rho, R=R, name=env.name + "_rotated")
    return env


# ============================================================================
# Policies
# ============================================================================
#
# Stochastic policies, evaluation, and policy optimization.
#
# A policy is an (S, K) row-stochastic array: pi[s, a] = P(action a | context s).
# With linear rewards, everything about a policy that matters is its feature
# occupancy  z_pi = sum_s rho(s) sum_a pi[s, a] phi(s, a),  and
# labeler i's utility is  J_i(pi) = <theta_i, z_pi>  (computed exactly, no sampling).
#
# Optimizing a *linear* objective <theta, z> is solved exactly by the greedy policy
# (best action per context). Every RLHF algorithm in this project instead maximizes a
# *concave* function of z (a min over labelers, or a pessimistic min over confidence
# sets). ``maximize_concave`` solves those exactly as a convex program in cvxpy.

def occupancy(env: ContextualBandit, pi: np.ndarray) -> np.ndarray:
    """Feature occupancy z_pi in R^d."""
    return np.einsum("s,sk,skd->d", env.rho, pi, env.phi)


def utilities(env: ContextualBandit, thetas: np.ndarray, pi: np.ndarray) -> np.ndarray:
    """J_i(pi) = <theta_i, z_pi> for a stack of parameters (k, d) -> (k,), or one (d,) -> scalar."""
    return np.asarray(thetas, dtype=float) @ occupancy(env, pi)


def uniform_policy(env: ContextualBandit) -> np.ndarray:
    return np.full((env.n_contexts, env.n_actions), 1.0 / env.n_actions)


def greedy_policy(env: ContextualBandit, theta: np.ndarray) -> np.ndarray:
    """Deterministic policy maximizing <theta, z>: best action per context.

    Ties go to the lowest action index (fixed, label-independent tie-breaking).
    """
    best = env.rewards(theta).argmax(axis=1)
    pi = np.zeros((env.n_contexts, env.n_actions))
    pi[np.arange(env.n_contexts), best] = 1.0
    return pi


def optimal_value(env: ContextualBandit, theta: np.ndarray) -> float:
    """max_pi <theta, z_pi> = sum_s rho(s) max_a r_theta(s, a)."""
    return float(env.rho @ env.rewards(theta).max(axis=1))


def occupancy_expr(env: ContextualBandit, P: cp.Variable) -> cp.Expression:
    """cvxpy expression for z_P, linear in the policy variable P (S, K)."""
    S, K, d = env.phi.shape
    weighted = (env.rho[:, None, None] * env.phi).reshape(S * K, d)
    return weighted.T @ cp.reshape(P, (S * K,), order="C")


def maximize_concave(
    env: ContextualBandit,
    objective: Callable[[cp.Expression], cp.Expression],
    solver: str | None = None,
) -> np.ndarray:
    """Return the stochastic policy maximizing a concave objective of z_pi.

    ``objective`` receives the cvxpy expression for z (shape (d,)) and must return a
    concave scalar expression, e.g. ``lambda z: cp.min(thetas @ z)`` for max-min.
    """
    P = cp.Variable((env.n_contexts, env.n_actions), nonneg=True)
    z = occupancy_expr(env, P)
    problem = cp.Problem(cp.Maximize(objective(z)), [cp.sum(P, axis=1) == 1])
    # Try the requested solver, then fall back if it fails or reports an inaccurate solution.
    candidates = [solver] if solver else []
    candidates += [s for s in ("CLARABEL", "HIGHS", "SCS") if s in cp.installed_solvers() and s != solver]
    status = None
    for name in candidates:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)   # "may be inaccurate": handled below
                problem.solve(solver=name)
        except (cp.error.SolverError, ValueError):   # e.g. an LP-only solver on a conic problem
            continue
        status = problem.status
        if status == "optimal":
            break
    if P.value is None or status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"policy optimization failed: status={status}")
    pi = np.clip(P.value, 0.0, None)
    return pi / pi.sum(axis=1, keepdims=True)


# ============================================================================
# Labelers
# ============================================================================
#
# Labeler populations: each labeler's true reward parameter theta_i*.
#
# Strategic reporting (theta_tilde) is in the next section; this one draws only the truth.
# Parameters are drawn in latent coordinates, then mapped into the env's feature basis
# so that rotation never changes a reward.
#
# symmetric  theta_i = mu + spread*B*u_i: a shared random center mu (length center_norm*B)
#            plus an individual offset u_i uniform in the unit ball. Symmetric around mu,
#            so the coordinate-wise median approaches the mean as k grows: the regime
#            the paper's welfare bound assumes.
# explicit   theta_i listed in the config.

@dataclass
class Population:
    thetas: np.ndarray        # (k, d) true parameters, in the env's feature basis

    @property
    def k(self) -> int:
        return self.thetas.shape[0]


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _uniform_ball(rng: np.random.Generator, d: int) -> np.ndarray:
    """One point uniform in the d-dimensional unit ball."""
    return _unit(rng.standard_normal(d)) * rng.uniform() ** (1.0 / d)


def sample_population(cfg: LabelerConfig, env: ContextualBandit, streams: Streams) -> Population:
    k, d, B = cfg.k, env.d, cfg.theta_norm
    if k % 2 == 0:
        warnings.warn(f"labelers.k = {k} is even: the coordinate-wise median averages the two "
                      "middle values, which is itself manipulable", stacklevel=2)

    if cfg.population == "explicit":
        latent = np.asarray(cfg.thetas, dtype=float)
        if latent.shape != (k, d):
            raise ValueError(f"labelers.thetas must be ({k}, {d}); got {latent.shape}")

    elif cfg.population == "symmetric":
        if cfg.center_norm + cfg.spread > 1 + 1e-9:
            raise ValueError("need center_norm + spread <= 1 so that ||theta|| <= B")
        mu = cfg.center_norm * B * _unit(streams.rng("population_center").standard_normal(d))
        latent = np.stack([mu + cfg.spread * B * _uniform_ball(streams.rng("labeler_theta", i), d)
                           for i in range(k)])
    else:
        raise ValueError(f"unknown labelers.population {cfg.population!r}")

    thetas = env.to_feature_basis(latent)
    if np.linalg.norm(thetas, axis=1).max() > B + 1e-9:
        raise ValueError("a labeler parameter exceeds theta_norm")
    return Population(thetas=thetas)


# ============================================================================
# Strategic reporting
# ============================================================================
#
# Strategic reporting: who lies, and how.
#
# A strategic labeler never touches the examples it is shown; it labels them with a
# *reported* parameter theta_tilde instead of its true theta* (the paper's model).
# Lies are computed from true parameters: strategic labelers are assumed to know the
# population's true preferences (a well-informed manipulator, as in the DSIC
# definition), but they act independently and do not coordinate.
#
# Every strategic labeler uses the same rule:
#
# exaggerate  theta_tilde = strength * theta*.  Keeps every sign, so under PMoMLE this
#             is the paper's predicted (weakly) dominant strategy, and under pooled BTL
#             it is exactly Alice's move in the intro example.
# bury        Keep your true weights on the features you care about most; on the half you
#             care about least, report strong opposition to what everyone else wants.
#             This is the paper's counterexample against Pessimistic Social Welfare.
#             It only pays when features are coupled: in the box env it is pure spite.

def strategic_order(k: int, streams: Streams) -> np.ndarray:
    """Random order of labelers per seed. The first m are strategic (nested in m)."""
    return streams.rng("strategic_order").permutation(k)


def n_strategic(fraction: float, k: int) -> int:
    return int(np.floor(fraction * k + 1e-9))


def bury(theta: np.ndarray, others_mean: np.ndarray, strength: float) -> np.ndarray:
    d = theta.shape[0]
    least = np.argsort(np.abs(theta), kind="stable")[: d // 2 if d > 1 else 1]
    out = theta.copy()
    sign = np.sign(others_mean[least])
    sign[sign == 0] = 1.0
    out[least] = -strength * np.abs(theta).max() * sign
    return out


def strategic_report(rule: str, strength: float, i: int, true_thetas: np.ndarray) -> np.ndarray:
    theta = true_thetas[i]
    if rule == "truthful":
        return theta.copy()
    if rule == "exaggerate":
        return strength * theta
    if rule == "bury":
        others = np.delete(true_thetas, i, axis=0)
        return bury(theta, others.mean(axis=0), strength)
    raise ValueError(f"unknown labelers.strategy {rule!r}; options: truthful, exaggerate, bury")


# ============================================================================
# Preference data and estimation
# ============================================================================
#
# Offline preference data and per-labeler estimation.
#
# Data (the paper's offline setting)
# ----------------------------------
# Each labeler i gets n examples (s, a0, a1): context s ~ rho, two distinct actions
# uniformly at random. The examples are fixed in advance; a labeler controls only
# the labels. A labeler reporting parameter theta labels with Bradley-Terry:
#
#     P(a0 preferred over a1) = sigmoid(<theta, phi(s, a0) - phi(s, a1)>)
#
# Label convention: y = 1 means a0 was preferred.
#
# Common random numbers: each label is  y = 1[u < sigmoid(<theta, x>)]  with a fixed
# uniform u per example. Changing the reported theta changes labels only where it has
# to, and leaves every other labeler's data untouched. Every stream is prefix-stable,
# so the data for n = 100 are the first 100 examples of the data for n = 1000.
#
# Estimation (shared by all data-based algorithms)
# ------------------------------------------------
# MLE       argmin (1/n) sum_j logloss_j(theta) + (ridge/2) ||theta||^2   (Newton's method)
# Sigma_i   (1/n) sum_j x_j x_j^T + ridge * I          (label-independent)
# C_i       {theta : ||theta - mle_i||_{Sigma_i} <= r},  r = c * sqrt((d + log(k/delta)) / n)

# ----------------------------------------------------------------------------
# Logistic MLE
# ----------------------------------------------------------------------------

def _objective(theta, X, y, ridge):
    z = X @ theta
    return np.mean(np.logaddexp(0.0, z) - y * z) + 0.5 * ridge * theta @ theta


def fit_bradley_terry(X: np.ndarray, y: np.ndarray, ridge: float, max_iter: int = 100) -> np.ndarray:
    """Ridge-regularized Bradley-Terry MLE via damped Newton's method (strictly convex)."""
    n, d = X.shape
    theta = np.zeros(d)
    f = _objective(theta, X, y, ridge)
    for _ in range(max_iter):
        p = expit(X @ theta)
        grad = X.T @ (p - y) / n + ridge * theta
        H = (X.T * (p * (1 - p))) @ X / n + ridge * np.eye(d)
        step = np.linalg.solve(H, grad)
        t = 1.0
        while True:  # backtracking line search
            new = theta - t * step
            f_new = _objective(new, X, y, ridge)
            if f_new <= f - 1e-4 * t * grad @ step or t < 1e-10:
                break
            t *= 0.5
        theta, f_old, f = new, f, f_new
        if abs(f_old - f) < 1e-12 and np.linalg.norm(t * step) < 1e-9 * (1 + np.linalg.norm(theta)):
            break
    return theta


def confidence_radius(conf_scale: float, d: int, n: int, k: int, delta: float) -> float:
    return conf_scale * np.sqrt((d + np.log(k / delta)) / n)


# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------

@dataclass
class PreferenceData:
    ctx: np.ndarray        # (k, n) context index
    a0: np.ndarray         # (k, n) first action
    a1: np.ndarray         # (k, n) second action
    X: np.ndarray          # (k, n, d) phi(s, a0) - phi(s, a1)
    u: np.ndarray          # (k, n) fixed uniforms used to draw labels
    y: np.ndarray          # (k, n) 1 if a0 preferred
    _mle: dict = field(default_factory=dict, repr=False)   # cache: (i, ridge) -> estimate

    @property
    def k(self) -> int:
        return self.X.shape[0]

    @property
    def n(self) -> int:
        return self.X.shape[1]

    @property
    def d(self) -> int:
        return self.X.shape[2]

    def with_labels(self, i: int, theta: np.ndarray) -> "PreferenceData":
        """Copy in which labeler i labels the same examples according to ``theta``.

        Everyone else's labels (and cached estimates) are shared, not recomputed.
        """
        new = copy.copy(self)
        new.y = self.y.copy()
        new.y[i] = draw_labels(self.X[i], self.u[i], theta)
        new._mle = {key: v for key, v in self._mle.items() if key[0] != i and key[0] != "pooled"}
        return new

    def mle(self, i: int, ridge: float) -> np.ndarray:
        key = (i, ridge)
        if key not in self._mle:
            self._mle[key] = fit_bradley_terry(self.X[i], self.y[i], ridge)
        return self._mle[key]

    def mles(self, ridge: float) -> np.ndarray:
        return np.stack([self.mle(i, ridge) for i in range(self.k)])

    def pooled_mle(self, ridge: float) -> np.ndarray:
        key = ("pooled", ridge)
        if key not in self._mle:
            self._mle[key] = fit_bradley_terry(self.X.reshape(-1, self.d), self.y.reshape(-1), ridge)
        return self._mle[key]

    def covariance(self, i: int, ridge: float) -> np.ndarray:
        Xi = self.X[i]
        return Xi.T @ Xi / self.n + ridge * np.eye(self.d)


def draw_labels(X: np.ndarray, u: np.ndarray, theta: np.ndarray) -> np.ndarray:
    return (u < expit(X @ np.asarray(theta, dtype=float))).astype(float)


def generate_data(env: ContextualBandit, reported_thetas: np.ndarray, n: int,
                  streams: Streams) -> PreferenceData:
    """Examples for every labeler, labeled with each labeler's *reported* parameter."""
    k = reported_thetas.shape[0]
    S, K = env.n_contexts, env.n_actions
    ctx = np.empty((k, n), dtype=int)
    a0 = np.empty((k, n), dtype=int)
    a1 = np.empty((k, n), dtype=int)
    u = np.empty((k, n))
    for i in range(k):
        ctx[i] = streams.rng("pair_context", i).choice(S, size=n, p=env.rho)
        a0[i] = streams.rng("pair_first", i).integers(0, K, size=n)
        a1[i] = (a0[i] + streams.rng("pair_second", i).integers(1, K, size=n)) % K   # distinct from a0
        u[i] = streams.rng("label_uniform", i).random(n)
    X = env.phi[ctx, a0] - env.phi[ctx, a1]
    y = np.stack([draw_labels(X[i], u[i], reported_thetas[i]) for i in range(k)])
    return PreferenceData(ctx=ctx, a0=a0, a1=a1, X=X, u=u, y=y)


# ============================================================================
# Metrics
# ============================================================================
#
# Evaluation of a learned policy against the labelers' TRUE parameters.
#
# Metrics are always computed with true theta_i*, never the reported ones.
#
# welfare             W(pi) = mean_i J_i(pi)
# suboptimality       W(pi*) - W(pi)
# norm_suboptimality  (W(pi*) - W(pi)) / (W(pi*) - W(uniform)):
#                     0 = optimal, 1 = no better than a uniformly random policy.
#                     Comparable across environments; use as the main y-axis.
# approx_ratio        W(pi) / W(pi*), only when W(pi*) is clearly positive (else NaN)
# worst_off_utility   min_i J_i(pi), the quantity MaxMin-RLHF optimizes
# honest_utility / strategic_utility        mean utility of each group
# strategic_advantage strategic_utility - honest_utility (who the policy ends up serving;
#                     note it can rise because honest labelers lose, not only because liars gain)

@dataclass
class References:
    w_star: float        # optimal social welfare
    w_uniform: float     # welfare of the uniformly random policy


def references(env: ContextualBandit, true_thetas: np.ndarray) -> References:
    # Welfare is linear in the mean parameter, so pi* is the greedy policy for it.
    w_star = optimal_value(env, true_thetas.mean(axis=0))
    w_uniform = float(utilities(env, true_thetas, uniform_policy(env)).mean())
    return References(w_star=w_star, w_uniform=w_uniform)


def evaluate(
    env: ContextualBandit,
    true_thetas: np.ndarray,
    pi: np.ndarray,
    refs: References,
    strategic: np.ndarray | None = None,
) -> dict[str, float]:
    u = utilities(env, true_thetas, pi)
    W = float(u.mean())
    gap = refs.w_star - refs.w_uniform
    scale = max(abs(refs.w_star), abs(refs.w_uniform), 1e-12)
    out = {
        "welfare": W,
        "suboptimality": refs.w_star - W,
        "norm_suboptimality": (refs.w_star - W) / gap if gap > 1e-6 * scale else np.nan,
        "approx_ratio": W / refs.w_star if refs.w_star > 1e-6 else np.nan,
        "worst_off_utility": float(u.min()),
    }
    if strategic is not None and strategic.any():
        out["strategic_utility"] = float(u[strategic].mean())
        if (~strategic).any():
            out["honest_utility"] = float(u[~strategic].mean())
            out["strategic_advantage"] = out["strategic_utility"] - out["honest_utility"]
    return out
