"""Sanity checks. Run before trusting the results.

    python -m sprlhf.sanity            # ~1 minute; writes results/sanity/report.md + figures

Every check states what theory or construction predicts and what the code produced.
"""
from __future__ import annotations

import argparse
import time
import warnings
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.special import expit, logit  # noqa: E402

from . import algorithms  # noqa: E402
from .algorithms import Problem  # noqa: E402
from .plot import FIXED_COLORS, GRID, INK_2, MUTED, SURFACE, _color_map  # noqa: E402
from .run import from_dict, run_experiment, run_trial  # noqa: E402
from .simulation import (EnvConfig, EstimationConfig, LabelerConfig, Streams, bury,  # noqa: E402
                         confidence_radius, fit_bradley_terry, generate_data, make_env, make_explicit,
                         optimal_value, sample_population, strategic_order, utilities)

RESULTS: list[tuple[str, str, str, bool]] = []


def check(name: str, expected: str, observed: str, ok: bool) -> None:
    RESULTS.append((name, expected, observed, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {observed}")


def _population(env_type: str, seed: int, k: int = 11, rotate: bool = False):
    st = Streams(seed)
    env = make_env(EnvConfig(type=env_type, rotate=rotate), st)
    pop = sample_population(LabelerConfig(k=k), env, st)
    return st, env, pop


# ----------------------------------------------------------------------------- data

def check_data():
    st, env, pop = _population("box", 0)
    big = generate_data(env, pop.thetas, 1000, st)
    small = generate_data(env, pop.thetas, 100, st)
    check("Data: increasing n only adds examples",
          "first 100 examples identical for n = 100 and n = 1000",
          f"identical: {np.array_equal(small.X, big.X[:, :100]) and np.array_equal(small.y, big.y[:, :100])}",
          np.array_equal(small.X, big.X[:, :100]) and np.array_equal(small.y, big.y[:, :100]))

    lie = big.with_labels(3, 5 * pop.thetas[3])
    others_same = np.array_equal(np.delete(lie.y, 3, 0), np.delete(big.y, 3, 0))
    check("Data: a lie changes only the liar's labels",
          "labels of the other 10 labelers unchanged",
          f"unchanged: {others_same}; liar's labels changed: {int((lie.y[3] != big.y[3]).sum())} of 1000",
          others_same and (lie.y[3] != big.y[3]).any())

    p = expit(np.einsum("knd,kd->kn", big.X, pop.thetas))
    gap = abs(big.y.mean() - p.mean())
    check("Data: labels follow Bradley-Terry",
          "empirical preference rate = model probability (within sampling error)",
          f"|empirical - model| = {gap:.4f} over 11,000 labels", gap < 0.015)

    # label predictability at the defaults, for the README
    pred = np.mean(np.maximum(p, 1 - p))
    check("Data: label noise is realistic at the defaults (B = 3)",
          "labels agree with the stronger preference 60-75% of the time (human annotators: ~65-75%)",
          f"{pred:.0%}", 0.6 <= pred <= 0.78)


# ----------------------------------------------------------------------------- estimation

def check_estimation(seeds: int, fig_axes):
    ns = [100, 300, 1000, 3000, 10000]
    errors = {n: [] for n in ns}
    scales = [1.0, 2.0, 3.0, 4.0]
    coverage = {(n, c): [] for n in ns for c in scales}
    simultaneous = {n: [] for n in ns}
    for seed in range(seeds):
        st, env, pop = _population("box", seed)
        data = generate_data(env, pop.thetas, max(ns), st)
        for n in ns:
            for i in range(pop.k):
                X, y = data.X[i, :n], data.y[i, :n]
                mle = fit_bradley_terry(X, y, 1e-3)
                errors[n].append(np.linalg.norm(mle - pop.thetas[i]) / np.linalg.norm(pop.thetas[i]))
                S = X.T @ X / n + 1e-3 * np.eye(env.d)
                dist = np.sqrt((mle - pop.thetas[i]) @ S @ (mle - pop.thetas[i]))
                for c in scales:
                    coverage[(n, c)].append(dist <= confidence_radius(c, env.d, n, pop.k, 0.1))
            simultaneous[n].append(all(coverage[(n, 3.0)][-pop.k:]))
    med = np.array([np.median(errors[n]) for n in ns])
    slope = np.polyfit(np.log(ns), np.log(med), 1)[0]
    check("Estimation: MLE is consistent",
          "relative error shrinks like n^-1/2 (log-log slope ~ -0.5)",
          "median relative error " + ", ".join(f"{e:.2f}" for e in med) + f" at n = {ns}; slope {slope:.2f}",
          -0.65 < slope < -0.35)
    cov3 = [np.mean(coverage[(n, 3.0)]) for n in ns]
    sim3 = [np.mean(simultaneous[n]) for n in ns]
    check("Estimation: confidence sets are valid at c = 3 (the default)",
          "true parameter inside its confidence set >= 95% of the time, at every n",
          "per labeler " + ", ".join(f"{c:.2f}" for c in cov3)
          + "; all 11 at once " + ", ".join(f"{c:.2f}" for c in sim3),
          min(cov3) >= 0.95)
    cov1 = np.mean([np.mean(coverage[(n, 1.0)]) for n in ns])
    check("Estimation: the paper's radius without constants (c = 1) is too small",
          "documents why c is calibrated rather than set to 1",
          f"per-labeler coverage at c = 1: {cov1:.2f}", cov1 < 0.5)

    ax1, ax2 = fig_axes
    ax1.loglog(ns, med, marker="o", color=FIXED_COLORS["pmomle"], linewidth=2, label="median relative error")
    ax1.loglog(ns, med[0] * (np.array(ns) / ns[0]) ** -0.5, color=MUTED, linestyle="--", linewidth=1.5,
               label="n^-1/2 reference")
    ax1.set_xlabel("Comparisons per labeler n", color=INK_2)
    ax1.set_title("MLE error shrinks like n^-1/2", loc="left", fontsize=10, color=INK_2)
    ax1.set_ylabel("Relative error ||mle - theta|| / ||theta||", color=INK_2)
    for c, colr in zip(scales, ["#e34948", "#eda100", "#2a78d6", "#1baf7a"]):
        ax2.semilogx(ns, [np.mean(coverage[(n, c)]) for n in ns], marker="o", linewidth=2, color=colr,
                     label=f"c = {c:g}")
    ax2.axhline(0.95, color=MUTED, linestyle="--", linewidth=1)
    ax2.set_ylim(-0.03, 1.03)
    ax2.set_xlabel("Comparisons per labeler n", color=INK_2)
    ax2.set_title("Confidence-set coverage by scale c", loc="left", fontsize=10, color=INK_2)
    ax2.set_ylabel("Share of true parameters inside", color=INK_2)


# ----------------------------------------------------------------------------- paper examples

def _two_labeler_gain(rule: str, true_thetas, lie, n: int):
    """The paper's Prop. 3.3 instance, plus a dominated third action (0, 0) so that
    comparisons identify both coordinates (two actions alone identify only one)."""
    env = make_explicit(EnvConfig(type="explicit", d=2, features=[[0.5, 0.5], [0.75, 0.0], [0.0, 0.0]]))
    st = Streams(0)
    reported = true_thetas.copy()
    reported[0] = lie
    fit = algorithms.get(rule)
    honest_data = generate_data(env, true_thetas, n, st)
    lie_data = honest_data.with_labels(0, lie)
    est = EstimationConfig()
    honest = fit(Problem(env, true_thetas, honest_data, est))
    lying = fit(Problem(env, reported, lie_data, est))
    return float(utilities(env, true_thetas[0], lying) - utilities(env, true_thetas[0], honest))


def check_paper_examples():
    n = 50_000
    g = _two_labeler_gain("psw", np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([1.0, -1.0]), n)
    check("Paper Prop. 3.3: Pessimistic Social Welfare is manipulable (with data)",
          "lie (1, -1) raises labeler 1's utility by 0.25 (1/2 -> 3/4)", f"gain {g:.3f}", abs(g - 0.25) < 0.02)
    g = _two_labeler_gain("maxmin", np.array([[1.0, 0.0], [0.5, 0.5]]), np.array([1.0, -1.0]), n)
    check("Paper Prop. 3.3: MaxMin-RLHF is manipulable (with data)",
          "lie raises labeler 1's utility; with stochastic policies by 0.25 x 4/7 = 0.143",
          f"gain {g:.3f}", abs(g - 0.25 * 4 / 7) < 0.02)

    # Alice and Bob from the writeup's introduction
    env = make_explicit(EnvConfig(type="explicit", d=1, features=[[1.0], [0.0]]))   # actions C, D
    truth = np.array([[0.4], [-0.85]])
    data = generate_data(env, truth, 100_000, Streams(1))
    pooled = data.pooled_mle(1e-3)[0]
    lied = data.with_labels(0, np.array([0.95])).pooled_mle(1e-3)[0]
    pred_truth = logit((expit(0.4) + expit(-0.85)) / 2)
    pred_lie = logit((expit(0.95) + expit(-0.85)) / 2)
    ok = abs(pooled - pred_truth) < 0.02 and abs(lied - pred_lie) < 0.02 and pooled < 0 < lied
    check("Writeup example: Alice's exaggeration flips pooled BTL from D to C",
          f"pooled r(C) - r(D) = {pred_truth:.3f} (D wins), then {pred_lie:.3f} after the lie (C wins)",
          f"estimated {pooled:.3f}, then {lied:.3f}", ok)


# ----------------------------------------------------------------------------- algorithms

LIMITS = {"maxmin": "maxmin_rule", "mean_of_mles": "mean_rule", "psw": "mean_rule",
          "median_of_mles": "median_rule", "pmomle": "median_rule"}


def check_algorithms(seeds: int, ax):
    ns = [300, 1000, 3000, 10000]
    algos = sorted(set(LIMITS) | set(LIMITS.values()) | {"pooled_btl"})
    cfg = from_dict({"name": "sanity_limits", "seeds": seeds, "algorithms": algos,
                     "sweep": {"data.n_per_labeler": ns}})
    df = run_experiment(cfg)
    w = df[df.metric == "welfare"].pivot_table(index=["data.n_per_labeler", "seed"], columns="algorithm",
                                                values="value")
    ws = df[df.metric == "optimal_welfare"].groupby(["data.n_per_labeler", "seed"])["value"].first()
    rows = []
    colors = _color_map(list(LIMITS))
    for algo, limit in LIMITS.items():
        gaps = ((w[algo] - w[limit]).abs() / ws).groupby(level=0).mean()
        rows.append((algo, limit, gaps))
        ax.semilogx(ns, gaps.to_numpy(), marker="o", linewidth=2,
                    color=colors[algo], label=f"{algo} → {limit}")
    ax.set_xlabel("Comparisons per labeler n", color=INK_2)
    ax.set_title("Data-based algorithms approach their limits", loc="left", fontsize=10, color=INK_2)
    ax.set_ylabel("Welfare gap / optimal welfare", color=INK_2)
    for algo, limit, gaps in rows:
        g = gaps.to_numpy()
        check(f"Algorithms: {algo} converges to {limit} as n grows",
              "welfare gap (as a fraction of optimal welfare) shrinks with n",
              "gap " + ", ".join(f"{v:.3f}" for v in g) + f" at n = {ns}",
              g[-1] < g[0] or g[-1] < 0.01)

    # pessimism trades welfare for robustness; record its size at the defaults
    sub = df[(df.metric == "norm_suboptimality") & (df["data.n_per_labeler"] == 3000)]
    table = sub.groupby("algorithm")["value"].mean()
    check("Algorithms: truthful welfare at n = 3000 (normalized suboptimality, 0 = optimal)",
          "pooled BTL and mean of MLEs ~0; pessimistic methods pay for their caution",
          ", ".join(f"{a} {table[a]:.3f}" for a in ["pooled_btl", "psw", "maxmin", "pmomle", "median_of_mles"]),
          table["pooled_btl"] < 0.02 and table["mean_of_mles"] < 0.02)


def check_rotation(seeds: int):
    invariant = ["pooled_btl", "maxmin", "psw", "mean_of_mles"]
    coordinatewise = ["pmomle", "median_of_mles"]
    base = from_dict({"name": "rot", "algorithms": invariant + coordinatewise,
                      "data": {"n_per_labeler": 500}})
    max_diff = {a: 0.0 for a in invariant + coordinatewise}
    for seed in range(seeds):
        rows = {}
        for rotate in (False, True):
            cfg = replace(base, env=replace(base.env, rotate=rotate))
            for r in run_trial(cfg, {}, seed):
                if r["metric"] == "welfare":
                    rows[(rotate, r["algorithm"])] = r["value"]
        for a in max_diff:
            max_diff[a] = max(max_diff[a], abs(rows[(True, a)] - rows[(False, a)]))
    worst_invariant = max(max_diff[a] for a in invariant)
    check("Rotation: basis-independent algorithms are unaffected by rotating the box",
          "identical welfare with and without rotation (solver tolerance)",
          "max difference " + ", ".join(f"{a} {max_diff[a]:.1e}" for a in invariant), worst_invariant < 1e-4)
    check("Rotation: coordinate-wise median methods do change",
          "PMoMLE and median of MLEs depend on the basis",
          "max difference " + ", ".join(f"{a} {max_diff[a]:.3f}" for a in coordinatewise),
          max(max_diff[a] for a in coordinatewise) > 1e-3)


def _liar_gains(env, thetas, data, liar, lies, algos, est):
    """Liar's gain from each lie, as a share of their utility range (best minus worst policy)."""
    theta = thetas[liar]
    u_range = optimal_value(env, theta) + optimal_value(env, -theta)
    out = {}
    for a in algos:
        fit = algorithms.get(a)
        u_honest = utilities(env, theta, fit(Problem(env, thetas, data, est)))
        for name, lie in lies.items():
            reported = thetas.copy()
            reported[liar] = lie
            lie_data = data.with_labels(liar, lie) if data is not None else None
            u_lie = utilities(env, theta, fit(Problem(env, reported, lie_data, est)))
            out[(a, name)] = float(u_lie - u_honest) / u_range
    return out


def check_incentives(seeds: int):
    """One random labeler lies, everyone else is truthful (box env)."""
    est = EstimationConfig()
    median_max, mean_steer, with_data = 0.0, [], {}
    for seed in range(seeds):
        st, env, pop = _population("box", seed)
        liar = strategic_order(pop.k, st)[0]
        theta = pop.thetas[liar]
        others = np.delete(pop.thetas, liar, axis=0)
        lies = {f"scale_x{f:g}": f * theta for f in (0.5, 2, 5, 10)}
        lies["bury"] = bury(theta, others.mean(axis=0), 5.0)
        lies["steer_mean"] = pop.k * theta - others.sum(axis=0)   # makes the mean of reports = theta
        rng = st.rng("sanity_random_lies")
        random_lies = {f"random_{r}": 3.0 * rng.uniform(0.3, 3) * rng.standard_normal(env.d) for r in range(40)}

        g = _liar_gains(env, pop.thetas, None, liar, {**lies, **random_lies}, ["median_rule"], est)
        median_max = max(median_max, max(g.values()))
        mean_steer.append(_liar_gains(env, pop.thetas, None, liar, {"steer_mean": lies["steer_mean"]},
                                      ["mean_rule"], est)[("mean_rule", "steer_mean")])

        data = generate_data(env, pop.thetas, 3000, st)   # the experiment's n
        g = _liar_gains(env, pop.thetas, data, liar,
                        {"exaggerate": lies["scale_x5"], "steer_mean": lies["steer_mean"]},
                        ["pmomle", "psw", "pooled_btl"], est)
        for key, v in g.items():
            with_data.setdefault(key, []).append(v)

    mean_steer = np.array(mean_steer)
    check("Incentives (box env, perfect data): the median rule is exactly strategyproof",
          "no lie helps the liar (paper's Case 1); 7 structured + 40 random lies per seed",
          f"largest gain from any lie: {median_max:.1e}", median_max <= 1e-9)
    check("Incentives (box env, perfect data): the mean rule is not",
          "steering the mean gains a clearly positive share of the liar's utility range",
          f"gain {mean_steer.mean():.3f} ± {mean_steer.std(ddof=1) / np.sqrt(len(mean_steer)):.3f}",
          mean_steer.mean() > 2 * mean_steer.std(ddof=1) / np.sqrt(len(mean_steer)))
    m = {key: (np.mean(v), np.std(v, ddof=1) / np.sqrt(len(v))) for key, v in with_data.items()}
    pm = max(abs(m[("pmomle", "exaggerate")][0]), abs(m[("pmomle", "steer_mean")][0]))
    check("Incentives (box env, n = 3000, as in the experiment): lying doesn't pay under PMoMLE but does under PSW",
          "PMoMLE ~ 0 (< 2% of utility range) for exaggerating x5 and steering the mean; PSW > 5%",
          "; ".join(f"{lie}: " + ", ".join(f"{a} {m[(a, lie)][0]:.3f} ± {m[(a, lie)][1]:.3f}"
                                           for a in ["pmomle", "pooled_btl", "psw"])
                    for lie in ["exaggerate", "steer_mean"]),
          pm < 0.02 and m[("psw", "steer_mean")][0] > 0.05)


def check_reproducibility():
    cfg = from_dict({"name": "rep", "algorithms": ["pooled_btl", "pmomle"],
                     "labelers": {"strategic_fraction": 0.3, "strategy": "bury"}})
    a, b = run_trial(cfg, {}, 7), run_trial(cfg, {}, 7)
    same = all(x["value"] == y["value"] for x, y in zip(a, b) if x["metric"] != "runtime_s")
    t0 = from_dict({"name": "t", "algorithms": ["pooled_btl"], "labelers": {"strategy": "exaggerate"}})
    t1 = from_dict({"name": "t", "algorithms": ["pooled_btl"], "labelers": {"strategy": "bury"}})
    w0 = [r["value"] for r in run_trial(t0, {}, 3) if r["metric"] == "welfare"]
    w1 = [r["value"] for r in run_trial(t1, {}, 3) if r["metric"] == "welfare"]
    check("Reproducibility: same seed, same numbers",
          "rerunning a trial reproduces every value; with 0% strategic the strategy setting is irrelevant",
          f"rerun identical: {same}; 0%-strategic identical across strategies: {w0 == w1}", same and w0 == w1)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--out", default="results/sanity")
    args = p.parse_args(argv)
    warnings.filterwarnings("ignore")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), facecolor=SURFACE)
    check_data()
    check_estimation(args.seeds, axes[:2])
    check_paper_examples()
    check_algorithms(args.seeds, axes[2])
    check_rotation(min(args.seeds, 6))
    check_incentives(max(args.seeds, 20))
    check_reproducibility()

    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.grid(color=GRID, linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=9)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(out / "sanity.png", dpi=170, facecolor=SURFACE)

    n_pass = sum(r[3] for r in RESULTS)
    lines = ["# Sanity report\n", f"{n_pass} of {len(RESULTS)} checks passed "
             f"({time.perf_counter() - t0:.0f}s, {args.seeds} seeds).\n",
             "| | Check | Expected | Observed |", "|---|---|---|---|"]
    lines += [f"| {'PASS' if ok else '**FAIL**'} | {n} | {e} | {o} |" for n, e, o, ok in RESULTS]
    (out / "report.md").write_text("\n".join(lines) + "\n\n![sanity figures](sanity.png)\n")
    print(f"\n{n_pass}/{len(RESULTS)} checks passed -> {out / 'report.md'}")


if __name__ == "__main__":
    main()
