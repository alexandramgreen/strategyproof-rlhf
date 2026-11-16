"""Run an experiment: every sweep point x every seed x every algorithm.

    python -m sprlhf.run configs/strategic_fraction.yaml
    python -m sprlhf.run configs/strategic_fraction.yaml --seeds 5 --set labelers.k=21

In each trial, floor(strategic_fraction * k) labelers lie with one rule, and every
algorithm is scored on welfare, worst-off, honest and strategic utility.

Writes one long-format table to results/<name>.csv.gz (one row per
sweep point x seed x algorithm x metric, with every config value as a column) plus
the resolved config next to it. Tables from different runs share columns, so
plot.py can overlay them.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import itertools
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

from . import algorithms
from .algorithms import DATA_BASED, Problem
from .simulation import (DataConfig, EnvConfig, EstimationConfig, LabelerConfig, Streams, evaluate,
                         generate_data, make_env, n_strategic, references, sample_population,
                         strategic_order, strategic_report)


# ============================================================================
# Experiment configuration
# ============================================================================
#
# Experiment configuration: the experiment dataclass, YAML loading, and sweep expansion.
# The simulation settings it contains (env, labelers, data, estimation) are defined in
# simulation.py.
#
# A config file looks like:
#
#     name: strategic_fraction
#     seeds: 30
#     data: {n_per_labeler: 3000}
#     algorithms: [pooled_btl, maxmin, psw, pmomle]
#     sweep:
#       env: [{type: box}, {type: box, rotate: true}, {type: random_features}]
#       labelers.strategy: [exaggerate, bury]
#       labelers.strategic_fraction: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
#
# Every field has a default, so a config only lists what it changes. Unknown keys
# raise an error (catches typos). ``sweep`` maps dotted keys to value lists; the
# experiment runs the Cartesian product of all lists, times ``seeds`` seeds.

@dataclass
class ExperimentConfig:
    name: str = "experiment"
    seeds: int = 30
    seed_offset: int = 0
    n_jobs: int = -1               # parallel workers for the runner (-1 = all cores)
    env: EnvConfig = field(default_factory=EnvConfig)
    labelers: LabelerConfig = field(default_factory=LabelerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    estimation: EstimationConfig = field(default_factory=EstimationConfig)
    algorithms: list = field(default_factory=lambda: ["pooled_btl", "maxmin", "psw", "pmomle"])
    algo_params: dict = field(default_factory=dict)   # {algorithm_name: {param: value}}
    sweep: dict = field(default_factory=dict)         # {"dotted.key": [values]}


_SECTIONS = {"env": EnvConfig, "labelers": LabelerConfig, "data": DataConfig,
             "estimation": EstimationConfig}


def _build(cls, values: dict, where: str):
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise KeyError(f"unknown key(s) in {where}: {sorted(unknown)}; valid: {sorted(known)}")
    return cls(**values)


def from_dict(raw: dict) -> ExperimentConfig:
    raw = copy.deepcopy(raw or {})
    sections = {name: _build(cls, raw.pop(name, {}) or {}, name) for name, cls in _SECTIONS.items()}
    return _build(ExperimentConfig, {**raw, **sections}, "top level")


def load(path: str | Path) -> ExperimentConfig:
    with open(path) as f:
        return from_dict(yaml.safe_load(f))


def with_overrides(cfg: ExperimentConfig, overrides: dict[str, Any]) -> ExperimentConfig:
    """Copy of cfg with dotted-key overrides applied, e.g. {"labelers.k": 21}."""
    raw = asdict(cfg)
    for key, value in overrides.items():
        node = raw
        *parents, leaf = key.split(".")
        free_form = parents[:1] == ["algo_params"]   # algorithm params may be new keys
        for p in parents:
            if free_form:
                node = node.setdefault(p, {})
            elif p in node and isinstance(node[p], dict):
                node = node[p]
            else:
                raise KeyError(f"bad override key {key!r}")
        if leaf not in node and not free_form:
            raise KeyError(f"bad override key {key!r}")
        if isinstance(value, dict) and isinstance(node.get(leaf), dict):
            node[leaf] = {**node[leaf], **value}   # sweep over presets: env: [{type: box}, ...]
        else:
            node[leaf] = value
    return from_dict(raw)


def expand_sweep(cfg: ExperimentConfig) -> list[tuple[dict, ExperimentConfig]]:
    """All (overrides, config) points of the sweep; a single point if no sweep."""
    if not cfg.sweep:
        return [({}, cfg)]
    keys = list(cfg.sweep)
    points = []
    for values in itertools.product(*(cfg.sweep[k] for k in keys)):
        overrides = dict(zip(keys, values))
        points.append((overrides, with_overrides(cfg, overrides)))
    return points


_SKIP = {"name", "seeds", "seed_offset", "n_jobs", "algorithms", "sweep",
         "env.features", "env.context_weights", "labelers.thetas"}


def flat_params(cfg: ExperimentConfig) -> dict[str, Any]:
    """Scalar config values as {"dotted.key": value}, used as result-table columns."""
    out: dict[str, Any] = {}

    def walk(prefix: str, node):
        if isinstance(node, dict):
            for k, v in node.items():
                walk(f"{prefix}.{k}" if prefix else k, v)
        elif prefix not in _SKIP:
            out[prefix] = str(node) if isinstance(node, (list, tuple)) else node

    walk("", asdict(cfg))
    return out


# ============================================================================
# Running
# ============================================================================

def _base_row(cfg: ExperimentConfig, overrides: dict, seed: int, env) -> dict:
    return {
        "experiment": cfg.name,
        "seed": seed,
        "env_name": env.name,   # box | box_rotated | random_features | explicit
        "sweep_point": ", ".join(f"{k}={v}" for k, v in overrides.items()) or "base",
        **flat_params(cfg),
    }


def run_trial(cfg: ExperimentConfig, overrides: dict, seed: int) -> list[dict]:
    """One sweep point and one seed: strategic reports -> data -> every algorithm -> scores."""
    streams = Streams(seed)
    env = make_env(cfg.env, streams)
    pop = sample_population(cfg.labelers, env, streams)

    m = n_strategic(cfg.labelers.strategic_fraction, pop.k)
    strategic = np.zeros(pop.k, dtype=bool)
    strategic[strategic_order(pop.k, streams)[:m]] = True
    reported = pop.thetas.copy()
    for i in np.flatnonzero(strategic):
        reported[i] = strategic_report(cfg.labelers.strategy, cfg.labelers.strategy_strength, i, pop.thetas)

    needs_data = any(name in DATA_BASED for name in cfg.algorithms)
    data = generate_data(env, reported, cfg.data.n_per_labeler, streams) if needs_data else None
    problem = Problem(env=env, reported_thetas=reported, dataset=data, estimation=cfg.estimation)

    refs = references(env, pop.thetas)
    base = {**_base_row(cfg, overrides, seed, env),
            "n_strategic": m, "actual_strategic_fraction": m / pop.k}
    rows = []
    for name in cfg.algorithms:
        t0 = time.perf_counter()
        pi = algorithms.get(name)(problem, **cfg.algo_params.get(name, {}))
        metrics = evaluate(env, pop.thetas, pi, refs, strategic=strategic)
        metrics["runtime_s"] = time.perf_counter() - t0
        metrics["optimal_welfare"] = refs.w_star
        rows += [{**base, "algorithm": name, "metric": k, "value": float(v)} for k, v in metrics.items()]
    return rows


def run_experiment(cfg: ExperimentConfig, n_jobs: int | None = None, verbose: int = 0) -> pd.DataFrame:
    seeds = range(cfg.seed_offset, cfg.seed_offset + cfg.seeds)
    tasks = [(point_cfg, overrides, s) for overrides, point_cfg in expand_sweep(cfg) for s in seeds]
    n_jobs = cfg.n_jobs if n_jobs is None else n_jobs
    if n_jobs == 1:
        chunks = [run_trial(*t) for t in tasks]
    else:
        chunks = Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(run_trial)(*t) for t in tasks)
    return pd.DataFrame([row for chunk in chunks for row in chunk])


def save(df: pd.DataFrame, cfg: ExperimentConfig, out_dir: str | Path = "results") -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{cfg.name}.csv.gz"
    df.to_csv(path, index=False)
    with open(out_dir / f"{cfg.name}.config.yaml", "w") as f:
        yaml.safe_dump(asdict(cfg), f, sort_keys=False)
    return path


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", help="path to an experiment YAML")
    p.add_argument("--seeds", type=int, help="override the number of seeds")
    p.add_argument("--n-jobs", type=int, help="parallel workers (-1 = all cores, 1 = serial)")
    p.add_argument("--out", default="results", help="output directory (default: results/)")
    p.add_argument("--name", help="override the experiment name (and output file name)")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="dotted-key overrides, e.g. --set labelers.k=21 env.rotate=true")
    args = p.parse_args(argv)

    cfg = load(args.config)
    overrides = {}
    for item in args.set:
        key, _, value = item.partition("=")
        overrides[key] = yaml.safe_load(value)
    if args.seeds is not None:
        overrides["seeds"] = args.seeds
    if args.name:
        overrides["name"] = args.name
    if overrides:
        cfg = with_overrides(cfg, overrides)

    n_points = len(expand_sweep(cfg))
    print(f"[{cfg.name}] {n_points} sweep point(s) x {cfg.seeds} seeds x {len(cfg.algorithms)} algorithms")
    t0 = time.perf_counter()
    df = run_experiment(cfg, n_jobs=args.n_jobs)
    path = save(df, cfg, args.out)
    print(f"[{cfg.name}] {len(df):,} rows -> {path}  ({time.perf_counter() - t0:.1f}s)")


if __name__ == "__main__":
    main()
