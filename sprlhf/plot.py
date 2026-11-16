"""Plot results: mean over seeds with 95% confidence bands.

    python -m sprlhf.plot results/strategic_fraction.csv.gz \
        --x actual_strategic_fraction --y norm_suboptimality --col env_name \
        --row labelers.strategy --out figures/norm_suboptimality.png

Each line is the mean over seeds with a 95% confidence band (t-interval; seeds are
paired across algorithms, so differences between lines are tighter than the bands
suggest). Any result-table column can be --x, --hue, --style, --col or --row, so
results files from different runs overlay as long as they share the x column.

From a notebook:
    from sprlhf.plot import load_results, plot_lines
    df = load_results(["results/strategic_fraction.csv.gz"])
    plot_lines(df, x="actual_strategic_fraction", y="norm_suboptimality",
               col="env_name", row="labelers.strategy")
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from scipy import stats  # noqa: E402

# Fixed categorical order (validated colorblind-safe palette). Colors follow the
# algorithm, never its position in a plot, so the same algorithm has the same color
# in every figure.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
FIXED_COLORS = {
    "pmomle": PALETTE[0], "pooled_btl": PALETTE[1], "psw": PALETTE[2], "maxmin": PALETTE[3],
    "median_rule": PALETTE[4], "mean_rule": PALETTE[5], "maxmin_rule": PALETTE[6],
    "median_of_mles": PALETTE[7],
}
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "h"]
LINESTYLES = ["-", "--", ":", "-."]
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"

LABELS = {
    "norm_suboptimality": "Normalized suboptimality (0 = optimal, 1 = uniform)",
    "suboptimality": "Suboptimality  W(π*) − W(π)",
    "welfare": "Social welfare",
    "worst_off_utility": "Worst-off labeler's utility",
    "approx_ratio": "Approximation ratio  W(π) / W(π*)",
    "labelers.k": "Number of labelers k",
    "labelers.strategic_fraction": "Strategic fraction",
    "data.n_per_labeler": "Comparisons per labeler n",
    "actual_strategic_fraction": "Share of labelers who lie",
    "strategic_advantage": "Strategic minus honest utility",
    "honest_utility": "Honest labelers' mean utility",
    "strategic_utility": "Strategic labelers' mean utility",
}
FACET_NAMES = {"labelers.population": "population", "env_name": "env", "env.type": "env",
               "labelers.strategy": "strategy",
               "env.rotate": "rotated", "labelers.k": "k", "data.n_per_labeler": "n"}


def load_results(paths: list[str | Path]) -> pd.DataFrame:
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


def apply_filters(df: pd.DataFrame, where: list[str]) -> pd.DataFrame:
    """Keep rows matching every KEY=VALUE (VALUE parsed as YAML: true, 3, 0.5, box)."""
    for cond in where or []:
        key, _, raw = cond.partition("=")
        if key not in df.columns:
            raise KeyError(f"--where column {key!r} not in results")
        value = yaml.safe_load(raw)
        mask = df[key] == value
        if not mask.any():
            mask = df[key].astype(str).str.lower() == str(raw).lower()
        df = df[mask]
    return df


def _color_map(names: list[str]) -> dict[str, str]:
    cmap, free = {}, [c for c in PALETTE if c not in {FIXED_COLORS.get(n) for n in names}]
    for n in names:
        cmap[n] = FIXED_COLORS.get(n) or (free.pop(0) if free else MUTED)
    return cmap


def summarize(df: pd.DataFrame, y: str, by: list[str]) -> pd.DataFrame:
    """Mean and 95% t-interval half-width over seeds for each group."""
    d = df[df["metric"] == y]
    if d.empty:
        raise ValueError(f"no rows for metric {y!r}; available: {sorted(df['metric'].unique())}")
    g = d.groupby(by, dropna=False)["value"]
    out = g.agg(mean="mean", std="std", n="count").reset_index()
    tcrit = stats.t.ppf(0.975, np.maximum(out["n"] - 1, 1))
    out["ci"] = np.where(out["n"] > 1, tcrit * out["std"] / np.sqrt(out["n"]), 0.0)
    return out


def plot_lines(df: pd.DataFrame, x: str, y: str, hue: str = "algorithm", style: str | None = None,
               col: str | None = None, row: str | None = None, logx: bool = False,
               title: str | None = None, out: str | Path | None = None):
    groups = [c for c in (x, hue, style, col, row) if c]
    s = summarize(df, y, groups)
    numeric_x = pd.api.types.is_numeric_dtype(s[x])

    col_vals = sorted(s[col].unique(), key=str) if col else [None]
    row_vals = sorted(s[row].unique(), key=str) if row else [None]
    hue_vals = sorted(s[hue].unique(), key=str)
    style_vals = sorted(s[style].unique(), key=str) if style else [None]
    colors = _color_map(hue_vals) if hue == "algorithm" else dict(zip(hue_vals, PALETTE + [MUTED] * 99))
    if len(hue_vals) > len(PALETTE) + 1:
        print(f"warning: {len(hue_vals)} hue groups; consider --where or facets to cut series")

    fig, axes = plt.subplots(len(row_vals), len(col_vals), squeeze=False, sharey=True,
                             figsize=(4.6 * len(col_vals) + 1.6, 3.4 * len(row_vals) + 0.6),
                             facecolor=SURFACE)
    x_levels = sorted(s[x].unique(), key=lambda v: (str(type(v)), v)) if not numeric_x else None
    for ri, rv in enumerate(row_vals):
        for ci, cv in enumerate(col_vals):
            ax = axes[ri][ci]
            ax.set_facecolor(SURFACE)
            panel = s
            if col:
                panel = panel[panel[col] == cv]
            if row:
                panel = panel[panel[row] == rv]
            n_hue = len(hue_vals)
            for hi, hv in enumerate(hue_vals):
                for si, sv in enumerate(style_vals):
                    line = panel[panel[hue] == hv]
                    if style:
                        line = line[line[style] == sv]
                    if line.empty:
                        continue
                    line = line.sort_values(x)
                    label = str(hv) if not style or len(style_vals) == 1 else f"{hv} · {sv}"
                    kw = dict(color=colors[hv], marker=MARKERS[hi % len(MARKERS)], markersize=6,
                              linewidth=2, linestyle=LINESTYLES[si % len(LINESTYLES)], label=label,
                              markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3)
                    if numeric_x:
                        xs = line[x].to_numpy(float)
                        ax.plot(xs, line["mean"], **kw)
                        ax.fill_between(xs, line["mean"] - line["ci"], line["mean"] + line["ci"],
                                        color=colors[hv], alpha=0.14, linewidth=0, zorder=2)
                    else:  # categorical x: dodged points with error bars, no connecting lines
                        offset = (hi - (n_hue - 1) / 2) * 0.8 / max(n_hue, 1)
                        xs = np.array([x_levels.index(v) for v in line[x]]) + offset
                        kw.pop("linestyle")
                        ax.errorbar(xs, line["mean"], yerr=line["ci"], linestyle="none",
                                    elinewidth=1.5, capsize=0, **kw)
            if not numeric_x:
                ax.set_xticks(range(len(x_levels)), [str(v) for v in x_levels])
            if logx and numeric_x:
                ax.set_xscale("log")
                xticks = sorted(s[x].unique())
                ax.set_xticks(xticks, [f"{v:g}" for v in xticks])
                ax.minorticks_off()
            ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
            if y in ("norm_suboptimality", "suboptimality", "strategic_advantage"):
                ax.axhline(0, color="#c3c2b7", linewidth=1, zorder=1)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color("#c3c2b7")
            ax.tick_params(colors=MUTED, labelcolor=INK_2, labelsize=9)
            facet = " · ".join(f"{FACET_NAMES.get(k, k)} = {v}" for k, v in ((row, rv), (col, cv)) if k)
            if facet:
                ax.set_title(facet, fontsize=10, color=INK_2, loc="left")
            if ri == len(row_vals) - 1:
                ax.set_xlabel(LABELS.get(x, x), fontsize=10, color=INK_2)

    handles, labels = axes[0][0].get_legend_handles_labels()
    seen = dict(zip(labels, handles))
    for a in axes.flat:  # collect labels from every panel
        for h, l in zip(*a.get_legend_handles_labels()):
            seen.setdefault(l, h)
    fig.legend(seen.values(), seen.keys(), loc="upper left", bbox_to_anchor=(1.0, 0.95),
               frameon=False, fontsize=9, labelcolor=INK_2)
    fig.supylabel(LABELS.get(y, y), fontsize=10, color=INK_2)
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout()
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=SURFACE)
        print(f"saved {out}")
    return fig


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results", nargs="+", help="one or more results/*.csv.gz files")
    p.add_argument("--x", required=True)
    p.add_argument("--y", default="norm_suboptimality", help="metric name")
    p.add_argument("--hue", default="algorithm")
    p.add_argument("--style", help="column mapped to line style, e.g. experiment")
    p.add_argument("--col", help="column for side-by-side panels")
    p.add_argument("--row", help="column for stacked panels")
    p.add_argument("--where", nargs="*", default=[], metavar="KEY=VALUE", help="row filters")
    p.add_argument("--algorithms", nargs="*", help="only plot these algorithms")
    p.add_argument("--logx", action="store_true")
    p.add_argument("--title")
    p.add_argument("--out", default="figures/plot.png")
    args = p.parse_args(argv)

    df = apply_filters(load_results(args.results), args.where)
    if args.algorithms:
        df = df[df["algorithm"].isin(args.algorithms)]
    plot_lines(df, x=args.x, y=args.y, hue=args.hue, style=args.style, col=args.col, row=args.row,
               logx=args.logx, title=args.title, out=args.out)


if __name__ == "__main__":
    main()
