# sprlhf: contextual-bandit simulations of strategyproof RLHF

Simulations of Pessimistic Median of MLEs (Kleine Buening et al., NeurIPS 2025)
against standard RLHF baselines, in the paper's own setting: an offline contextual
bandit with linear rewards and Bradley–Terry comparisons. The experiment asks what
happens to welfare as a growing share of labelers lie.

## Quick start

```bash
pip install -e ".[dev]"          # Python >= 3.10; needs cvxpy
pytest -q                        # unit checks, a few seconds
python -m sprlhf.sanity          # 22 end-to-end checks, ~1 min -> results/sanity/report.md

python -m sprlhf.run configs/strategic_fraction.yaml   # the experiment, ~2 min on 2 cores
bash make_figures.sh                                       # -> figures/norm_suboptimality.png
```

Try anything quickly with `--seeds 4`; change any setting with `--set key=value`.

## The experiment

`configs/strategic_fraction.yaml`: 0–5 of 11 labelers report strategically, all with
the same rule (exaggerate 5× or bury). Run in three environments: `box` (the paper's box
assumption holds), `box_rotated` (the same box in a rotated basis), and `random_features`
(the assumption fails). 3 environments × 2 strategies × 6 liar shares × 30 seeds.

Metrics (always computed with true preferences):
- `norm_suboptimality`: welfare lost, 0 = optimal, 1 = no better than random (the figure)
- `honest_utility`, `strategic_utility`, `strategic_advantage` (= strategic − honest)
- `worst_off_utility`, `welfare`, `suboptimality`

## Plotting

```bash
python -m sprlhf.plot results/strategic_fraction.csv.gz \
    --x actual_strategic_fraction --y norm_suboptimality --col env_name --row labelers.strategy \
    --algorithms pooled_btl maxmin psw pmomle --out figures/norm_suboptimality.png
```

Flags: `--where KEY=VALUE` (filter), `--col` / `--row` (panels), `--style experiment`
(overlay several results files), `--title`. In a notebook:
`from sprlhf.plot import load_results, plot_lines`. Bands are 95% CIs over seeds; seeds
are paired across algorithms (identical data), so differences between algorithms are
more certain than overlapping bands suggest.

## Algorithms

| Name | What it is |
|---|---|
| `pooled_btl` | one Bradley–Terry model on all comparisons pooled (InstructGPT recipe) |
| `maxmin` | MaxMin-RLHF: maximize the worst-off labeler's estimated utility (known identities, no KL) |
| `psw` | Pessimistic Social Welfare (utilitarian): worst-case average utility over confidence sets |
| `pmomle` | Pessimistic Median of MLEs: worst-case utility of the feature-wise median over confidence sets |
| `median_of_mles`, `mean_of_mles` | ablations without pessimism (used by the sanity checks) |
| `mean_rule`, `median_rule`, `maxmin_rule` | direct access to reported preferences: the infinite-data limits |

**PMoMLE implementation note.** The exact worst case over all medians is intractable.
Each labeler's confidence ellipsoid is replaced by its bounding box, which makes the set
of possible medians a box: the same coordinate-by-coordinate reasoning the paper's proofs
use, and at least as cautious as the exact algorithm.

## Calibrated defaults (from the sanity checks)

- **Preference strength `theta_norm = 3`:** labels agree with the labeler's stronger
  preference ~65% of the time, close to real annotator agreement.
- **Confidence scale `conf_scale = 3`:** with the paper's radius taken literally (c = 1),
  true preferences fall inside their confidence sets only ~9% of the time; with c = 3, 95–99%.
- **Pessimism has a real cost:** with valid confidence sets, PMoMLE's truthful welfare loss
  is ~0.2 at n = 3000 (pooled BTL ~0) and much larger at small n, which is why the
  experiment uses n = 3000.

## Layout

| File | What it does |
|---|---|
| `sprlhf/simulation.py` | the simulated world, in sections: random streams, settings, environments (box, rotated, random features), policies and exact utilities, labelers' true preferences, strategic reporting (`exaggerate`, `bury`), comparison data and Bradley–Terry estimation, metrics |
| `sprlhf/algorithms.py` | pooled BTL, MaxMin, PSW, PMoMLE, two ablations, and the infinite-data rules |
| `sprlhf/run.py` | experiment config (YAML, sweeps) and the runner → long-format results table |
| `sprlhf/plot.py` | the figure: mean ± 95% CI |
| `sprlhf/sanity.py` | end-to-end checks |
| `tests/test_sprlhf.py` | unit checks |
| `configs/base.yaml` | every setting with its default, documented |

## Design decisions

- **Exact utilities.** Every policy is summarized by its average feature vector
  z_π = E_s E_{a~π} φ(s, a); every utility is ⟨θ_i, z_π⟩, computed exactly. Policies are stochastic.
- **Paired randomness.** Each random source has its own stream keyed by (seed, name, index):
  every algorithm sees identical data, a lie changes only the liar's labels, and changing one
  setting leaves everything else fixed.
- **Strategic labelers act independently**, know the population's true preferences (a
  worst-case manipulator), and all use the same rule. The strategic set is a random subset
  per seed, nested across fractions.
- **Metrics always use true preferences,** never reported ones.
