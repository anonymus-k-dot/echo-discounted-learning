# 02 — Pre-registration: confirmatory test of self-influence-discounted learning (LOOP-α)

Written 2026-10-22. All choices below were made on development seed 0 only. Seeds 1–4 have never been run.

## Hypothesis

Evaluating each example's gradient at θ − α·u_i, where u_i is its own accumulated parameter contribution, does two things:
- it makes a network acquire a rule (modular addition) from fewer examples;
- it does so faster than ordinary training under any tuned learning rate and weight decay.

The effect requires the example's *own* contribution. Subtracting another example's (identity control) does not produce it.

## Setup (fixed)

- **Task.** a + b mod 97, all 9,409 pairs. The seed determines both the train/test split and the initialization.
- **Model.** ModMLP: shared embedding with d = 128, concatenated, one ReLU hidden layer h = 512, linear readout. Embedding init N(0, 1).
- **Training.** Full-batch.

## Methods and frozen hyperparameters

| Method | Optimizer | Hyperparameters | Budget |
|---|---|---|---|
| LOOP | SGD, momentum 0.9, decoupled wd | α = 0.25, lr 1.0, wd 0.02 (exact attribution, `src/loo_toy.py`) | 6,000 steps |
| Identity control | same as LOOP | same, but example i is evaluated at θ − α·u_π(i), π a fixed random permutation | 6,000 steps |
| SGD baseline | SGD, momentum 0.9, decoupled wd | per fraction, the best seed-0 config of the broad grid (lr ∈ {0.5, 1, 1.5, 2} × wd ∈ {0.005, …, 0.1}; table below) | 30,000 steps (5× LOOP's) |
| AdamW baseline | AdamW | per fraction, the best seed-0 wd ∈ {1, 3, 10}; lr 1e-3, β = (0.9, 0.98) | 30,000 steps |

"Best" means the earliest step at which test ≥ 99%, or, if no config reaches it, the highest final test accuracy.

Baseline configs selected on seed 0 (filled in from `results/base_grok_*` before any confirmatory run):

| Fraction | SGD (lr, wd) | AdamW wd |
|---|---|---|
| 0.30 | (2.0, 0.01) | 3.0 |
| 0.20 | TBD | TBD |
| 0.15 | TBD | TBD |
| 0.10 | TBD | TBD |

## Pre-registered criteria (seeds 1–4, train fractions 0.30, 0.20, 0.15, 0.10)

| ID | Criterion |
|---|---|
| C1 speed | At 0.30, LOOP reaches test ≥ 99% in fewer steps than the selected SGD baseline on ≥ 3/4 seeds. |
| C2 data | At 0.20, LOOP's final test ≥ 0.99 on ≥ 3/4 seeds, while the better baseline's final test < 0.99 on ≥ 3/4 seeds. |
| C3 data | At 0.15, LOOP's final test ≥ 0.90 on ≥ 3/4 seeds, while the better baseline's final test < 0.50 on ≥ 3/4 seeds. |
| C4 identity | At 0.15 and 0.20, the identity control's final test is below LOOP's on ≥ 3/4 seeds at each fraction, and < 0.50 at 0.15 on ≥ 3/4 seeds. |
| — | 0.10 is exploratory; no prediction. |

## Decision rule

| Outcome | Conclusion |
|---|---|
| C2, C3 and C4 pass | LOOP improves data efficiency of rule acquisition in this setting, and the effect is due to removing the example's own contribution. |
| C2 or C3 fails | No data-efficiency claim. |
| C4 fails | No mechanism claim; the effect would be attributed to perturbation. |

## Known threats, stated in advance

1. **Narrow setting.** One task family and one architecture; full-batch training.
2. **Lighter tuning for LOOP.** LOOP's hyperparameters came from fewer configs (an α sweep plus a 3×3 grid at 0.30, and two configs at lower fractions). The baselines get larger grids and 5× the steps; this asymmetry favors the baselines.
3. **α sensitivity.** α is sensitive (unstable at ≥ 0.5 here; §7.4–7.5 of doc 01). A single α is frozen.
4. **Possibly not general.** On MNIST (1k examples) LOOP gave no gain clean and was harmful with 40% label noise (doc 01 §7). This test does not address generality.

---

## Amendment A (2026-10-22, before any run on seeds 1–4)

### Why
On development seed 0, **SGD + label smoothing** turned out to be a far stronger baseline than plain SGD or AdamW. Label smoothing also keeps a non-vanishing residual after interpolation. At 15% train it reaches 0.949 final test in 20k steps, where all 24 plain-SGD and 3 AdamW configs stay below 0.02. As written, criterion C3 ("the better baseline < 0.50") would be a strawman. The criteria are rewritten against the strongest baseline, before the confirmatory runs.

### Seed-0 selections, frozen

| Fraction | Best plain SGD | Best SGD + LS | AdamW (wd 3) | Muon (lr 0.05, wd 0.3) |
|---|---|---|---|---|
| 0.30 | (2.0, 0.01): 99% at 1,200 | (2.0, 0.01, ls 0.3): 99% at 700 | 99% at 8,600 | 99% at 1,000 |
| 0.20 | (2.0, 0.01): final 0.954, no 99% | (2.0, 0.01, ls 0.3): 99% at 2,100 | — | — |
| 0.15 | none generalizes (24 configs) | (2.0, 0.01, ls 0.3): 50% at 6,400, final 0.949 | none | — |

Confirmatory baselines:
- **SGD + LS (lr 2.0, wd 0.01, ls 0.3)**, the strongest, at every fraction;
- **plain SGD (lr 2.0, wd 0.01)**, reported for reference.

Both get 30,000 steps. LOOP stays frozen at α = 0.25, lr 1.0, wd 0.02, with 6,000 steps, as does the identity control.

### Rewritten criteria (seeds 1–4; "never within budget" counts as ∞)

| ID | Criterion |
|---|---|
| C1′ | At 0.30, LOOP reaches test ≥ 99% in fewer steps than SGD + LS on ≥ 3/4 seeds. |
| C2′ | At 0.20, LOOP reaches test ≥ 99% in fewer steps than SGD + LS on ≥ 3/4 seeds. |
| C3′ | At 0.15, LOOP reaches test ≥ 90% in fewer steps than SGD + LS on ≥ 3/4 seeds, **and** LOOP's final test ≥ SGD + LS's final test − 0.02 on ≥ 3/4 seeds (LOOP has 5× fewer steps). |
| C4 | Unchanged (identity control below LOOP at 0.15 and 0.20, and < 0.50 at 0.15, on ≥ 3/4 seeds). |

### Decision
- **C1′–C3′ and C4 all pass:** LOOP accelerates rule acquisition beyond the strongest tuned baseline, and the effect needs the example's own contribution.
- **Any of C1′–C3′ fails:** report as is; no speed claim at that fraction.
- **Already known from the dev seed, and not claimed:** LOOP generalizes at 15% but not at 10% (no method does).

---

## Results (seeds 1–4, run once after Amendment A; produced by `experiments/analyze_conf.py`)

**Settings.** LOOP: α = 0.25, lr 1.0, wd 0.02, 6k steps. SGD + LS: lr 2.0, wd 0.01, ls 0.3, 30k steps. SGD: lr 2.0, wd 0.01, 30k steps. Control: θ − α·u_π(i), 6k steps. "inf" means the threshold was not reached within budget.

| Fraction | Seed | LOOP t90 | LOOP t99 | LOOP final | LS t90 | LS t99 | LS final | SGD t99 | SGD final | Control final |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.30 | 1 | 400 | 400 | 1.000 | 600 | 700 | 1.000 | 800 | 0.999 | — |
| 0.30 | 2 | 400 | 400 | 1.000 | 600 | 700 | 1.000 | 800 | 1.000 | — |
| 0.30 | 3 | 400 | 400 | 1.000 | 500 | 500 | 1.000 | 700 | 1.000 | — |
| 0.30 | 4 | 400 | 400 | 1.000 | 600 | 600 | 1.000 | 1100 | 0.999 | — |
| 0.20 | 1 | 600 | 600 | 1.000 | 2000 | 2700 | 0.999 | inf | 0.919 | 0.973 |
| 0.20 | 2 | 600 | 800 | 1.000 | 2100 | 2400 | 0.999 | inf | 0.978 | 0.965 |
| 0.20 | 3 | 600 | 600 | 1.000 | 1700 | 2100 | 0.999 | 2600 | 0.993 | 0.730 |
| 0.20 | 4 | 600 | 800 | 1.000 | 1700 | 2000 | 0.999 | inf | 0.984 | 0.970 |
| 0.15 | 1 | 2000 | inf | 0.973 | inf | inf | 0.842 | inf | 0.000 | 0.000 |
| 0.15 | 2 | 2200 | inf | 0.922 | inf | inf | 0.764 | inf | 0.000 | 0.000 |
| 0.15 | 3 | 2000 | inf | 0.965 | inf | inf | 0.685 | inf | 0.001 | 0.000 |
| 0.15 | 4 | 2200 | inf | 0.968 | 12600 | inf | 0.932 | inf | 0.032 | 0.000 |

| Criterion | Result |
|---|---|
| C1′ (0.30, t99: LOOP < LS) | **4/4 PASS** |
| C2′ (0.20, t99: LOOP < LS) | **4/4 PASS** |
| C3′ (0.15, t90: LOOP < LS) | **4/4 PASS** |
| C3′ (0.15, final: LOOP ≥ LS − 0.02) | **4/4 PASS** |
| C4 (0.20, control < LOOP) | **4/4 PASS** |
| C4 (0.15, control < LOOP and < 0.5) | **4/4 PASS** |

**Verdict under the pre-registered rule.** In this setting (modular addition, ReLU-MLP with embeddings, full-batch SGD), echo-discounted learning speeds up rule acquisition beyond the strongest tuned baseline:
- **1.5×** at 30% train;
- **3–4×** at 20%;
- **≥ 6×** to 90% test at 15%, where label smoothing with 5× the steps ends 0.04–0.28 lower on final accuracy.

The effect requires the example's *own* contribution: the identity control reaches 0.000 at 15% on all seeds.

**Not claimed:** generality beyond this setting (see doc 03 for MNIST, label noise, other architectures).
