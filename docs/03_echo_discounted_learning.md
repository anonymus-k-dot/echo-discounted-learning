# 03 — Echo-discounted learning (EDL): what was found

EDL is the final form of the mechanism called LOOP in docs 01–02. This document states it, the evidence for it, and its limits. Generality results are appended in §8 as they arrive.

## 1. The principle

In multi-epoch training, the gradient of example i is computed by a network whose parameters already contain i's own past influence (its "echo"). Once i has been fitted by i-specific changes, its loss is small and its learning signal fades, though nothing that predicts i from the *other* examples has been learned.

EDL computes each example's learning signal at the point where a fraction α of its own echo is removed:

  g̃_i = ∇ℓ_i(θ − α·u_i),   u_i ← (1 − ηλ)·u_i − (η/(1−μ))·g̃_i/B   (for i in the batch)

- u_i is the exact first-order cumulative displacement of θ caused by i's gradients, under SGD with heavy-ball momentum μ and decoupled weight decay λ.
- The step itself is ordinary: θ follows SGD on mean_i g̃_i.

**Credit is reassigned, not rescaled.** An example teaches the network only what the rest of the network does not already explain about it. It never teaches by confirming its own memorization.

## 2. Theory

**Exact kernel-regime dynamics.** With linearized f, squared loss, gradient flow and kernel K, let z̃ be the predictions at the discounted points and D = diag(K). Then

  dz̃/dt = −η (K − αD)(z̃ − y),   ż = −η K (z̃ − y).

EDL is therefore gradient descent whose tangent kernel has its **diagonal (self-influence) reduced by α**.

**Spectral reading.** Take the normalized kernel C = D^{-1/2} K D^{-1/2}.
- A mode with eigenvalue λ is learned at rate ∝ (λ − α).
- Self-dominated modes (example-specific, small λ) are slowed or reversed.
- Shared structure (large λ) is barely affected.

**Stability.** K − αD ⪰ 0 ⟺ α ≤ λ_min(C), with a margin from weight decay.
- Measured: λ_min(C) ≈ 0.40–0.45 during memorization (`results/a1/`).
- Observed: α ≤ 0.35 stable, α ≥ 0.5 unstable (doc 01 §7.4).
- λ_min drops to about 0.015 once the network generalizes, which explains the slow late windup at fixed α.

**Interpretation that matches the probes.** α is the fraction of the naive path-integrated self-contribution that is *really* example-specific. At α ≈ 0.25 the discounted point is "test-like":
- under ordinary SGD, accuracy at θ − 0.25·u_i equals true test accuracy at the transition (0.68 vs 0.683, step 1000);
- at α = 1 it is badly pessimistic (0.09 when test = 1.0), because the rest of training would have compensated (§7.3 of doc 01).

EDL's update is then approximately the gradient of a *held-out* loss, computed from training data alone.

## 3. Main result (pre-registered, fresh seeds; doc 02)

Setup: modular addition (p = 97); ReLU MLP with a shared embedding; full batch. EDL is frozen at α = 0.25, lr 1.0, wd 0.02, with 6k steps. The baselines were selected on the development seed from 30+ configurations and get 30k steps.

| Train fraction | Metric | EDL | SGD + label smoothing (strongest baseline) | Plain SGD | AdamW | Identity control |
|---|---|---|---|---|---|---|
| 30% | steps to test ≥ 99% | **400** (4/4 seeds) | 500–700 | 700–1,100 | 8,600 (dev) | — |
| 20% | steps to test ≥ 99% | **600–800** | 2,000–2,700 | ∞ on 3/4 seeds | ∞ (dev) | final 0.73–0.97 |
| 15% | steps to test ≥ 90% | **2,000–2,200** | ≥ 12,600 (∞ on 3/4) | ∞ | ∞ | — |
| 15% | final test | **0.922–0.973** | 0.685–0.932 (5× steps) | 0.00–0.03 | 0.00 | **0.000** |
| 10% | — | fails (dev) | — | fails | fails | — |

All six pre-registered criteria pass on 4/4 seeds.

## 4. Mechanism evidence

1. **Identity.** Subtracting a *random other* example's echo of the same size gives no gain. At 30% train (dev) it reaches 99% at 1,700 steps, against 1,600 for SGD. At 15% it gives 0.000 on all seeds. The gain needs the example's own contribution.
2. **Parameter space, not logits.** The function-space approximation keeps per-example logit offsets that integrate the discounted residual, with a constant self-kernel. It is weaker: best 800 steps at 30% vs 400, and at most 0.73 at 15% vs 0.94–0.97. So removing the echo from the *parameters*, i.e. from the features and embedding rows at which the gradient is computed, matters. EDL is not "integral target control".
3. **Not a non-vanishing residual.** Label smoothing also keeps the residual alive. It explains part of the data-efficiency gain over plain SGD but is 3–6× slower than EDL. Combining the two gives no synergy (0.90–0.94 at 15% vs EDL's 0.98).
4. **The update points at the test loss earlier.** Oracle probe, analysis only (cosine between the step's mean gradient and the test-loss gradient, 15% train):
   - plain SGD: −0.65 throughout (it actively hurts test loss once memorized);
   - label smoothing: turns positive at about step 5,600;
   - EDL: turns positive at about step 1,600 and settles near +0.4.
5. **Stability boundary predicted by the spectrum.** See §2.

## 5. Where EDL fails or does not apply (all measured)

| Setting | Outcome | Reading |
|---|---|---|
| Adam / AdamW | diverges (loss 1e16) | Linear attribution of Adam's normalized step inflates echoes, because Adam's preconditioner comes from the *mean* gradient, whose components cancel across examples. EDL needs an optimizer that is linear in the gradient. |
| MNIST, 1k examples, clean (factored MLP) | no gain (0.900 vs 0.900); unstable at α ≥ 0.25 late in training | Generalization does not lag memorization here, so there is no gap for the held-out signal to close. |
| MNIST, 1k examples, 40% label noise | **harmful**: final 0.46–0.51 vs 0.65 (best epoch slightly higher) | Predicted by §2: labels the others cannot explain have a persistent discounted residual, so EDL keeps pushing them into memory (windup). |
| 10% train, modular addition | fails, like every baseline | — |
| Untied first layer (factored implementation) | works at α = 0.1 (99% at 700 vs SGD never within 3k); unstable at 0.25 | The stability bound depends on the parameterization; α must follow λ_min (§8.2). |

## 6. Novelty assessment (searches 2026-10-22; see doc 00 §5)

- **Training-data attribution** (influence functions, TracIn, TRAK, SOURCE, influence imprints) stores or estimates per-example contributions *after* training, for analysis. None uses them to move the point where a training gradient is evaluated.
- **MAT** (Bayat et al., ICLR 2025) uses predictions from separately trained held-out models to shift logits. EDL uses no extra model and changes the gradient's evaluation point.
- **AND-mask and GAF** filter gradients by agreement. **SAGA** stores per-example gradients for variance reduction. **SAM** and random weight perturbation evaluate gradients at adversarial or random points. EDL evaluates at a deterministic, example-specific "un-learned" point.
- **Grokking accelerators** (weight decay, Grokfast, ⊥Grad, egalitarian GD / Muon-style, GrokAlign, NeuralGrok) act on the aggregate gradient or the weights. None is example-specific.
- In kernel terms, EDL learns with kernel K − αD (a *negative* diagonal adjustment). Ridge regularization adds a diagonal; jackknife and PRESS use leave-one-out residuals only for model selection.

**No prior method found does what EDL does.** This is evidence from searching, not proof.

## 7. Honest scope

EDL is a new credit-assignment rule with a clear theory and a reproducible, controlled gain in one important regime: memorization that precedes generalization (grokking). It has not been shown to help ordinary supervised learning. It is harmful with label noise, and it needs a linear optimizer.
