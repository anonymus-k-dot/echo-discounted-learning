# 01 — Leave-one-out-point learning (working name LOOP)

Sections 1–5 were written before any LOOP result (only the standard-AdamW meter at step 1000 had been seen).

## 1. Principle

Each example's learning signal is computed by the network *minus that example's own past influence*.

Notation:
- θ_t: parameters at step t.
- u_i(t): the cumulative displacement of θ caused by example i's own gradients ("self-echo").

LOOP replaces each example's gradient ∇ℓ_i(θ_t) by

  g̃_i = ∇ℓ_i(θ_t − u_i(t)),

and attributes the resulting step back to u_i. To first order (the same order at which TracIn and influence functions operate), θ_t − u_i is the network trained on everything except i. The signal for i is then "what the rest of the data has failed to teach about i". The example can never be used up by memorizing it. Ordinary SGD/Adam asks a different question, "how to fit i further, given that i is already partly fitted by itself".

## 2. Algorithm (AdamW form used in the experiments)

State: θ, Adam moments (m, v), and u_i for each training example. The u_i are stored in factored form wherever the architecture allows it (§4).

For each step:
1. For each example i in the batch: run the forward pass at θ − α u_i, then compute g̃_i = ∇ℓ_i(θ − α u_i).
2. ḡ = mean_i g̃_i. Run AdamW on ḡ: θ ← (1 − ηλ)θ − η m̂/(√v̂ + ε). Let P = 1/(√v̂ + ε).
3. Attribute the step immediately: u_i ← (1 − ηλ) u_i − (η/B) P ⊙ g̃_i for i in the batch. For every other j, u_j ← (1 − ηλ) u_j.

Invariant (exact without momentum lag): θ_t = decayed(θ_0) + Σ_i u_i(t).

- α = 1 is the principle; α is kept as an ablation knob.
- Standard training is α = 0 (u is then only an internal meter).

## 3. Analysis in the kernel regime

Linearize: f(x; θ) = f_0(x) + J(x)(θ − θ_0) with fixed J. Use squared loss, gradient flow, no weight decay, and K_ij = J_i J_jᵀ. Write z for training predictions, z̃ for the leave-one-out-point predictions, and D = diag(K).

Standard GD:  ż = −ηK(z − y), and z̃ = z − (self part).

LOOP:  dz̃/dt = −η (K − D)(z̃ − y)  and  ż = −η K (z̃ − y).

Consequences:
- **(a) Fixed point.** If K − D is nonsingular, the fixed point is z̃ = y: every training label is reproduced by the *other* examples' contributions. The test predictor is f(x) = f_0(x) + k(x)ᵀα with α = (K − D)⁻¹(y − f_0). This is the off-diagonal ("jackknife") interpolant, versus K⁻¹ for GD and (K + λI)⁻¹ for ridge.
- **(b) Instability.** K − D has zero trace, so it is indefinite whenever K is not diagonal. Error components along its negative eigen-directions *grow*. These are the components of y that the other examples cannot reproduce: noise, or structure the current kernel cannot express. Weight decay acts as anti-windup: with decay rate λ′ the dynamics are stable iff λ′ > −η λ_min(K − D). So **LOOP needs anti-windup**, either weight decay or a cap on ||u_i||. Without it, unexplainable labels drive the self-echo up without bound.
- **(c) Integral control.** z − z̃ = −ηD ∫(z̃ − y)dt. LOOP is GD whose targets are inflated by the *integrated leave-one-out error*: integral control on the leave-one-out error. Standard GD is proportional control on the training error. Integral action removes the steady-state error that is not seen in the training residual, which here is the generalization error. Its known cost is windup; the cap is the classical anti-windup fix.
- **(d) Feature learning.** In a feature-learning network, K changes. The leave-one-out residual stays large while i is fitted only by its own echo. So representation learning keeps receiving signal after interpolation, which is exactly when ordinary training stops receiving it.

## 4. Cost

- **Modular-arithmetic MLP (toy).** u_i is stored exactly, about 194k floats × 2,822 examples = 2.2 GB. The analytic per-example forward/backward costs about 35 ms/step at p = 97.
- **MLPs at scale.** Every linear-layer gradient is rank-1 per example: u_i^l ≈ c_i^l ā_i^{lᵀ}, which needs out + in floats per layer per example. The leave-one-out forward then costs one extra dot product per layer: (W − c āᵀ)a = Wa − c(āᵀa).
- **Convolutions.** Per-example contributions are not rank-1; out of scope until the toy results justify the effort.

## 5. Pre-registered predictions for E1 (modular addition, p = 97, AdamW lr 1e-3, wd 1, β = (0.9, 0.98), full batch, seed 0)

| ID | Prediction | Kill / falsification |
|---|---|---|
| P1 meter | Under standard AdamW, leave-one-out-point training accuracy tracks test accuracy (mean \|LOO − test\| ≤ 0.10 over checkpoints). The random-identity control stays near training accuracy. | If the meter tracks training accuracy instead, the self-echo is not where memorization lives, and the premise fails. |
| P2 speed | LOOP reaches test ≥ 99% in ≤ 0.5× the steps of AdamW (AdamW: 11,250 at 30% train). | — |
| P3 identity | The random-identity control (θ − u_π(i)) is within ±30% of AdamW's time. | If the control does as well as LOOP, the gain comes from perturbation, not from removing the self-echo. |
| P4 data | LOOP generalizes (test ≥ 99%) at 15% train within 50k steps, where AdamW does not. | — |
| P5 windup | Training logits under LOOP grow larger than under AdamW, and ||u|| saturates under wd = 1. With wd = 0 the dynamics blow up (§3b). | If wd = 0 is stable and helpful, §3b is wrong for this regime. |

Decision:
- P2 and P4 both fail: LOOP does not accelerate generalization on its flagship case, and I record why.
- P3 fails (control ≈ LOOP): the mechanism claim is withdrawn.

## 6. Novelty: still to check

- "self-influence correction during training"
- "prequential multi-epoch training"
- "jackknife gradient"
- "training at unlearned parameters"
- "negative ridge" / "off-diagonal kernel" interpolants in kernel regression

Also to check: SOURCE (unrolled attribution) as a training-time tool, and online unlearning-based regularizers.

---

## 7. Results log (appended after runs; nothing above was edited)

### 7.1 AdamW: LOOP diverges (2026-10-22)
- **Run.** α = 1, AdamW lr 1e-3, wd 1, 30% train.
- **Outcome.** The leave-one-out loss grows from 55 to 3e16. Training accuracy stays at chance.
- **Cause.** Adam's preconditioner P is computed from the *mean* gradient, and per-example gradients largely cancel. Linear attribution P·g_i/N therefore inflates every u_i by about 1/|mean gradient|, so θ − u_i lies far from θ and the leave-one-out gradients grow. That is positive feedback.
- **Conclusion.** Per-example attribution is only sound for an optimizer that is linear in the gradient. All further runs use SGD with heavy-ball momentum, where attribution is exact (verified to 1e-7 on the first layer of the scalable MLP version).

### 7.2 Baselines (modular addition, p = 97, 30% train, seed 0; steps to test ≥ 99%)

| Method | Steps |
|---|---|
| AdamW wd 1 | 11,200 |
| AdamW wd 3 | 8,600 |
| Grokfast (best of 6 settings) | 10,600 |
| ⊥Grad | did not generalize (likely needs its StableMax companion) |
| **full-batch SGD + momentum 0.9, lr 1.0, wd 0.01 (decoupled)** | **1,700** (best of a 3×3 grid; 1,600 in the LOO trainer) |

### 7.3 P1 (meter) fails
- Under ordinary SGD the network reaches 100% test accuracy by step 2,100.
- The leave-one-out meter at α = 1 stays at about 0.09 the whole time, so it tracks neither test nor training accuracy.
- **Diagnosis.** The path-integrated self-contribution J_i u_i is not the leave-one-out displacement. It counts the full self-kernel push ∫K_ii r_i and ignores how the other examples would have compensated without i. In kernel ridge terms, it is z_i − K_ii α_i rather than the PRESS form (z_i − h_ii y_i)/(1 − h_ii).
- **The principle as stated in §1 is therefore wrong at α = 1:** θ − u_i is *not* the network without i.

### 7.4 LOOP-α with SGD (lr 1.0, wd 0.01, 30% train, seed 0)

| α | Steps to test ≥ 99% | Behavior |
|---|---|---|
| 0 (SGD) | 1,600 | test 0.00 at step 300 |
| 0.25 | **600** | test 0.41 at step 300; stable |
| 0.5 | none | reaches 0.91 test, then oscillates and collapses to about 0.25 |
| 1.0 | none | collapses to chance; loss frozen |
| random identity, α = 1 | none | NaN |

### 7.5 Revised theory (replaces §3's reading of α)
- **Dynamics.** With an evaluation point θ − αu_i, the kernel-regime dynamics are exactly dz̃/dt = −η (K − αD)(z̃ − y). LOOP-α is gradient descent with the tangent kernel's *diagonal* (self-influence) reduced by α.
- **Spectral reading.** Write C = D^{-1/2} K D^{-1/2} for the normalized kernel. An eigendirection with eigenvalue λ is learned at rate ∝ (λ − α). Directions dominated by self-influence (example-specific memorization, small λ) are slowed or stopped; shared structure (large λ) is barely affected.
- **Stability.** Stable iff K − αD ⪰ 0, i.e. α ≤ λ_min(C), up to weight decay and feature learning. **Prediction:** the critical α* ≈ λ_min of the normalized empirical NTK on the training set. This is to be measured.
- **What the method is now claimed to be.** "Learn as if a fraction α of your own past influence on this example were absent" = self-influence-discounted learning, a dynamic diagonal reduction of the NTK. It no longer claims to be leave-one-out.
