# 00 — Prior art, fundamental limitations, and where a new learning principle can still live

Written 2026-10-22, before any result on the mechanism of doc 01.

## 1. Goal and the bar

Goal: a learning principle that changes how a network acquires knowledge, assigns credit, or evolves its representations. A new optimizer, schedule or regularizer does not count, and neither does a renamed known method. The bar a candidate must clear:
1. It is derived from a stated limitation, not assembled from parts.
2. It is not equivalent (exactly or to first/second order) to a published method.
3. It beats strong, tuned baselines in controlled experiments, with seeds and controls that isolate the claimed mechanism.
4. Its failure modes are reported.

## 2. Map of existing learning principles

| Family | What drives the update | Representative work |
|---|---|---|
| Gradient descent and preconditioning | ∂L/∂θ, optionally rescaled by a metric | SGD, Adam, natural gradient, K-FAC, Shampoo, Muon (spectral-norm steepest descent), schedule-free |
| Second order and saddle handling | curvature | Newton, Gauss–Newton, Hessian-free, saddle-free Newton |
| Aggregation of per-example credit | nonlinear or robust combination of per-example gradients | AND-mask ("hard to vary"), GAF (gradient agreement filtering), coherent gradients / winsorized GD, Fishr, geometric median, PCGrad |
| Gradient alignment / meta-objectives | maximize transfer between examples or tasks | Reptile, MAML, MER, Fish, Lookahead; SAM and flatness |
| Temporal credit on updates | validate past steps on new data | IDBD (Sutton 1992), hypergradient descent, Lookahead |
| Backprop alternatives (different signal) | random feedback, inverses, energies, local goodness | FA/DFA, target propagation, predictive coding, equilibrium propagation, least-control principle, prospective configuration (2024), Forward-Forward, PEPITA, NoProp (2025), Mono-Forward |
| Backprop-free feature learning | predictor sensitivity | Recursive Feature Machines / AGOP (2024), OPG/MAVE sufficient dimension reduction (2002) |
| Memorization and example-level credit | which examples to trust | co-teaching, ELR, SOP (per-example noise parameters), RHO-LOSS, MAT (held-out predictions shift logits, ICLR 2025), data maps, forgetting events |
| Plasticity and consolidation | keep or restore the ability to learn | EWC, SI, MAS, continual backprop, ReDo, L2-to-init, shrink & perturb, UPGD, Parseval |
| Structure | grow, split, prune or route | splitting steepest descent, GradMax, RigL, MoE, cascade-correlation, RAN/ART |
| Grokking accelerators | post-interpolation dynamics | weight decay / Omnigrok, Grokfast (amplify slow gradients), ⊥Grad |
| Data attribution (analysis only) | per-example contributions to θ | influence functions, TracIn, TRAK, SOURCE (unrolled attribution, NeurIPS 2024) |
| Learning theory of data reuse | label nonlinearity from reusing a batch | Dandi et al. 2024, Lee et al. 2024 (reuse breaks the information-exponent barrier) |

## 3. Results that rule out large parts of the design space

I tried to derive candidate rules from first principles. Most collapsed onto a known family. The derivations are recorded here because they tell us where *not* to look.

**C1. First-order spread is fixed by the kernel.** Suppose an update uses only example i's gradient (any preconditioner P). Then its effect on any other input x′ is Δf(x′) = −η J(x′) P J(x_i)ᵀ ∇ℓ_i, i.e. the preconditioned tangent kernel K_P(x′, x_i). How a single learning event generalizes is therefore fixed by the representation and the metric, so per-example rules differ only in P. That puts them in the preconditioning family.

**C2. Aggregation geometry collapses to the Schatten family.** Suppose we maximize the first-order transfer of a layer update ΔW = G Q onto unseen examples under a Frobenius budget on Q. The optimum is Q ∝ GᵀG, and with both Kronecker factors, ΔW ∝ U S^p Vᵀ. Here p = 0 is Muon, p = 1 is SGD, and p → ∞ is a rank-one top-singular step. All are steepest descent under a Schatten norm (known). Minimizing interference instead yields natural gradient.

**C3. Sums of pathways split credit by kernel share.** Training a sum of pathways (a "rule" net plus an "exception" memory) with SGD splits every error between them in proportion to their tangent kernels. So "route typical examples to shared weights and exceptions to a memory" reduces to an architecture choice plus SGD (cf. SOP).

**C4. Training for good spread reduces to gradient alignment.** The total helpful first-order transfer in a batch is Σ_{i≠j} g_iᵀ g_j. Maximizing it is the implicit objective of Reptile, MER and Fish. At a minimum it equals −Σ_i ||g_i||², i.e. per-example flatness (the SAM family).

**C5. Validating past updates on new data reduces to IDBD.** The first-order test is g_{t+1}ᵀ u_t, which is IDBD's or hypergradient descent's signal.

**C6. Exponentially forgetting old updates is L2 regularization to the initial weights.**

**C7. A leave-one-out training objective approximated by influence functions reduces to a per-example gradient-norm penalty** (flatness family): ℓ(z_i − c K_ii ∇ℓ_i) ≈ ℓ_i + c ||g_i||².

**Consequence.** Within "one differentiable network, gradient signals, i.i.d. data, first or second order", the principled derivations land on known methods. A new principle needs at least one of the following:
1. information that is not in the batch gradient or curvature;
2. a non-gradient computation in the forward pass;
3. discrete structure;
4. a *state* that changes what the learning signal is, not just how it is scaled.

## 4. The limitation this project targets

**Self-echo.** In multi-epoch training, example i's gradient at θ_t is computed by a model whose parameters already contain the cumulative effect of i's own past gradients. Once the network has fit i with i-specific changes (memorization), i's loss is small and its learning signal fades, even though nothing has been learned that predicts i from the *other* data. The learner hears its own echo and takes it for understanding. Three known phenomena follow:
- **Grokking.** After interpolation the residual is near zero, and only weak forces (weight decay, noise) drive the slow move to the generalizing solution.
- **Memorization kills the incentive to learn robust features.** This is the mechanism MAT identifies for spurious correlations.
- **Lock-in.** Examples fitted early through poor features stop shaping the representation.

**Measured here** (`experiments/diag_grok_readout.py`, modular addition p = 97, 30% train, AdamW):
- At step 1000 the network fits 100% of training pairs, and test accuracy is 0.0%.
- A ridge readout refit on the training features also gives about 0% test accuracy. A ridge readout fit on half the training pairs predicts the *other half of training pairs* at 62–94%.
- So memorization lives in the **features**, and readout-level cross-validation is fooled by it. An honest internal signal must remove an example's influence from the *whole network*.

Existing remedies all pay for that honesty with extra models or held-out data. MAT and cross-fitting train held-out models. Influence-based leave-one-out is used only post hoc (TracIn, TRAK, SOURCE). Weight decay does not target the echo at all.

## 5. What is new in the proposal (doc 01), relative to the closest work

| Closest work | What it does | Difference |
|---|---|---|
| TracIn / TRAK / SOURCE / influence imprints | store or approximate per-example contributions to θ, *after* training, for attribution | we use the stored contribution *during* training to move the point at which each example's learning signal is computed |
| MAT (Bayat et al. 2025) | held-out *models* give predictions that shift logits | no extra models; the leave-one-out point of the *same* network is maintained online; the gradient itself, not a logit offset, changes |
| AND-mask / GAF | filter per-example or per-microbatch gradients by agreement | no filtering; every example keeps full credit, evaluated where its own echo is removed |
| SAGA / SAG | store per-example *gradients* for variance reduction | we store per-example *cumulative parameter displacements* and evaluate gradients elsewhere |
| Influence-function leave-one-out penalties (C7) | second-order Taylor expansion → gradient-norm penalty | no Taylor expansion of the loss; exact forward at θ − u_i; the dynamics are non-conservative (not a gradient flow) |

The novelty search (web, 2026-10-22) found no training method that evaluates each example's gradient at the parameters minus that example's own accumulated contribution. This is a claim about the search, not proof; doc 01 §6 lists what is still to check.
