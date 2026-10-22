# Echo-discounted learning (EDL)

A credit-assignment rule in which each training example's gradient is evaluated at the parameters minus a fraction α of that example's *own* accumulated past influence:

  g̃_i = ∇ℓ_i(θ − α·u_i),   with u_i the exact (first-order) cumulative displacement of θ caused by example i.

**Why.** In multi-epoch training, an example that the network has memorized stops producing a learning signal, even though nothing that predicts it from the *other* data has been learned. EDL removes the example's own echo before asking it what to teach. In the kernel regime this is exactly gradient descent with the tangent kernel's diagonal reduced by α. It is stable for α < λ_min of the normalized NTK.

**What was found.**
- **Grokking.** On modular addition, pre-registered and run on fresh seeds, EDL speeds up rule acquisition beyond the strongest tuned baseline (SGD + label smoothing): 1.5× at 30% train, 3–4× at 20%, and ≥ 6× to 90% test at 15%.
- **Mechanism.** The gain needs the example's own echo; subtracting another example's echo does nothing.
- **Where it does not help.**
  - It needs an optimizer that is linear in the gradient (it diverges with Adam).
  - It gives no gain on clean small MNIST.
  - It is harmful with label noise.
  - Full details are in `docs/03`.

## Documents

| File | Content |
|---|---|
| `docs/00_prior_art_and_limitations.md` | Map of existing learning principles. Derivations showing that many first-principles candidates collapse onto known methods. The targeted limitation ("self-echo"). |
| `docs/01_loo_point_learning.md` | First formulation (leave-one-out point), pre-registered predictions, and the full log of what failed and why. The theory was revised to K − αD. |
| `docs/02_confirmatory_preregistration.md` | Frozen design, Amendment A (adds the label-smoothing baseline before any confirmatory run), and the confirmatory results. |
| `docs/03_echo_discounted_learning.md` | Consolidated results, mechanism evidence, failures, novelty assessment, generality. |

## Code

| File | Content |
|---|---|
| `src/loo_toy.py` | Exact analytic per-example echo for the modular-arithmetic MLP (fast; used for E1 and the confirmatory runs). |
| `src/echo_layers.py` | Exact echo bookkeeping for any network built from `EchoLinear` / `EchoTable` (MLP, transformer); verified against `loo_toy` to 1e-9. |
| `src/loop_mlp.py` | Scalable factored (rank-1 per layer) echo for MLPs on dense inputs (MNIST). |
| `experiments/e1_loo_grok.py` | Main toy experiment (modes `std` / `loo` / `rand`, α, adaptive κ, probes). |
| `experiments/base_grok.py` | Baselines: AdamW, SGD, label smoothing, Grokfast, ⊥Grad, Muon-style, function-space EDL. |
| `experiments/analyze_conf.py` | Evaluates the pre-registered criteria. |

## Reproduce the confirmatory result (one seed, 30% train)

```bash
python experiments/e1_loo_grok.py --opt sgd --lr 1.0 --wd 0.02 --mode loo --alpha 0.25 --frac 0.3 --seed 1 --steps 6000 --every 200
```

```bash
python experiments/base_grok.py --method adamw --opt sgd --lr 2.0 --wd 0.01 --ls 0.3 --frac 0.3 --seed 1 --steps 30000 --full
```

Data: MNIST is read from local raw files in `data/MNIST/raw`. Modular arithmetic is generated. Nothing is downloaded.

## Setup and data

```bash
pip install -r requirements.txt
```

The scripts assume a CUDA GPU. Modular-arithmetic data is generated in code. The MNIST experiments (`e2`, `e6`, `a2`) read the four standard raw IDX files from `data/MNIST/raw/`:
- `train-images-idx3-ubyte`
- `train-labels-idx1-ubyte`
- `t10k-images-idx3-ubyte`
- `t10k-labels-idx1-ubyte`

They are not included in this repository; obtain them from the standard MNIST distribution (for example, `torchvision.datasets.MNIST(root="data", download=True)` creates this layout).

All numbers in the docs come from the files in `results/`.

## License

MIT; see `LICENSE`.
