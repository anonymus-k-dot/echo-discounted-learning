"""E2: scalable LOOP (factored rank-1 self-echo, SGD+momentum attribution) on MNIST MLPs.

Regimes: small training sets and symmetric label noise (memorization matters most there).
Logged per epoch: test acc, train acc (given labels), internal LOO-meter acc, and on noisy runs the fraction of
corrupted examples predicted with their corrupted label (memorization of noise)."""
import sys, os, json, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch, torch.nn.functional as F
from data import mnist, subsample_balanced, corrupt_labels
from loop_mlp import LoopMLP, LoopTrainer

ap = argparse.ArgumentParser()
ap.add_argument("--mode", default="loop", choices=["std", "loop", "rand"])
ap.add_argument("--n", type=int, default=200, help="examples per class")
ap.add_argument("--noise", type=float, default=0.0)
ap.add_argument("--width", type=int, default=512)
ap.add_argument("--depth", type=int, default=2)
ap.add_argument("--lr", type=float, default=0.05)
ap.add_argument("--wd", type=float, default=5e-4)
ap.add_argument("--mom", type=float, default=0.9)
ap.add_argument("--alpha", type=float, default=1.0)
ap.add_argument("--cap", type=float, default=None)
ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--ls", type=float, default=0.0, help="label smoothing (std mode baseline)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--tag", default="")
args = ap.parse_args()

torch.manual_seed(args.seed)
xtr_all, ytr_all, xte, yte = mnist()
xtr, ytr_clean = subsample_balanced(xtr_all, ytr_all, args.n, seed=args.seed)
mu = xtr.mean(0, keepdim=True)
xtr, xte = xtr - mu, xte - mu
ytr, is_noisy = corrupt_labels(ytr_clean, args.noise, seed=args.seed) if args.noise > 0 else (ytr_clean, None)
m = LoopMLP(784, (args.width,) * args.depth, 10).cuda()
tr = LoopTrainer(m, xtr, ytr, lr=args.lr, wd=args.wd, betas=(args.mom, 0.999), mode=args.mode,
                 alpha=args.alpha, cap=args.cap, seed=args.seed, opt="sgd")
if args.ls > 0:   # label smoothing applies to the loss used for the gradient
    _ce = F.cross_entropy
    F.cross_entropy = lambda z, y, reduction="mean": _ce(z, y, reduction=reduction, label_smoothing=args.ls)
N = xtr.shape[0]
rows, t0 = [], time.time()
gen = torch.Generator(device="cuda").manual_seed(args.seed)
for ep in range(1, args.epochs + 1):
    perm = torch.randperm(N, device="cuda", generator=gen)
    tot = 0.0
    for s in range(0, N, args.bs):
        tot += tr.step(perm[s:s + args.bs])
    if ep % max(1, args.epochs // 20) == 0 or ep == args.epochs:
        with torch.no_grad():
            te = (m(xte).argmax(1) == yte).float().mean().item()
            ptr = m(xtr).argmax(1)
            tra = (ptr == ytr).float().mean().item()
            row = dict(ep=ep, loss=tot / (N // args.bs + 1), test=te, train=tra, loo=tr.accuracy_at("own"))
            if is_noisy is not None:
                row["mem_noise"] = (ptr[is_noisy] == ytr[is_noisy]).float().mean().item()
                row["clean_fit"] = (ptr[~is_noisy] == ytr[~is_noisy]).float().mean().item()
                row["loo_clean_true"] = None
            rows.append(row)
best = max(r["test"] for r in rows)
out = dict(args=vars(args), final=rows[-1], best_test=best, rows=rows, secs=round(time.time() - t0, 1))
os.makedirs("results/e2", exist_ok=True)
name = f"results/e2/{args.mode}_n{args.n}_noise{args.noise}_lr{args.lr}_wd{args.wd}_a{args.alpha}_ls{args.ls}_s{args.seed}{args.tag}.json"
json.dump(out, open(name, "w"))
print(json.dumps(dict(mode=args.mode, n=args.n, noise=args.noise, lr=args.lr, wd=args.wd, alpha=args.alpha, ls=args.ls,
                      seed=args.seed, final_test=round(rows[-1]["test"], 4), best_test=round(best, 4),
                      final_train=round(rows[-1]["train"], 3), loo=round(rows[-1]["loo"], 3),
                      mem_noise=rows[-1].get("mem_noise"), secs=out["secs"])))
