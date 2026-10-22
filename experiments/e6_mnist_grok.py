"""E6: real-data grokking (Omnigrok-style): MNIST, 1k training examples, MLP with large initialization (x scale),
cross-entropy, mini-batch SGD + momentum, decoupled weight decay. Compares ordinary SGD, EDL (factored echo,
src/loop_mlp.py) and the identity control. Generalization is known to lag memorization in this regime."""
import sys, os, json, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch, torch.nn.functional as F
from data import mnist, subsample_balanced
from loop_mlp import LoopMLP, LoopTrainer

ap = argparse.ArgumentParser()
ap.add_argument("--mode", default="loop", choices=["std", "loop", "rand"])
ap.add_argument("--n", type=int, default=100, help="per class")
ap.add_argument("--init", type=float, default=8.0)
ap.add_argument("--width", type=int, default=200)
ap.add_argument("--depth", type=int, default=3)
ap.add_argument("--lr", type=float, default=0.05)
ap.add_argument("--wd", type=float, default=0.01)
ap.add_argument("--alpha", type=float, default=0.1)
ap.add_argument("--ls", type=float, default=0.0)
ap.add_argument("--bs", type=int, default=200)
ap.add_argument("--steps", type=int, default=20000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--tag", default="")
args = ap.parse_args()

torch.manual_seed(args.seed)
xa, ya, xte, yte = mnist()
xtr, ytr = subsample_balanced(xa, ya, args.n, seed=args.seed)
m = LoopMLP(784, (args.width,) * args.depth, 10).cuda()
with torch.no_grad():
    for l in m.lins:
        l.weight.mul_(args.init)
tr = LoopTrainer(m, xtr, ytr, lr=args.lr, wd=args.wd, betas=(0.9, 0.999), mode=args.mode, alpha=args.alpha,
                 seed=args.seed, opt="sgd")
if args.ls > 0:
    _ce = F.cross_entropy
    F.cross_entropy = lambda z, y, reduction="mean": _ce(z, y, reduction=reduction, label_smoothing=args.ls)
N = xtr.shape[0]
gen = torch.Generator(device="cuda").manual_seed(args.seed)
rows, first, t0 = [], {}, time.time()
perm, pos = torch.randperm(N, device="cuda", generator=gen), 0
for s in range(args.steps + 1):
    if pos + args.bs > N:
        perm, pos = torch.randperm(N, device="cuda", generator=gen), 0
    idx = perm[pos:pos + args.bs]; pos += args.bs
    loss = tr.step(idx)
    if s % 250 == 0:
        with torch.no_grad():
            tra = (m(xtr).argmax(1) == ytr).float().mean().item()
            tea = (m(xte).argmax(1) == yte).float().mean().item()
            wn = sum(p.pow(2).sum() for p in m.parameters()).sqrt().item()
        rows.append(dict(step=s, loss=loss, train=tra, test=tea, wnorm=wn))
        for key, thr in (("train99", .99), ("test80", .8), ("test85", .85), ("test88", .88)):
            val = tra if key == "train99" else tea
            if key not in first and val >= thr:
                first[key] = s
out = dict(args=vars(args), first=first, final_test=rows[-1]["test"], best_test=max(r["test"] for r in rows),
           rows=rows, secs=round(time.time() - t0, 1))
os.makedirs("results/e6", exist_ok=True)
json.dump(out, open(f"results/e6/{args.mode}_init{args.init}_lr{args.lr}_wd{args.wd}_a{args.alpha}_ls{args.ls}_s{args.seed}{args.tag}.json", "w"))
print(json.dumps(dict(mode=args.mode, init=args.init, lr=args.lr, wd=args.wd, alpha=args.alpha, ls=args.ls, seed=args.seed,
                      first=first, final=round(out["final_test"], 4), best=round(out["best_test"], 4), secs=out["secs"])))
