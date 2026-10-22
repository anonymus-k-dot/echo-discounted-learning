"""E3: the scalable factored LOOP (src/loop_mlp.py) on modular arithmetic with one-hot inputs.

Checks that the effect seen with exact per-example storage (E1) survives the factored rank-1 approximation,
and whether it holds for deeper MLPs and other operations. Full-batch or mini-batch SGD + momentum."""
import sys, os, json, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch, torch.nn.functional as F
from data import modular
from loop_mlp import LoopMLP, LoopTrainer

ap = argparse.ArgumentParser()
ap.add_argument("--mode", default="loop", choices=["std", "loop", "rand"])
ap.add_argument("--op", default="add")
ap.add_argument("--p", type=int, default=97)
ap.add_argument("--frac", type=float, default=0.3)
ap.add_argument("--widths", type=int, nargs="+", default=[512])
ap.add_argument("--lr", type=float, default=1.0)
ap.add_argument("--wd", type=float, default=0.02)
ap.add_argument("--alpha", type=float, default=0.25)
ap.add_argument("--bs", type=int, default=0, help="0 = full batch")
ap.add_argument("--steps", type=int, default=6000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--tag", default="")
args = ap.parse_args()

torch.manual_seed(args.seed)
xtr_i, ytr, xte_i, yte = modular(args.p, args.op, args.frac, args.seed)
onehot = lambda X: torch.cat([F.one_hot(X[:, 0], args.p), F.one_hot(X[:, 1], args.p)], 1).float()
xtr, xte = onehot(xtr_i), onehot(xte_i)
# same function class as E1's ModMLP: [one-hot a | one-hot b] -> block-diagonal shared embedding is replaced by a
# linear layer to 2*d (untied embeddings), then ReLU layers
d = 128
m = LoopMLP(2 * args.p, (2 * d, *args.widths), args.p, linear_first=True).cuda()
with torch.no_grad():
    W = torch.zeros(2 * d, 2 * args.p); E = torch.randn(args.p, d)
    W[:d, :args.p] = E.T; W[d:, args.p:] = E.T      # tied at init like a shared embedding table
    m.lins[0].weight.copy_(W); m.lins[0].bias.zero_()
tr = LoopTrainer(m, xtr, ytr, lr=args.lr, wd=args.wd, betas=(0.9, 0.999), mode=args.mode, alpha=args.alpha,
                 seed=args.seed, opt="sgd")
N = xtr.shape[0]
bs = N if args.bs == 0 else args.bs
first, rows, t0 = {}, [], time.time()
gen = torch.Generator(device="cuda").manual_seed(args.seed)
for s in range(args.steps + 1):
    idx = torch.arange(N, device="cuda") if bs == N else torch.randint(0, N, (bs,), device="cuda", generator=gen)
    loss = tr.step(idx)
    if s % 100 == 0:
        with torch.no_grad():
            tra = (m(xtr).argmax(1) == ytr).float().mean().item()
            tea = (m(xte).argmax(1) == yte).float().mean().item()
        rows.append(dict(step=s, loss=loss, train=tra, test=tea))
        for key, thr in (("train99", .99), ("test50", .5), ("test90", .9), ("test99", .99)):
            val = tra if key == "train99" else tea
            if key not in first and val >= thr:
                first[key] = s
out = dict(args=vars(args), first=first, final_test=rows[-1]["test"], rows=rows, secs=round(time.time() - t0, 1))
os.makedirs("results/e3", exist_ok=True)
w = "x".join(map(str, args.widths))
json.dump(out, open(f"results/e3/{args.mode}_{args.op}_frac{args.frac}_w{w}_a{args.alpha}_lr{args.lr}_wd{args.wd}_bs{args.bs}_s{args.seed}{args.tag}.json", "w"))
print(json.dumps(dict(mode=args.mode, op=args.op, frac=args.frac, widths=args.widths, alpha=args.alpha, lr=args.lr,
                      wd=args.wd, bs=args.bs, seed=args.seed, first=first, final_test=round(rows[-1]["test"], 4),
                      secs=out["secs"])))
