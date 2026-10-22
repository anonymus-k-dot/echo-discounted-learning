"""E5: echo-discounted learning with exact layer-wise bookkeeping (src/echo_layers.py) for an MLP or a
one-layer transformer on modular arithmetic. SGD + heavy-ball momentum, decoupled weight decay, full batch."""
import sys, os, json, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch, torch.nn.functional as F
from data import modular
from echo_layers import EchoMLP, EchoTransformer, EchoTrainer, EchoUntiedMLP

ap = argparse.ArgumentParser()
ap.add_argument("--arch", default="tf", choices=["mlp", "tf", "untied"])
ap.add_argument("--d", type=int, default=128)
ap.add_argument("--opt", default="sgd")
ap.add_argument("--freeze_at", type=int, default=1000)
ap.add_argument("--mode", default="edl", choices=["std", "edl", "rand"])
ap.add_argument("--op", default="add")
ap.add_argument("--frac", type=float, default=0.3)
ap.add_argument("--lr", type=float, default=1.0)
ap.add_argument("--wd", type=float, default=0.02)
ap.add_argument("--mom", type=float, default=0.9)
ap.add_argument("--alpha", type=float, default=0.25)
ap.add_argument("--kappa", type=float, default=None)
ap.add_argument("--ls", type=float, default=0.0)
ap.add_argument("--steps", type=int, default=4000)
ap.add_argument("--every", type=int, default=100)
ap.add_argument("--chunk", type=int, default=1024)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--tag", default="")
args = ap.parse_args()

torch.manual_seed(args.seed)
xtr, ytr, xte, yte = modular(97, args.op, args.frac, args.seed)
m = {"mlp": lambda: EchoMLP(97), "untied": lambda: EchoUntiedMLP(97),
     "tf": lambda: EchoTransformer(97, d=args.d, d_mlp=4 * args.d)}[args.arch]().cuda()
tr = EchoTrainer(m, xtr, ytr, lr=args.lr, wd=args.wd, mom=args.mom, mode=args.mode, alpha=args.alpha,
                 chunk=args.chunk, seed=args.seed, ls=args.ls, kappa=args.kappa,
                 opt=args.opt, freeze_at=args.freeze_at)
first, rows, t0 = {}, [], time.time()
for s in range(args.steps + 1):
    tr.step()
    if s % args.every == 0:
        with torch.no_grad():
            tra = (m(xtr).argmax(1) == ytr).float().mean().item()
            tea = (m(xte).argmax(1) == yte).float().mean().item()
        rows.append(dict(step=s, train=tra, test=tea, alpha=tr.alpha, loo=tr.last_loo_acc))
        for key, thr in (("train99", .99), ("test50", .5), ("test90", .9), ("test99", .99)):
            val = tra if key == "train99" else tea
            if key not in first and val >= thr:
                first[key] = s
        if s % (args.every * 10) == 0:
            print(json.dumps(rows[-1]), f"{time.time()-t0:.0f}s", flush=True)
        if not (tra == tra):   # NaN guard
            break
out = dict(args=vars(args), first=first, final_test=rows[-1]["test"], rows=rows, lam=tr.lam_hist,
           secs=round(time.time() - t0, 1))
os.makedirs("results/e5", exist_ok=True)
json.dump(out, open(f"results/e5/{args.arch}{args.d if args.arch == 'tf' else ''}_{args.opt}_{args.mode}_{args.op}_frac{args.frac}_lr{args.lr}_wd{args.wd}_a{args.alpha}"
                    f"_k{args.kappa}_ls{args.ls}_s{args.seed}{args.tag}.json", "w"))
print("FIRST", json.dumps(dict(arch=args.arch, mode=args.mode, op=args.op, frac=args.frac, lr=args.lr, wd=args.wd,
                               alpha=args.alpha, kappa=args.kappa, ls=args.ls, first=first,
                               final=round(rows[-1]["test"], 4), secs=out["secs"])))
