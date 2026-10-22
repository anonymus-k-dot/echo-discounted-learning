"""E4: echo-discounted learning with exact torch.func bookkeeping, for any architecture (MLP or transformer)."""
import sys, os, json, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch, torch.nn.functional as F
from data import modular
from models import ModMLP, ModTransformer
from loo_generic import EchoTrainer

ap = argparse.ArgumentParser()
ap.add_argument("--arch", default="tf", choices=["mlp", "tf"])
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
ap.add_argument("--chunk", type=int, default=256)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--tag", default="")
args = ap.parse_args()

torch.manual_seed(args.seed)
xtr, ytr, xte, yte = modular(97, args.op, args.frac, args.seed)
m = (ModMLP(97) if args.arch == "mlp" else ModTransformer(97)).cuda()
tr = EchoTrainer(m, xtr, ytr, lr=args.lr, wd=args.wd, mom=args.mom, mode=args.mode, alpha=args.alpha,
                 chunk=args.chunk, seed=args.seed, ls=args.ls, kappa=args.kappa)
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
out = dict(args=vars(args), first=first, final_test=rows[-1]["test"], rows=rows, lam=tr.lam_hist,
           secs=round(time.time() - t0, 1))
os.makedirs("results/e4", exist_ok=True)
json.dump(out, open(f"results/e4/{args.arch}_{args.mode}_{args.op}_frac{args.frac}_lr{args.lr}_wd{args.wd}_a{args.alpha}"
                    f"_k{args.kappa}_ls{args.ls}_s{args.seed}{args.tag}.json", "w"))
print("FIRST", json.dumps(dict(arch=args.arch, mode=args.mode, frac=args.frac, lr=args.lr, wd=args.wd, alpha=args.alpha,
                               kappa=args.kappa, first=first, final=round(rows[-1]["test"], 4), secs=out["secs"])))
