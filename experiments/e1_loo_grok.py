"""E1: leave-one-out-point (LOO) learning on modular addition, exact first-order self-contributions.

Questions:
 (a) In ordinary AdamW, does the internal LOO meter (accuracy of f(x_i; theta - u_i) on training examples)
     track true test accuracy, i.e. does the stored self-contribution capture memorization?
 (b) If each example's gradient is computed at its own LOO point, does generalization arrive earlier,
     and at train fractions where AdamW does not grok within budget?
 (c) Control: subtracting another example's contribution (same perturbation size, wrong identity).
"""
import sys, os, json, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch, torch.nn.functional as F
from data import modular
from models import ModMLP
from loo_toy import LOOTrainer

ap = argparse.ArgumentParser()
ap.add_argument("--mode", default="loo", choices=["std", "loo", "rand"])
ap.add_argument("--p", type=int, default=97)
ap.add_argument("--frac", type=float, default=0.3)
ap.add_argument("--wd", type=float, default=1.0)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--alpha", type=float, default=1.0)
ap.add_argument("--steps", type=int, default=12000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--every", type=int, default=500)
ap.add_argument("--tag", default="")
ap.add_argument("--opt", default="adam")
ap.add_argument("--ls", type=float, default=0.0)
ap.add_argument("--kappa", type=float, default=None)
ap.add_argument("--op", default="add")
ap.add_argument("--meter_alphas", type=float, nargs="*", default=[])
ap.add_argument("--mom", type=float, default=0.9)
args = ap.parse_args()

torch.manual_seed(args.seed)
xtr, ytr, xte, yte = modular(args.p, args.op, args.frac, args.seed)
m = ModMLP(args.p).cuda()
theta0 = {k: v.detach().clone() for k, v in dict(emb=m.emb.weight, w1=m.fc1.weight, b1=m.fc1.bias,
                                                  w2=m.fc2.weight, b2=m.fc2.bias).items()}
tr = LOOTrainer(m, xtr, ytr, lr=args.lr, wd=args.wd, mode=args.mode, alpha=args.alpha, seed=args.seed, opt=args.opt, mom=args.mom, ls=args.ls, adapt_kappa=args.kappa)
rows, t0 = [], time.time()
first = {}
for s in range(args.steps + 1):
    loss = tr.step()
    if s % args.every == 0:
        with torch.no_grad():
            tra = (m(xtr).argmax(1) == ytr).float().mean().item()
            tea = (m(xte).argmax(1) == yte).float().mean().item()
        row = dict(step=s, loss=loss, train=tra, test=tea, alpha=tr.alpha,
                   loo_own=(tr.last_loo_acc if args.mode == "loo" else None))
        # oracle probe (analysis only): cosine between this step's mean gradient and the test-loss gradient
        m.zero_grad(set_to_none=True)
        F.cross_entropy(m(xte), yte).backward()
        names = dict(emb=m.emb.weight, w1=m.fc1.weight, b1=m.fc1.bias, w2=m.fc2.weight, b2=m.fc2.bias)
        num = sum((tr.last_gbar[k] * v.grad).sum() for k, v in names.items())
        den = (sum(tr.last_gbar[k].pow(2).sum() for k in names) * sum(v.grad.pow(2).sum() for v in names.values())).sqrt()
        row["cos_test"] = (num / (den + 1e-30)).item()
        m.zero_grad(set_to_none=True)
        with torch.no_grad():     # mechanism probes: Fourier concentration of embeddings, weight norm
            Fe = torch.fft.rfft(m.emb.weight, dim=0).abs().pow(2).sum(1)[1:]
            row["fourier_top5"] = (Fe.sort(descending=True).values[:5].sum() / Fe.sum()).item()
            row["wnorm"] = sum(q.pow(2).sum() for q in m.parameters()).sqrt().item()
        if s % 500 == 0:
            a0 = tr.alpha
            for am in args.meter_alphas:
                tr.alpha = am
                row[f"meter_a{am}"] = tr.loo_accuracy("own")
            tr.alpha = a0
            row["loo_rand"] = tr.loo_accuracy("rand")
        for key, val in (("train99", tra), ("test50", tea), ("test90", tea), ("test99", tea)):
            thr = {"test50": 0.5, "test90": 0.9}.get(key, 0.99)
            if key not in first and val >= thr:
                first[key] = s
        rows.append(row)
        if s % 1000 == 0:
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}),
                  f"{time.time()-t0:.0f}s", flush=True)
out = dict(args=vars(args), first=first, rows=rows, secs=time.time() - t0)
os.makedirs("results/e1", exist_ok=True)
name = f"results/e1/{args.opt}_{args.mode}{'' if args.op == 'add' else '_' + args.op}_frac{args.frac}_lr{args.lr}_wd{args.wd}_a{args.alpha}_ls{args.ls}_k{args.kappa}_s{args.seed}{args.tag}.json"
json.dump(out, open(name, "w"))
print("FIRST", json.dumps(first), f"{time.time()-t0:.0f}s")
