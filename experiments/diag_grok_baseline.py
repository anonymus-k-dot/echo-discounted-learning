"""Baseline grokking dynamics on modular addition: how long does memorization-before-generalization last
under AdamW, as a function of weight decay and train fraction? Establishes the testbed and its timing."""
import sys, os, json, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch, torch.nn.functional as F
from data import modular
from models import ModMLP

ap = argparse.ArgumentParser()
ap.add_argument("--p", type=int, default=97)
ap.add_argument("--frac", type=float, default=0.3)
ap.add_argument("--wd", type=float, default=1.0)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--steps", type=int, default=20000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--act", default="relu")
args = ap.parse_args()

torch.manual_seed(args.seed)
xtr, ytr, xte, yte = modular(args.p, "add", args.frac, args.seed)
m = ModMLP(args.p, act=args.act).cuda()
opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.98))
t0 = time.time()
hist = []
first_tr99 = first_te99 = None
for s in range(args.steps + 1):
    loss = F.cross_entropy(m(xtr), ytr)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    if s % 250 == 0:
        with torch.no_grad():
            tra = (m(xtr).argmax(1) == ytr).float().mean().item()
            tea = (m(xte).argmax(1) == yte).float().mean().item()
        if first_tr99 is None and tra > 0.99: first_tr99 = s
        if first_te99 is None and tea > 0.99: first_te99 = s
        hist.append((s, loss.item(), tra, tea))
        if s % 2000 == 0:
            print(f"step {s:6d} loss {loss.item():.4f} train {tra:.3f} test {tea:.3f}  ({time.time()-t0:.0f}s)", flush=True)
print(json.dumps(dict(args=vars(args), first_train99=first_tr99, first_test99=first_te99, secs=time.time() - t0)))
