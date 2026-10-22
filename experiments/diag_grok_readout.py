"""During the grokking plateau, where does memorization live: in the readout or in the features?

At checkpoints we compare
  (A) the trained network's test accuracy,
  (B) a ridge readout re-fit on the *training* hidden features (closed form, training data only),
  (C) a ridge readout fit on a random half of the training pairs and scored on the other half
      (internal cross-validation; training data only).
If (B)/(C) >> (A) during the plateau, generalizing features exist early and the SGD-trained readout hides them.
"""
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
ap.add_argument("--steps", type=int, default=12000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--every", type=int, default=500)
args = ap.parse_args()


def ridge_fit(H, y, p, lam):
    Y = F.one_hot(y, p).float() - 1.0 / p
    Hc = torch.cat([H, torch.ones(H.shape[0], 1, device=H.device)], 1)
    A = Hc.T @ Hc + lam * torch.eye(Hc.shape[1], device=H.device)
    return torch.linalg.solve(A, Hc.T @ Y)


def ridge_acc(W, H, y):
    Hc = torch.cat([H, torch.ones(H.shape[0], 1, device=H.device)], 1)
    return ((Hc @ W).argmax(1) == y).float().mean().item()


torch.manual_seed(args.seed)
xtr, ytr, xte, yte = modular(args.p, "add", args.frac, args.seed)
m = ModMLP(args.p).cuda()
opt = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.98))
g = torch.Generator(device="cuda").manual_seed(123)
half = torch.rand(xtr.shape[0], generator=g, device="cuda") < 0.5
rows = []
for s in range(args.steps + 1):
    loss = F.cross_entropy(m(xtr), ytr)
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    if s % args.every == 0:
        with torch.no_grad():
            Htr, Hte = m.hidden(xtr), m.hidden(xte)
            scale = Htr.pow(2).mean().item()
            A = (m(xte).argmax(1) == yte).float().mean().item()
            row = dict(step=s, train=(m(xtr).argmax(1) == ytr).float().mean().item(), test=A)
            for lam in (1e-3, 1e-1, 1e1):
                l = lam * scale * Htr.shape[0]
                W = ridge_fit(Htr, ytr, args.p, l)
                row[f"B_refit_test_lam{lam}"] = ridge_acc(W, Hte, yte)
                Wc = ridge_fit(Htr[half], ytr[half], args.p, l)
                row[f"C_cv_lam{lam}"] = ridge_acc(Wc, Htr[~half], ytr[~half])
            rows.append(row)
            print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
os.makedirs("results", exist_ok=True)
json.dump(rows, open(f"results/diag_grok_readout_seed{args.seed}.json", "w"))
