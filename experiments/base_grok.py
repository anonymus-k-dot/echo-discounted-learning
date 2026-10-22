"""Strong baselines for accelerating grokking on the same setup as E1:
  adamw     : plain AdamW (reference)
  grokfast  : Grokfast-EMA (Lee et al. 2024): g <- g + lam * EMA_alpha(g), then AdamW
  orthograd : ⊥Grad (Prieto et al. 2025): remove the gradient component parallel to each weight tensor
Hyperparameters of each baseline are swept; the best setting per method is reported."""
import sys, os, json, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch, torch.nn.functional as F
from data import modular
from models import ModMLP

ap = argparse.ArgumentParser()
ap.add_argument("--method", default="grokfast")
ap.add_argument("--p", type=int, default=97)
ap.add_argument("--frac", type=float, default=0.3)
ap.add_argument("--wd", type=float, default=1.0)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--gf_alpha", type=float, default=0.98)
ap.add_argument("--gf_lam", type=float, default=2.0)
ap.add_argument("--steps", type=int, default=20000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--op", default="add")
ap.add_argument("--opt", default="adamw")
ap.add_argument("--mom", type=float, default=0.9)
ap.add_argument("--ls", type=float, default=0.0)
ap.add_argument("--full", action="store_true")
ap.add_argument("--alpha", type=float, default=0.25)
ap.add_argument("--frate", type=float, default=1.0, help="floop: integration rate of the per-example logit echo")
args = ap.parse_args()

torch.manual_seed(args.seed)
xtr, ytr, xte, yte = modular(args.p, args.op, args.frac, args.seed)
m = ModMLP(args.p).cuda()
opt = (torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.98)) if args.opt == "adamw"
       else torch.optim.SGD(m.parameters(), lr=args.lr, momentum=args.mom, weight_decay=0.0))
muon_buf = {n: torch.zeros_like(p) for n, p in m.named_parameters()}
ema = {n: torch.zeros_like(p) for n, p in m.named_parameters()}
first, t0 = {}, time.time()
for s in range(args.steps + 1):
    if args.method == "floop":
        if s == 0:
            echo = torch.zeros(xtr.shape[0], args.p, device="cuda")
        z = m(xtr)
        zt = z - args.alpha * echo                      # logits as if a fraction alpha of own echo were absent
        loss = F.cross_entropy(zt, ytr, label_smoothing=args.ls)
        with torch.no_grad():                           # integrate the residual at the discounted point
            res = torch.softmax(zt, 1); res[torch.arange(len(ytr)), ytr] -= 1.0
            decay = (1 - args.lr * args.wd) if args.opt == "sgd" else (1 - args.lr * args.wd)
            echo.mul_(decay).add_(res, alpha=-args.frate)
    else:
        loss = F.cross_entropy(m(xtr), ytr, label_smoothing=args.ls)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if args.method == "grokfast":
                ema[n].mul_(args.gf_alpha).add_(p.grad, alpha=1 - args.gf_alpha)
                p.grad.add_(ema[n], alpha=args.gf_lam)
            elif args.method == "orthograd":
                w, g = p.flatten(), p.grad.flatten()
                proj = (w @ g) / (w @ w + 1e-30)
                g_orth = g - proj * w
                p.grad.copy_((g_orth * (g.norm() / (g_orth.norm() + 1e-30))).view_as(p))
    if args.method == "muon":   # orthogonalized momentum updates for matrices (embedding treated as a matrix too)
        with torch.no_grad():
            for n, p_ in m.named_parameters():
                buf = muon_buf[n]; buf.mul_(args.mom).add_(p_.grad)
                g = p_.grad.add(buf, alpha=args.mom)          # Nesterov
                if g.dim() == 2:
                    U, S, Vh = torch.linalg.svd(g, full_matrices=False)
                    upd = (U @ Vh) * max(1.0, g.shape[0] / g.shape[1]) ** 0.5
                else:
                    upd = g / (g.norm() + 1e-12) * g.numel() ** 0.5 * 0.1
                p_.mul_(1 - args.lr * args.wd)
                p_.add_(upd, alpha=-args.lr)
    else:
        if args.opt == "sgd":
            with torch.no_grad():
                for p_ in m.parameters(): p_.mul_(1 - args.lr * args.wd)
        opt.step()
    if s % 100 == 0:
        with torch.no_grad():
            tra = (m(xtr).argmax(1) == ytr).float().mean().item()
            tea = (m(xte).argmax(1) == yte).float().mean().item()
        for key, val, thr in (("train99", tra, .99), ("test50", tea, .5), ("test90", tea, .9), ("test99", tea, .99)):
            if key not in first and val >= thr:
                first[key] = s
        if "test99" in first and s >= first["test99"] + 500 and not args.full:
            break
print(json.dumps(dict(args=vars(args), first=first, final_test=tea, secs=round(time.time() - t0, 1))))
