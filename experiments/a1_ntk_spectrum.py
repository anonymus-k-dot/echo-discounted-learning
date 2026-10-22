"""A1: test the stability prediction alpha* ~ lambda_min of the normalized tangent kernel C = D^-1/2 K D^-1/2.

K is computed exactly on all training pairs from analytic per-example factors (no materialized gradients):
  K_loss[i,j] = g_i . g_j          with g_i = grad of CE loss of example i   (residual-weighted kernel)
  K_lab[i,j]  = J_i,y_i . J_j,y_j  (Jacobian of the true-class logit)
along an ordinary SGD run (lr 1.0, wd 0.01, momentum 0.9, full batch), at several checkpoints."""
import sys, os, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch, torch.nn.functional as F
from data import modular
from models import ModMLP

ap = argparse.ArgumentParser()
ap.add_argument("--frac", type=float, default=0.3)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--lr", type=float, default=1.0)
ap.add_argument("--wd", type=float, default=0.01)
ap.add_argument("--checkpoints", type=int, nargs="*", default=[0, 50, 100, 200, 300, 600, 1000, 1600, 2500])
args = ap.parse_args()

torch.manual_seed(args.seed)
xtr, ytr, xte, yte = modular(97, "add", args.frac, args.seed)
m = ModMLP(97).cuda()
opt = torch.optim.SGD(m.parameters(), lr=args.lr, momentum=0.9)


def factors(dz):
    """Per-example backward factors given dL/dz rows dz (N, C) at the current parameters."""
    a, b = xtr[:, 0], xtr[:, 1]
    e = torch.cat([m.emb.weight[a], m.emb.weight[b]], 1)
    hp = e @ m.fc1.weight.T + m.fc1.bias
    hh = F.relu(hp)
    d1 = (dz @ m.fc2.weight) * (hp > 0).float()
    de = d1 @ m.fc1.weight
    d = e.shape[1] // 2
    return e, hh, d1, de[:, :d], de[:, d:]


def gram(dz):
    e, hh, d1, dea, deb = factors(dz)
    a, b = xtr[:, 0], xtr[:, 1]
    K = (dz @ dz.T) * (hh @ hh.T + 1) + (d1 @ d1.T) * (e @ e.T + 1)
    eq = lambda u, v: (u[:, None] == v[None, :]).float()
    K += (dea @ dea.T) * eq(a, a) + (dea @ deb.T) * eq(a, b) + (deb @ dea.T) * eq(b, a) + (deb @ deb.T) * eq(b, b)
    return K


def lam_min_normalized(K):
    dg = K.diagonal().clamp_min(1e-30).rsqrt()
    C = dg[:, None] * K * dg[None, :]
    ev = torch.linalg.eigvalsh(C.double())
    return ev[0].item(), ev[int(0.01 * len(ev))].item(), ev[-1].item()


rows = []
for s in range(max(args.checkpoints) + 1):
    if s in args.checkpoints:
        with torch.no_grad():
            z = m(xtr)
            p = torch.softmax(z, 1)
            dz_loss = p.clone(); dz_loss[torch.arange(len(ytr)), ytr] -= 1
            dz_lab = F.one_hot(ytr, 97).float()
            r = dict(step=s, test=(m(xte).argmax(1) == yte).float().mean().item(),
                     train=(z.argmax(1) == ytr).float().mean().item())
            for name, dz in (("loss", dz_loss), ("label", dz_lab)):
                lo, p1, hi = lam_min_normalized(gram(dz))
                r[f"{name}_lmin"], r[f"{name}_p01"], r[f"{name}_lmax"] = lo, p1, hi
            rows.append(r)
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
    loss = F.cross_entropy(m(xtr), ytr)
    opt.zero_grad(); loss.backward()
    with torch.no_grad():
        for q in m.parameters():
            q.mul_(1 - args.lr * args.wd)
    opt.step()
os.makedirs("results/a1", exist_ok=True)
json.dump(rows, open(f"results/a1/ntk_spectrum_frac{args.frac}_s{args.seed}.json", "w"))
