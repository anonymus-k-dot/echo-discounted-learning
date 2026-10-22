"""A2: lambda_min of the normalized per-example loss-gradient Gram at initialization, C0 = D^-1/2 K0 D^-1/2,
for several architectures. Hypothesis (from the tied MLP): the working alpha is ~0.6 * lambda_min(C0).
Per-example gradients via torch.func (shared parameters, vmapped over examples)."""
import sys, os, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import torch, torch.nn as nn, torch.nn.functional as F
from torch.func import functional_call, vmap, grad
from data import modular, mnist, subsample_balanced
from models import ModMLP, ModTransformer
from loop_mlp import LoopMLP


def lam_min_init(model, x, y, chunk=256):
    params = {n: p.detach() for n, p in model.named_parameters()}
    loss_one = lambda pr, xi, yi: F.cross_entropy(functional_call(model, pr, (xi.unsqueeze(0),)), yi.unsqueeze(0))
    pe = vmap(grad(loss_one), in_dims=(None, 0, 0))
    G = []
    for s in range(0, x.shape[0], chunk):
        g = pe(params, x[s:s + chunk], y[s:s + chunk])
        G.append(torch.cat([v.flatten(1) for v in g.values()], 1))
    G = torch.cat(G, 0)
    K = G @ G.T
    dg = K.diagonal().clamp_min(1e-30).rsqrt()
    ev = torch.linalg.eigvalsh((dg[:, None] * K * dg[None, :]).double())
    return ev[0].item(), ev[int(0.01 * len(ev))].item()


def untied_mlp(p=97, d=128, widths=(512,), seed=0):
    torch.manual_seed(seed)
    m = LoopMLP(2 * p, (2 * d, *widths), p, linear_first=True)
    with torch.no_grad():
        W = torch.zeros(2 * d, 2 * p); E = torch.randn(p, d)
        W[:d, :p] = E.T; W[d:, p:] = E.T
        m.lins[0].weight.copy_(W); m.lins[0].bias.zero_()
    return m.cuda()


rows = []
xtr, ytr, _, _ = modular(97, "add", 0.3, 0)
onehot = torch.cat([F.one_hot(xtr[:, 0], 97), F.one_hot(xtr[:, 1], 97)], 1).float()
cases = [
    ("tied MLP (E1; alpha 0.25 works, 0.5 unstable)", lambda: (torch.manual_seed(0), ModMLP(97).cuda())[1], xtr),
    ("untied MLP (E3; alpha 0.1 works, 0.25 unstable)", lambda: untied_mlp(), onehot),
    ("transformer d=128", lambda: (torch.manual_seed(0), ModTransformer(97).cuda())[1], xtr),
    ("transformer d=64", lambda: (torch.manual_seed(0), ModTransformer(97, d=64, d_mlp=256).cuda())[1], xtr),
]
for f in (0.2, 0.15):
    xf, yf, _, _ = modular(97, "add", f, 0)
    cases.append((f"tied MLP frac {f}", lambda: (torch.manual_seed(0), ModMLP(97).cuda())[1], xf))
for name, mk, x in cases:
    y = ytr if x.shape[0] == xtr.shape[0] else modular(97, "add", {1881: 0.2, 1411: 0.15}[x.shape[0]], 0)[1]
    lo, p1 = lam_min_init(mk(), x, y)
    rows.append(dict(case=name, n=x.shape[0], lam_min=lo, lam_p01=p1, alpha_pred=0.6 * lo))
    print(json.dumps(rows[-1]), flush=True)
xa, ya, _, _ = mnist()
xm, ym = subsample_balanced(xa, ya, 100, seed=0)
xm = xm - xm.mean(0, keepdim=True)
torch.manual_seed(0)
lo, p1 = lam_min_init(LoopMLP(784, (512, 512), 10).cuda(), xm, ym)
rows.append(dict(case="MNIST MLP 784-512-512-10 (1k)", n=1000, lam_min=lo, lam_p01=p1, alpha_pred=0.6 * lo))
print(json.dumps(rows[-1]), flush=True)
os.makedirs("results/a2", exist_ok=True)
json.dump(rows, open("results/a2/lambda_init.json", "w"), indent=1)
