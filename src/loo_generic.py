"""Exact per-example echo bookkeeping for ANY nn.Module (small datasets), using torch.func.

Echo-discounted learning (EDL, formerly "LOOP"): each example's gradient is evaluated at theta - alpha * u_i,
where u_i is the cumulative parameter displacement attributed to that example's own past gradients.
Optimizer: full-batch or mini-batch SGD with heavy-ball momentum mu and decoupled weight decay; the eventual
effect of one gradient, lr/(1-mu), is attributed immediately (exact up to the momentum tail).

mode: "std" (alpha only used by meters), "edl" (gradients at own echo-discounted point), "rand" (another
example's echo; identity control)."""
import math
import torch
import torch.nn.functional as F
from torch.func import functional_call, vmap, grad


class EchoTrainer:
    def __init__(self, model, xtr, ytr, lr=1.0, wd=0.02, mom=0.9, mode="edl", alpha=0.25, chunk=256, seed=0,
                 ls=0.0, kappa=None, adapt_every=100, alpha_max=1.0):
        self.m, self.x, self.y = model, xtr, ytr
        self.N = xtr.shape[0]
        self.lr, self.wd, self.mu, self.mode, self.alpha, self.chunk, self.ls = lr, wd, mom, mode, alpha, chunk, ls
        self.kappa, self.adapt_every, self.alpha_max = kappa, adapt_every, alpha_max
        self.names = [n for n, _ in model.named_parameters()]
        self.P = dict(model.named_parameters())
        dev = xtr.device
        self.U = {n: torch.zeros(self.N, *p.shape, device=dev) for n, p in self.P.items()}
        self.buf = {n: torch.zeros_like(p) for n, p in self.P.items()}
        self.t = 0
        gen = torch.Generator(device="cpu").manual_seed(seed + 999)
        self.perm = torch.randperm(self.N, generator=gen).to(dev)
        self.last_loo_acc, self.lam_hist = None, []

        def loss_one(params, x, y):
            z = functional_call(self.m, params, (x.unsqueeze(0),))
            return F.cross_entropy(z, y.unsqueeze(0), label_smoothing=self.ls), z.squeeze(0)

        self._pe_grad = vmap(grad(loss_one, has_aux=True), in_dims=(0, 0, 0))
        self._pe_fwd = vmap(lambda params, x: functional_call(self.m, params, (x.unsqueeze(0),)).squeeze(0),
                            in_dims=(0, 0))

    def _params_for(self, idx, which, alpha):
        if which == "theta":
            return {n: p.detach().unsqueeze(0).expand(len(idx), *p.shape) for n, p in self.P.items()}
        src = idx if which == "own" else self.perm[idx]
        return {n: p.detach().unsqueeze(0) - alpha * self.U[n][src] for n, p in self.P.items()}

    def step(self, idx=None):
        which = {"std": "theta", "edl": "own", "rand": "rand"}[self.mode]
        idx = torch.arange(self.N, device=self.x.device) if idx is None else idx
        B = len(idx)
        gsum = {n: torch.zeros_like(p) for n, p in self.P.items()}
        G = {n: torch.empty(B, *p.shape, device=p.device) for n, p in self.P.items()}
        correct, flat = 0, []
        for s in range(0, B, self.chunk):
            sub = idx[s:s + self.chunk]
            g, z = self._pe_grad(self._params_for(sub, which, self.alpha), self.x[sub], self.y[sub])
            correct += (z.argmax(1) == self.y[sub]).sum().item()
            for n in self.names:
                G[n][s:s + len(sub)] = g[n]
                gsum[n] += g[n].sum(0)
        if self.kappa is not None and self.t % self.adapt_every == 0:
            flatG = torch.cat([G[n].flatten(1) for n in self.names], 1)
            K = flatG @ flatG.T
            dg = K.diagonal().clamp_min(1e-30).rsqrt()
            lam = torch.linalg.eigvalsh((dg[:, None] * K * dg[None, :]).double())[0].item()
            self.lam_hist.append((self.t, lam))
            self.alpha = float(min(self.alpha_max, self.kappa * lam))
            del flatG, K
        self.t += 1
        self.last_loo_acc = correct / B if which == "own" else None
        gain, dec = self.lr / (1 - self.mu), 1 - self.lr * self.wd
        with torch.no_grad():
            for n, p in self.P.items():
                g = gsum[n] / B
                p.mul_(dec)
                self.buf[n].mul_(self.mu).add_(g)
                p.add_(self.buf[n], alpha=-self.lr)
                self.U[n].mul_(dec)
                self.U[n][idx] -= (gain / B) * G[n]
        return None

    @torch.no_grad()
    def accuracy_at(self, which="own", alpha=None):
        al = self.alpha if alpha is None else alpha
        correct = 0
        for s in range(0, self.N, self.chunk):
            idx = torch.arange(s, min(s + self.chunk, self.N), device=self.x.device)
            z = self._pe_fwd(self._params_for(idx, which, al), self.x[idx])
            correct += (z.argmax(1) == self.y[idx]).sum().item()
        return correct / self.N
