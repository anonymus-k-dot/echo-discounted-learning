"""Scalable leave-one-out-point (LOOP) learning for MLPs with factored self-echo.

Per-example gradient of a linear layer is rank-1: G_i = delta_i a_i^T. Under Adam the attributed step is
-(lr/B) * P ⊙ (delta_i a_i^T) with P the elementwise preconditioner. We store, per example and layer,
    c_i  = sum_t -(lr/B) delta_{i,t}      (decayed by weight decay; the bias self-echo is P_b ⊙ c_i)
    abar_i = activation a_i at the latest visit
and use U_i ≈ P ⊙ (c_i abar_i^T) with the *current* P. The leave-one-out-point layer is then
    (W - alpha*U_i) a + b - alpha*P_b⊙c_i  =  W a + b - alpha*[ c_i ⊙ (P (abar_i ⊙ a)) + P_b ⊙ c_i ]
which costs one extra matmul per layer. For the first layer abar_i = x_i exactly.
Weight decay on the echo is applied lazily: c_i is stored with the step of its last update.

mode: "std" (alpha used only for the meter), "loop" (gradients at LOO point), "rand" (other example's echo).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoopMLP(nn.Module):
    """MLP; with linear_first=True the first layer has no nonlinearity (an embedding when inputs are one-hot)."""

    def __init__(self, din=784, widths=(256, 256), dout=10, linear_first=False):
        super().__init__()
        dims = [din, *widths, dout]
        self.lins = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.linear_first = linear_first

    def act(self, k, z):
        if k == len(self.lins) - 1 or (k == 0 and self.linear_first):
            return z
        return F.relu(z)

    def forward(self, x):
        for k, l in enumerate(self.lins):
            x = self.act(k, l(x))
        return x


class LoopTrainer:
    def __init__(self, model, xtr, ytr, lr=1e-3, wd=0.0, betas=(0.9, 0.999), eps=1e-8, mode="loop",
                 alpha=1.0, cap=None, seed=0, store_first_input=False, opt="adam"):
        self.m, self.x, self.y = model, xtr, ytr
        self.N = xtr.shape[0]
        self.lr, self.wd, self.b1, self.b2, self.eps = lr, wd, betas[0], betas[1], eps
        self.mode, self.alpha, self.cap, self.opt = mode, alpha, cap, opt
        dev = xtr.device
        self.L = len(model.lins)
        self.c = [torch.zeros(self.N, l.out_features, device=dev) for l in model.lins]
        # abar for layer 0 is the input itself unless asked to store it
        self.abar = [None if (k == 0 and not store_first_input) else torch.zeros(self.N, l.in_features, device=dev)
                     for k, l in enumerate(model.lins)]
        self.last = torch.zeros(self.N, dtype=torch.long, device=dev)   # step of last echo update
        self.params = [p for l in model.lins for p in (l.weight, l.bias)]
        self.mom = [torch.zeros_like(p) for p in self.params]
        self.var = [torch.zeros_like(p) for p in self.params]
        self.t = 0
        gen = torch.Generator(device="cpu").manual_seed(seed + 999)
        self.perm = torch.randperm(self.N, generator=gen).to(dev)

    # ---- preconditioner (current) ----
    def precond(self):
        if self.opt == "sgd":
            return [torch.ones_like(p) for p in self.params]
        if self.t == 0:
            return [torch.ones_like(p) for p in self.params]
        bc2 = 1 - self.b2 ** self.t
        return [1.0 / (torch.sqrt(v / bc2) + self.eps) for v in self.var]

    def _decay(self, idx):
        if self.wd == 0:
            return None
        return (1 - self.lr * self.wd) ** (self.t - self.last[idx]).float()

    def forward_at(self, idx, which, alpha=None, keep=False):
        """Logits for examples idx at theta ('theta') or at its LOO point ('own') / other's echo ('rand')."""
        al = self.alpha if alpha is None else alpha
        src = idx if which == "own" else (self.perm[idx] if which == "rand" else None)
        a = self.x[idx]
        P = self.precond() if src is not None else None
        dec = self._decay(src) if src is not None else None
        acts, pres = [], []
        for k, lin in enumerate(self.m.lins):
            z = lin(a)
            if src is not None:
                c = self.c[k][src]
                if dec is not None:
                    c = c * dec.unsqueeze(1)
                ab = self.x[src] if self.abar[k] is None else self.abar[k][src]
                PW, Pb = P[2 * k], P[2 * k + 1]
                z = z - al * (c * ((ab * a) @ PW.T) + c * Pb)
            if keep:
                z.retain_grad()
            acts.append(a); pres.append(z)
            a = self.m.act(k, z)
        return (a, acts, pres) if keep else a

    def step(self, idx):
        which = {"std": "theta", "loop": "own", "rand": "rand"}[self.mode]
        B = idx.shape[0]
        logits, acts, pres = self.forward_at(idx, which, keep=True)
        loss = F.cross_entropy(logits, self.y[idx], reduction="sum") / B
        for p in self.params:
            p.grad = None
        loss.backward()
        with torch.no_grad():
            self.t += 1
            bc1, bc2 = 1 - self.b1 ** self.t, 1 - self.b2 ** self.t
            for p, mo, va in zip(self.params, self.mom, self.var):
                g = p.grad
                p.mul_(1 - self.lr * self.wd)
                if self.opt == "sgd":                       # heavy-ball momentum, mu = betas[0]
                    mo.mul_(self.b1).add_(g)
                    p.add_(-self.lr * mo)
                else:
                    mo.mul_(self.b1).add_(g, alpha=1 - self.b1)
                    va.mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
                    p.add_(-(self.lr / bc1) * mo / (torch.sqrt(va / bc2) + self.eps))
            gain = self.lr / (1 - self.b1) if self.opt == "sgd" else self.lr   # eventual effect of one gradient
            # attribute: per-example delta = B * dL/dz (loss is a mean over the batch)
            dec = self._decay(idx)
            for k in range(self.L):
                delta = pres[k].grad * B
                c = self.c[k][idx]
                if dec is not None:
                    c = c * dec.unsqueeze(1)
                self.c[k][idx] = c - (gain / B) * delta
                if self.abar[k] is not None:
                    self.abar[k][idx] = acts[k].detach()
            self.last[idx] = self.t
            if self.cap is not None:
                self._cap(idx)
        return loss.item()

    def _cap(self, idx):
        n = torch.zeros(idx.shape[0], device=self.x.device)
        for k in range(self.L):
            ab = self.x[idx] if self.abar[k] is None else self.abar[k][idx]
            n += self.c[k][idx].pow(2).sum(1) * (1 + ab.pow(2).sum(1))
        s = torch.clamp(self.cap / (n.sqrt() + 1e-12), max=1.0).unsqueeze(1)
        for k in range(self.L):
            self.c[k][idx] *= s

    @torch.no_grad()
    def accuracy_at(self, which="own", chunk=4096, alpha=None):
        correct = 0
        for s in range(0, self.N, chunk):
            idx = torch.arange(s, min(s + chunk, self.N), device=self.x.device)
            correct += (self.forward_at(idx, which, alpha=alpha).argmax(1) == self.y[idx]).sum().item()
        return correct / self.N
