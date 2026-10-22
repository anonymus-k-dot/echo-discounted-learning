"""Exact (first-order) per-example self-contribution tracking for the modular-arithmetic MLP
(embedding -> concat -> fc1 -> relu -> fc2), with an analytic per-example forward/backward.

For every training example i we store u_i, the cumulative parameter displacement attributed to that
example's own gradients. With immediate attribution of an AdamW step,
    u_i <- (1 - lr*wd) u_i - lr * P_t * g_i / N,      P_t = Adam preconditioner (bias-corrected),
so theta_t ~= decayed(theta_0) + sum_i u_i, and theta_t - u_i is a first-order estimate of the parameters
the network would have without example i ("leave-one-out point").
Storage: example i only touches embedding rows a_i, b_i, so its embedding contribution is 2 x d.

mode = "std":  gradients at theta (ordinary AdamW); u is tracked only as an internal LOO meter.
mode = "loo":  each example's gradient is evaluated at its own LOO point theta - alpha * u_i.
mode = "rand": control; example i is evaluated at theta - alpha * u_pi(i) (fixed random permutation pi):
               same perturbation statistics, wrong identity.
"""
import math
import torch
import torch.nn.functional as F


class LOOTrainer:
    def __init__(self, model, xtr, ytr, lr=1e-3, wd=1.0, betas=(0.9, 0.98), eps=1e-8,
                 mode="loo", alpha=1.0, chunk=1024, seed=0, cap=None, opt="adam", mom=0.9, ls=0.0,
                 adapt_kappa=None, adapt_every=100, alpha_max=1.0):
        self.m, self.x, self.y = model, xtr, ytr
        self.N, self.p = xtr.shape[0], model.fc2.out_features
        self.lr, self.wd, self.b1, self.b2, self.eps = lr, wd, betas[0], betas[1], eps
        self.mode, self.alpha, self.chunk, self.cap = mode, alpha, chunk, cap
        self.opt, self.mu, self.ls = opt, mom, ls
        self.kappa, self.adapt_every, self.alpha_max = adapt_kappa, adapt_every, alpha_max
        self.lam_hist = []
        self.P = dict(emb=model.emb.weight, w1=model.fc1.weight, b1=model.fc1.bias,
                      w2=model.fc2.weight, b2=model.fc2.bias)
        d, h = model.emb.weight.shape[1], model.fc1.out_features
        dev = xtr.device
        self.U = dict(emb=torch.zeros(self.N, 2, d, device=dev), w1=torch.zeros(self.N, h, 2 * d, device=dev),
                      b1=torch.zeros(self.N, h, device=dev), w2=torch.zeros(self.N, self.p, h, device=dev),
                      b2=torch.zeros(self.N, self.p, device=dev))
        self.mom = {k: torch.zeros_like(v) for k, v in self.P.items()}
        self.var = {k: torch.zeros_like(v) for k, v in self.P.items()}
        self.t = 0
        gen = torch.Generator(device="cpu").manual_seed(seed + 999)
        self.perm = torch.randperm(self.N, generator=gen).to(dev)
        self.same = (xtr[:, 0] == xtr[:, 1]).float()
        self.last_loo_acc = None

    def _src(self, idx, which):
        if which == "own":
            return slice(int(idx[0]), int(idx[-1]) + 1)   # idx is always a contiguous range
        return self.perm[idx]

    def forward(self, idx, which, need_cache=False):
        """Logits of examples idx at theta (which='theta') or at theta - alpha*u_src."""
        P, x = self.P, self.x[idx]
        a, b = x[:, 0], x[:, 1]
        ea, eb = P["emb"][a], P["emb"][b]
        W1, B1, W2, B2 = P["w1"], P["b1"], P["w2"], P["b2"]
        if which == "theta":
            e = torch.cat([ea, eb], 1)
            hp = e @ W1.T + B1
            hh = F.relu(hp)
            z = hh @ W2.T + B2
            cache = (None, e, hp, hh)
        else:
            src, al = self._src(idx, which), self.alpha
            ue = self.U["emb"][src]
            sm = self.same[idx].unsqueeze(1)
            ea = ea - al * (ue[:, 0] + sm * ue[:, 1])
            eb = eb - al * (ue[:, 1] + sm * ue[:, 0])
            e = torch.cat([ea, eb], 1)
            hp = e @ W1.T + B1 - al * (torch.einsum("chk,ck->ch", self.U["w1"][src], e) + self.U["b1"][src])
            hh = F.relu(hp)
            z = hh @ W2.T + B2 - al * (torch.einsum("cph,ch->cp", self.U["w2"][src], hh) + self.U["b2"][src])
            cache = (src, e, hp, hh)
        return (z, cache) if need_cache else z

    def step(self):
        which = {"std": "theta", "loo": "own", "rand": "rand"}[self.mode]
        gb = {k: torch.zeros_like(v) for k, v in self.P.items()}
        pending, tot, correct = [], 0.0, 0
        with torch.no_grad():
            for s in range(0, self.N, self.chunk):
                idx = torch.arange(s, min(s + self.chunk, self.N), device=self.x.device)
                z, (src, e, hp, hh) = self.forward(idx, which, need_cache=True)
                y = self.y[idx]
                lp = torch.log_softmax(z, 1)
                tgt = torch.full_like(lp, self.ls / self.p); tgt[torch.arange(len(idx)), y] += 1.0 - self.ls
                tot += -(tgt * lp).sum().item()
                correct += (z.argmax(1) == y).sum().item()
                dz = lp.exp() - tgt                                               # dL/dz per example
                dh = dz @ self.P["w2"]
                if src is not None:
                    dh = dh - self.alpha * torch.einsum("cp,cph->ch", dz, self.U["w2"][src])
                d1 = dh * (hp > 0).float()
                de = d1 @ self.P["w1"]
                if src is not None:
                    de = de - self.alpha * torch.einsum("ch,chk->ck", d1, self.U["w1"][src])
                dd = e.shape[1] // 2
                dea, deb = de[:, :dd], de[:, dd:]
                x = self.x[idx]
                gb["w2"] += dz.T @ hh; gb["b2"] += dz.sum(0)
                gb["w1"] += d1.T @ e;  gb["b1"] += d1.sum(0)
                gb["emb"].index_add_(0, x[:, 0], dea); gb["emb"].index_add_(0, x[:, 1], deb)
                pending.append((idx, dz, hh, d1, e, dea, deb))
            if self.kappa is not None and self.t % self.adapt_every == 0:
                lam = self._lam_min(pending)
                self.lam_hist.append((self.t, lam))
                self.alpha = float(min(self.alpha_max, self.kappa * lam))
            self.t += 1
            self.last_gbar = {k: v / self.N for k, v in gb.items()}
            self.last_loo_acc = correct / self.N if which == "own" else None
            bc1, bc2 = 1 - self.b1 ** self.t, 1 - self.b2 ** self.t
            pre = {}
            for k, v in self.P.items():
                g = gb[k] / self.N
                v.mul_(1 - self.lr * self.wd)
                if self.opt == "sgd":
                    self.mom[k].mul_(self.mu).add_(g)
                    v.add_(-self.lr * self.mom[k])
                    pre[k] = torch.ones((), device=v.device)
                else:
                    self.mom[k].mul_(self.b1).add_(g, alpha=1 - self.b1)
                    self.var[k].mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
                    pre[k] = 1.0 / (torch.sqrt(self.var[k] / bc2) + self.eps)
                    v.add_(-(self.lr / bc1) * self.mom[k] * pre[k])
            gain = self.lr / (1 - self.mu) if self.opt == "sgd" else self.lr
            dec, c = 1 - self.lr * self.wd, gain / self.N
            for idx, dz, hh, d1, e, dea, deb in pending:
                x, sl = self.x[idx], slice(int(idx[0]), int(idx[-1]) + 1)
                self.U["w2"][sl].mul_(dec).addcmul_((-c * dz).unsqueeze(2) * pre["w2"], hh.unsqueeze(1))
                self.U["b2"][sl].mul_(dec).add_(-c * dz * pre["b2"])
                self.U["w1"][sl].mul_(dec).addcmul_((-c * d1).unsqueeze(2) * pre["w1"], e.unsqueeze(1))
                self.U["b1"][sl].mul_(dec).add_(-c * d1 * pre["b1"])
                pa = pre["emb"][x[:, 0]] if pre["emb"].dim() else pre["emb"]
                pb = pre["emb"][x[:, 1]] if pre["emb"].dim() else pre["emb"]
                self.U["emb"][sl, 0].mul_(dec).add_(-c * dea * pa)
                self.U["emb"][sl, 1].mul_(dec).add_(-c * deb * pb)
            if self.cap is not None:
                self._apply_cap()
        return tot / self.N

    def _lam_min(self, pending):
        """lambda_min of the normalized per-example loss-gradient Gram at the evaluation points (all examples)."""
        cat = lambda j: torch.cat([q[j] for q in pending], 0)
        idx, dz, hh, d1, e, dea, deb = (cat(j) for j in range(7))
        x = self.x[idx]; a, b = x[:, 0], x[:, 1]
        K = (dz @ dz.T) * (hh @ hh.T + 1) + (d1 @ d1.T) * (e @ e.T + 1)
        eq = lambda u, v: (u[:, None] == v[None, :]).float()
        K += (dea @ dea.T) * eq(a, a) + (dea @ deb.T) * eq(a, b) + (deb @ dea.T) * eq(b, a) + (deb @ deb.T) * eq(b, b)
        dg = K.diagonal().clamp_min(1e-30).rsqrt()
        return torch.linalg.eigvalsh((dg[:, None] * K * dg[None, :]).double())[0].item()

    def _apply_cap(self):
        n2 = sum(self.U[k].flatten(1).pow(2).sum(1) for k in self.U)
        scale = torch.clamp(self.cap / (n2.sqrt() + 1e-12), max=1.0)
        for k in self.U:
            self.U[k].mul_(scale.view(-1, *[1] * (self.U[k].dim() - 1)))

    @torch.no_grad()
    def loo_accuracy(self, which="own"):
        correct = 0
        for s in range(0, self.N, self.chunk):
            idx = torch.arange(s, min(s + self.chunk, self.N), device=self.x.device)
            correct += (self.forward(idx, which).argmax(1) == self.y[idx]).sum().item()
        return correct / self.N

    @torch.no_grad()
    def invariant_error(self, theta0):
        """|| theta - decayed(theta0) - sum_i u_i || / || theta - decayed(theta0) || (attribution check)."""
        dec = (1 - self.lr * self.wd) ** self.t
        num = den = 0.0
        for k, v in self.P.items():
            d = v - dec * theta0[k]
            if k == "emb":
                s = torch.zeros_like(v)
                s.index_add_(0, self.x[:, 0], self.U["emb"][:, 0]); s.index_add_(0, self.x[:, 1], self.U["emb"][:, 1])
            else:
                s = self.U[k].sum(0)
            num += (d - s).pow(2).sum().item()
            den += d.pow(2).sum().item()
        return math.sqrt(num / max(den, 1e-30))
