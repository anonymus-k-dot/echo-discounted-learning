"""Exact per-example echo bookkeeping for networks built from EchoLinear / EchoParam layers.

Echo-discounted learning (EDL): example i's gradient is evaluated at theta - alpha * u_i, u_i = cumulative
parameter displacement attributed to i's own past gradients (SGD + heavy-ball momentum, decoupled weight decay,
immediate attribution of a gradient's eventual effect lr/(1-mu)).

Instead of materializing per-example parameters, every layer applies its own echo correction:
  EchoLinear:  y = x W^T + b - alpha * (U_i x + ub_i)            (x may carry extra token dims)
  EchoParam:   a parameter tensor P that is *indexed* by the forward pass (embedding tables, position tables);
               the module returns P[ids] - alpha * (echo of i on those rows).
Per-example gradients are captured with tensor hooks and summed over token positions, so the stored echo is
exact (first order) regardless of sequence length.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class EchoCtx:
    def __init__(self):
        self.active = False      # apply echo corrections in forward
        self.capture = False     # record per-example gradients
        self.src = None          # per-example rows of the echo tables to use (LongTensor, batch-aligned)
        self.alpha = 0.0


class EchoLinear(nn.Linear):
    def setup(self, ctx, N):
        self.ctx = ctx
        self.U = torch.zeros(N, self.out_features, self.in_features, device=self.weight.device)
        self.Ub = torch.zeros(N, self.out_features, device=self.weight.device) if self.bias is not None else None
        self.G = self.Gb = None

    def forward(self, x):
        ctx = self.ctx
        y = F.linear(x, self.weight, self.bias)
        if ctx.active:
            U = self.U[ctx.src]                               # (B, out, in)
            xs = x.reshape(x.shape[0], -1, x.shape[-1])       # (B, T, in)
            corr = torch.einsum("boi,bti->bto", U, xs)
            if self.Ub is not None:
                corr = corr + self.Ub[ctx.src].unsqueeze(1)
            y = y - ctx.alpha * corr.reshape(y.shape)
        if ctx.capture and y.requires_grad:
            xin = x.detach().reshape(x.shape[0], -1, x.shape[-1])

            def hook(g, xin=xin):
                self.fac = (xin, g.reshape(g.shape[0], -1, g.shape[-1]))   # (B,T,in), (B,T,out)
            y.register_hook(hook)
        return y


class EchoTable(nn.Module):
    """Learnable table P (R, d) read by row ids; ids may repeat within an example."""

    def __init__(self, R, d, init_std, T):
        super().__init__()
        self.P = nn.Parameter(torch.randn(R, d) * init_std)
        self.T = T

    def setup(self, ctx, N):
        self.ctx = ctx
        T = self.T
        self.U = torch.zeros(N, T, self.P.shape[1], device=self.P.device)   # echo on the row read at each slot
        self.G = None

    def forward(self, ids):                       # ids: (B, T)
        ctx = self.ctx
        rows = self.P[ids]                        # (B, T, d)
        if ctx.active:
            same = (ids.unsqueeze(2) == ids.unsqueeze(1)).float()            # (B, T, T): slots sharing a row
            echo = torch.einsum("bst,btd->bsd", same, self.U[ctx.src])
            rows = rows - ctx.alpha * echo
        if ctx.capture and rows.requires_grad:
            def hook(g, ids=ids):
                self.fac = (ids, g)               # gradient wrt the row read at each slot, (B, T, d)
            rows.register_hook(hook)
        return rows


class EchoTransformer(nn.Module):
    """One-layer transformer for (a op b) mod p, tokens [a, b, '='], no LayerNorm, readout at '='."""

    def __init__(self, p, d=128, heads=4, d_mlp=512):
        super().__init__()
        self.p, self.d, self.h = p, d, heads
        self.tok = EchoTable(p + 1, d, d ** -0.5, T=3)
        self.pos = EchoTable(3, d, d ** -0.5, T=3)
        self.qkv = EchoLinear(d, 3 * d, bias=False)
        self.o = EchoLinear(d, d, bias=False)
        self.fc1 = EchoLinear(d, d_mlp)
        self.fc2 = EchoLinear(d_mlp, d)
        self.unembed = EchoLinear(d, p, bias=False)

    def echo_modules(self):
        return [self.tok, self.pos, self.qkv, self.o, self.fc1, self.fc2, self.unembed]

    def forward(self, x):
        B = x.shape[0]
        t = torch.cat([x, torch.full((B, 1), self.p, dtype=x.dtype, device=x.device)], 1)
        r = self.tok(t) + self.pos(torch.arange(3, device=x.device).expand(B, 3))
        hd = self.d // self.h
        q, k, v = self.qkv(r).split(self.d, dim=-1)
        q, k, v = (z.view(B, 3, self.h, hd).transpose(1, 2) for z in (q, k, v))
        att = torch.softmax(q @ k.transpose(-1, -2) / hd ** 0.5, -1)
        r = r + self.o((att @ v).transpose(1, 2).reshape(B, 3, self.d))
        r = r + self.fc2(F.relu(self.fc1(r)))
        return self.unembed(r[:, -1])


class EchoMLP(nn.Module):
    """Same function class as ModMLP: shared embedding table, concat, ReLU hidden layers, linear readout."""

    def __init__(self, p, d=128, widths=(512,)):
        super().__init__()
        self.p = p
        self.emb = EchoTable(p, d, 1.0, T=2)
        dims = [2 * d, *widths]
        self.hid = nn.ModuleList(EchoLinear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.out = EchoLinear(dims[-1], p)

    def echo_modules(self):
        return [self.emb, *self.hid, self.out]

    def forward(self, x):
        h = self.emb(x).flatten(1)
        for l in self.hid:
            h = F.relu(l(h))
        return self.out(h)


class EchoTrainer:
    """Full-batch or mini-batch SGD + heavy-ball momentum with exact echo bookkeeping.
    mode: 'std' | 'edl' (own echo) | 'rand' (another example's echo)."""

    def __init__(self, model, xtr, ytr, lr=1.0, wd=0.02, mom=0.9, mode="edl", alpha=0.25, chunk=1024, seed=0,
                 ls=0.0, kappa=None, adapt_every=100, alpha_max=1.0, opt="sgd", freeze_at=200, beta2=0.98,
                 edl_start=0):
        self.m, self.x, self.y, self.N = model, xtr, ytr, xtr.shape[0]
        self.lr, self.wd, self.mu, self.mode, self.alpha, self.chunk, self.ls = lr, wd, mom, mode, alpha, chunk, ls
        self.kappa, self.adapt_every, self.alpha_max = kappa, adapt_every, alpha_max
        self.ctx = EchoCtx()
        for mod in model.echo_modules():
            mod.setup(self.ctx, self.N)
        self.params = list(model.parameters())
        self.buf = [torch.zeros_like(p) for p in self.params]
        self.opt, self.freeze_at, self.beta2, self.edl_start = opt, freeze_at, beta2, edl_start
        self.var = [torch.zeros_like(p) for p in self.params]
        self.pidx = {id(p): k for k, p in enumerate(self.params)}
        self.t, self.lam_hist, self.last_loo_acc = 0, [], None
        gen = torch.Generator(device="cpu").manual_seed(seed + 999)
        self.perm = torch.randperm(self.N, generator=gen).to(xtr.device)

    def _set(self, idx, which, alpha):
        self.ctx.active = which != "theta"
        self.ctx.src = idx if which == "own" else (self.perm[idx] if which == "rand" else None)
        self.ctx.alpha = alpha

    def _precond(self):
        """Diagonal preconditioner per parameter for the current step (ones for SGD)."""
        if self.opt == "sgd":
            return [None] * len(self.params)
        n = min(self.t + 1, self.freeze_at)
        bc2 = 1 - self.beta2 ** n
        return [1.0 / (torch.sqrt(v / bc2) + 1e-8) for v in self.var]

    def step(self, idx=None):
        which = {"std": "theta", "edl": "own", "rand": "rand"}[self.mode]
        if self.t < self.edl_start:
            which = "theta"
        full = idx is None
        idx = torch.arange(self.N, device=self.x.device) if idx is None else idx
        B = len(idx)
        for p in self.params:
            p.grad = None
        mods = self.m.echo_modules()
        facs, subs, correct = [], [], 0
        self.ctx.capture = True
        for s in range(0, B, self.chunk):
            sub = idx[s:s + self.chunk]
            self._set(sub, which, self.alpha)
            z = self.m(self.x[sub])
            loss = F.cross_entropy(z, self.y[sub], reduction="sum", label_smoothing=self.ls) / B
            loss.backward()
            correct += (z.argmax(1) == self.y[sub]).sum().item()
            chunk_fac = []
            for mod in mods:
                if isinstance(mod, EchoTable):
                    ids, g = mod.fac
                    chunk_fac.append(("table", ids, g * B))
                else:
                    xin, gs = mod.fac
                    chunk_fac.append(("lin", xin, gs * B, mod.Ub is not None))
                mod.fac = None
            facs.append(chunk_fac); subs.append(sub)
        self.ctx.capture = False
        self.ctx.active = False
        if self.kappa is not None and self.t % self.adapt_every == 0:
            lam = self._lam_min(facs)
            self.lam_hist.append((self.t, lam))
            self.alpha = float(min(self.alpha_max, self.kappa * lam))
        dec = 1 - self.lr * self.wd
        with torch.no_grad():
            # optimizer step
            if self.opt != "sgd" and self.t < self.freeze_at:
                for p, v in zip(self.params, self.var):
                    v.mul_(self.beta2).addcmul_(p.grad, p.grad, value=1 - self.beta2)
            P = self._precond()
            for p, b, pc in zip(self.params, self.buf, P):
                p.mul_(dec)
                b.mul_(self.mu).add_(p.grad)
                if pc is None:
                    p.add_(b, alpha=-self.lr)
                else:
                    p.add_(-self.lr * (1 - self.mu) * pc * b)
            gain = self.lr / (1 - self.mu) if self.opt == "sgd" else self.lr     # eventual effect of one gradient
            # attribution with this step's preconditioner
            if not full:
                for mod in mods:
                    mod.U.mul_(dec)
                    if getattr(mod, "Ub", None) is not None:
                        mod.Ub.mul_(dec)
            for sub, chunk_fac in zip(subs, facs):
                for mod, f in zip(mods, chunk_fac):
                    if f[0] == "table":
                        _, ids, g = f
                        pc = P[self.pidx[id(mod.P)]]
                        upd = g if pc is None else g * pc[ids]
                        U = mod.U
                    else:
                        _, xin, gs, hasb = f
                        upd = torch.einsum("bto,bti->boi", gs, xin)
                        pc = P[self.pidx[id(mod.weight)]]
                        if pc is not None:
                            upd = upd * pc
                        U = mod.U
                    if full:
                        U[sub] = dec * U[sub] - (gain / B) * upd
                    else:
                        U[sub] -= (gain / B) * upd
                    if f[0] == "lin" and f[3]:
                        gb = gs.sum(1)
                        pcb = P[self.pidx[id(mod.bias)]]
                        if pcb is not None:
                            gb = gb * pcb
                        if full:
                            mod.Ub[sub] = dec * mod.Ub[sub] - (gain / B) * gb
                        else:
                            mod.Ub[sub] -= (gain / B) * gb
                    del upd
        self.t += 1
        self.last_loo_acc = correct / B if which == "own" else None

    @staticmethod
    def _lam_min(facs):
        """lambda_min of the normalized per-example gradient Gram, computed from per-layer factors:
        linear layer: sum_{t,t'} (d_it . d_jt')(x_it . x_jt') [+ bias (sum_t d_it).(sum_t' d_jt')];
        table: sum_{t,t'} [row_it == row_jt'] (g_it . g_jt')."""
        n_layers = len(facs[0])
        K = None
        for l in range(n_layers):
            items = [c[l] for c in facs]
            if items[0][0] == "table":
                ids = torch.cat([it[1] for it in items]); g = torch.cat([it[2] for it in items])
                T = ids.shape[1]
                Kl = 0
                for t in range(T):
                    for u in range(T):
                        Kl = Kl + (g[:, t] @ g[:, u].T) * (ids[:, t:t + 1] == ids[:, u].unsqueeze(0)).float()
            else:
                xin = torch.cat([it[1] for it in items]); gs = torch.cat([it[2] for it in items])
                T = xin.shape[1]
                Kl = 0
                for t in range(T):
                    for u in range(T):
                        Kl = Kl + (gs[:, t] @ gs[:, u].T) * (xin[:, t] @ xin[:, u].T)
                if items[0][3]:
                    sg = gs.sum(1); Kl = Kl + sg @ sg.T
            K = Kl if K is None else K + Kl
        dg = K.diagonal().clamp_min(1e-30).rsqrt()
        return torch.linalg.eigvalsh((dg[:, None] * K * dg[None, :]).double())[0].item()

    @torch.no_grad()
    def accuracy_at(self, which="own", alpha=None):
        al = self.alpha if alpha is None else alpha
        correct = 0
        for s in range(0, self.N, self.chunk):
            sub = torch.arange(s, min(s + self.chunk, self.N), device=self.x.device)
            self._set(sub, which, al)
            correct += (self.m(self.x[sub]).argmax(1) == self.y[sub]).sum().item()
        self.ctx.active = False
        return correct / self.N


class EchoUntiedMLP(nn.Module):
    """Untied-embedding MLP on one-hot [a | b] inputs: linear 2p -> 2d (no nonlinearity), ReLU hidden, readout.
    Same function class as experiments/e3 (LoopMLP, linear_first=True), but with *exact* echo bookkeeping."""

    def __init__(self, p, d=128, widths=(512,)):
        super().__init__()
        self.p = p
        self.inp = EchoLinear(2 * p, 2 * d)
        with torch.no_grad():
            W = torch.zeros(2 * d, 2 * p); E = torch.randn(p, d)
            W[:d, :p] = E.T; W[d:, p:] = E.T
            self.inp.weight.copy_(W); self.inp.bias.zero_()
        dims = [2 * d, *widths]
        self.hid = nn.ModuleList(EchoLinear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.out = EchoLinear(dims[-1], p)

    def echo_modules(self):
        return [self.inp, *self.hid, self.out]

    def forward(self, x):
        oh = torch.cat([F.one_hot(x[:, 0], self.p), F.one_hot(x[:, 1], self.p)], 1).float()
        h = self.inp(oh)
        for l in self.hid:
            h = F.relu(l(h))
        return self.out(h)
