import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModMLP(nn.Module):
    """Embed a and b (shared table), concatenate, one hidden layer, linear readout."""

    def __init__(self, p, d=128, h=512, act="relu"):
        super().__init__()
        self.emb = nn.Embedding(p, d)
        self.fc1 = nn.Linear(2 * d, h)
        self.fc2 = nn.Linear(h, p)
        self.act = act
        nn.init.normal_(self.emb.weight, std=1.0)

    def hidden(self, x):
        e = self.emb(x).flatten(1)
        u = self.fc1(e)
        return F.relu(u) if self.act == "relu" else u * u

    def forward(self, x):
        return self.fc2(self.hidden(x))


class MLP(nn.Module):
    def __init__(self, din=784, widths=(256, 256), dout=10):
        super().__init__()
        dims = [din, *widths]
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.head = nn.Linear(dims[-1], dout)

    def features(self, x):
        for l in self.layers:
            x = F.relu(l(x))
        return x

    def forward(self, x):
        return self.head(self.features(x))


class ModTransformer(nn.Module):
    """One-layer transformer for (a op b) mod p in the style of Nanda et al. (2023): tokens [a, b, '='],
    learned position embeddings, multi-head attention, ReLU MLP, no LayerNorm, readout at the '=' position.
    Written with plain tensor ops so torch.func.vmap can batch per-example parameters."""

    def __init__(self, p, d=128, heads=4, d_mlp=512, n_layers=1):
        super().__init__()
        self.p, self.d, self.h = p, d, heads
        self.tok = nn.Parameter(torch.randn(p + 1, d) / math.sqrt(d))
        self.pos = nn.Parameter(torch.randn(3, d) / math.sqrt(d))
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            blk = nn.Module()
            blk.qkv = nn.Linear(d, 3 * d, bias=False)
            blk.o = nn.Linear(d, d, bias=False)
            blk.fc1 = nn.Linear(d, d_mlp)
            blk.fc2 = nn.Linear(d_mlp, d)
            self.layers.append(blk)
        self.unembed = nn.Linear(d, p, bias=False)

    def forward(self, x):                       # x: (B, 2) token ids
        B = x.shape[0]
        eq = torch.full((B, 1), self.p, dtype=x.dtype, device=x.device)
        t = torch.cat([x, eq], 1)
        r = self.tok[t] + self.pos.unsqueeze(0)  # (B, 3, d)
        hd = self.d // self.h
        for blk in self.layers:
            q, k, v = blk.qkv(r).split(self.d, dim=-1)
            q = q.view(B, 3, self.h, hd).transpose(1, 2)
            k = k.view(B, 3, self.h, hd).transpose(1, 2)
            v = v.view(B, 3, self.h, hd).transpose(1, 2)
            att = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(hd), -1)
            a = (att @ v).transpose(1, 2).reshape(B, 3, self.d)
            r = r + blk.o(a)
            r = r + blk.fc2(F.relu(blk.fc1(r)))
        return self.unembed(r[:, -1])
