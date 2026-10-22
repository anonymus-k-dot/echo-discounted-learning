"""Datasets used in this project. Everything is generated or read locally; no downloads.

- MNIST from raw idx files (data/MNIST/raw), kept entirely on the GPU.
- Modular arithmetic (a op b mod p), the standard grokking testbed.
"""
import os
import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def _read_idx(path):
    with open(path, "rb") as f:
        magic = int.from_bytes(f.read(4), "big")
        ndim = magic & 0xFF
        shape = [int.from_bytes(f.read(4), "big") for _ in range(ndim)]
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)


def mnist(device="cuda"):
    """Returns (xtr, ytr, xte, yte); x is float in [0,1], shape (N, 784)."""
    raw = os.path.join(ROOT, "MNIST", "raw")
    xtr = _read_idx(os.path.join(raw, "train-images-idx3-ubyte")).reshape(-1, 784)
    ytr = _read_idx(os.path.join(raw, "train-labels-idx1-ubyte"))
    xte = _read_idx(os.path.join(raw, "t10k-images-idx3-ubyte")).reshape(-1, 784)
    yte = _read_idx(os.path.join(raw, "t10k-labels-idx1-ubyte"))
    f = lambda a: torch.tensor(a, dtype=torch.float32, device=device) / 255.0
    g = lambda a: torch.tensor(a.astype(np.int64), device=device)
    return f(xtr), g(ytr), f(xte), g(yte)


def subsample_balanced(x, y, n_per_class, seed, n_classes=10):
    gen = np.random.default_rng(seed)
    yc = y.cpu().numpy()
    idx = np.concatenate([gen.choice(np.where(yc == c)[0], n_per_class, replace=False)
                          for c in range(n_classes)])
    gen.shuffle(idx)
    idx = torch.tensor(idx, device=x.device)
    return x[idx], y[idx]


def corrupt_labels(y, frac, seed, n_classes=10):
    """Symmetric noise: a `frac` fraction of labels replaced by a uniformly drawn *different* class.
    Returns (noisy_labels, is_noisy mask)."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    n = y.shape[0]
    flip = torch.rand(n, generator=gen) < frac
    shift = torch.randint(1, n_classes, (n,), generator=gen)
    yn = y.clone().cpu()
    yn[flip] = (yn[flip] + shift[flip]) % n_classes
    return yn.to(y.device), flip.to(y.device)


def modular(p=97, op="add", train_frac=0.3, seed=0, device="cuda"):
    """All p*p pairs (a, b) -> (a op b) mod p, split into train/test."""
    a = torch.arange(p).repeat_interleave(p)
    b = torch.arange(p).repeat(p)
    if op == "add":
        c = (a + b) % p
    elif op == "sub":
        c = (a - b) % p
    elif op == "mul":
        c = (a * b) % p
    elif op == "x2y2":
        c = (a * a + b * b) % p
    else:
        raise ValueError(op)
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(p * p, generator=gen)
    ntr = int(train_frac * p * p)
    tr, te = perm[:ntr], perm[ntr:]
    X = torch.stack([a, b], 1).to(device)
    Y = c.to(device)
    return X[tr], Y[tr], X[te], Y[te]
