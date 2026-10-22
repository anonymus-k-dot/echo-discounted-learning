"""Evaluate the pre-registered confirmatory criteria (docs/02, Amendment A) on seeds 1-4."""
import json, glob, math

INF = math.inf


def loop_runs(mode):
    out = {}
    for f in glob.glob(f"results/e1/sgd_{mode}_frac*_lr1.0_wd0.02_a0.25_*_conf.json"):
        d = json.load(open(f)); a = d["args"]
        out[(a["frac"], a["seed"])] = dict(first=d["first"], final=d["rows"][-1]["test"],
                                           maxtest=max(r["test"] for r in d["rows"]))
    return out


def base_runs():
    out = {}
    for l in open("results/conf/baselines.jsonl"):
        if not l.startswith("{"):
            continue
        d = json.loads(l); a = d["args"]
        key = "ls" if a["ls"] > 0 else "sgd"
        out[(key, a["frac"], a["seed"])] = dict(first=d["first"], final=d["final_test"])
    return out


L, R, B = loop_runs("loo"), loop_runs("rand"), base_runs()
step = lambda r, k: r["first"].get(k, INF) if r else None
rows = []
print(f"{'frac':>5} {'seed':>4} | {'LOOP t90':>8} {'t99':>6} {'final':>6} | {'LS t90':>7} {'t99':>6} {'final':>6} | "
      f"{'SGD t99':>7} {'final':>6} | {'ctrl final':>10}")
for frac in (0.3, 0.2, 0.15):
    for seed in (1, 2, 3, 4):
        l, ls, sg, c = L.get((frac, seed)), B.get(("ls", frac, seed)), B.get(("sgd", frac, seed)), R.get((frac, seed))
        fmt = lambda v: "-" if v is None else ("inf" if v == INF else (f"{v:.3f}" if isinstance(v, float) and v <= 1 else str(v)))
        print(f"{frac:>5} {seed:>4} | {fmt(step(l,'test90')):>8} {fmt(step(l,'test99')):>6} {fmt(l and l['final']):>6} | "
              f"{fmt(step(ls,'test90')):>7} {fmt(step(ls,'test99')):>6} {fmt(ls and ls['final']):>6} | "
              f"{fmt(step(sg,'test99')):>7} {fmt(sg and sg['final']):>6} | {fmt(c and c['final']):>10}")


def count(pred, frac):
    n = ok = 0
    for seed in (1, 2, 3, 4):
        l, ls, c = L.get((frac, seed)), B.get(("ls", frac, seed)), R.get((frac, seed))
        if l is None or ls is None:
            continue
        n += 1; ok += bool(pred(l, ls, c))
    return ok, n


print()
c1 = count(lambda l, ls, c: step(l, "test99") < step(ls, "test99"), 0.3)
c2 = count(lambda l, ls, c: step(l, "test99") < step(ls, "test99"), 0.2)
c3a = count(lambda l, ls, c: step(l, "test90") < step(ls, "test90"), 0.15)
c3b = count(lambda l, ls, c: l["final"] >= ls["final"] - 0.02, 0.15)
c4a = count(lambda l, ls, c: c is not None and c["final"] < l["final"], 0.2)
c4b = count(lambda l, ls, c: c is not None and c["final"] < l["final"] and c["final"] < 0.5, 0.15)
for name, (ok, n) in (("C1' frac0.3 t99 LOOP<LS", c1), ("C2' frac0.2 t99 LOOP<LS", c2), ("C3' frac0.15 t90 LOOP<LS", c3a),
                      ("C3' frac0.15 final LOOP>=LS-0.02", c3b), ("C4 frac0.2 ctrl<LOOP", c4a), ("C4 frac0.15 ctrl<LOOP & <0.5", c4b)):
    print(f"{name:36s} {ok}/{n}  {'PASS' if n == 4 and ok >= 3 else ('pending' if n < 4 else 'FAIL')}")
