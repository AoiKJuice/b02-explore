"""两套参数的配对比较：同样的场景各跑一遍，按目标做配对 bootstrap，给组平均总分差的 95% 置信区间。

单场景只有 30 个左右目标时，一两个目标的成败就能让总分差 1~2 分，
所以只有置信区间不跨 0 的差别才当真。

用法：
    python compare.py '{"assoc_mode":"llr"}'                 # 和默认参数比
    python compare.py '{"assoc_mode":"learn"}' --base '{"assoc_mode":"llr"}' --scenes s4 v05
"""
import argparse, glob, json, os, sys
from multiprocessing import Pool
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
W = dict(m=0.3, fa=0.1, frag=0.2, tim=0.2, lng=0.2)


def per_target(truth_path, output):
    """每个真目标的掌握率、及时性、最长跟踪、零碎度（没跟上为 nan），以及场景级虚警分。口径同 score.py。"""
    t = json.load(open(truth_path))
    lab = {(l["CircleID"], l["PntID"]): l["BatchID"] for l in t["labels"]}
    tgt = {}
    for l in t["labels"]:
        if l["BatchID"] > 0:
            tgt.setdefault(l["BatchID"], []).append(l["CircleID"])
    ids = sorted(tgt)
    total = sum(len(v) for v in tgt.values())
    per, first, false_pts = {i: {} for i in ids}, {}, 0
    for k, pid, bid in output:
        g = lab.get((k, pid), 0)
        if g < 0:            # 属性未确定的点：不计
            continue
        if g == 0:
            false_pts += 1
            continue
        per[g][bid] = per[g].get(bid, 0) + 1
        first[g] = min(first.get(g, 1e9), k)
    m, tim, lng, frag = [], [], [], []
    for i in ids:
        n = len(tgt[i])
        m.append(sum(per[i].values()) / n)
        rs = sorted(tgt[i])
        if i in first and n >= 3:
            tim.append(1 - max(min((first[i] - rs[2]) * 5.0, 50), 0) / 50)
        else:
            tim.append(0.0)
        lng.append(min(1, (max(per[i].values()) if per[i] else 0) / max(1, n - 2)))
        frag.append(1 / len(per[i]) if per[i] else np.nan)
    return np.array(m), np.array(tim), np.array(lng), np.array(frag), max(0.0, 1 - false_pts / total)


def total(m, tim, lng, frag, fa, idx):
    f = frag[idx]
    f = f[~np.isnan(f)]
    return 100 * (W["m"] * m[idx].mean() + W["tim"] * tim[idx].mean() + W["lng"] * lng[idx].mean()
                  + W["frag"] * (f.mean() if len(f) else 0) + W["fa"] * fa)


def run(arg):
    key, over_a, over_b, root, mod = arg
    import importlib
    tracker = importlib.import_module(mod)
    res = []
    for over in (over_a, over_b):
        out = tracker.run_scene(os.path.join(root, "csv", f"{key}_points.csv"), dict(tracker.P, **over))
        res.append(per_target(os.path.join(root, "answers", f"{key}_truth.json"), out))
    return key, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("new", help="新参数 JSON")
    ap.add_argument("--base", default="{}", help="基准参数 JSON（默认 = tracker.P）")
    ap.add_argument("--scenes", nargs="*", default=None)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--tracker", default="tracker_v0", help="跟踪模块名（模块里要有 P 和 run_scene）")
    ap.add_argument("--official", default="", help="官方数据的划分名（tune / val），数据在 official/conv/；"
                    "两套参数都以 official_data.PARAMS 为底")
    a = ap.parse_args()
    pre = {}
    if a.official:
        import official_data
        pre = official_data.PARAMS
        root = official_data.ROOT
        keys = a.scenes or official_data.keys(a.official)
    else:
        root = HERE
        keys = a.scenes or sorted(os.path.basename(f)[:-11] for f in glob.glob(os.path.join(HERE, "csv", "*_points.csv")))
    with Pool(a.jobs) as pool:
        res = dict(pool.map(run, [(k, dict(pre, **json.loads(a.base)), dict(pre, **json.loads(a.new)), root, a.tracker) for k in keys]))
    rng = np.random.default_rng(0)
    groups = ({f"官方 {a.official}": keys} if a.official else
              {"s 组": [k for k in keys if k[0] == "s"], "v01~v14": [k for k in keys if k[0] == "v" and k != "v15"]})
    print("场景   基准    新    差   单场景 95% 区间（按目标 bootstrap）")
    for k in keys:
        A, B = res[k]
        n = len(A[0])
        ta, tb = total(*A, np.arange(n)), total(*B, np.arange(n))
        bd = []
        for _ in range(500):
            idx = rng.integers(0, n, n)
            bd.append(total(*B, idx) - total(*A, idx))
        lo, hi = np.percentile(bd, [2.5, 97.5])
        print(f"{k:4s} {ta:6.1f} {tb:6.1f} {tb - ta:+5.1f}   [{lo:+5.1f}, {hi:+5.1f}]" + (" *" if lo > 0 or hi < 0 else ""))
    for g, ks in groups.items():
        ks = [k for k in ks if k in res]
        if not ks:
            continue
        diffs = np.zeros(a.boot)
        point = np.mean([total(*res[k][1], np.arange(len(res[k][1][0]))) - total(*res[k][0], np.arange(len(res[k][0][0])))
                         for k in ks])
        for b in range(a.boot):
            d = []
            for k in ks:
                A, B = res[k]
                idx = rng.integers(0, len(A[0]), len(A[0]))
                d.append(total(*B, idx) - total(*A, idx))
            diffs[b] = np.mean(d)
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        print(f"[{g}] 平均差 {point:+.2f}，95% 置信区间 [{lo:+.2f}, {hi:+.2f}]" + ("  ← 显著" if lo > 0 or hi < 0 else ""))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
