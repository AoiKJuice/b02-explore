"""按赛题 5 项公式打分。匹配用点的真值标签（输出点的 PntID 属于哪个真目标）。
BatchID > 0 是目标，0 是杂波；< 0（官方数据里的 −1“属性未确定”）不计：输出了不算假点，也不算任何目标的真点。

用法：
    python score.py                         # 仿真 s/v 组全部场景
    python score.py s2 v05 --params '{"q":0.001}'
    python score.py --official val          # 官方数据的验证组（划分见 official_data.py），多进程；自动套用 official_data.PARAMS
"""
import json, sys, time, glob, os, importlib, argparse
from multiprocessing import Pool

PERIOD_MS = 5000
W = dict(掌握率=0.3, 虚警=0.1, 零碎度=0.2, 及时性=0.2, 最长跟踪=0.2)


def score(truth_path, output):
    t = json.load(open(truth_path))
    lab = {(l["CircleID"], l["PntID"]): l["BatchID"] for l in t["labels"]}
    tgt = {}
    for l in t["labels"]:
        if l["BatchID"] > 0:
            tgt.setdefault(l["BatchID"], []).append(l["CircleID"])
    total_true = sum(len(v) for v in tgt.values())
    false_pts = 0
    per = {i: {} for i in tgt}          # target -> {batch: count}
    first = {}                          # target -> earliest round matched
    for k, pid, bid in output:
        g = lab.get((k, pid), 0)
        if g < 0:            # 属性未确定的点：不计
            continue
        if g == 0:
            false_pts += 1
            continue
        per[g][bid] = per[g].get(bid, 0) + 1
        first[g] = min(first.get(g, 1e9), k)
    T = len(tgt)
    m = sum(sum(per[i].values()) / len(tgt[i]) for i in tgt) / T * 100
    fa = max(0.0, 1 - false_pts / total_true) * 100
    tracked = [i for i in tgt if per[i]]
    frag = (sum(1 / len(per[i]) for i in tracked) / len(tracked) * 100) if tracked else 0
    tim = 0
    for i in tgt:
        rs = sorted(tgt[i])
        if i in first and len(rs) >= 3:
            delay = (first[i] - rs[2]) * PERIOD_MS / 1000
            tim += 1 - max(min(delay, 50), 0) / 50
    tim = tim / T * 100
    lng = sum(min(1, (max(per[i].values()) if per[i] else 0) / max(1, len(tgt[i]) - 2)) for i in tgt) / T * 100
    s = dict(掌握率=m, 虚警=fa, 零碎度=frag, 及时性=tim, 最长跟踪=lng)
    s["总分"] = sum(W[k] * s[k] for k in W)
    s["_真目标"] = T
    s["_跟上的目标"] = len(tracked)
    s["_假点/真点"] = f"{false_pts}/{total_true}"
    return s


def run_one(arg):
    tracker_name, params, root, key = arg
    mod = importlib.import_module(tracker_name)
    t0 = time.time()
    out = mod.run_scene(os.path.join(root, "csv", f"{key}_points.csv"), params)
    return key, score(os.path.join(root, "answers", f"{key}_truth.json"), out), time.time() - t0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="跑 tracker 并按赛题公式打分")
    ap.add_argument("scenes", nargs="*", help="场景 key，如 s1 v03；不写则跑全部")
    ap.add_argument("--tracker", default="tracker_v0", help="跟踪模块名（模块里要有 P 和 run_scene），默认是示例基线 tracker_v0")
    ap.add_argument("--params", default="{}", help='覆盖参数的 JSON，如 {"q":0.001}')
    ap.add_argument("--official", default="", help="官方数据的划分名（tune / val / model / point），数据在 official/conv/")
    ap.add_argument("--jobs", type=int, default=1, help="并行进程数（--official 时默认 6）")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    mod = importlib.import_module(a.tracker)
    params = dict(mod.P, **json.loads(a.params))
    if a.official:
        import official_data
        params = {**mod.P, **official_data.PARAMS, **json.loads(a.params)}   # 官方预设，--params 在它上面覆盖
        root = official_data.ROOT
        keys = a.scenes or official_data.keys(a.official)
        jobs = a.jobs if a.jobs > 1 else 6
    else:
        root = here
        keys = a.scenes or sorted(os.path.basename(f)[:-11] for f in glob.glob(os.path.join(root, "csv", "*_points.csv")))
        jobs = a.jobs
    args = [(a.tracker, params, root, k) for k in keys]
    if jobs > 1:
        with Pool(jobs) as pool:
            res = pool.map(run_one, args)
    else:
        res = [run_one(x) for x in args]
    tot = {}
    for key, s, dt in res:
        g = a.official or key[0]
        tot.setdefault(g, []).append(s["总分"])
        print(f"{key:4s} 总分={s['总分']:5.1f} | " + " ".join(f"{k}={s[k]:5.1f}" for k in W),
              f"| 目标 {s['_跟上的目标']}/{s['_真目标']} 假点/真点 {s['_假点/真点']} 耗时 {dt:.1f}s")
    for g, v in tot.items():
        print(f"[{g} 组] {len(v)} 个场景 平均总分 {sum(v) / len(v):.2f}")
