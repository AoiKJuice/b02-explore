"""本地模拟官方收发程序：按官方客户端的数据格式逐圈调用 submit/src/algorithm.py 的 process_circle，打分并统计耗时。

不经过 serve.exe（它在 Docker + Wine 里跑），但 process_circle 收到的点和官方客户端给的完全同格式：
Time / CircleID / PntID 是 int，其余是 float，CircleID 从 0 开始，每包新建一个 state。
同时检查返回值是否满足官方要求（字段齐全、点号属于本圈、每点只报一次、BatchID > 0、坐标原样回填）。

用法：
    python submit_check.py o181 o182          # 官方训练数据里的指定文件（o 编号同 official_data.py）
    python submit_check.py --official val     # 整个验证组，多进程
    python submit_check.py o181 --params '{"q":0.001}'   # 在提交包参数上再覆盖（试新配置的耗时，不用改包）
"""
import argparse, csv, json, os, sys, time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
FIELDS = ("Time", "Azimuth", "Elevation", "Distance", "BatchID", "CircleID", "PntID")


def read_official(key):
    """官方原始 CSV → {CircleID: [点字典, ...]}，格式同官方客户端交给 process_circle 的 points。"""
    path = os.path.join(HERE, "official", "train", "data", f"train_{key[1:]}.csv")
    rounds = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            p = dict(Time=int(float(r["Pt_RadarTime"])), Azimuth=float(r["Pt_Azi"]), Elevation=float(r["Pt_El"]),
                     Distance=float(r["Pt_Distance"]), Amp=float(r["Pt_Amp"]), SNR=float(r["Pt_SNR"]),
                     CircleID=int(float(r["Pt_Circle"])), PntID=int(float(r["Pt_PtID"])))
            rounds.setdefault(p["CircleID"], []).append(p)
    return rounds


def run(arg):
    key, over = arg
    sys.path.insert(0, os.path.join(HERE, "submit", "src"))
    sys.path.insert(1, HERE)
    import algorithm
    algorithm.PARAMS.update(over)
    import official_data
    from score import score
    rounds = read_official(key)
    state, out, times = {}, [], []
    for k in range(min(rounds), max(rounds) + 1):     # 空圈也要调用
        pts = rounds.get(k, [])
        t0 = time.perf_counter()
        res = algorithm.process_circle(pts, state)
        times.append(time.perf_counter() - t0)
        assert isinstance(res, list), "process_circle 必须返回列表"
        by = {p["PntID"]: p for p in pts}
        seen = set()
        for tr in res:
            assert set(tr) == set(FIELDS), f"字段不对：{sorted(tr)}"
            pid = tr["PntID"]
            assert pid in by and pid not in seen, f"第 {k} 圈点号 {pid} 不属于本圈或重复上报"
            seen.add(pid)
            assert isinstance(tr["BatchID"], int) and 0 < tr["BatchID"] <= 4294967295, "BatchID 必须是 1~4294967295 的 int"
            p = by[pid]
            assert all(tr[f] == p[f] for f in ("Time", "Azimuth", "Elevation", "Distance", "CircleID")), "坐标/时间/圈号必须原样回填"
            out.append((k + 1, pid, tr["BatchID"]))      # 真值文件里的圈号是官方圈号 + 1（见 official_data.py）
    s = score(os.path.join(official_data.ROOT, "answers", f"{key}_truth.json"), out)
    n_in = sum(len(v) for v in rounds.values())
    return key, s, sum(times), max(times), len(times), n_in, len(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("keys", nargs="*")
    ap.add_argument("--official", default="", help="官方划分名（tune / val）")
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--params", default="{}", help="在提交包参数（algorithm.PARAMS）上再覆盖的 JSON")
    a = ap.parse_args()
    import official_data
    keys = a.keys or official_data.keys(a.official)
    args = [(k, json.loads(a.params)) for k in keys]
    if len(keys) > 1 and a.jobs > 1:
        with Pool(min(a.jobs, len(keys))) as pool:
            res = pool.map(run, args)
    else:
        res = [run(x) for x in args]
    for key, s, tot, mx, n, n_in, n_out in res:
        print(f"{key} 总分={s['总分']:5.1f} | " + " ".join(f"{k}={s[k]:5.1f}" for k in ("掌握率", "虚警", "零碎度", "及时性", "最长跟踪"))
              + f" | {n} 圈，输入 {n_in} 点、输出 {n_out} 点（可和 serve.exe 的 run_summary.json 核对）"
              + f" | 每圈平均 {tot / n * 1000:.0f} ms、最长 {mx * 1000:.0f} ms（时限 30000 ms）")
    print(f"[{len(res)} 个文件] 平均总分 {sum(r[1]['总分'] for r in res) / len(res):.2f}，接口检查全部通过")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
