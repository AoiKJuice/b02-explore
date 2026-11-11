"""官方训练数据转成本项目的场景格式，让 score.py / compare.py / 诊断和训练脚本直接复用。

官方 CSV（official/train/data/train_NNN.csv）字段：
    Pt_RadarTime, Pt_Azi, Pt_El, Pt_Distance, Pt_Amp, Pt_SNR, Pt_Circle（从 0 开始）, Pt_PtID, Batch_ID
转换后（official/conv/）：
    csv/oNNN_points.csv      Time, Azimuth, Elevation, Distance, Amp, SNR, CircleID, PntID
    answers/oNNN_truth.json  {"n_targets": 目标数, "labels": [{"CircleID", "PntID", "BatchID"}, ...]}
- CircleID = Pt_Circle + 1：tracker 内部圈计数从 1 开始，训练脚本用它和真值对齐（仿真数据也从 1 开始）。
- Azimuth = Pt_Azi × AZ_SPAN_DEG / 360：官方方位 0~1 只对应约 120° 的扇区（匀速直线拟合得到 115°~122°），而 tracker / 点级网络按“方位 × 2π”换算（仿真是整圈）。在这里先缩放，两边都不用改。
  **在线接口（提交包）里也必须用 az_to_tracker() 做同样的换算**，输出时仍回填原始方位。距离按 100 km，不用改。
- BatchID 原样保留：> 0 目标，0 杂波，−1 属性未确定。打分和训练都把 −1 当“不计”（既不算真点也不算假点）。

场景划分（按文件，互不重叠）：
    o001~o080  点级网络（o071~o080 作它的验证）
    o081~o160  航迹分类器、关联模型
    o161~o180  调参用（相当于仿真的 s 组）
    o181~o200  只看不调的验证（相当于仿真的 v 组）

用法：.venv/Scripts/python.exe official_data.py [--jobs 8]
"""
import argparse, csv, glob, json, os, sys
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "official", "train", "data")
ROOT = os.path.join(HERE, "official", "conv")
AZ_SPAN_DEG = 120.0     # 官方方位 0~1 对应的角度

# 官方数据上的参数预设：score.py / compare.py 加 --official 时，在跟踪模块的 P 之上覆盖这些。
# 主线的预设不在这个仓库里；新方法需要官方数据专用的参数时写在这里，或写在自己模块的 P 里。
PARAMS = {}


def az_to_tracker(az):
    """官方归一化方位 → tracker 约定的归一化方位（tracker 按 × 2π 换算成弧度）。"""
    return az * AZ_SPAN_DEG / 360.0


SPLITS = dict(point=range(1, 81), point_val=range(71, 81), model=range(81, 161), tune=range(161, 181), val=range(181, 201))


def keys(name):
    return [f"o{i:03d}" for i in SPLITS[name]]


def convert(path):
    key = "o" + os.path.basename(path)[6:9]
    rows, labels, tg = [], [], set()
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            k, pid, b = int(float(r["Pt_Circle"])) + 1, int(float(r["Pt_PtID"])), int(float(r["Batch_ID"]))
            rows.append([int(float(r["Pt_RadarTime"])), repr(az_to_tracker(float(r["Pt_Azi"]))), r["Pt_El"], r["Pt_Distance"],
                         r["Pt_Amp"], r["Pt_SNR"], k, pid])
            labels.append({"CircleID": k, "PntID": pid, "BatchID": b})
            if b > 0:
                tg.add(b)
    with open(os.path.join(ROOT, "csv", f"{key}_points.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Time", "Azimuth", "Elevation", "Distance", "Amp", "SNR", "CircleID", "PntID"])
        w.writerows(rows)
    with open(os.path.join(ROOT, "answers", f"{key}_truth.json"), "w") as f:
        json.dump(dict(n_targets=len(tg), labels=labels), f)
    return key, len(rows), len(tg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(SRC, "train_*.csv")))
    assert files, f"没找到官方训练数据：{SRC}（先把训练数据集 zip 解压到 official/train/）"
    os.makedirs(os.path.join(ROOT, "csv"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "answers"), exist_ok=True)
    with Pool(a.jobs) as pool:
        res = pool.map(convert, files)
    print(f"转换 {len(res)} 个文件到 {ROOT}，共 {sum(r[1] for r in res)} 个点、{sum(r[2] for r in res)} 个目标")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
