"""参赛方处理函数：把官方逐圈点迹交给本项目的 tracker，返回本圈的航迹点。

接口见官方样例 README 第 4 节：每圈调用一次 process_circle(points, state)；
state 在同一数据包内保留，换包时由客户端清空，所以每个数据包会新建一个 Tracker。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# 单线程计算：每圈只有几百个点，多线程没有好处；CPU 紧张时（容器、共享机器），OpenMP 线程在同步点上互相等，
# 小批量预测会被拖慢上百倍（实测 Docker 里宿主机 CPU 占满时单圈从 0.05 s 变成 13~42 s，超过 29 s 客户端会停掉整个数据包）。
# 必须在导入 numpy / torch / sklearn 之前设置。
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import importlib                        # noqa: E402
import official_data                    # noqa: E402
from official_data import az_to_tracker  # noqa: E402

# 跟踪模块：要有 P（参数字典）和 Tracker(params).step(pts) -> [(PntID, BatchID), ...]。默认是示例基线 tracker_v0。
tracker = importlib.import_module(os.environ.get("B02_TRACKER", "tracker_v0"))

# 提交用参数：模块的 P + official_data.PARAMS + 下面的覆盖。
OVERRIDES = {}
PARAMS = {**tracker.P, **official_data.PARAMS, **OVERRIDES}

# 预热：导入时（连接收发软件之前）先加载模型、跑一圈假数据，第一圈就不用再等 torch 初始化。
tracker.Tracker(PARAMS).step([(1, 0.1, 0.05, 0.5, 0.3, 0.3), (2, 0.2, 0.05, 0.6, 0.3, 0.3)])


def process_circle(points, state):
    tr = state.get("tracker")
    if tr is None:
        tr = state["tracker"] = tracker.Tracker(PARAMS)
    # 官方方位 0~1 只对应约 120° 扇区，换成 tracker 约定的方位（见 official_data.py）
    pts = [(p["PntID"], az_to_tracker(p["Azimuth"]), p["Elevation"], p["Distance"], p["Amp"], p["SNR"])
           for p in points]
    by_id = {p["PntID"]: p for p in points}
    out, seen = [], set()
    for pid, bid in tr.step(pts):
        if pid in seen or pid not in by_id:   # 同一个点每圈只报一次
            continue
        seen.add(pid)
        p = by_id[pid]
        out.append({
            "Time": int(p["Time"]),
            "Azimuth": p["Azimuth"],          # 输出原始值，不是换算后的
            "Elevation": p["Elevation"],
            "Distance": p["Distance"],
            "BatchID": int(bid),
            "CircleID": int(p["CircleID"]),
            "PntID": int(pid),
        })
    return out
