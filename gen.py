"""仿真雷达点迹生成器：输出官方字段格式的点迹 + 单独保存的真值。

用法：
    python gen.py            # 生成全部 s/v 场景
    python gen.py s1 v03     # 只生成指定场景
    python gen.py --train 240    # 生成 240 个随机化训练场景 t001~t240 到 train/
                                 # 点级网络用 t001~t060 + t121~t180，航迹分类器用 t061~t120 + t181~t240（两者分开）
    python gen.py --train 120 120  # 只生成 t121~t240（前 120 个不变：同一随机序列）

场景分两组：
- s1~s4：最初的 4 个场景，参数一直用来调 tracker（相当于“训练集”）。
- v01~：不同预设的扩展场景，用来检验泛化、防止只在 s1~s4 上过拟合（相当于“验证集”）。
所有可调项见 DEFAULTS；场景配置里没写的就用默认值。默认值下随机数抽取顺序和旧版完全一致，s1~s4 的数据不变。
"""
import json, math, os, csv, sys
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
PERIOD = 5.0          # 秒/圈
RMAX = 100.0          # km，距离归一化上限
EL_MAX = math.radians(30)  # 俯仰归一化上限

DEFAULTS = dict(
    start_at_1=0.5,               # 目标在第 1 圈就出现的概率
    p_full=0.7,                   # 目标一直持续到最后一圈的概率
    speed_bands=[(0.02, 0.06), (0.1, 0.3)],   # 速度档 km/s，等概率选一档
    tgt_range=(15, 90),           # 目标初始距离 km
    pd_weak=(0.6, 0.8), pd_strong=(0.9, 0.99),
    snr_weak=(9, 13), snr_strong=(15, 25),     # 40 km 处 SNR dB
    turn=(3, 9), man_len=(3, 8),  # 机动转弯率 °/s、持续圈数
    form_off=0.8, form_n=(1, 3),  # 伴飞：相对长机偏移 ±km、僚机数 [lo, hi)
    az_sig=0.25, el_sig=0.3, r_sig=0.06,      # 量测噪声：方位 °、俯仰 °、距离 km
    clutter_mode="area",          # 随机虚警：area = 面积均匀；range = 距离均匀（近区更密）
    patch_rad=(3, 9),             # 杂波区半径 km
    patch_drift=0.0,              # 杂波区漂移速度 km/s（>0 模拟气象杂波）
)


def snr_norm(db):
    return float(np.clip(db / 40.0, 0, 1))


class Target:
    def __init__(self, rng, tid, start, end, pos, vel, pd, snr0, maneuver=None):
        self.id, self.start, self.end = tid, start, end
        self.p = np.array(pos, float)   # x,y,z km
        self.v = np.array(vel, float)   # km/s
        self.pd, self.snr0 = pd, snr0
        self.maneuver = maneuver or []  # [(round_from, round_to, turn_rate_deg_s, accel_factor)]

    def step(self, k):
        w, acc = 0.0, 1.0
        for a, b, tr, af in self.maneuver:
            if a <= k < b:
                w, acc = math.radians(tr), af
        if w:
            th = w * PERIOD
            c, s = math.cos(th), math.sin(th)
            vx, vy = self.v[0], self.v[1]
            self.v[0], self.v[1] = c * vx - s * vy, s * vx + c * vy
        self.v[:2] *= acc
        self.p += self.v * PERIOD


def to_polar(p):
    x, y, z = p
    g = math.hypot(x, y)
    r = math.sqrt(x * x + y * y + z * z)
    az = (math.atan2(x, y) / (2 * math.pi)) % 1.0
    el = math.atan2(z, g)
    return az, el, r


def rand_dir(rng, speed_kms):
    a = rng.uniform(0, 2 * math.pi)
    return np.array([speed_kms * math.cos(a), speed_kms * math.sin(a), rng.normal(0, 0.002)])


def rand_pos(rng, rmin=15, rmax=90):
    r = math.sqrt(rng.uniform(rmin ** 2, rmax ** 2))
    a = rng.uniform(0, 2 * math.pi)
    return [r * math.sin(a), r * math.cos(a), rng.uniform(0.1, 8)]


def make_scene(cfg):
    cfg = dict(DEFAULTS, **cfg)
    rng = np.random.default_rng(cfg["seed"])
    N = cfg["rounds"]
    targets = []
    tid = 1

    def add(**kw):
        nonlocal tid
        t = Target(rng, tid, **kw)
        targets.append(t)
        tid += 1
        return t

    for i in range(cfg["n_targets"]):
        start = 1 if rng.random() < cfg["start_at_1"] else int(rng.integers(1, N - 12))
        end = N if rng.random() < cfg["p_full"] else int(rng.integers(start + 10, N + 1))
        weak = rng.random() < cfg["p_weak"]
        speed = rng.choice([rng.uniform(*b) for b in cfg["speed_bands"]])
        man = []
        if rng.random() < cfg["p_maneuver"]:
            a = int(rng.integers(start + 3, max(start + 4, end - 3)))
            man.append((a, a + int(rng.integers(*cfg["man_len"])),
                        float(rng.choice([-1, 1]) * rng.uniform(*cfg["turn"])),
                        float(rng.uniform(0.97, 1.05))))
        base = add(start=start, end=end, pos=rand_pos(rng, *cfg["tgt_range"]), vel=rand_dir(rng, speed),
                   pd=rng.uniform(*cfg["pd_weak"]) if weak else rng.uniform(*cfg["pd_strong"]),
                   snr0=rng.uniform(*cfg["snr_weak"]) if weak else rng.uniform(*cfg["snr_strong"]), maneuver=man)
        if rng.random() < cfg["p_formation"]:  # 伴飞
            fo = cfg["form_off"]
            for _ in range(int(rng.integers(*cfg["form_n"]))):
                off = np.array([rng.uniform(-fo, fo), rng.uniform(-fo, fo), rng.uniform(-0.2, 0.2)])
                add(start=base.start, end=base.end, pos=base.p + off, vel=base.v.copy(),
                    pd=base.pd, snr0=base.snr0 + rng.normal(0, 1), maneuver=list(base.maneuver))

    # 杂波区（固定位置附近反复出现；patch_drift>0 时整体缓慢漂移）
    patches = []
    for _ in range(cfg["n_patches"]):
        c = rand_pos(rng, 10, 85)
        patches.append([np.array(c[:2]), rng.uniform(*cfg["patch_rad"]), int(rng.integers(*cfg["patch_pts"])), None])
    if cfg["patch_drift"] > 0:
        for pt in patches:
            a = rng.uniform(0, 2 * math.pi)
            pt[3] = cfg["patch_drift"] * PERIOD * np.array([math.cos(a), math.sin(a)])

    az_sig, el_sig, r_sig = cfg["az_sig"] / 360, math.radians(cfg["el_sig"]), cfg["r_sig"]
    rounds, truth = [], []
    for k in range(1, N + 1):
        pts = []  # (az, el, r, amp, snr, tid)
        for t in targets:
            if t.start <= k <= t.end:
                if rng.random() < t.pd:
                    az, el, r = to_polar(t.p)
                    if 0.5 < r < RMAX:
                        az += rng.normal(0, az_sig)
                        el += rng.normal(0, el_sig)
                        r += rng.normal(0, r_sig)
                        db = t.snr0 - 40 * math.log10(max(r, 5) / 40) + rng.normal(0, 2.5)
                        db = max(db, 4 + rng.random())
                        pts.append((az % 1, el, r, db + rng.normal(0, 1.5), db, t.id))
            if k >= t.start:
                t.step(k)
        # 均匀虚警
        nfa = rng.poisson(cfg["clutter"])
        for _ in range(nfa):
            if cfg["clutter_mode"] == "range":
                r = rng.uniform(1, RMAX)
            else:
                r = math.sqrt(rng.uniform(1, RMAX ** 2))
            db = 4 + rng.exponential(cfg["clutter_db"])
            pts.append((rng.random(), abs(rng.normal(0.03, 0.04)), r, db + rng.normal(0, 1.5), db, 0))
        # 杂波区
        for pt in patches:
            c, rad, n, drift = pt
            for _ in range(rng.poisson(n)):
                q = c + rng.normal(0, rad / 2, 2)
                az, el, r = to_polar([q[0], q[1], rng.uniform(0, 0.5)])
                db = 5 + rng.exponential(cfg["clutter_db"] * 1.3)
                pts.append((az, el, r, db + rng.normal(0, 1.5), db, 0))
            if drift is not None:
                pt[0] = c + drift
        pts.sort(key=lambda p: p[0])
        rnd, tr = [], []
        for j, (az, el, r, amp, db, tid_) in enumerate(pts, 1):
            row = dict(Time=int(((k - 1) + az) * PERIOD * 1000), Azimuth=round(az, 5),
                       Elevation=round(float(np.clip(el / EL_MAX, 0, 1)), 5),
                       Distance=round(min(r / RMAX, 1.0), 5), Amp=round(snr_norm(amp), 4),
                       SNR=round(snr_norm(db), 4), CircleID=k, PntID=j)
            rnd.append(row)
            tr.append(dict(CircleID=k, PntID=j, BatchID=tid_))
        rounds.append(rnd)
        truth.extend(tr)
    return rounds, truth, len(targets)


SCENES = [
    # ---- 原始 4 个场景（调参用）----
    dict(key="s1", name="小规模 · 入门", seed=11, rounds=40, n_targets=4, p_weak=0, p_maneuver=0.25,
         p_formation=0, n_patches=0, patch_pts=(1, 2), clutter=20, clutter_db=3, start_at_1=0.8),
    dict(key="s2", name="小规模 · 困难", seed=22, rounds=50, n_targets=5, p_weak=0.4, p_maneuver=0.5,
         p_formation=0.4, n_patches=1, patch_pts=(20, 30), clutter=70, clutter_db=4.5, start_at_1=0.6),
    dict(key="s3", name="大规模 · 中等", seed=33, rounds=60, n_targets=35, p_weak=0.2, p_maneuver=0.3,
         p_formation=0.2, n_patches=2, patch_pts=(25, 45), clutter=220, clutter_db=4, start_at_1=0.5),
    dict(key="s4", name="大规模 · 困难", seed=44, rounds=80, n_targets=80, p_weak=0.35, p_maneuver=0.4,
         p_formation=0.3, n_patches=4, patch_pts=(40, 80), clutter=550, clutter_db=5, start_at_1=0.4),
]

# ---- 扩展场景（验证泛化用）：在“中等”基础上每个只突出一两个难点 ----
_MID = dict(rounds=60, n_targets=25, p_weak=0.25, p_maneuver=0.3, p_formation=0.2,
            n_patches=2, patch_pts=(25, 45), clutter=250, clutter_db=4, start_at_1=0.5)
_V = [
    ("v01", "稀疏杂波 · 高精度", dict(clutter=60, n_patches=0, patch_pts=(1, 2), az_sig=0.15, r_sig=0.04, clutter_db=3)),
    ("v02", "密集随机虚警", dict(clutter=1200, n_patches=0, patch_pts=(1, 2))),
    ("v03", "大片杂波区", dict(n_patches=7, patch_pts=(50, 100), patch_rad=(5, 12))),
    ("v04", "漂移气象杂波", dict(n_patches=4, patch_pts=(40, 80), patch_rad=(6, 12), patch_drift=0.015)),
    ("v05", "密集编队", dict(p_formation=0.8, form_off=0.4, form_n=(2, 4), n_targets=12)),
    ("v06", "强机动", dict(p_maneuver=0.9, turn=(6, 15), man_len=(4, 10))),
    ("v07", "弱目标为主", dict(p_weak=0.8, pd_weak=(0.5, 0.7), snr_weak=(7, 11))),
    ("v08", "低慢小 · 近区", dict(speed_bands=[(0.005, 0.03), (0.03, 0.06)], tgt_range=(5, 40),
                                p_weak=0.6, snr_weak=(5, 10), clutter_mode="range")),
    ("v09", "高速目标", dict(speed_bands=[(0.2, 0.35), (0.35, 0.6)], p_maneuver=0.2)),
    ("v10", "近区虚警更密", dict(clutter_mode="range", clutter=400)),
    ("v11", "量测噪声大", dict(az_sig=0.5, el_sig=0.6, r_sig=0.15)),
    ("v12", "长时 · 目标进出频繁", dict(rounds=150, n_targets=40, start_at_1=0.2, p_full=0.3)),
    ("v13", "高目标密度", dict(n_targets=120, tgt_range=(15, 60), clutter=300)),
    ("v14", "大规模综合困难", dict(rounds=80, n_targets=70, p_weak=0.4, p_maneuver=0.5, p_formation=0.35,
                                  form_off=0.6, n_patches=5, patch_pts=(40, 80), patch_drift=0.008,
                                  clutter=700, clutter_db=5, clutter_mode="range")),
    ("v15", "超高速目标", dict(speed_bands=[(0.6, 0.8), (0.8, 1.0)], p_maneuver=0.2)),
]
for i, (key, name, over) in enumerate(_V):
    SCENES.append(dict(_MID, key=key, name=name, seed=100 + i, **over))


def make_train_cfgs(n, seed=7):
    """训练集（给学习模块用）：每个场景的参数在覆盖 s/v 组的范围内随机抽取（域随机化）。
    与 s/v 组的种子不重叠，学习模块只在 train/ 上训练，在 s/v 组上评估。"""
    rng = np.random.default_rng(seed)
    U = lambda a, b: float(rng.uniform(a, b))
    LU = lambda a, b: float(math.exp(rng.uniform(math.log(a), math.log(b))))   # 对数均匀
    bands = [[(0.02, 0.06), (0.1, 0.3)], [(0.005, 0.03), (0.03, 0.06)], [(0.2, 0.35), (0.35, 0.6)],
             [(0.005, 0.05), (0.1, 0.3), (0.3, 0.6)], [(0.3, 0.6), (0.6, 1.0)]]
    out = []
    for i in range(n):
        az = U(0.15, 0.5)
        pw_lo, sw_lo, tr_lo, pp_lo = U(0.45, 0.65), U(5, 10), U(3, 8), int(U(15, 60))
        cfg = dict(
            key=f"t{i + 1:03d}", name="训练·随机", seed=1000 + i,
            rounds=int(U(40, 80)), n_targets=int(LU(5, 90)), start_at_1=U(0.2, 0.8), p_full=U(0.3, 0.8),
            p_weak=U(0, 0.8), pd_weak=(pw_lo, pw_lo + 0.2), snr_weak=(sw_lo, sw_lo + 4),
            p_maneuver=U(0, 0.9), turn=(tr_lo, tr_lo + U(3, 9)), man_len=(3, int(U(6, 11))),
            p_formation=U(0, 0.6), form_off=U(0.3, 1.0), form_n=[(1, 3), (2, 4)][int(rng.integers(2))],
            n_patches=int(rng.integers(0, 8)), patch_pts=(pp_lo, 2 * pp_lo), patch_rad=(U(3, 6), U(8, 12)),
            patch_drift=0.0 if rng.random() < 0.6 else U(0.003, 0.02),
            clutter=int(LU(30, 1200)), clutter_db=U(3, 5.5), clutter_mode=["area", "range"][int(rng.integers(2))],
            az_sig=az, el_sig=az * 1.2, r_sig=U(0.04, 0.15),
            speed_bands=bands[int(rng.integers(len(bands)))],
            tgt_range=[(15, 90), (15, 90), (5, 40), (15, 60)][int(rng.integers(4))],
        )
        out.append(cfg)
    return out


def write_scene(cfg, base):
    rounds, truth, ntrue = make_scene(cfg)
    fields = ["Time", "Azimuth", "Elevation", "Distance", "Amp", "SNR", "CircleID", "PntID"]
    with open(f"{base}/csv/{cfg['key']}_points.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for rnd in rounds:
            w.writerows(rnd)
    with open(f"{base}/answers/{cfg['key']}_truth.json", "w") as f:
        json.dump(dict(n_targets=ntrue, labels=truth), f)
    return rounds, truth, ntrue


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--train":   # python gen.py --train 60 [起始序号，默认 0]
        base = f"{OUT}/train"
        os.makedirs(f"{base}/csv", exist_ok=True)
        os.makedirs(f"{base}/answers", exist_ok=True)
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 60
        start = int(sys.argv[3]) if len(sys.argv) > 3 else 0
        for cfg in make_train_cfgs(n + start)[start:]:
            rounds, truth, ntrue = write_scene(cfg, base)
            npts = [len(r) for r in rounds]
            print(cfg["key"], "rounds", len(rounds), "targets", ntrue, "pts/round", round(sum(npts) / len(npts)))
        sys.exit()
    want = set(sys.argv[1:])
    os.makedirs(f"{OUT}/site", exist_ok=True)
    os.makedirs(f"{OUT}/answers", exist_ok=True)
    os.makedirs(f"{OUT}/csv", exist_ok=True)
    pub = []
    for cfg in SCENES:
        if want and cfg["key"] not in want:
            continue
        rounds, truth, ntrue = write_scene(cfg, OUT)
        if cfg["key"].startswith("s"):
            # 网页只放点迹，不放真值：每点 [az, el, dist, amp, snr]
            pub.append(dict(key=cfg["key"], name=cfg["name"],
                            rounds=[[[p["Azimuth"], p["Elevation"], p["Distance"], p["Amp"], p["SNR"]] for p in rnd]
                                    for rnd in rounds]))
        npts = [len(r) for r in rounds]
        ntp = sum(1 for t in truth if t["BatchID"])
        print(cfg["key"], cfg["name"], "rounds", len(rounds), "targets", ntrue,
              "pts/round avg", round(sum(npts) / len(npts)), "max", max(npts), "true-pt ratio", round(ntp / sum(npts), 3))
    if pub and len(pub) == sum(1 for c in SCENES if c["key"].startswith("s")):
        with open(f"{OUT}/site/scenes.js", "w") as f:
            f.write("window.SCENES=" + json.dumps(pub, separators=(",", ":"), ensure_ascii=False) + ";")
        print("scenes.js MB", round(os.path.getsize(f"{OUT}/site/scenes.js") / 1e6, 2))
