"""经典多目标跟踪基线：杂波图 + 卡尔曼(CV) + 波门 + 匈牙利全局分配 + M/N 起批 + 外推。
逐圈在线处理，只用当前圈和历史数据。"""
import math, csv, sys, json
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

DT = 5.0
RMAX = 100.0
P = dict(
    snr_min=0.0,        # 信噪比预筛门限（归一化）
    q=0.0025,           # 过程噪声加速度谱密度 (km/s^2)^2*s，兼顾机动
    vmax=0.36,          # 起批时最大速度 km/s
    gate=16.0,          # 马氏距离平方门限
    confirm_m=3, confirm_n=4,   # N 圈内命中 M 次确认
    coast=5,            # 确认航迹允许连续丢失圈数
    cmap_cell=2.0,      # 杂波图格子 km
    cmap_alpha=0.15,    # 杂波图更新速率
    cmap_thr=0.6,       # 格子平均点数超过此值视为杂波区，禁止起批
    confirm_snr=0.0,    # 确认时航迹平均信噪比门限
    tent_life=6,        # 未确认航迹最长存活圈数
)


def to_xy(az, dist):
    r = dist * RMAX
    t = az * 2 * math.pi
    return r * math.sin(t), r * math.cos(t), r


def meas_cov(r):
    s_r, s_a = 0.08, math.radians(0.3) * max(r, 1)
    s = max(s_r, s_a)
    return np.eye(2) * s * s


F = np.array([[1, 0, DT, 0], [0, 1, 0, DT], [0, 0, 1, 0], [0, 0, 0, 1]], float)
H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)


def Qm(q):
    g = np.array([[DT ** 3 / 3, DT ** 2 / 2], [DT ** 2 / 2, DT]]) * q
    Q = np.zeros((4, 4))
    Q[np.ix_([0, 2], [0, 2])] = g
    Q[np.ix_([1, 3], [1, 3])] = g
    return Q


class Track:
    def __init__(self, x, Pm, tid=None):
        self.x, self.P = x, Pm
        self.hits = [1]          # 最近命中记录
        self.snr = []
        self.age = 0
        self.miss = 0
        self.confirmed = False
        self.id = tid


class Tracker:
    def __init__(self, p=P):
        self.p = dict(p)
        self.Q = Qm(self.p["q"])
        self.tracks = []
        self.singles = []        # 上一圈未关联的孤立点 (x,y,r)
        self.next_id = 1
        self.cmap = {}

    def cell(self, x, y):
        c = self.p["cmap_cell"]
        return (int(math.floor(x / c)), int(math.floor(y / c)))

    def step(self, pts):
        """pts: list of (pntid, az, el, dist, amp, snr). 返回 [(pntid, batch_id)]."""
        p = self.p
        meas = [(pid, *to_xy(az, d), snr) for pid, az, el, d, amp, snr in pts if snr >= p["snr_min"]]
        # 杂波图更新
        cnt = {}
        for _, x, y, r, _ in meas:
            k = self.cell(x, y)
            cnt[k] = cnt.get(k, 0) + 1
        for k in set(cnt) | set(self.cmap):
            self.cmap[k] = (1 - p["cmap_alpha"]) * self.cmap.get(k, 0) + p["cmap_alpha"] * cnt.get(k, 0)
            if self.cmap[k] < 1e-3:
                del self.cmap[k]

        # 预测
        for t in self.tracks:
            t.x = F @ t.x
            t.P = F @ t.P @ F.T + self.Q

        # 关联（全局匈牙利，确认航迹优先）
        out = []
        used = set()
        order = sorted(range(len(self.tracks)), key=lambda i: not self.tracks[i].confirmed)
        for group in (True, False):
            idx = [i for i in order if self.tracks[i].confirmed == group]
            free = [j for j in range(len(meas)) if j not in used]
            if not idx or not free:
                for i in idx:
                    self._miss(self.tracks[i])
                continue
            XY = np.array([[meas[j][1], meas[j][2]] for j in free])
            tree = cKDTree(XY)
            C = np.full((len(idx), len(free)), 1e6)
            for a, i in enumerate(idx):
                t = self.tracks[i]
                hx = H @ t.x
                S = H @ t.P @ H.T + meas_cov(math.hypot(hx[0], hx[1]))
                rad = math.sqrt(p["gate"] * max(S[0, 0], S[1, 1]))
                cand = tree.query_ball_point(hx, rad)
                if not cand:
                    continue
                V = XY[cand] - hx
                d2 = np.einsum('ij,jk,ik->i', V, np.linalg.inv(S), V)
                for b, d in zip(cand, d2):
                    if d < p["gate"]:
                        C[a, b] = d
            keep_r = np.where((C < 1e6).any(1))[0]
            keep_c = np.where((C < 1e6).any(0))[0]
            C = C[np.ix_(keep_r, keep_c)]
            idx_k = [idx[a] for a in keep_r]
            free = [free[b] for b in keep_c]
            ra, cb = linear_sum_assignment(C) if C.size else ([], [])
            got = set()
            for a, b in zip(ra, cb):
                if C[a, b] < 1e6:
                    t = self.tracks[idx_k[a]]
                    j = free[b]
                    _, x, y, r, _ = meas[j]
                    R = meas_cov(r)
                    S = H @ t.P @ H.T + R
                    K = t.P @ H.T @ np.linalg.inv(S)
                    t.x = t.x + K @ (np.array([x, y]) - H @ t.x)
                    t.P = (np.eye(4) - K @ H) @ t.P
                    t.hits.append(1)
                    t.miss = 0
                    used.add(j)
                    got.add(idx_k[a])
                    t.snr.append(meas[j][4])
                    if not t.confirmed and sum(t.hits[-p["confirm_n"]:]) >= p["confirm_m"] and np.mean(t.snr) >= p["confirm_snr"]:
                        t.confirmed = True
                        t.id = self.next_id
                        self.next_id += 1
                    if t.confirmed:
                        out.append((meas[j][0], t.id))
            for i in idx:
                if i not in got:
                    self._miss(self.tracks[i])
        self.tracks = [t for t in self.tracks if not t.dead]

        # 用孤立点两点起始（杂波区内禁止）
        rest = [meas[j] for j in range(len(meas)) if j not in used]
        new_singles = []
        taken = set()
        stree = cKDTree(np.array([[a[0], a[1]] for a in self.singles])) if self.singles else None
        for k, (pid, x, y, r, snr) in enumerate(rest):
            if self.cmap.get(self.cell(x, y), 0) > p["cmap_thr"]:
                continue
            best, bd = None, None
            if stree is not None:
                for c in stree.query_ball_point([x, y], p["vmax"] * DT + 0.5):
                    if c in taken:
                        continue
                    s0 = self.singles[c]
                    d = math.hypot(x - s0[0], y - s0[1])
                    if bd is None or d < bd:
                        best, bd = c, d
            if best is not None:
                taken.add(best)
                s0 = self.singles[best]
                v = [(x - s0[0]) / DT, (y - s0[1]) / DT]
                R = meas_cov(r)
                Pm = np.diag([R[0, 0], R[1, 1], 2 * R[0, 0] / DT ** 2 + 0.01, 2 * R[1, 1] / DT ** 2 + 0.01])
                t = Track(np.array([x, y, *v]), Pm)
                t.hits = [1, 1]
                t.snr = [s0[3], snr]
                self.tracks.append(t)
            else:
                new_singles.append((x, y, r, snr))
        self.singles = new_singles
        return out

    def _miss(self, t):
        t.hits.append(0)
        t.miss += 1
        lim = self.p["coast"] if t.confirmed else 1
        t.dead = t.miss > lim or (not t.confirmed and len(t.hits) > self.p["tent_life"])

Track.dead = False


def run_scene(csv_path, params=P):
    rows = list(csv.DictReader(open(csv_path)))
    by = {}
    for r in rows:
        by.setdefault(int(r["CircleID"]), []).append(
            (int(r["PntID"]), float(r["Azimuth"]), float(r["Elevation"]), float(r["Distance"]),
             float(r["Amp"]), float(r["SNR"])))
    tr = Tracker(params)
    out = []
    for k in sorted(by):
        for pid, bid in tr.step(by[k]):
            out.append((k, pid, bid))
    return out
