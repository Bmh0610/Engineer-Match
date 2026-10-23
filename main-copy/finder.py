"""物料圆心检测前端: 检测 / 选最近 / 稳定帧判据 / 守株待兔。

物料俯视面是圆形(gcxl 的"圆环"即此), bbox 中心即圆心 —— 纠偏与抓取都用它
(gcxl 同思路, 霍夫圆精化留 backlog, 实测落点分布后再决定)。
像素→mm 用工作高度的比例尺(config.PX_PER_MM, tools/calib_ratio.py 标定;
观察位/中心位同高只需一套, 高度若不同再拆分)。
守株待兔等的是"转盘停歇": 转盘每转有停留, 物料静止且连帧同色才出手。
"""
import math
import time
from dataclasses import dataclass

import cv2

import calib
import config as C

_CALIB = calib.load() if C.USE_INTRINSICS else None
# USE_INTRINSICS=False(无畸变镜头相机): 不加载内参, 全链路回退画面中心+免去畸变


@dataclass
class Material:
    color: str
    conf: float
    box: tuple            # (x1, y1, x2, y2) 像素
    cx: float
    cy: float
    dx: float             # 相对主点(或画面中心)偏差 px(右+ 下+)
    dy: float
    dx_mm: float          # 像素偏差换算 mm
    dy_mm: float
    hough: tuple = None   # 霍夫精化命中时的 (cx, cy, r) 原图像素; None=未命中/未启用
    ellipse: tuple = None  # 顶面椭圆定位命中: 原图坐标 ((cx,cy),(MA,ma),ang)
    confirmed: bool = False   # 跨帧后处理: 连续 N 帧确认通过
    holding: bool = False     # 跨帧后处理: 漏检保持中(输出上次圆心)


_PX_WARNED = False


def _warn_px_ratio(px_per_mm):
    global _PX_WARNED
    if px_per_mm is None:
        if not _PX_WARNED:   # 只告警一次, 别在逐帧循环里刷屏
            _PX_WARNED = True
            print("!! 比例尺未标定(config.PX_PER_MM), 回退 1.0 px/mm —— 偏差数值不可信, 先跑 tools/calib_ratio.py !!")
        return 1.0
    return px_per_mm


def _center_to_dev(cx_raw, cy_raw, frame_shape):
    """圆心像素(带畸变) → (修正后 cx, cy, 相对主点偏差 dx, dy)。"""
    h, w = frame_shape[:2]
    if _CALIB:
        cx, cy = _CALIB.undistort_point(cx_raw, cy_raw)
        cx0, cy0 = _CALIB.principal_point(w, h)
    else:
        cx, cy, cx0, cy0 = cx_raw, cy_raw, w / 2, h / 2
    return cx, cy, cx - cx0, cy - cy0
def refine_material(frame, m, px_per_mm=None):
    """霍夫圆精化物料圆心: 就地更新 m 的 cx/cy/dx/dy/dx_mm/dy_mm(bbox 不变)。
    返回 True=霍夫命中; False=检不出(bbox 中心保持原值, 调用方无需回退)。"""
    ok, _ = refine_material_stages(frame, m, px_per_mm)
    return ok


def refine_material_stages(frame, m, px_per_mm=None):
    """物料圆心精化, 三级定位链: 椭圆顶面(主) → 霍夫正圆(备) → bbox(兜底)。

    透视下顶面投影为椭圆, fitEllipse 的中心即圆心投影, 对弱边缘/椭圆鲁棒;
    霍夫正圆仅在顶面接近正圆时有效。返回 (是否命中非bbox, stages 含 source)。
    就地更新 m 的 cx/cy/dx/dy/dx_mm/dy_mm 与 ellipse/hough 标记。
    """
    px_per_mm = _warn_px_ratio(px_per_mm)
    m.hough, m.ellipse = None, None
    stages = {}

    def _apply(cx, cy, source):
        acx, acy, dx, dy = _center_to_dev(cx, cy, frame.shape)
        m.cx, m.cy, m.dx, m.dy = acx, acy, dx, dy
        m.dx_mm, m.dy_mm = dx / px_per_mm, dy / px_per_mm
        stages["source"] = source
        return True

    # 1) 圆盘中心(主, 近垂直俯视下顶/底同心)
    disc, est = fit_top_center(frame, m.box)
    stages.update(est)
    if disc is not None:
        cx, cy, r = disc
        stages["source"] = "disc"
        m.ellipse = ((cx, cy), (2 * r, 2 * r), 0)   # render 画正圆
        return _apply(cx, cy, "disc"), stages
    # 2) bbox 中心(兜底): 带透视系统偏差, 仅保流程不断
    #    (霍夫正圆已退役: 斜视时顶面投影为椭圆, 正圆假设不成立)
    stages["source"] = "bbox"
    return False, stages


def find_materials(backend, frame, targets=None, conf=0.7, px_per_mm=1.0):
    """看一帧, 返回白名单内物料列表。

    targets: 颜色名列表; None = 全部 6 色。
    px_per_mm: 当前工作高度的比例尺(像素/mm); None = 回退 1.0 并告警。
    """
    px_per_mm = _warn_px_ratio(px_per_mm)
    out = []
    for d in backend.infer(frame, classes=C.MATERIAL_CLS):
        color = C.CLS2COLOR[d.cls_id]
        if targets and color not in targets:
            continue
        if d.conf < conf:
            continue
        cx, cy = (d.xyxy[0] + d.xyxy[2]) / 2, (d.xyxy[1] + d.xyxy[3]) / 2
        cx, cy, dx, dy = _center_to_dev(cx, cy, frame.shape)
        out.append(Material(color, d.conf, d.xyxy, cx, cy, dx, dy,
                            dx / px_per_mm, dy / px_per_mm))
    return out


def nearest(materials):
    """选离车最近的物料 = 画面最下方(cy 最大; gcxl raw_area 选 y 最大同思路)。"""
    return max(materials, key=lambda m: m.cy) if materials else None


def to_body(obj):
    """图像偏差(像素) → 车体系 (dy_right, dx_fwd), 2x2 矩阵 config.IMG2BODY。

    ⚠️ IMG2BODY 标定的是像素→车体映射(标定响应÷100mm), 必须喂像素 obj.dx/dy;
    喂 mm 会差 PX_PER_MM 倍(实车 3 倍纠偏误差根因)。"""
    a, b, c, d = C.IMG2BODY
    return a * obj.dx + b * obj.dy, c * obj.dx + d * obj.dy


def nearest_to_axis(materials):
    """选离主点(相机光轴正下方)最近的物料: 垂直视角、畸变与透视偏移最小,
    定位基准首选(多物料同框时, 边缘斜视角的物料误差可达数 mm~cm)。"""
    if not materials:
        return None
    if _CALIB:
        px, py = _CALIB.principal_point()
    else:
        return nearest(materials)   # 无内参无法定主点, 回退画面最下
    return min(materials, key=lambda m: (m.cx - px) ** 2 + (m.cy - py) ** 2)


def find_stable(finder, cam, timeout=3.0, need=None, tol_px=None):
    """循环取帧直到目标连续 need 帧中心位移 < tol_px(稳定判据, gcxl 同款)。

    finder: callable(frame) -> 带 .cx/.cy 的对象 或 None。
    返回 (obj, frame); 超时/检不出返回 (None, None)。
    """
    need = need or C.STABLE_FRAMES
    tol_px = tol_px if tol_px is not None else C.STABLE_TOL_PX
    deadline = time.time() + timeout
    last = None
    stable = 0
    while time.time() < deadline:
        frame = cam.read_latest(timeout=0.2)
        if frame is None:
            continue
        obj = finder(frame)
        if obj is None:
            stable, last = 0, None
            time.sleep(0.03)
            continue
        if last is not None and abs(obj.cx - last.cx) < tol_px \
                and abs(obj.cy - last.cy) < tol_px:
            stable += 1
            if stable >= need:
                return obj, frame
        else:
            stable = 1
        last = obj
    return None, None


def confirm_color(backend, cam, color, px_per_mm=None, conf=None, frames=None):
    """时序一致性: 连续 N 帧该色存在(conf≥门槛)且中心几乎不动(转盘停歇)
    才返回 Material, 否则 None。"""
    conf = C.CONFIRM_CONF if conf is None else conf
    frames = frames or C.CONFIRM_FRAMES
    px_per_mm = _warn_px_ratio(px_per_mm)
    last = None
    for _ in range(frames):
        f = cam.read_latest()
        if f is None:
            return None
        hits = find_materials(backend, f, [color], conf=conf, px_per_mm=px_per_mm)
        if not hits:
            return None
        m = max(hits, key=lambda x: x.conf)
        if last is not None and (abs(m.cx - last.cx) > C.STABLE_TOL_PX
                                 or abs(m.cy - last.cy) > C.STABLE_TOL_PX):
            return None   # 还在动(转盘没停), 不认
        last = m
        time.sleep(C.FRAME_GAP)
    return last


def wait_for_color(backend, cam, color, px_per_mm=None, timeout=None,
                   window=None, conf=None):
    """守株待兔: 循环看帧, 目标色中心进入抓取窗(画面归一化 x0,x1,y0,y1)
    且连帧同色+中心稳定(转盘停歇) → 返回 (Material, frame); 超时 (None, None)。
    臂提前守在窗上方由调度层做, 这里只答"目标色到了没、在哪"。"""
    timeout = C.WAIT_COLOR_TIMEOUT if timeout is None else timeout
    window = C.GRAB_WINDOW if window is None else window
    conf = C.CONFIRM_CONF if conf is None else conf
    px_per_mm = _warn_px_ratio(px_per_mm)
    deadline = time.time() + timeout
    while time.time() < deadline:
        f = cam.read_latest()
        if f is None:
            time.sleep(C.FRAME_GAP)
            continue
        h, w = f.shape[:2]
        for m in find_materials(backend, f, [color], conf=conf, px_per_mm=px_per_mm):
            if (window[0] * w <= m.cx <= window[1] * w
                    and window[2] * h <= m.cy <= window[3] * h):
                m2 = confirm_color(backend, cam, color, px_per_mm, conf)
                if m2:
                    return m2, f
        time.sleep(C.FRAME_GAP)
    return None, None


# ---- 跨帧后处理(移植沈航 common.py 思路): 消歧 → 关联 → 确认 → 低通 → 保持 ----

# 易混对裁决赢家: blue/light_blue 按沈航实测经验归 light_blue(更特殊类);
# 不在表中的易混对归 conf 高者(保守)。键 = frozenset(色对)。
_PAIR_WINNER = {frozenset(("blue", "light_blue")): "light_blue"}


def _cover_ratio(box_a, box_b):
    """B 框被 A 框覆盖的比例 = 交面积 / B 面积。"""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    area_b = max((bx2 - bx1) * (by2 - by1), 1e-6)
    return iw * ih / area_b


def disambiguate_pairs(mats, cover=None):
    """易混对覆盖率消歧(返回新列表): RISKY_PAIRS 两色框互相覆盖≥门槛判同一
    物体双检, 保留赢家(light_blue 优先 / 其余对 conf 高者), 删另一个。"""
    cover = C.DISAMBIG_COVER if cover is None else cover
    pairs = [frozenset(p) for p in C.RISKY_PAIRS]
    out = list(mats)
    i = 0
    while i < len(out):
        j = i + 1
        while j < len(out):
            a, b = out[i], out[j]
            if frozenset((a.color, b.color)) in pairs and \
                    max(_cover_ratio(a.box, b.box),
                        _cover_ratio(b.box, a.box)) >= cover:
                winner_color = _PAIR_WINNER.get(frozenset((a.color, b.color)))
                keep = (a if a.color == winner_color else b) \
                    if winner_color else (a if a.conf >= b.conf else b)
                out.remove(b if keep is a else a)
                continue   # 不前移, 重新比较保留者与下一个
            j += 1
        i += 1
    return out


class MaterialTracker:
    """YOLO 检测结果的跨帧后处理: 关联 → 确认 → 低通 → 漏检保持。

    update(mats, frame_shape, px_per_mm) 每帧调用, 返回稳定物料列表
    (Material 子集, 带 confirmed/holding 标记):
      关联: 与已有 Track 贪心近邻(圆心距 < TRACK_MATCH_PX, 同色);
      确认: 连续 TRACK_CONFIRM_FRAMES 帧命中才 confirmed;
      低通: 圆心 = α·新帧 + (1-α)·旧(只平滑位置, 出手定位够稳且滞后小);
      保持: 确认过的 Track 漏检 ≤ TRACK_HOLD_FRAMES 帧时继续输出上次圆心
            (holding=True, 防转盘短暂遮挡误判消失), 超时删除。
    """

    def __init__(self):
        self.tracks = []   # dict: color/cx/cy/conf/box/hough/hits/missed/...

    def reset(self):
        self.tracks = []

    def update(self, mats, frame_shape, px_per_mm=None):
        px_per_mm = _warn_px_ratio(px_per_mm)
        h, w = frame_shape[:2]
        cx0, cy0 = (_CALIB.principal_point(w, h) if _CALIB else (w / 2, h / 2))
        unmatched = list(self.tracks)
        for m in mats:
            best, bd = None, C.TRACK_MATCH_PX
            for t in unmatched:
                if t["color"] != m.color:
                    continue
                d = math.hypot(m.cx - t["cx"], m.cy - t["cy"])
                if d < bd:
                    best, bd = t, d
            if best is not None:
                a = C.TRACK_ALPHA
                best["cx"] = a * m.cx + (1 - a) * best["cx"]
                best["cy"] = a * m.cy + (1 - a) * best["cy"]
                best["conf"], best["box"], best["hough"] = m.conf, m.box, m.hough
                best["hits"] += 1
                best["missed"] = 0
                best["confirmed"] = best["hits"] >= C.TRACK_CONFIRM_FRAMES
                best["holding"] = False
                unmatched.remove(best)
            else:
                self.tracks.append(dict(color=m.color, cx=m.cx, cy=m.cy,
                                        conf=m.conf, box=m.box, hough=m.hough,
                                        hits=1, missed=0, confirmed=False,
                                        holding=False))
        for t in unmatched:
            t["missed"] += 1
            if t["missed"] > C.TRACK_HOLD_FRAMES:
                self.tracks.remove(t)
            else:
                t["holding"] = t["confirmed"]
        out = []
        for t in self.tracks:
            if not (t["confirmed"] or t["holding"]):
                continue
            dx, dy = t["cx"] - cx0, t["cy"] - cy0
            out.append(Material(t["color"], t["conf"], t["box"], t["cx"], t["cy"],
                                dx, dy, dx / px_per_mm, dy / px_per_mm,
                                hough=t["hough"], confirmed=t["confirmed"],
                                holding=t["holding"]))
        return out


# ---- 顶面椭圆定位(透视下顶面投影为椭圆, 霍夫正圆不适; 主定位法) ----

def fit_top_center(frame, box, pad=0.10):
    """ROI 内物料剪影 → 最大轮廓 → 最小外接圆 → 圆盘中心 (cx, cy, r)。

    近垂直俯视下物料顶/底面同心, 圆盘中心即抓取基准点(斜视时顶/底投影
    圆心有高度×视角的透视偏移, 随俯角减小而缩小)。
    六色通用剪影: 彩色(高饱和)或暗(低亮度)=物料, 白背景排除。
    毛刺防护: 轮廓面积应占外接圆一半以上, 否则拒绝。
    返回 ((cx, cy, r), stages); 失败 (None, stages)。
    """
    import numpy as np
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    H, W = frame.shape[:2]
    rx1, ry1 = max(int(x1 - w * pad), 0), max(int(y1 - h * pad), 0)
    rx2, ry2 = min(int(x2 + w * pad), W), min(int(y2 + h * pad), H)
    roi = frame[ry1:ry2, rx1:rx2]
    stages = {"offset": (rx1, ry1)}
    if roi.size == 0:
        return None, stages
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    s_ch, v_ch = hsv[:, :, 1], hsv[:, :, 2]
    body = cv2.inRange(hsv, (0, 60, 40), (180, 255, 255))     # 彩色: S≥60
    body |= cv2.inRange(hsv, (0, 0, 0), (180, 255, 89))       # 黑/暗: V<90
    body = cv2.morphologyEx(body, cv2.MORPH_OPEN,
                            np.ones((3, 3), np.uint8))
    stages["body_mask"] = body
    contours, _ = cv2.findContours(body, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, stages
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area < 150:
        return None, stages
    (cx, cy), r = cv2.minEnclosingCircle(cnt)
    if area < 0.5 * np.pi * r * r:   # 毛刺防护: 轮廓应占外接圆大半
        return None, stages
    stages["circle"] = ((cx + rx1, cy + ry1), r)   # 原图坐标
    return (float(cx) + rx1, float(cy) + ry1, float(r)), stages


