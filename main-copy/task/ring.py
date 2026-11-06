"""Step_Ring: 载物盘 → 粗加工区数字环放置(定一式: 对画面中间的环整车纠偏, 查表放三)。

独立于 grab.py: 自带通信原语与环检测前端(不 import 物料检测/解码代码)——
环检测=OpenCV(vision/ring_cv.py: 黑线同心圆环+中心数字模板匹配+排列恢复,
无需模型/权重; YOLO 环检测已退役 2026-11-02);
转盘标定(IMG2BODY/PX_PER_MM)与环面标定(IMG2BODY_RING/PX_PER_MM_RING)严格
分开: 环纠偏只用 *_RING, 未填即拒跑(无回退, 防两套标定互相污染)。

流程(车到粗加工区, 臂在复位位 OBSERVE=(0,2090,0)):
  摆圆环纠偏位 RING_ALIGN → 找数字环 → 取画面 cx 居中者(物理中间环)
  做"定一"目标 → 整车看-动-再看纠偏(骨架同 grab.align: 死区3mm ≤5轮, 超时
  带误差继续) → 收敛帧数字→左中右位置映射(未检出数字按默认次序 1,2,3 填空位
  +告警) → 任务码颜色序: 盘上取料 → 对应位置示教位姿 RING_SLOT_L/M/R 放置
  (每颗 8 回执: 盘上4+环上4) → 收臂回复位。
软失败(未见环/某色没上车/环号超 3/回执未满)=告警+跳过, 料留盘上, 保完赛;
硬失败(串口静默 LinkError)=抛出终止(同 grab —— 臂/车位姿不明继续盲编排不安全)。
"""
import math
import time
from dataclasses import dataclass

import config as C
from communication.protocol import Mode
from vision.finder import find_stable


_DEFAULT_ORDER = (1, 2, 3)   # 未检出数字的默认次序(用户口径"未检出默认123")


@dataclass
class Ring:
    digit: int           # 1/2/3(=贴纸数字, 由中心数字识别/排列恢复)
    conf: float
    box: tuple
    cx: float
    cy: float
    dx: float            # 相对画面中心偏差 px(右+ 下+)
    dy: float
    bbox_cx: float = 0.0  # YOLO bbox 中心(两阶段纠偏粗段用; 几何中心=cx/cy 精段用)
    bbox_cy: float = 0.0
    precise: bool = True  # cx/cy 来自几何精定位; False=降级 bbox 中心(精段纠偏丢弃)


def _dump_miss(frame, tag):
    """0 检出存证: 现场帧覆盖写 /tmp/ring_miss.jpg(定案取数, 静默兜底)。"""
    try:
        import cv2
        cv2.imwrite("/tmp/ring_miss.jpg", frame)
        print(f"  [判环] 存证 /tmp/ring_miss.jpg ({tag})")
    except Exception as e:
        print(f"  [判环] 存证失败: {e}")


def _roi_precise_center(frame, xyxy, expand=0.25):
    """bbox 内 ring_cv 几何精定位环中心(2026-11-05 用户定: YOLO 定身份+
    OpenCV 定中心): bbox 外扩 expand → fit_ellipses+同心成组+多层加权估中心
    → 精确中心映射回全图坐标; ROI 内无候选/异常 → None(调用方降级 bbox 中心)。"""
    try:
        from vision import ring_cv
        x1, y1, x2, y2 = xyxy
        bw, bh = x2 - x1, y2 - y1
        mx, my = int(bw * expand), int(bh * expand)
        X0, Y0 = max(0, int(x1) - mx), max(0, int(y1) - my)
        X1, Y1 = min(frame.shape[1], int(x2) + mx), min(frame.shape[0], int(y2) + my)
        roi = frame[Y0:Y1, X0:X1]
        if roi.size == 0:
            return None
        ellipses = ring_cv.fit_ellipses(ring_cv.software_relight(roi),
                                        ring_cv.MAJOR_LO, ring_cv.MAJOR_HI)
        if not ellipses:
            return None
        markers = ring_cv.group_concentric(ellipses, ring_cv.MIN_RINGS)
        if not markers:
            return None
        center, outer = max(markers, key=lambda m: m[1][0])[:2]
        return (center[0] + X0, center[1] + Y0)   # marker[0]=(cx,cy) 已是加权精中心
    except Exception:
        return None


def find_rings(frame, digit=None, backend=None):
    """一帧 → 数字环列表(Ring: digit 1/2/3, 位置=画面中心偏差 px)。

    混合定位(2026-11-05 用户定): YOLO 定身份(几号环, 类别映射), OpenCV 定中心
    (bbox 内 ring_cv 几何精定位——多层同心加权, 比 bbox 中心准); 几何失败降级
    bbox 中心。YOLO 全空 → 全画面 ring_cv 几何 + x 排序兜底(左1中2右3, 用户
    定死口径) → 仍 0 存证 /tmp/ring_miss.jpg。
    backend None(引擎未加载/缺文件)→返回 [](判环跳过, 保完赛)。"""
    if backend is None:
        return []
    h, w = frame.shape[:2]
    ox, oy = C.RING_CAM_CENTER_OFFSET_PX   # 相机中心点偏移(纠偏基准)
    cls2digit = {v: k for k, v in C.RING_CLS.items()}
    out = []
    for d in backend.infer(frame, classes=list(C.RING_CLS.values())):
        if d.conf < C.RING_CONF or d.cls_id not in cls2digit:
            continue
        digit_d = cls2digit[d.cls_id]
        if digit is not None and digit_d != digit:
            continue
        bx_cx = (d.xyxy[0] + d.xyxy[2]) / 2
        bx_cy = (d.xyxy[1] + d.xyxy[3]) / 2
        # 几何精定位(bbox 内 ring_cv); 失败降级 bbox 中心
        precise = _roi_precise_center(frame, d.xyxy)
        cx, cy = precise if precise else (bx_cx, bx_cy)
        if not precise:
            print(f"  [判环] {digit_d} 号环几何降级 bbox 中心")
        out.append(Ring(digit_d, d.conf, d.xyxy, cx, cy,
                        cx - (w / 2 + ox), cy - (h / 2 + oy),
                        bbox_cx=bx_cx - (w / 2 + ox), bbox_cy=bx_cy - (h / 2 + oy),
                        precise=precise is not None))
    if not out and digit is None:
        # YOLO 全空 → 全画面 ring_cv 几何 + x 排序兜底(身份来自排列, 左1中2右3)
        try:
            from vision import ring_cv
            results, _ = ring_cv.analyze(frame, keep_unknown=True)
            if len(results) >= 3:
                top = sorted(results, key=lambda r: -r[1][0])[:3]
                srt = sorted(top, key=lambda r: r[0][0])
                out = [Ring(k + 1, r[7] if r[5] else 0.0,
                            (r[0][0] - 40, r[0][1] - 40, r[0][0] + 40, r[0][1] + 40),
                            r[0][0], r[0][1],
                            r[0][0] - (w / 2 + ox), r[0][1] - (h / 2 + oy),
                            bbox_cx=r[0][0] - (w / 2 + ox),
                            bbox_cy=r[0][1] - (h / 2 + oy))
                       for k, r in enumerate(srt)]
                print(f"  [判环] YOLO 未检出, 全画面几何+x 排序兜底 {len(out)} 环")
        except Exception:
            pass
    if not out and digit is None:
        _dump_miss(frame, "ring")
    return out


class StepRing:
    """环放置: 摆观测位 → 对中环整车纠偏(定一) → 数字查位 → 盘→环 放三 → 收臂。"""

    def __init__(self, link, cam):
        self.lk, self.cam = link, cam
        self.stats = {"place_ok": 0, "place_total": 3}

    # ---- 通信原语(同 grab.py 口径: 单命令在途, 等对应完成码) ----

    def move_arm(self, reach, height, theta, label=""):
        self.lk.send_and_wait(
            lambda: self.lk.move_a(reach, height, theta),
            Mode.ACK_ARM, C.CMD_TIMEOUT,
            f"move_a({reach:.0f},{height:.0f},{theta:+.0f}) {label}")

    def grip(self, close):
        ang, t = C.GRIP_CLOSE if close else C.GRIP_OPEN
        self.lk.send_and_wait(lambda: self.lk.gripper(ang, t),
                              Mode.ACK_GRIPPER, C.CMD_TIMEOUT,
                              f"grip({'闭合' if close else '张开'})")

    def _move_v(self, dx_fwd, dy_right):
        """车体系微调: 前进 dx_fwd, 右移 dy_right (mm), 环纠偏用。"""
        self.lk.send_and_wait(lambda: self.lk.move_v(round(dy_right), round(dx_fwd), 0),
                              Mode.ACK_CHASSIS, C.CMD_TIMEOUT,
                              f"move_v(前{dx_fwd:+.0f}, 右{dy_right:+.0f})")

    # ---- 标定防混淆: `*_RING` 必填 + 量级自检 ----

    def _calib_check(self):
        """环面标定可用性: 未填=拒跑(无回退); 宽松合理带=填反/填同告警。

        比例尺比 = PX_PER_MM_RING/PX_PER_MM, 环观测位更高 → 应明显更小;
        精确期望比依赖两处标定面高度(均靠尺量, 不确切), 只卡 0.4~0.9 带宽。"""
        if C.PX_PER_MM_RING is None or C.IMG2BODY_RING is None:
            print("!! 环面标定未填(PX_PER_MM_RING/IMG2BODY_RING): 在环观测位跑 "
                  "tools/calib_ratio.py --ring 与 tests/test_img2body_cal.py --ring 填 config")
            return False
        ratio = C.PX_PER_MM_RING / C.PX_PER_MM
        big_ring = max(abs(v) for v in C.IMG2BODY_RING)
        big_ref = max(abs(v) for v in C.IMG2BODY)
        if not 0.4 < ratio < 0.9 or big_ring < 1.05 * big_ref:
            print(f"!! 环标定疑似填错位置: 比例尺比转盘 {ratio:.2f}(合理带 0.4~0.9), "
                  f"矩阵量级 {big_ring:.4f} vs 转盘 {big_ref:.4f}"
                  f"(环位更高, mm/px 应更大) —— 核对 config 环放置段")
        return True

    # ---- 定一: 对画面中间的环整车纠偏 ----

    @staticmethod
    def _to_body(ring):
        """图像偏差(px) → 车体系 (dy_right, dx_fwd), 环面矩阵 IMG2BODY_RING。"""
        a, b, c, d = C.IMG2BODY_RING
        return a * ring.dx + b * ring.dy, c * ring.dx + d * ring.dy

    def _ring_observe(self, frame):
        """find_stable 回调: 检环 → 返回 cx 居中者(定一目标, 与数字无关)。"""
        rings = find_rings(frame)
        if not rings:
            return None
        return min(rings, key=lambda r: abs(r.cx - frame.shape[1] / 2))

    def align_mid(self):
        """看-动-再看, 直到 cx 居中环进死区(骨架=grab.align, 映射用环面标定)。

        返回 数字→位置映射 或 None(全程未检出环)。收敛/超时带误差都返回最后
        一帧的映射(保完赛继续放); 位置排序对车微调鲁棒(环间距 >> 纠偏幅度)。"""
        if not self._calib_check():
            return None   # test_place_ring 直接调本方法, 标定未填须同样拒跑
        deadline = time.time() + C.ALIGN_TIMEOUT
        residual = None
        dmap = None
        for i in range(1, C.ALIGN_MAX_ITER + 1):
            obj, frame = find_stable(self._ring_observe, self.cam,
                                     timeout=max(0.5, deadline - time.time()))
            if obj is None:
                print(f"  [环纠偏] 第{i}轮未检出数字环, 放弃对齐")
                return dmap   # 之前轮检出过则仍有映射可用
            dy_right, dx_fwd = self._to_body(obj)
            residual = math.hypot(dx_fwd, dy_right)
            dmap = self._digit_map(frame)
            if residual <= C.ALIGN_TOL_MM:
                print(f"  ✓ 环纠偏结束 (残差 {residual:.1f}mm, {i} 轮)")
                return dmap
            if time.time() > deadline:
                break
            print(f"  [环纠偏] 第{i}轮: 前{dx_fwd:+.1f} 右{dy_right:+.1f} "
                  f"(残差 {residual:.1f}mm) → move_v")
            self._move_v(dx_fwd, dy_right)
            time.sleep(C.ALIGN_SETTLE)   # 等车沉降再取帧
        print(f"  环纠偏结束(带误差 {residual if residual is not None else -1:.1f}mm, 保完赛)")
        return dmap

    def _digit_map(self, frame):
        """收敛帧 数字→位置 {1/2/3: 'L'/'M'/'R'}: 检出环按画面位置分档(以画面
        中心为 M 基准, 半间距阈值 RING_PITCH_PX/2 —— 与"定一对中间环"同先验);
        未检出数字按默认次序(1,2,3)从左到右填空位+告警(用户口径"未检出默认123")。"""
        pos = {}
        for r in find_rings(frame):
            pos[r.digit] = ("L" if r.dx < -C.RING_PITCH_PX / 2
                            else "R" if r.dx > C.RING_PITCH_PX / 2 else "M")
        rest = [d for d in _DEFAULT_ORDER if d not in pos]
        empty = [p for p in "LMR" if p not in pos.values()]
        for d, p in zip(rest, empty):
            pos[d] = p
            print(f"  !! 数字 {d} 未检出, 按默认次序补 {p} 位")
        return pos

    # ---- 布点放置(查表) ----

    @staticmethod
    def slot_pose(digit, dmap):
        """数字 → 示教臂位姿: dmap[数字] 给位置, 查 RING_SLOT_L/M/R。"""
        return {"L": C.RING_SLOT_L, "M": C.RING_SLOT_M,
                "R": C.RING_SLOT_R}[dmap[digit]]

    def place_one(self, carrier, pose, ack_counter, color=""):
        """盘上取一颗 → 放 pose 环位。8 回执: 盘上(上方→下探→夹→抬)
        + 环上(上方→下探→张→抬)。盘上下探深度=放置下探同款(Y-200, 放多深取多深)。"""
        r, h, t = carrier
        self.move_arm(r, C.H_LIFT, t, label=f"盘上上方({color})")
        self.move_arm(r, h - 200, t, label="下探取料")
        self.grip(close=True)
        self.move_arm(r, C.H_LIFT, t, label="抬到最高")
        rr, rh, rt = pose
        self.move_arm(rr, C.H_LIFT, rt, label=f"环位上方({color})")
        self.move_arm(rr, C.H_PLACE_RING, rt, label="下探放环")
        self.grip(close=False)
        self.move_arm(rr, C.H_LIFT, rt, label="抬到最高")
        ack_counter[0] += 8

    def run_place3(self, task, placed_colors=None):
        """放3: 摆观测位 → 对中环纠偏(定一) → 任务码颜色序 盘→环 查表放置 → 收臂。

        placed_colors: 抓3实际放上盘的颜色序(run_grab3 返回值, 决定每色从盘上
        哪个点位取); None=按任务码假定三颗都在盘上。
        """
        if not self._calib_check():
            return []
        colors, rings = task["batch1_colors"], task["batch1_rings"]
        on_carrier = list(placed_colors) if placed_colors is not None else list(colors)
        pts = [C.CARRIER_P1, C.CARRIER_P2, C.CARRIER_P3]
        carrier_of = {}
        for i, c in enumerate(on_carrier[:3]):
            carrier_of.setdefault(c, []).append(pts[i])
        order = [(c, r) for c, r in zip(colors, rings) if carrier_of.get(c)]
        for c, r in zip(colors, rings):
            if not carrier_of.get(c):
                print(f"  !! {c} 不在载物盘(抓3被跳过), {r} 号环跳过")
        if not order:
            print("  !! 盘上无料可放, 放3环节跳过")
            return []
        self.move_arm(*C.RING_ALIGN, label="到圆环纠偏位")
        dmap = self.align_mid()
        if dmap is None:
            print("  !! 环区未见数字环(未到位/识别失败), 放置跳过 —— 料保留在盘上")
            self.move_arm(*C.OBSERVE, label="收臂回复位")
            return []
        counter = [0]
        done = []
        for color, ring_no in order:
            if ring_no not in (1, 2, 3):
                print(f"  !! 环号 {ring_no} 超出 1~3(模型与默认次序均不含), 该颗跳过")
                continue
            carrier = carrier_of[color].pop(0)
            pose = self.slot_pose(ring_no, dmap)
            print(f"  放 {color} → {ring_no} 号环({dmap[ring_no]} 位) 臂位 {pose}")
            self.place_one(carrier, pose, counter, color)
            done.append(color)
        self.move_arm(*C.OBSERVE, label="收臂回复位")
        total = counter[0] + 1   # +收臂归位回执
        if total >= C.RING_ACK_TOTAL:
            print(f"  ✓ 放3环节结束: 回执 {total}/{C.RING_ACK_TOTAL}")
        else:
            print(f"  !! 看门狗: 放3 回执 {total}/{C.RING_ACK_TOTAL} 未满 —— 保完赛继续")
        self.stats["place_ok"], self.stats["place_total"] = len(done), 3
        return done

    def run(self, task=None, placed_colors=None):
        """摆观测位 → 对中环纠偏 → 放3。task=qr.parse_task 结果或 None(默认码)。"""
        if task is None:
            from vision.qr import parse_task
            task = parse_task(C.DEFAULT_TASK_CODE)
        print(f"== Step_Ring: 环纠偏 → 放3 {task['batch1_rings']} ==")
        return self.run_place3(task, placed_colors)
