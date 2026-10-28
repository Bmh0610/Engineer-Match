"""整场流程联调·守株待兔抓取版(2026-10-27): 与 test_full_seq.py 唯一区别=抓料换笨办法。

抓料(笨办法): 定一纠偏(复位位, 建锚)之后车一动不动(账本恒零, 无前伸/无±90送位/
无逐色纠偏), 臂停复位位盯夹爪正下方——任务码第1组下一色进抓取窗(GRAB_WINDOW)
且连帧稳定(转盘停歇, finder.wait_for_color) → 下探(0,1500,0)抓 → 放盘 P_i
(三段爪 1225/1150/800) → 回复位等下一色。单色等待超时默认 60s(--wait 可改;
比赛值按实车转盘一圈周期调 config WAIT_COLOR_TIMEOUT, 建议≥1.5圈), 超时跳过
该色槽位不推进(保完赛)。

其余(腿/扫码/放环臂回转查表 左1中2右3/抓环纯臂回转无底盘/纠偏前延时3s)
与 test_full_seq.py 完全同款, 全文见那份。

流程: 右移150→前进1000→扫码→顺转90→前进850→顺转90→前进875→逆转90→【守株抓料】
→逆转90→前进920→前进850→逆转90→【放环】【抓环】(粗加工区)→后退820→顺转90
→后退800(暂存区)→【放环】【抓环】→后退850→顺转90→后退850→后退800→逆转90
→左移50→后退50→完成(回启停区)。

用法: python3 tests/test_full_seq_wait.py [--skip-to start|rough|buffer|home]
      [--wait 60]
每步回车执行, 打印帧 hex, 等对应完成回执, 条间 2s; Ctrl+C 退出。
"""
import math
import sys
import time
from pathlib import Path

BASE = str(Path(__file__).resolve().parent.parent)   # main-copy 根(双端通用)
sys.path.insert(0, BASE)

import config as C  # noqa: E402
from communication.protocol import Mode, frame  # noqa: E402
from vision import finder  # noqa: E402
from vision.qr import parse_task  # noqa: E402
from task.ring import find_rings  # noqa: E402

ACK_NAME = {Mode.ACK_CHASSIS: "0x0011底盘", Mode.ACK_ARM: "0x0012臂",
            Mode.ACK_GRIPPER: "0x0013夹爪"}


_STEP = [0]


class ChassisLedger:
    """底盘记账器: 工作区所有 move_v 入账, 回中=发反向和+清零。
    不变量: 回中(含复核)后账本为零; 账本非零=不在锚点。"""

    def __init__(self):
        self.dx = 0.0   # 右向累计 mm
        self.dy = 0.0   # 前向累计 mm

    def add(self, dy_right, dx_fwd):
        self.dx += dy_right
        self.dy += dx_fwd

    def zero(self):
        self.dx = self.dy = 0.0

    @property
    def at_anchor(self):
        return abs(self.dx) < 1e-6 and abs(self.dy) < 1e-6


_LEDGER = ChassisLedger()


def confirm(msg):
    try:
        input(f"\n[{_STEP[0]}] <- 回车: {msg}")
    except EOFError:
        pass


def run_step(lk, label, tx, ack, auto=False):
    """发帧等回执; auto=True 跳过回车确认直接执行(连续段), 默认人工门控。"""
    _STEP[0] += 1
    if not auto:
        confirm(f"{label}\n  hex={tx.hex(' ').upper()}")
    else:
        print(f"[{_STEP[0]}] {label}")
    lk.flush()
    lk.ser.write(tx)
    t0 = time.time()
    got = None
    while time.time() - t0 < C.CMD_TIMEOUT:
        for f in lk.poll():
            print(f"    RX {f}")
            if f.mode == ack and got is None:
                got = f
        if got:
            break
        time.sleep(0.03)
    if got:
        print(f"    ✓ {ACK_NAME[ack]}完成 ({time.time()-t0:.1f}s)")
    else:
        print(f"    !! {C.CMD_TIMEOUT:.0f}s 无 {ACK_NAME[ack]} 完成回执")
    time.sleep(2)


def arm(lk, label, pose, auto=False):
    run_step(lk, f"{label} {tuple(pose)}", frame(Mode.MOVE_A, *pose),
             Mode.ACK_ARM, auto=auto)


def chassis(lk, label, dx, dy, auto=False):
    """底盘送位(入账): 右 dx, 前 dy mm——只用于工作区内微调。"""
    _LEDGER.add(dx, dy)
    run_step(lk, f"{label} (右{dx:+.0f},前{dy:+.0f})mm",
             frame(Mode.MOVE_V, dx, dy, 0), Mode.ACK_CHASSIS, auto=auto)


def leg_move(lk, dx, dy, dz, label):
    """行程腿 move_v(不入账——腿间移动, 非工作区微调); 连续跑不回车。"""
    run_step(lk, f"[腿] {label} (右{dx:+.0f} 前{dy:+.0f} 转{dz:+.0f})",
             frame(Mode.MOVE_V, dx, dy, dz), Mode.ACK_CHASSIS, auto=True)


def return_to_anchor(lk, tag, auto=False):
    """回中: 发记账反向和并清零。账零自动跳过; 不依赖视觉(死账)。"""
    if _LEDGER.at_anchor:
        print(f"  · [{tag}]已在中位(记账为零), 跳过回中")
        return
    dy, dx = -_LEDGER.dx, -_LEDGER.dy
    run_step(lk, f"回中[{tag}] (记账 右{_LEDGER.dx:+.0f} 前{_LEDGER.dy:+.0f}"
                 f" → 反向 右{dy:+.0f} 前{dx:+.0f})",
             frame(Mode.MOVE_V, round(dy), round(dx), 0), Mode.ACK_CHASSIS,
             auto=auto)
    _LEDGER.zero()


def grip(lk, close, auto=False):
    ang, t = C.GRIP_CLOSE if close else C.GRIP_OPEN
    run_step(lk, f"夹爪{'闭合' if close else '张开'}({ang})",
             frame(Mode.GRIPPER, ang, t), Mode.ACK_GRIPPER, auto=auto)


# ---- 纠偏(看-动-再看): 纠偏 move_v 直发并入账 ----

def _send_move_v(lk, dy_right, dx_fwd):
    _LEDGER.add(dy_right, dx_fwd)
    lk.flush()
    lk.ser.write(frame(Mode.MOVE_V, round(dy_right), round(dx_fwd), 0))
    t0 = time.time()
    while time.time() - t0 < C.CMD_TIMEOUT:
        for f in lk.poll():
            if f.mode == Mode.ACK_CHASSIS:
                return True
        time.sleep(0.03)
    print("  !! 纠偏 move_v 无回执(继续)")
    return False


def _align(lk, cam, find_fn, matrix, label, timeout, auto=False):
    """整车纠偏: 看-动-再看, 死区3mm ≤5轮, 超时带误差继续(保完赛)。
    auto=True 不弹门控直接跑(定一用); 默认回车触发, s=跳过(人工兜底)。"""
    if not auto:
        try:
            ans = input(f"\n<- 回车={label}(看-动-再看) / s=跳过: ").strip().lower()
        except EOFError:
            print("  (无输入) 跳过纠偏")
            return
        if ans == "s":
            print("  !! 跳过纠偏(人工兜底)")
            return
    print(f"  · 纠偏前延时 {C.ALIGN_PRE_DELAY:.0f}s…")
    time.sleep(C.ALIGN_PRE_DELAY)
    deadline = time.time() + C.ALIGN_TIMEOUT
    for i in range(1, C.ALIGN_MAX_ITER + 1):
        obj, _fr = finder.find_stable(find_fn, cam,
                                      timeout=max(0.5, deadline - time.time()))
        if obj is None:
            print(f"  [纠偏] 第{i}轮未检出目标, 放弃(保完赛继续)")
            return
        a, b, c, d = matrix
        dy_right = a * obj.dx + b * obj.dy
        dx_fwd = c * obj.dx + d * obj.dy
        residual = math.hypot(dx_fwd, dy_right)
        if residual <= C.ALIGN_TOL_MM:
            print(f"  ✓ 纠偏结束 (残差 {residual:.1f}mm, {i} 轮)")
            return
        if time.time() > deadline:
            break
        print(f"  [纠偏] 第{i}轮: 前{dx_fwd:+.1f} 右{dy_right:+.1f} "
              f"(残差 {residual:.1f}mm) → move_v")
        _send_move_v(lk, dy_right, dx_fwd)
        time.sleep(C.ALIGN_SETTLE)
    print("  纠偏结束(带误差, 保完赛继续)")


def find_material_fn(backend):
    def find(f):
        mats = finder.find_materials(backend, f, conf=0.5, px_per_mm=C.PX_PER_MM)
        return finder.nearest_to_axis(mats, f.shape)
    return find


def find_ring_mid_fn(backend):
    """中环=离画面中心最近的数字环(建锚用; 建锚时环空数字可见)。"""
    def find(f):
        rings = find_rings(backend, f)
        return min(rings, key=lambda r: r.dx ** 2 + r.dy ** 2) if rings else None
    return find


def ring_slot_pose(ring_no):
    """环号→(下探位姿, 是否侧环): 左1=示教点 / 中2=x=0直下(H_PLACE_RING) / 右3=示教点。
    圆环从左到右=1,2,3(定死); 抓环同 reach/theta, 高度换 H_GRAB_RING。"""
    if ring_no == 2:
        return (C.RESET[0], C.H_PLACE_RING, C.RESET[2]), False
    pose = C.RING_SLOT_L if ring_no == 1 else C.RING_SLOT_R
    return pose, True


# ---- 载物舵盘(2026-10-27 改型: 舵机转盘选料, 固定取放点) ----

_DIAL_AT = 0      # 舵盘当前对准固定点的料位号(代码顺序记忆: 第i颗料在第i格)
_DIAL_HOMED = False


def rotate_dial(lk, slot, auto=False):
    """舵盘转位: 把 slot 号料位送到固定取放点。首次调用先回零找结束位标志。
    协议未定(DIAL_CMD_HOME/MODE=None)时只打印占位记账, 不发送。"""
    global _DIAL_AT, _DIAL_HOMED
    if slot == _DIAL_AT:
        return
    if not _DIAL_HOMED:
        print("  · [占位] 舵盘回零找结束位标志… (协议 MODE 待电控定)")
        _DIAL_HOMED = True
    ang = C.DIAL_SLOT_ANGLES[slot]
    label = f"舵盘转位: 料位{_DIAL_AT}→料位{slot} (舵机{ang:.0f})"
    if C.DIAL_CMD_MODE is None:
        print(f"  · [占位] {label} —— 协议未定, 只记账不发送")
    else:
        run_step(lk, label, frame(C.DIAL_CMD_MODE, int(ang), 0, 0),
                 Mode.ACK_CHASSIS, auto=auto)   # ⚠️回执常数待电控定, 暂按底盘
    _DIAL_AT = slot


def place_to_carrier(lk, slot, auto=False):
    """盘上放置(舵盘版): 转位到槽 → 固定点悬停 → 下探 → 张800 → 转下一格预备。"""
    rotate_dial(lk, slot, auto=auto)
    arm(lk, "到固定取放点悬停", C.DIAL_PICK_HOVER, auto=auto)
    arm(lk, "下探放料", C.DIAL_PICK_WORK, auto=auto)
    grip(lk, close=False, auto=auto)
    arm(lk, "升回观察位", C.OBSERVE, auto=auto)
    rotate_dial(lk, (slot + 1) % 3, auto=auto)   # 放完即转, 预备下一空槽


def pick_from_carrier(lk, slot, auto=False):
    """盘上取料(舵盘版): 转位到槽 → 固定点悬停 → 下探 → 夹1400 → 抬悬停 → 回复位。"""
    rotate_dial(lk, slot, auto=auto)
    arm(lk, "到固定取放点悬停", C.DIAL_PICK_HOVER, auto=auto)
    arm(lk, "下探取料", C.DIAL_PICK_WORK, auto=auto)
    grip(lk, close=True, auto=auto)
    arm(lk, "抬回固定点悬停", C.DIAL_PICK_HOVER, auto=auto)
    arm(lk, "回复位", C.RESET, auto=auto)


# ---- 动作组①: 抓料(转盘→盘, 守株待兔: 定一后车不动, 原地等色) ----

def wait_and_grab(lk, cam, backend, color, timeout):
    """原地等目标色: 进抓取窗+连帧稳定→报告看到的色→你确认对才抓;
    答"不对"回到等待(用剩余时间); 超时返回 False(槽位不推进, 保完赛)。
    抓取段连贯无回车; 下探带相机-夹爪补偿(GRIP_REACH_COMP)。"""
    print(f"\n<等 {color}: 转进窗+停歇即报, ≤{timeout:.0f}s>")
    deadline = time.time() + timeout
    while True:
        mat, _fr = finder.wait_for_color(backend, cam, color,
                                         px_per_mm=C.PX_PER_MM,
                                         timeout=max(0.5, deadline - time.time()))
        if mat is None:
            print(f"  !! {timeout:.0f}s 未等到 {color}, 跳过该色(保完赛)")
            return False
        a, b, c, d = C.IMG2BODY
        print(f"  ✓ 看到: {mat.color} (偏差 右{a*mat.dx+b*mat.dy:+.1f} "
              f"前{c*mat.dx+d*mat.dy:+.1f}mm)")
        try:
            ok = input("  <- 看到得对吗? 回车=对, 抓 / n=不对, 继续等: ").strip().lower()
        except EOFError:
            ok = ""
        if ok == "n":
            print("  · 继续等…")
            continue
        reach = C.GRAB_NEAR[0] + C.GRIP_REACH_COMP   # 相机比夹爪靠外15mm→多伸补偿
        arm(lk, "下探抓取(离地150,含补偿)", (reach, C.GRAB_NEAR[1], C.GRAB_NEAR[2]),
            auto=True)
        grip(lk, close=True, auto=True)
        arm(lk, "升回复位", C.RESET, auto=True)
        return True


def action_grab_materials(lk, cam, backend, task, wait):
    """抓料×3轮(笨办法): 定一(复位位, 建锚)后车一动不动, 原地等色+稳定即抓→放盘。"""
    print("\n===== 动作组①: 抓料 (守株待兔: 定一后原地等色, 车全程不动) =====")
    arm(lk, "缩回复位(定一纠偏位)", C.RESET)
    _align(lk, cam, find_material_fn(backend), C.IMG2BODY,
           "定一纠偏(建锚)", wait, auto=True)   # 定一免确认, 直接跑
    _LEDGER.zero()
    slot = 0   # 槽位轮转: 第 i 颗抓到的放 P_{i+1}, 只在实际抓到时推进
    for color in task["batch1_colors"]:
        if wait_and_grab(lk, cam, backend, color, wait):
            return_to_anchor(lk, color)   # 车不动, 恒跳过(统一口径)
            place_to_carrier(lk, slot, auto=True)   # 抓取动作连贯, 无逐步回车
            slot += 1
            arm(lk, "回复位(继续等下一色)", C.RESET, auto=True)
        else:
            print(f"  ({color} 未抓, 槽位不推进)")
    print("\n== 抓料完成 (车全程在锚点位, 账本为零) ==")


# ---- 动作组②: 放环(盘→环, 臂回转查表: 左1/中2/右3) ----

def _recheck(lk, cam, backend, timeout):
    """回中后视觉复核(自动直跑): 目标=画面中心最近的物料(=中环那颗),
    消记账残差; 后账清零。"""
    _align(lk, cam, find_material_fn(backend), C.IMG2BODY,
           "回中复核", timeout, auto=True)
    _LEDGER.zero()


def action_place_rings(lk, cam, backend, task, timeout):
    """放环×3轮: 对中环纠偏建锚(唯一底盘动作) → 环号定槽位(左1/中2/右3,
    圆环从左到右=1,2,3) → 臂回转查表放置(全连续, 无底盘/无回中/无数字识别)。"""
    print("\n===== 动作组②: 放环 (臂回转查表: 左1/中2/右3) =====")
    arm(lk, "回复位", C.RESET, auto=True)
    _align(lk, cam, find_ring_mid_fn(backend), C.IMG2BODY_RING,
           "对中环纠偏(建锚)", timeout, auto=True)
    _LEDGER.zero()
    for i, (color, ring_no) in enumerate(
            zip(task["batch1_colors"], task["batch1_rings"])):
        pose, side = ring_slot_pose(ring_no)
        pos_name = {1: "左", 2: "中", 3: "右"}[ring_no]
        print(f"\n----- 放 {color} → {ring_no} 号环({pos_name}位, 取盘P{i + 1}) -----")
        if side:
            arm(lk, "回转到环位悬停", (pose[0], float(C.H_LIFT), pose[2]), auto=True)
        pick_from_carrier(lk, i, auto=True)
        if side:
            arm(lk, "回环位悬停", (pose[0], float(C.H_LIFT), pose[2]), auto=True)
        arm(lk, "下探放环", pose, auto=True)
        grip(lk, close=False, auto=True)
        if side:
            arm(lk, "升回悬停", (pose[0], float(C.H_LIFT), pose[2]), auto=True)
        else:
            arm(lk, "升回复位", C.RESET, auto=True)
    print("\n== 放环完成 (车在中环锚点位) ==")


# ---- 动作组③: 抓环(环→盘, 纯记忆不开视觉) ----

def action_grab_rings(lk, task):
    """抓环×3轮: 纯记忆不开视觉——环号定槽位(左1/中2/右3), 臂回转查表下探抓→放盘。"""
    print("\n===== 动作组③: 抓环 (臂回转查表, 无底盘无视觉) =====")
    for i, (color, ring_no) in enumerate(
            zip(task["batch1_colors"], task["batch1_rings"])):
        pose, side = ring_slot_pose(ring_no)
        pos_name = {1: "左", 2: "中", 3: "右"}[ring_no]
        print(f"\n----- 抓 {ring_no} 号环({pos_name}位) → 盘P{i + 1} -----")
        if side:
            arm(lk, "回转到环位悬停", (pose[0], float(C.H_LIFT), pose[2]), auto=True)
            arm(lk, "下探抓环(离地65)", (pose[0], C.H_GRAB_RING, pose[2]), auto=True)
        else:
            arm(lk, "下探抓环(离地65)", (C.RESET[0], C.H_GRAB_RING, C.RESET[2]),
                auto=True)
        grip(lk, close=True, auto=True)
        if side:
            arm(lk, "升回悬停", (pose[0], float(C.H_LIFT), pose[2]), auto=True)
            arm(lk, "回复位", C.RESET, auto=True)
        else:
            arm(lk, "升回复位", C.RESET, auto=True)
        place_to_carrier(lk, i, auto=True)
    print("\n== 抓环完成 (车在中环锚点位, 纯记忆) ==")


# ---- 扫码 ----

def scan_step(cam):
    """扫码: 末端相机尝试 SCAN_TIMEOUT(扫码相机未接), 超时自动用兜底码; 无回车门控。"""
    from vision.qr import QrScanner
    print(f"<扫码: ≤{C.SCAN_TIMEOUT:.0f}s, 超时自动用兜底码>")
    task = QrScanner().scan_from_camera(cam, C.SCAN_TIMEOUT)
    if task is None:
        task = parse_task(C.DEFAULT_TASK_CODE)
        print(f"  !! 未扫到, 用兜底码 {C.DEFAULT_TASK_CODE}")
    print(f"  任务码: {task['raw']} | 色序 {task['batch1_colors']} "
          f"环号 {task['batch1_rings']}")
    return task


# ---- 主流程 ----

def main():
    import argparse
    ap = argparse.ArgumentParser(description="整场流程联调·守株待兔抓取版")
    ap.add_argument("--port", default=C.SERIAL_PORT)
    ap.add_argument("--model", default=C.MODEL_PATH)
    ap.add_argument("--skip-to", choices=("start", "rough", "buffer", "home"),
                    default="start",
                    help="start=全流程(默认); rough/buffer=从该区动作组起跑"
                         "(车人工摆位); home=只跑回程腿")
    ap.add_argument("--timeout", type=float, default=5.0,
                    help="纠偏单轮等稳定帧超时(秒)")
    ap.add_argument("--wait", type=float, default=60.0,
                    help="单色守株等待超时(秒, 默认60; 比赛值按转盘一圈周期调) "
                         "-- 门控可回车续等")
    args = ap.parse_args()

    from vision.camera import Camera
    from vision.detect import YOLOBackend
    from communication.protocol import Link

    cam = Camera(C.CAMERA_ID_END, "end")
    cam.open()
    print(f"加载模型 {args.model} ...")
    backend = YOLOBackend(args.model)
    lk = Link(args.port)
    print(f"口径: 观察{C.OBSERVE} 复位{C.RESET} 最大伸{C.ARM_REACH_MAX} "
          f"行距{C.TURN_STEP_MM} 环距{C.RING_PITCH_MM:.0f} "
          f"放/抓环高{C.H_PLACE_RING}/{C.H_GRAB_RING} | Ctrl+C 退出")

    task = None
    if args.skip_to == "start":
        leg_move(lk, 150, 0, 0, "右移150(起步对位)")
        leg_move(lk, 0, 1000, 0, "前进1000(到扫码位)")
        task = scan_step(cam)
        leg_move(lk, 0, 0, -90, "顺时针转90")
        leg_move(lk, 0, 850, 0, "前进850")
        leg_move(lk, 0, 0, -90, "顺时针转90")
        leg_move(lk, 0, 875, 0, "前进875")
        leg_move(lk, 0, 0, 90, "逆时针转90(机械臂朝物料盘)")
        action_grab_materials(lk, cam, backend, task, args.wait)
        leg_move(lk, 0, 0, 90, "逆时针转90")
        leg_move(lk, 0, 920, 0, "前进920")
        leg_move(lk, 0, 850, 0, "前进850")
        leg_move(lk, 0, 0, 90, "逆时针转90")
        leg_move(lk, 0, 75, 0, "前进75(到守株抓取位)")
    if args.skip_to in ("start", "rough"):
        if task is None:
            task = parse_task(C.DEFAULT_TASK_CODE)
        action_place_rings(lk, cam, backend, task, args.timeout)
        action_grab_rings(lk, task)
        leg_move(lk, 0, -820, 0, "后退820")
        leg_move(lk, 0, 0, -90, "顺时针转90")
        leg_move(lk, 0, -800, 0, "后退800(到暂存区)")
    if args.skip_to in ("start", "rough", "buffer"):
        if task is None:
            task = parse_task(C.DEFAULT_TASK_CODE)
        action_place_rings(lk, cam, backend, task, args.timeout)
        action_grab_rings(lk, task)
    if args.skip_to in ("start", "rough", "buffer", "home"):
        leg_move(lk, 0, -850, 0, "后退850")
        leg_move(lk, 0, 0, -90, "顺时针转90")
        leg_move(lk, 0, -850, 0, "后退850")
        leg_move(lk, 0, -800, 0, "后退800")
        leg_move(lk, 0, 0, 90, "逆时针转90")
        leg_move(lk, -50, 0, 0, "左移50")
        leg_move(lk, 0, -50, 0, "后退50(回启停区)")
    print("\n== 全部完成 (账本为零=车在锚点位) ==")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n中止")
