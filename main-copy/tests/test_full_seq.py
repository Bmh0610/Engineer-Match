"""整场流程联调(2026-10-27 口径): 腿序列 + 三动作组 + 记账回中, 车会真动。

流程: 前进1000→扫码(末端相机尝试→兜底码)→顺转90→前进850→右移950
→【抓料: 定一(复位位纠偏建锚)→升观察位判色→按任务码颜色序分槽执行】
→逆转90→前进920→前进850→逆转90
→【放环: 对中环纠偏建锚(环空数字可见)→按任务码环号对数字环纠偏→盘→环】
→【抓环: 不开视觉——沿用放环的账本与数字→位置映射, ±150 记账送位, x=0 直下抓】
→前进820→顺转90→前进800→【放环→抓环】(暂存区同款重复)
→前进850→顺转90→前进850→前进800→顺转90→右移50→后退50→完成(回启停区)。

总原则: 锚点只在视觉可用时建一次(转盘=复位位定一/环区=放环开场环空数字可见);
之后全靠记账——每轮抓/放完回中(发反向和+清零), 账零=车在锚点; 环距150已知
(侧环=锚点±150); 行程腿 move_v 不入账; 抓环的数字→位置映射复用放环的检出
结果, 检不出按 右=1/中=2/左=3 兜底。

用法(Nano): python3 tests/test_full_seq.py [--skip-to start|rough|buffer|home]
  start=全流程(默认) rough/buffer=从该区动作组起跑(车人工摆位) home=只跑回程腿。
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

_DIGIT_POS_DEFAULT = {1: "right", 2: "mid", 3: "left"}

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


def run_step(lk, label, tx, ack):
    _STEP[0] += 1
    confirm(f"{label}\n  hex={tx.hex(' ').upper()}")
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


def arm(lk, label, pose):
    run_step(lk, f"{label} {tuple(pose)}", frame(Mode.MOVE_A, *pose), Mode.ACK_ARM)


def chassis(lk, label, dx, dy):
    """底盘送位(入账): 右 dx, 前 dy mm——只用于工作区内微调。"""
    _LEDGER.add(dx, dy)
    run_step(lk, f"{label} (右{dx:+.0f},前{dy:+.0f})mm",
             frame(Mode.MOVE_V, dx, dy, 0), Mode.ACK_CHASSIS)


def leg_move(lk, dx, dy, dz, label):
    """行程腿 move_v(不入账——腿间移动, 非工作区微调)。"""
    run_step(lk, f"[腿] {label} (右{dx:+.0f} 前{dy:+.0f} 转{dz:+.0f})",
             frame(Mode.MOVE_V, dx, dy, dz), Mode.ACK_CHASSIS)


def return_to_anchor(lk, tag):
    """回中: 发记账反向和并清零。账零自动跳过; 不依赖视觉(死账)。"""
    if _LEDGER.at_anchor:
        print(f"  · [{tag}]已在中位(记账为零), 跳过回中")
        return
    dy, dx = -_LEDGER.dx, -_LEDGER.dy
    run_step(lk, f"回中[{tag}] (记账 右{_LEDGER.dx:+.0f} 前{_LEDGER.dy:+.0f}"
                 f" → 反向 右{dy:+.0f} 前{dx:+.0f})",
             frame(Mode.MOVE_V, round(dy), round(dx), 0), Mode.ACK_CHASSIS)
    _LEDGER.zero()


def grip(lk, close):
    ang, t = C.GRIP_CLOSE if close else C.GRIP_OPEN
    run_step(lk, f"夹爪{'闭合' if close else '张开'}({ang})",
             frame(Mode.GRIPPER, ang, t), Mode.ACK_GRIPPER)


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


def _align(lk, cam, find_fn, matrix, label, timeout):
    """整车纠偏: 回车触发, s=跳过; 死区3mm ≤5轮, 超时带误差继续(保完赛)。"""
    try:
        ans = input(f"\n<- 回车={label}(看-动-再看) / s=跳过: ").strip().lower()
    except EOFError:
        print("  (无输入) 跳过纠偏")
        return
    if ans == "s":
        print("  !! 跳过纠偏(人工兜底)")
        return
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


def find_material_fn(backend, color=None):
    """color=None 找全部取最近; 给色只看该色(颜色驱动抓取)。"""
    def find(f):
        mats = finder.find_materials(backend, f,
                                     targets=[color] if color else None,
                                     conf=0.5, px_per_mm=C.PX_PER_MM)
        return finder.nearest_to_axis(mats, f.shape)
    return find


def find_ring_mid_fn(backend):
    """中环=离画面中心最近的数字环(建锚用; 建锚时环空数字可见)。"""
    def find(f):
        rings = find_rings(backend, f)
        return min(rings, key=lambda r: r.dx ** 2 + r.dy ** 2) if rings else None
    return find


def ring_position(ring):
    """检出环的画面位置: 左/中/右(以画面中心为基准, 环距像素阈值为档)。"""
    half = C.RING_PITCH_PX / 2
    return ("left" if ring.dx < -half
            else "right" if ring.dx > half else "mid")


def prepare_ring_target(backend, cam, digit):
    """先探一帧定找法: digit 检出→按数字对位(并记实际位置); 未检出→右1中2左3 兜底。
    返回 (find_fn, position)。position 记入映射给抓环用(不开视觉也能知道去哪抓)。"""
    f = cam.read_latest(timeout=1.0)
    rings = find_rings(backend, f) if f is not None else []
    hit = next((r for r in rings if r.digit == digit), None)
    if hit is not None:
        pos = ring_position(hit)
        print(f"  · {digit} 号环已检出(在{pos}位), 按数字对位")

        def find(f_, _d=digit):
            rs = [r for r in find_rings(backend, f_) if r.digit == _d]
            return rs[0] if rs else None
        return find, pos
    pos = _DIGIT_POS_DEFAULT.get(digit, "mid")
    print(f"  !! 数字 {digit} 未检出, 按默认位置兜底: {pos} 位 (右=1/中=2/左=3)")

    def find(f_, _p=pos):
        rs = find_rings(backend, f_)
        if not rs:
            return None
        if _p == "left":
            return min(rs, key=lambda r: r.cx)
        if _p == "right":
            return max(rs, key=lambda r: r.cx)
        return min(rs, key=lambda r: r.dx ** 2 + r.dy ** 2)
    return find, pos


# ---- 盘上槽位与取放(2026-10-27 实测三点位; 爪子三段口径) ----

def carrier_pose(slot, work=True):
    """盘上槽位位姿: slot=0/1/2 → P1中间/P2车头/P3车尾; work=工作高, 否则悬停(H_LIFT)。"""
    p = (C.CARRIER_P1, C.CARRIER_P2, C.CARRIER_P3)[slot]
    return (p[0], p[1] if work else float(C.H_LIFT), p[2])


def grip_to(lk, ang_t, label):
    ang, t = ang_t
    run_step(lk, f"{label}({ang})", frame(Mode.GRIPPER, ang, t), Mode.ACK_GRIPPER)


def place_to_carrier(lk, slot):
    """盘上放置: 悬停 → 开1225(松一档) → 下探 → 开1150(放料) → 升回观察位 → 开800。"""
    arm(lk, "到盘槽位上方", carrier_pose(slot, work=False))
    grip_to(lk, C.GRIP_OPEN_WIDE, "盘上悬停松一档")
    arm(lk, "下探放料", carrier_pose(slot, work=True))
    grip_to(lk, C.GRIP_RELEASE, "放料小开度")
    arm(lk, "升回观察位", C.OBSERVE)
    grip_to(lk, C.GRIP_OPEN, "回复位开度")


def pick_from_carrier(lk, slot):
    """盘上取料: 悬停 → 下探 → 夹1400 → 抬悬停 → 开1225 → 回复位。"""
    arm(lk, "到盘槽位上方", carrier_pose(slot, work=False))
    arm(lk, "下探取料", carrier_pose(slot, work=True))
    grip(lk, close=True)
    arm(lk, "抬回盘上方", carrier_pose(slot, work=False))
    grip_to(lk, C.GRIP_OPEN_WIDE, "抬悬停松一档")
    arm(lk, "回复位", C.RESET)


# ---- 动作组①: 抓料(转盘→盘, 颜色驱动) ----

def classify_slots(mats, frame_shape):
    """三物料→{槽位: Material}: 近端=离画面中心最近; 画面左=车头端/右=车尾端。"""
    h, w = frame_shape[:2]
    near = min(mats, key=lambda m: (m.cx - w / 2) ** 2 + (m.cy - h / 2) ** 2)
    others = [m for m in mats if m is not near]
    left, right = sorted(others, key=lambda m: m.cx)
    if C.FRONT_REAR_SWAP:
        left, right = right, left
    return {"near": near, "front": left, "rear": right}


def observe_slots(cam, backend, timeout):
    """观察位判色: 检出三物料→{色: 槽位}; 失败/s→None(调用方按近/头/尾兜底)。"""
    try:
        ans = input("\n<- 回车=观察判色(定各色槽位) / s=跳过(按近/头/尾兜底): ").strip().lower()
    except EOFError:
        print("  (无输入) 跳过观察, 按近/头/尾兜底")
        return None
    if ans == "s":
        print("  !! 跳过观察判色")
        return None
    deadline = time.time() + timeout * 2
    while time.time() < deadline:
        f = cam.read_latest(timeout=0.2)
        if f is None:
            continue
        mats = finder.find_materials(backend, f, conf=0.5, px_per_mm=C.PX_PER_MM)
        if len(mats) >= 3:
            slots = classify_slots(mats, f.shape)
            print("  ✓ 判色: " + ", ".join(f"{m.color}={s}" for s, m in slots.items()))
            return {m.color: s for s, m in slots.items()}
        time.sleep(0.2)
    print("  !! 观察超时未凑齐三物料, 按近/头/尾兜底")
    return None


def action_grab_materials(lk, cam, backend, task, timeout):
    """抓料×3轮: 定一(复位位, 建锚)→观察位判色→按任务码颜色序分槽抓→回中→放盘。"""
    print("\n===== 动作组①: 抓料 (定一复位位 → 观察判色 → 按色分槽) =====")
    arm(lk, "缩回复位(定一纠偏位)", C.RESET)
    _align(lk, cam, find_material_fn(backend), C.IMG2BODY,
           "定一纠偏(建锚)", timeout)
    _LEDGER.zero()
    arm(lk, "升到观察位", C.OBSERVE)
    slots = observe_slots(cam, backend, timeout)
    colors = task["batch1_colors"]
    if slots is None:
        slots = dict(zip(colors, ("near", "front", "rear")))
        print(f"  !! 兜底槽位: {slots}")
    slot = 0   # 槽位轮转: 第 i 颗抓的放 P_{i+1} (P1中间→P2车头→P3车尾), 只在实际抓到时推进
    for color in colors:
        where = slots.get(color)
        if where is None:
            print(f"  !! {color} 不在观察结果, 跳过该轮(保完赛)")
            continue
        print(f"\n----- 抓 {color} ({where}) → 盘P{slot + 1} -----")
        if where == "near":
            arm(lk, "缩回复位", C.RESET)
            arm(lk, "下降抓取(离地150)", C.GRAB_NEAR)
            grip(lk, close=True)
            arm(lk, "升回复位", C.RESET)
            return_to_anchor(lk, color)
        else:
            dy = C.TURN_STEP_MM if where == "front" else -C.TURN_STEP_MM
            arm(lk, "前伸最大(145)", (C.ARM_REACH_MAX, C.H_LIFT, 0))
            chassis(lk, f"底盘粗送位(前{dy:+.0f})", 0, dy)
            _align(lk, cam, find_material_fn(backend, color), C.IMG2BODY,
                   f"{color}纠偏", timeout)
            arm(lk, "下降抓取(离地150)", (C.ARM_REACH_MAX, C.GRAB_NEAR[1], 0))
            grip(lk, close=True)
            arm(lk, "升到最高", (C.ARM_REACH_MAX, C.H_LIFT, 0))
            arm(lk, "回复位", C.RESET)
            return_to_anchor(lk, color)
        place_to_carrier(lk, slot)
        slot += 1
    print("\n== 抓料完成 (车在锚点=中间物料位) ==")


# ---- 动作组②: 放环(盘→环, 数字驱动) — 返回数字→位置映射给抓环 ----

def _recheck(lk, cam, backend, timeout):
    """回中后视觉复核: 目标=画面中心最近的物料(=中环那颗), 消记账残差; 后账清零。"""
    try:
        ans = input("  <- 回车=回中复核(对中环物料再纠偏) / s=跳过: ").strip().lower()
    except EOFError:
        print("  (无输入) 跳过复核")
    else:
        if ans == "s":
            print("  !! 跳过复核(人工兜底)")
        else:
            _align(lk, cam, find_material_fn(backend), C.IMG2BODY,
                   "回中复核", timeout)
    _LEDGER.zero()


def action_place_rings(lk, cam, backend, task, timeout):
    """放环×3轮: 对中环纠偏建锚 → 按任务码(色,环号)对数字环纠偏 → 盘→环 → 回中。"""
    print("\n===== 动作组②: 放环 (数字驱动, 建锚后每轮回中) =====")
    arm(lk, "回复位", C.RESET)
    _align(lk, cam, find_ring_mid_fn(backend), C.IMG2BODY_RING,
           "对中环纠偏(建锚)", timeout)
    _LEDGER.zero()
    ring_map = {}
    for i, (color, ring_no) in enumerate(
            zip(task["batch1_colors"], task["batch1_rings"])):
        print(f"\n----- 放 {color} → {ring_no} 号环 (取盘P{i + 1}) -----")
        find_fn, pos = prepare_ring_target(backend, cam, ring_no)
        ring_map[ring_no] = pos
        _align(lk, cam, find_fn, C.IMG2BODY_RING,
               f"{ring_no}号环纠偏", timeout)
        pick_from_carrier(lk, i)
        arm(lk, "下探放环(离地65)", (C.RESET[0], C.H_PLACE_RING, C.RESET[2]))
        grip(lk, close=False)
        arm(lk, "升回复位", C.RESET)
        return_to_anchor(lk, f"{ring_no}号环")
        if pos != "mid":
            _recheck(lk, cam, backend, timeout)
    print(f"\n== 放环完成 (数字→位置映射: {ring_map}, 车在中环锚点位) ==")
    return ring_map


# ---- 动作组③: 抓环(环→盘, 纯记忆不开视觉) ----

def action_grab_rings(lk, task, ring_map):
    """抓环×3轮: 不开视觉——账本(锚点=中环位)+放环记下的数字→位置映射。
    侧环=锚点±150 记账送位; 中环直接下探; 每轮抓完回中→放盘。"""
    print("\n===== 动作组③: 抓环 (纯记忆: 账本+位置映射, 无视觉) =====")
    for i, (color, ring_no) in enumerate(
            zip(task["batch1_colors"], task["batch1_rings"])):
        pos = ring_map.get(ring_no) or _DIGIT_POS_DEFAULT.get(ring_no, "mid")
        print(f"\n----- 抓 {ring_no} 号环({pos}位) → 盘P{i + 1} -----")
        arm(lk, "回复位", C.RESET)
        if pos != "mid":
            dx = C.RING_PITCH_MM if pos == "right" else -C.RING_PITCH_MM
            chassis(lk, f"底盘记账送位(右{dx:+.0f})", dx, 0)
        arm(lk, "下探抓环(离地65)", (C.RESET[0], C.H_GRAB_RING, C.RESET[2]))
        grip(lk, close=True)
        arm(lk, "升回复位", C.RESET)
        return_to_anchor(lk, f"{ring_no}号环")
        place_to_carrier(lk, i)
    print("\n== 抓环完成 (车在中环锚点位, 纯记忆) ==")


# ---- 扫码 ----

def scan_step(cam):
    """扫码: 末端相机尝试 SCAN_TIMEOUT(扫码相机未接), 超时用兜底码。"""
    try:
        ans = input(f"\n<- 回车=扫码({C.SCAN_TIMEOUT:.0f}s, 超时用兜底码) "
                    f"/ s=直接用兜底码: ").strip().lower()
    except EOFError:
        ans = ""
    task = None
    if ans != "s":
        from vision.qr import QrScanner
        task = QrScanner().scan_from_camera(cam, C.SCAN_TIMEOUT)
    if task is None:
        task = parse_task(C.DEFAULT_TASK_CODE)
        print(f"  !! 用兜底码 {C.DEFAULT_TASK_CODE}")
    print(f"  任务码: {task['raw']} | 色序 {task['batch1_colors']} "
          f"环号 {task['batch1_rings']}")
    return task


# ---- 主流程 ----

def main():
    import argparse
    ap = argparse.ArgumentParser(description="整场流程联调(腿+三动作组+记账回中)")
    ap.add_argument("--port", default=C.SERIAL_PORT)
    ap.add_argument("--model", default=C.MODEL_PATH)
    ap.add_argument("--skip-to", choices=("start", "rough", "buffer", "home"),
                    default="start",
                    help="start=全流程(默认); rough/buffer=从该区动作组起跑"
                         "(车人工摆位); home=只跑回程腿")
    ap.add_argument("--timeout", type=float, default=5.0,
                    help="纠偏单轮等稳定帧超时(秒)")
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
        leg_move(lk, 0, 1000, 0, "前进1000(到扫码位)")
        task = scan_step(cam)
        leg_move(lk, 0, 0, -90, "顺时针转90")
        leg_move(lk, 0, 850, 0, "前进850")
        leg_move(lk, 950, 0, 0, "右移950")
        action_grab_materials(lk, cam, backend, task, args.timeout)
        leg_move(lk, 0, 0, 90, "逆时针转90")
        leg_move(lk, 0, 920, 0, "前进920")
        leg_move(lk, 0, 850, 0, "前进850")
        leg_move(lk, 0, 0, 90, "逆时针转90")
    if args.skip_to in ("start", "rough"):
        if task is None:
            task = parse_task(C.DEFAULT_TASK_CODE)
        ring_map = action_place_rings(lk, cam, backend, task, args.timeout)
        action_grab_rings(lk, task, ring_map)
        leg_move(lk, 0, 820, 0, "前进820")
        leg_move(lk, 0, 0, -90, "顺时针转90")
        leg_move(lk, 0, 800, 0, "前进800(到暂存区)")
    if args.skip_to in ("start", "rough", "buffer"):
        if task is None:
            task = parse_task(C.DEFAULT_TASK_CODE)
        ring_map = action_place_rings(lk, cam, backend, task, args.timeout)
        action_grab_rings(lk, task, ring_map)
    if args.skip_to in ("start", "rough", "buffer", "home"):
        leg_move(lk, 0, 850, 0, "前进850")
        leg_move(lk, 0, 0, -90, "顺时针转90")
        leg_move(lk, 0, 850, 0, "前进850")
        leg_move(lk, 0, 800, 0, "前进800")
        leg_move(lk, 0, 0, -90, "顺时针转90")
        leg_move(lk, 50, 0, 0, "右移50")
        leg_move(lk, 0, -50, 0, "后退50(回启停区)")
    print("\n== 全部完成 (账本为零=车在锚点位) ==")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n中止")
