"""机械臂两动作组实车测试(底盘对位+纠偏+每轮回中, 2026-10-26/27 口径)。

A 抓料→放盘 ×3轮: 抓完(夹完)升回复位 → 回中 → 放盘 → 升回观察位;
B 取盘→放环 ×3轮: 全程复位位(无观察位), 对环纠偏把环送到复位位正下方后
x=0 直下放(离地65mm, 不前伸), 放完回中(左/右环回中后再复核一次)。

锚点与记账(整体流程同款架构): 锚点=A组测试开始位(人工对准近端物料)/
B组中环纠偏完成位(纠偏后账清零); 工作区内所有底盘位移(±90粗送位、纠偏
move_v)入账; 每轮结束发反向和回中并清零——回中后账本必为零, 末轮结束车
必在锚点(中间物料/中环位), 后续行程不压线; 纠偏失败的位移同样入账, 回中
不依赖视觉。B 组回中后可选视觉复核(目标=画面中心最近的物料=中环那颗),
把记账残差消到纠偏死区 3mm, 复核后账再清零(复核后位置=新锚点)。

用法(Nano):
  python3 tests/test_arm_seq.py                 # 两组连跑
  python3 tests/test_arm_seq.py --phase grab    # 只跑动作组A
  python3 tests/test_arm_seq.py --phase ring    # 只跑动作组B
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
from task.ring import find_rings  # noqa: E402

ACK_NAME = {Mode.ACK_CHASSIS: "0x0011底盘", Mode.ACK_ARM: "0x0012臂",
            Mode.ACK_GRIPPER: "0x0013夹爪"}

# ---- 盘上槽位与取放 ----

def carrier_pose(slot, work=True):
    """盘上槽位位姿: slot=0/1/2 → P1中间/P2车头/P3车尾; work=工作高, 否则悬停(H_LIFT)。"""
    p = (C.CARRIER_P1, C.CARRIER_P2, C.CARRIER_P3)[slot]
    return (p[0], p[1] if work else float(C.H_LIFT), p[2])


def place_to_carrier(lk, slot):
    """盘上放置: 悬停 → 下探 → 张800 → 升回观察位。"""
    arm(lk, "到盘槽位上方", carrier_pose(slot, work=False))
    arm(lk, "下探放料", carrier_pose(slot, work=True))
    grip(lk, close=False)
    arm(lk, "升回观察位", C.OBSERVE)


def pick_from_carrier(lk, slot):
    """盘上取料: 悬停 → 下探 → 夹1400 → 抬悬停 → 回复位。"""
    arm(lk, "到盘槽位上方", carrier_pose(slot, work=False))
    arm(lk, "下探取料", carrier_pose(slot, work=True))
    grip(lk, close=True)
    arm(lk, "抬回盘上方", carrier_pose(slot, work=False))
    arm(lk, "回复位", C.RESET)

_STEP = [0]


class ChassisLedger:
    """底盘记账器(比赛整体流程同款): 工作区所有 move_v 入账, 回中=发反向和+清零。
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
    """回车 → 发帧 → 等完成回执 → ✓/!! → 条间 2s。"""
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
    """底盘粗送位(入账): 右 dx, 前 dy mm。"""
    _LEDGER.add(dx, dy)
    run_step(lk, f"{label} (右{dx:+.0f},前{dy:+.0f})mm",
             frame(Mode.MOVE_V, dx, dy, 0), Mode.ACK_CHASSIS)


def return_to_anchor(lk, tag):
    """回中: 发记账反向和并清零。账本为零时自动跳过。回中不依赖视觉(死账)。"""
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


# ---- 纠偏(看-动-再看, 骨架=grab.align): 纠偏 move_v 直发并入账 ----

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
    """整车纠偏: 回车触发, s=跳过(人工兜底); 死区3mm ≤5轮, 超时带误差继续。"""
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


def find_ring_fn(backend, select):
    """select: mid=离画面中心最近(定一) / left=最左 / right=最右。"""
    def find(f):
        rings = find_rings(backend, f)
        if not rings:
            return None
        if select == "left":
            return min(rings, key=lambda r: r.cx)
        if select == "right":
            return max(rings, key=lambda r: r.cx)
        return min(rings, key=lambda r: r.dx ** 2 + r.dy ** 2)
    return find


# ---- 盘上取放 ----

# ---- 两动作组 ----

def action_grab(lk, cam, backend, timeout):
    """动作组A ×3轮: 近端/车头端/车尾端 抓料→放盘(槽位轮转 P1/P2/P3)。"""
    print("\n===== 动作组A: 抓料→放盘 (近端→车头端→车尾端, 每轮抓完回中) =====")
    arm(lk, "缩回复位", C.RESET)
    arm(lk, "下降抓取(离地150)", C.GRAB_NEAR)
    grip(lk, close=True)
    arm(lk, "升回复位", C.RESET)
    return_to_anchor(lk, "近端")
    place_to_carrier(lk, 0)   # 第1颗 → P1(中间)
    for i, (label, dy) in enumerate(
            (("车头端", C.TURN_STEP_MM), ("车尾端", -C.TURN_STEP_MM))):
        print(f"\n----- {label} -----")
        arm(lk, "前伸最大(145)", (C.ARM_REACH_MAX, C.H_LIFT, 0))
        chassis(lk, f"底盘粗送位(前{dy:+.0f})", 0, dy)
        _align(lk, cam, find_material_fn(backend), C.IMG2BODY,
               f"{label}物料纠偏", timeout)
        arm(lk, "下降抓取(离地150)", (C.ARM_REACH_MAX, C.GRAB_NEAR[1], 0))
        grip(lk, close=True)
        arm(lk, "升到最高", (C.ARM_REACH_MAX, C.H_LIFT, 0))
        arm(lk, "回复位", C.RESET)
        return_to_anchor(lk, label)
        place_to_carrier(lk, i + 1)   # 第2颗→P2(车头), 第3颗→P3(车尾)
    print("\n== 动作组A完成 (3轮, 每轮已回中, 车在近端物料锚点位) ==")


def _recheck(lk, cam, backend, timeout):
    """回中后视觉复核: 目标=画面中心最近的物料(B组=中环那颗), 消记账残差。
    复核后账清零——复核后的位置=新锚点。"""
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


def action_ring(lk, cam, backend, timeout):
    """动作组B ×3轮: 纠偏→取盘→放环(x=0直下)→回中(左/右环加复核)。全程复位位。"""
    print("\n===== 动作组B: 取盘→放环 (中环→左环→右环, 每轮放完回中) =====")
    for i, (label, select) in enumerate(
            (("中环", "mid"), ("左环", "left"), ("右环", "right"))):
        print(f"\n----- {label} -----")
        _align(lk, cam, find_ring_fn(backend, select), C.IMG2BODY_RING,
               f"{label}纠偏(把环送到复位位正下)", timeout)
        if i == 0:
            _LEDGER.zero()   # 中环纠偏完成=建锚(B组账本从此刻起算)
        pick_from_carrier(lk, i)
        arm(lk, "下探放环(离地65)", (C.RESET[0], C.H_PLACE_RING, C.RESET[2]))
        grip(lk, close=False)
        arm(lk, "升回复位", C.RESET)
        return_to_anchor(lk, label)
        if i > 0:
            _recheck(lk, cam, backend, timeout)
    print("\n== 动作组B完成 (3轮, 已回中+复核, 车在中环锚点位) ==")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="机械臂两动作组测试(底盘对位+纠偏+每轮回中)")
    ap.add_argument("--port", default=C.SERIAL_PORT)
    ap.add_argument("--model", default=C.MODEL_PATH)
    ap.add_argument("--phase", choices=("all", "grab", "ring"), default="all",
                    help="all=两组连跑(默认); grab=只动作组A; ring=只动作组B")
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
          f"行距{C.TURN_STEP_MM} 放环高{C.H_PLACE_RING} 每轮回中 | Ctrl+C 退出")
    if args.phase in ("all", "grab"):
        action_grab(lk, cam, backend, args.timeout)
    if args.phase in ("all", "ring"):
        action_ring(lk, cam, backend, args.timeout)
    print("\n== 全部完成 (账本为零=车在锚点位) ==")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n中止")
