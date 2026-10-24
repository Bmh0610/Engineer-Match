"""底盘→连续纠偏→机械臂抓放 全流程联调(每步回车执行, 条间 2s)。

序列:
  底盘: 前进100 → 右转90°(顺时针) → 前进100 → 右移100   (0x0001, 参数=预期×10)
  纠偏: 连续自动(圆盘定位, ≤5 轮, 死区 3mm) —— 无需回车
  机械臂: 伸出150(高2090) → 顺转30° → 降离地120 → 夹闭合 → 抬209
        → 转-180° → 缩回0 → 降离地18 → 夹张开(放料) → 抬209 → 转回0
        (0x0002/0x0003, 绝对目标, 参数=预期×10)
每步打印帧 hex, 等 0x0011/0x0012/0x0013 完成回执; Ctrl+C 退出。
"""
import sys
import time

BASE = "/home/ubuntu/main-copy"
sys.path.insert(0, BASE)

import config as C  # noqa: E402
from grab import StepGrab  # noqa: E402
from protocol import Link, Mode, frame  # noqa: E402

# 底盘段: (label, x, y, z) 参数=预期×10
CHASSIS = [
    ("前进 100mm", 0, 1000, 0),
    ("右转 90°(顺时针)", 0, 0, -900),
    ("前进 100mm", 0, 1000, 0),
    ("右移 100mm", 1000, 0, 0),
]

# 臂段: (label, tx, ack) 绝对目标, 参数=预期×10
ARM = [
    ("伸出 150mm(高2090)", frame(Mode.MOVE_A, 1500, 2090, 0), Mode.ACK_ARM),
    ("顺转 30°(参数-300)", frame(Mode.MOVE_A, 1500, 2090, -300), Mode.ACK_ARM),
    ("下降离地 120mm(参数1200)", frame(Mode.MOVE_A, 1500, 1200, -300), Mode.ACK_ARM),
    ("夹爪闭合(1300)", frame(Mode.GRIPPER, 1300, 500), Mode.ACK_GRIPPER),
    ("抬回 209mm(参数2090)", frame(Mode.MOVE_A, 1500, 2090, -300), Mode.ACK_ARM),
    ("回转 -180°(参数-1800)", frame(Mode.MOVE_A, 1500, 2090, -1800), Mode.ACK_ARM),
    ("缩回到 0", frame(Mode.MOVE_A, 0, 2090, -1800), Mode.ACK_ARM),
    ("下降离地 18mm(参数180)", frame(Mode.MOVE_A, 0, 180, -1800), Mode.ACK_ARM),
    ("夹爪张开放料(800)", frame(Mode.GRIPPER, 800, 500), Mode.ACK_GRIPPER),
    ("抬回 209mm(参数2090)", frame(Mode.MOVE_A, 0, 2090, -1800), Mode.ACK_ARM),
    ("转回 0°(归位)", frame(Mode.MOVE_A, 0, 2090, 0), Mode.ACK_ARM),
]


def run_step(lk, label, tx, ack, total, i):
    """执行一步: 回车 → 发帧 → 等完成码 → ✓/!!。条间 2s。"""
    try:
        input(f"\n[{i}/{total}] {label}\n  hex={tx.hex(' ').upper()}\n  <- 回车执行")
    except EOFError:
        pass
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
    name = {Mode.ACK_CHASSIS: "0x0011底盘", Mode.ACK_ARM: "0x0012臂",
            Mode.ACK_GRIPPER: "0x0013夹爪"}[ack]
    if got:
        print(f"    ✓ {name}完成 ({time.time()-t0:.1f}s)")
    else:
        print(f"    !! {C.CMD_TIMEOUT:.0f}s 无 {name} 完成回执")
    time.sleep(2)   # 条间 2s


def main():
    import argparse
    ap = argparse.ArgumentParser(description="底盘+纠偏+机械臂 全流程联调")
    ap.add_argument("--model", default="/home/ubuntu/main-copy/best.engine")
    args = ap.parse_args()

    from camera import Camera
    from detect import YOLOBackend

    cam = Camera(C.CAMERA_ID_END, "end")
    cam.open()
    print(f"加载模型 {args.model} ...")
    backend = YOLOBackend(args.model)
    lk = Link("/dev/ttyUSB0")
    print("串口已连。全流程: 底盘4步 → 自动纠偏(≤5轮) → 臂11步。每步回车, Ctrl+C 退出。")
    n = len(CHASSIS) + len(ARM)
    i = 0
    try:
        for label, x, y, z in CHASSIS:
            i += 1
            run_step(lk, f"底盘 {label}", frame(Mode.MOVE_V, x, y, z),
                     Mode.ACK_CHASSIS, n, i)
        print("\n-> 连续自动纠偏(圆盘定位, ≤5 轮, 死区 3mm)——无需回车")
        C.ALIGN_MAX_ITER = 5   # 本测试放宽到 5 轮
        StepGrab(lk, cam, backend).align()
        for label, tx, ack in ARM:
            i += 1
            run_step(lk, f"臂 {label}", tx, ack, n, i)
        print("\n== 全流程完成 ==")
    except KeyboardInterrupt:
        print("\n中止")


if __name__ == "__main__":
    main()
