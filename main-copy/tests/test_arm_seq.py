"""机械臂全流程序列测试(参数原值, 0.1mm/0.1° 单位=预期×10, 绝对目标)。

初始状态: 最高点(参数 Y=2090 = 预期 209mm)。两轮抓放:
  伸出500 → 降1200 → 夹闭合(1300) → 抬2090 → 转-1800 → 缩回X=0 → 降135
  → 张开(放料) → 抬2090 → 转回0(归初始朝向)
每步回车执行, 等 0x0012/0x0013 完成回执(打印全部回执帧), Ctrl+C 退出。
"""
import sys
import time
from pathlib import Path

BASE = str(Path(__file__).resolve().parent.parent)   # main-copy 根(双端通用)
sys.path.insert(0, BASE)
from communication.protocol import Link, Mode, frame  # noqa: E402

STEPS = [
    ("伸出 500(预期50mm)", frame(Mode.MOVE_A, 500, 2090, 0), Mode.ACK_ARM),
    ("下降离地 1200(预期120mm)", frame(Mode.MOVE_A, 500, 1200, 0), Mode.ACK_ARM),
    ("夹爪闭合抓取(1300)", frame(Mode.GRIPPER, 1300, 500), Mode.ACK_GRIPPER),
    ("抬回 2090(预期209mm)", frame(Mode.MOVE_A, 500, 2090, 0), Mode.ACK_ARM),
    ("回转到 -1800(预期−180°)", frame(Mode.MOVE_A, 500, 2090, -1800), Mode.ACK_ARM),
    ("缩回到 0(预期缩回50mm, X=0)", frame(Mode.MOVE_A, 0, 2090, -1800), Mode.ACK_ARM),
    ("下降到 135(预期13.5mm)", frame(Mode.MOVE_A, 0, 135, -1800), Mode.ACK_ARM),
    ("夹爪松开放料(800)", frame(Mode.GRIPPER, 800, 500), Mode.ACK_GRIPPER),
    ("抬回 2090(预期209mm)", frame(Mode.MOVE_A, 500, 2090, -1800), Mode.ACK_ARM),
    ("转回 0(归初始朝向)", frame(Mode.MOVE_A, 500, 2090, 0), Mode.ACK_ARM),
]

lk = Link("/dev/ttyUSB0")
print("串口已连。每步回车执行(清场+物料就位), Ctrl+C 退出。")
try:
    for i, (label, tx, ack) in enumerate(STEPS, 1):
        try:
            input(f"\n[{i}/14] {label}\n  hex={tx.hex(' ').upper()}\n  <- 回车执行")
        except EOFError:
            pass
        lk.flush()
        lk.ser.write(tx)
        t0 = time.time()
        got = None
        while time.time() - t0 < 35:
            for f in lk.poll():
                print(f"    RX {f}")
                if f.mode == ack and got is None:
                    got = f
            if got:
                break
            time.sleep(0.03)
        ackname = "0x0012臂" if ack == Mode.ACK_ARM else "0x0013夹爪"
        if got:
            print(f"    ✓ {ackname}完成回执 ({time.time()-t0:.1f}s) — 观察动作!")
        else:
            print(f"    !! 35s 无 {ackname} 完成回执")
except KeyboardInterrupt:
    print("\n退出")
