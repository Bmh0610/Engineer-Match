"""机械臂全流程序列测试(参数原值, 0.1mm/0.1° 单位=预期×10, 绝对目标)。

初始状态: 最高点(参数 Y=2090 = 预期 209mm)。两轮抓放:
  伸出500 → 降1200 → 夹闭合(1300) → 抬2090 → 转600 → 降625 → 张开(放①)
  → 抬2090 → 降580 → 夹闭合(抓②) → 抬2090 → 转回0(−60°) → 降1200 → 张开(放②)
每步回车执行, 等 0x0012/0x0013 完成回执(打印全部回执帧), Ctrl+C 退出。
"""
import sys
import time

sys.path.insert(0, "/home/ubuntu/main-copy")
from protocol import Link, Mode, frame  # noqa: E402

STEPS = [
    ("伸出 500(预期50mm)", frame(Mode.MOVE_A, 500, 2090, 0), Mode.ACK_ARM),
    ("下降离地 1200(预期120mm)", frame(Mode.MOVE_A, 500, 1200, 0), Mode.ACK_ARM),
    ("夹爪闭合抓取(1300)", frame(Mode.GRIPPER, 1300, 500), Mode.ACK_GRIPPER),
    ("抬回 2090(预期209mm)", frame(Mode.MOVE_A, 500, 2090, 0), Mode.ACK_ARM),
    ("回转 600(预期+60°)", frame(Mode.MOVE_A, 500, 2090, 600), Mode.ACK_ARM),
    ("降到 625(预期62.5mm)", frame(Mode.MOVE_A, 500, 625, 600), Mode.ACK_ARM),
    ("夹爪张开放料①(800)", frame(Mode.GRIPPER, 800, 500), Mode.ACK_GRIPPER),
    ("抬回 2090", frame(Mode.MOVE_A, 500, 2090, 600), Mode.ACK_ARM),
    ("下降到 580(预期58mm)", frame(Mode.MOVE_A, 500, 580, 600), Mode.ACK_ARM),
    ("夹爪闭合抓料②(1300)", frame(Mode.GRIPPER, 1300, 500), Mode.ACK_GRIPPER),
    ("抬到 2090", frame(Mode.MOVE_A, 500, 2090, 600), Mode.ACK_ARM),
    ("回转回 0(预期−60°)", frame(Mode.MOVE_A, 500, 2090, 0), Mode.ACK_ARM),
    ("下降到 1200(预期120mm)", frame(Mode.MOVE_A, 500, 1200, 0), Mode.ACK_ARM),
    ("夹爪松开放料②(800)", frame(Mode.GRIPPER, 800, 500), Mode.ACK_GRIPPER),
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
