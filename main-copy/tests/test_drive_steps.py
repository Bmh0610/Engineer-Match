"""实车运动逐步测试(Nano): 每条打印出来, 按回车执行, 等完成码, 再下一条。

纯 pyserial, 不依赖相机/YOLO。Ctrl+C 随时退出。
用法:
  python3 tests/test_drive_steps.py               # 端口默认 /dev/ttyUSB0
  python3 tests/test_drive_steps.py --port /dev/ttyTHS1
"""
import argparse
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import config as C       # noqa: E402
from communication.protocol import Mode  # noqa: E402

MOVES = [   # (x, y, z, 说明) 车体系: x 右(臂侧) y 前 z 旋转°(顺时针/右转为负)
    (0, 0, -90, "右转90°"),
    (0, 1000, 0, "前进1000mm"),
    (-100, 0, 0, "左移100mm"),
]


def main():
    ap = argparse.ArgumentParser(description="底盘运动逐步测试(实车)")
    ap.add_argument("--port", default=C.SERIAL_PORT)
    args = ap.parse_args()

    from communication.protocol import Link
    lk = Link(args.port)
    print(f"串口 {args.port} 已连。共 {len(MOVES)} 步, 每步按回车执行。")
    for i, (x, y, z, label) in enumerate(MOVES, 1):
        try:
            input(f"[{i}/{len(MOVES)}] {label}  move_v(x={x}, y={y}, z={z})  ← 回车执行")
        except EOFError:
            pass
        try:
            lk.send_and_wait(lambda x=x, y=y, z=z: lk.move_v(x, y, z),
                             Mode.ACK_CHASSIS, C.CMD_TIMEOUT, label)
            print("    ✓ 完成")
        except RuntimeError as e:
            print(f"    !! {e}")
            return 1
    print("全部完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
