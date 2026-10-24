"""实车跑图测试(纯底盘, 无视觉): 按 14 条动作序列走完整场地路线。

用法:
  python3 tests/test_paotu.py
  python3 tests/test_paotu.py --port /dev/ttyTHS1
前置: 整车上电 → /dev/ttyUSB0 在 → 清场 → 按一键启动 → 跑本脚本。
每条 (x, y, z, 说明) 车体系相对量: x 右(臂侧) y 前(车头) z 旋转°(顺时针负);
逐条等 0x0011 完成码, 35s 静默不重发(车停在最近完成的动作处, 重摆重跑)。
"""
import argparse
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import config as C       # noqa: E402
from protocol import Mode  # noqa: E402


def drive_route(lk,
                moves=((100, 0, 0, "右移100"),
                       (0, 1000, 0, "前进1000"),
                       (0, 0, -90, "右转90°"),
                       (0, 1000, 0, "前进1000"),
                       (0, 0, -90, "右转90°"),
                       (0, 1000, 0, "前进1000"),
                       (0, 0, 90, "左转90°"),
                       (-2000, 0, 0, "左移2000"),
                       (0, 0, -180, "右转180°"),
                       (0, -1000, 0, "后退1000"),
                       (0, 0, -90, "右转90°"),
                       (0, -1000, 0, "后退1000"),
                       (0, -1000, 0, "后退1000"),
                       (-2000, 0, 0, "左移2000"))):
    """按序列走完整场地路线(车体系相对量), 逐条发 move_v 等完成码, 条间 2s。"""
    for i, (x, y, z, label) in enumerate(moves, 1):
        print(f"[{i}/{len(moves)}] {label}  move_v(x={x}, y={y}, z={z})")
        t0 = time.time()
        lk.send_and_wait(lambda x=x, y=y, z=z: lk.move_v(x, y, z),
                         Mode.ACK_CHASSIS, C.CMD_TIMEOUT, label)
        print(f"    ✓ 完成 ({time.time() - t0:.1f}s)")
        if i < len(moves):
            time.sleep(2)   # 动作间 2s 延迟(等车稳/便于观察)


def main():
    ap = argparse.ArgumentParser(description="跑图路线测试(纯底盘)")
    ap.add_argument("--port", default=C.SERIAL_PORT)
    args = ap.parse_args()

    from protocol import Link
    lk = Link(args.port)
    input("确认: 已清场 + 已按一键启动。回车开始跑图, Ctrl+C 中止...")
    try:
        t0 = time.time()
        drive_route(lk)
        print(f"跑图完成, 总耗时 {time.time() - t0:.0f}s")
    except RuntimeError as e:   # 通信静默等硬失败
        print(f"!! 中止(硬失败): {e}")
        print("!! 车停在最近完成的动作处; 重摆位置后重跑(序列从头开始)")


if __name__ == "__main__":
    main()
