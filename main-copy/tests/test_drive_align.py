"""实车联调: 动作序列(函数参数内含) + 连续自动纠偏(Nano 上跑, 车会真动)。

前提: 臂已手动摆到观察位(末端相机看到离车最近的物料), 车前方有物料可纠偏。
用法:
  python3 tests/test_drive_align.py --model ~/main-copy/best.engine --skip-start
  python3 tests/test_drive_align.py --model ... --visualize   # 纠偏可视化(接屏/NoMachine)
流程: 逐条动作(等 0x0011, 条间 2s) → 连续自动纠偏(算偏差→发帧→等 0x0011
→ 再看, ≤3mm 结束, ≤4 轮, 超时带误差继续保完赛) —— 全程无需手动发帧。
"""
import argparse
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import config as C       # noqa: E402
from grab import StepGrab   # noqa: E402
from protocol import Mode  # noqa: E402


def drive_and_align(lk, cam, backend, visualize=False,
                    moves=((0, 0, -90, "右转90°"),
                           (0, 1000, 0, "前进1000mm"),
                           (-100, 0, 0, "左移100mm"))):
    """动作序列 + 连续自动纠偏, 一个函数跑完。

    moves: 每条 (x, y, z, 说明) 车体系相对量——x 右(臂侧) y 前(车头)
    z 旋转°(顺时针/右转为负); 逐条发 move_v 等完成码, 条间 2s。
    序列跑完后接 StepGrab 阶段A 连续纠偏, 返回 align 的收敛结果。
    """
    for i, (x, y, z, label) in enumerate(moves, 1):
        print(f"-> {label}")
        lk.send_and_wait(lambda x=x, y=y, z=z: lk.move_v(x, y, z),
                         Mode.ACK_CHASSIS, C.CMD_TIMEOUT, label)
        print(f"   ✓ 完成 ({i}/{len(moves)})")
        if i < len(moves):
            time.sleep(2)   # 动作间 2s 延迟(等车稳/便于观察)
    print("-> 连续自动纠偏(看-动-再看, ≤3mm 结束)")
    return StepGrab(lk, cam, backend).align(visualize=visualize)


def main():
    ap = argparse.ArgumentParser(description="底盘运动+连续自动纠偏(实车)")
    ap.add_argument("--port", default=C.SERIAL_PORT)
    ap.add_argument("--model", default=C.MODEL_PATH, help="YOLO 权重(Nano 上 .engine 路径)")
    ap.add_argument("--skip-start", action="store_true", help="跳过一键启动等待")
    ap.add_argument("--visualize", action="store_true",
                    help="纠偏可视化(Nano 接屏/NoMachine 下用)")
    args = ap.parse_args()

    from camera import Camera
    from detect import YOLOBackend
    from protocol import Link

    cam = Camera(C.CAMERA_ID_END, "end")
    cam.open()
    print(f"加载模型 {args.model} ...")
    backend = YOLOBackend(args.model)
    lk = Link(args.port)

    try:
        drive_and_align(lk, cam, backend, visualize=args.visualize)
        print("测试完成")
    except RuntimeError as e:   # 通信静默等硬失败
        print(f"!! 中止(硬失败): {e}")


if __name__ == "__main__":
    main()
