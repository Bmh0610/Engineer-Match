"""实车联调: 右转90° → 前进1000mm → 左移100mm → 视觉纠偏(Nano 上跑, 车会真动)。

前提: 臂已手动摆到观察位(末端相机看到离车最近的物料), 车前方有物料可纠偏。
用法:
  python3 tests/test_drive_align.py --model ~/best.pt   # 端口默认 /dev/ttyUSB0
每条运动等 0x0011 完成码再发下一条(单命令在途, 35s 静默不重发);
纠偏 = StepGrab 阶段A(最近物料圆心, 稳定帧, 看-动-再看, 死区 3mm)。
"""
import argparse
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import config as C       # noqa: E402
from protocol import Mode  # noqa: E402

MOVES = [   # (x, y, z, 说明) 车体系: x 右(臂侧) y 前 z 旋转°(顺时针/右转为负)
    (0, 0, -90, "右转90°"),
    (0, 1000, 0, "前进1000mm"),
    (-100, 0, 0, "左移100mm"),
]


def main():
    ap = argparse.ArgumentParser(description="底盘运动+纠偏联调(实车)")
    ap.add_argument("--port", default=C.SERIAL_PORT)
    ap.add_argument("--model", default=C.MODEL_PATH, help="YOLO 权重(Nano 上的路径)")
    args = ap.parse_args()

    from camera import Camera
    from detect import YOLOBackend
    from grab import StepGrab
    from protocol import Link

    cam = Camera(C.CAMERA_ID_END, "end")
    cam.open()
    print(f"加载模型 {args.model} ...")
    backend = YOLOBackend(args.model)
    lk = Link(args.port)

    try:
        for x, y, z, label in MOVES:
            print(f"-> {label}")
            lk.send_and_wait(lambda x=x, y=y, z=z: lk.move_v(x, y, z),
                             Mode.ACK_CHASSIS, C.CMD_TIMEOUT, label)
        print("-> 纠偏(观察位物料圆心, 看-动-再看)")
        StepGrab(lk, cam, backend).align()
        print("测试完成")
    except RuntimeError as e:   # 通信静默等硬失败
        print(f"!! 中止(硬失败): {e}")


if __name__ == "__main__":
    main()
