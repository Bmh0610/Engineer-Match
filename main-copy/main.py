"""main-copy 入口: 假设车已停在转盘面前(臂收在观察位) → Step_Grab。

用法:
  python3 main.py --model ~/main-copy/best.engine
参数: --port 串口(默认 /dev/ttyUSB0); --colors 抓取颜色顺序; --skip-start 跳过一键启动等待。
硬失败(通信静默 LinkError)统一收口打印; 软失败(视觉超时)grab.py 内部已兜底。
"""
import argparse
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

import config as C          # noqa: E402


def wait_start(lk):
    from protocol import Mode
    print("等待一键启动(0x0020)...")
    while True:
        for f in lk.poll():
            if f.mode == Mode.BUTTON_START:
                print("收到一键启动!")
                return
        time.sleep(0.05)


def main():
    ap = argparse.ArgumentParser(description="转盘定位纠偏+抓取(假设车已到位)")
    ap.add_argument("--port", default=C.SERIAL_PORT, help="串口设备")
    ap.add_argument("--model", default=C.MODEL_PATH, help="YOLO 权重路径(.engine/.pt)")
    ap.add_argument("--colors", default=",".join(C.GRAB_COLORS),
                    help="抓取颜色顺序, 逗号分隔")
    ap.add_argument("--skip-start", action="store_true", help="跳过一键启动等待")
    args = ap.parse_args()
    colors = [c.strip() for c in args.colors.split(",") if c.strip()]

    from camera import Camera
    from detect import YOLOBackend
    from protocol import Link

    cam = Camera(C.CAMERA_ID_END, "end")
    cam.open()
    print(f"末端相机已开(dev={C.CAMERA_ID_END}); 加载模型 {args.model} ...")
    backend = YOLOBackend(args.model)
    lk = Link(args.port)

    if not args.skip_start:
        wait_start(lk)

    from grab import StepGrab
    try:
        StepGrab(lk, cam, backend).run(colors)
    except RuntimeError as e:   # 硬失败: 通信静默(不重发策略)
        print(f"!! 中止(硬失败): {e}")
        return 1
    finally:
        cam.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
