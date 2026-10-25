"""环纠偏手动测试(粗加工区, test_align_manual 同款交互): 算偏差 → 回传协议式命令。

前提: 车已人工摆到粗加工区环前; config 环面标定(PX_PER_MM_RING/IMG2BODY_RING)已填。
用法: python3 tests/test_ring_align.py [--code 任务码]

流程(每轮):
  回车 → 等稳定帧(转盘/车静止) → YOLO 找数字环 → 取画面中间者(定一目标)
  → 算车体系偏差(前/右 mm, IMG2BODY_RING) → 打印 12 字节纠偏帧 + 一行流发送命令
  → 你自己发帧(脚本不自动动车) → 车停后回车测下一轮; 残差 ≤3mm 提示已收敛。
每轮同时打印"数字→左中右"识别映射(纠偏顺便核对 1/2/3 检出位置)。
q 退出。
"""
import math
import sys
from pathlib import Path

BASE = str(Path(__file__).resolve().parent.parent)   # main-copy 根(双端通用)
sys.path.insert(0, BASE)

import config as C  # noqa: E402
from communication.protocol import Mode, frame  # noqa: E402
from task.ring import StepRing  # noqa: E402
from vision.finder import find_stable  # noqa: E402
from vision.qr import parse_task  # noqa: E402


def main():
    import argparse
    ap = argparse.ArgumentParser(description="环纠偏手动测试(协议式命令回传)")
    ap.add_argument("--model-ring", default=C.MODEL_PATH_RING)
    ap.add_argument("--code", default=C.DEFAULT_TASK_CODE, help="任务码(打印放置环号)")
    ap.add_argument("--timeout", type=float, default=5.0, help="单轮等稳定帧超时(秒)")
    args = ap.parse_args()

    from vision.camera import Camera
    from vision.detect import YOLOBackend

    cam = Camera(C.CAMERA_ID_END, "end")
    cam.open()
    print(f"加载环模型 {args.model_ring} ...")
    sp = StepRing(None, cam, YOLOBackend(args.model_ring))   # 手动发帧: 不接串口
    if not sp._calib_check():
        return
    task = parse_task(args.code)
    if task:
        print(f"任务码放置环号: {task['batch1_rings']}"
              f" (放置按贴纸数字对位, 勿按左右猜)\n")

    print(f"回车=测一轮(偏差+纠偏帧); q 退出; 收敛=残差≤{C.ALIGN_TOL_MM:.0f}mm")
    rnd = 0
    while True:
        try:
            if input(f"\n[第{rnd + 1}轮] 回车测偏差 / q 退出: ").strip().lower() == "q":
                break
        except EOFError:
            break
        obj, fr = find_stable(sp._ring_observe, cam, timeout=args.timeout)
        if obj is None:
            print("  !! 未检出数字环(核对臂在环观测位/环区在画面内), 车不动, 回车重测")
            continue
        dy_right, dx_fwd = sp._to_body(obj)
        residual = math.hypot(dx_fwd, dy_right)
        rnd += 1
        print(f"  数字→位置: {sp._digit_map(fr)}")
        if residual <= C.ALIGN_TOL_MM:
            print(f"  ✓ 已收敛: 前{dx_fwd:+.1f} 右{dy_right:+.1f} "
                  f"残差 {residual:.1f}mm ≤ {C.ALIGN_TOL_MM:.0f}mm")
            continue
        tx = frame(Mode.MOVE_V, round(dy_right), round(dx_fwd), 0)
        print(f"  偏差: 前{dx_fwd:+.1f} 右{dy_right:+.1f}mm (残差 {residual:.1f}mm)")
        print(f"  纠偏帧: {tx.hex(' ').upper()}")
        print(f"  发送命令:\n    python3 -c \"import serial; "
              f"serial.Serial('{C.SERIAL_PORT}',115200,timeout=0.1)"
              f".write(bytes.fromhex('{tx.hex()}'))\"")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n退出")
