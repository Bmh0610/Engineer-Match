"""纠偏手动执行测试(笔记本跑): 回车一轮纠偏, 你发 12 字节纠偏帧, 直到残差<阈值。

流程: 回车 → 取稳定帧(连续2帧位移<2px) → YOLO+霍夫顶面小圆(30mm 实测校准)
→ 打印车体系偏差(X右/Y前, mm, 带±)与**完整 12 字节协议纠偏帧**(CRC 现算)
→ 你发帧 → 车动 → 停稳 → 再回车 → 下一轮; 残差 ≤ 3mm 提示收敛完成。
画面窗口全程实时刷新(select 非阻塞监听终端, 不会卡"未响应")。q 退出(两处均可)。

用法: python3 tests/test_align_manual.py --cam 2
"""
import argparse
import select
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import cv2   # noqa: E402

import config as C    # noqa: E402
import finder         # noqa: E402
from protocol import Mode, frame   # noqa: E402
from test_yolo_live import render   # noqa: E402  复用主画面渲染(同目录)

HOUGH_P2_SOFT = 20   # 实拍顶面边缘弱, 累加器门槛比线上(30)放宽


def one_round(backend, cam, conf, round_no):
    """一轮纠偏判定: 稳定帧 → 检测+霍夫顶面 → 打印偏差与 12 字节纠偏帧。
    返回叠加后的显示帧(供主循环持续刷新)。"""
    print("  检测中(取稳定帧, 最多 2s)...", flush=True)
    obj, frame_img = finder.find_stable(
        lambda f: finder.nearest_to_axis(finder.find_materials(
            backend, f, conf=conf, px_per_mm=C.PX_PER_MM)),
        cam, timeout=2.0)
    if obj is None:
        print("  !! 未检出稳定物料(视野/物料确认), 再回车重试")
        return frame_img
    hit, _st = finder.refine_material_stages(
        frame_img, obj, C.PX_PER_MM, r_lo=finder.CAP_R_LO, r_hi=finder.CAP_R_HI,
        param2=HOUGH_P2_SOFT)
    src = "霍夫顶面" if hit else "bbox中心(霍夫未命中)"
    dy_right = obj.dx_mm * C.IMG2BODY[0]
    dx_fwd = obj.dy_mm * C.IMG2BODY[1]
    residual = (dy_right ** 2 + dx_fwd ** 2) ** 0.5
    if residual <= C.ALIGN_TOL_MM:
        print(f"[r{round_no}] ✓ 已收敛: X右{dy_right:+.1f} Y前{dx_fwd:+.1f} "
              f"残差 {residual:.1f}mm ≤ {C.ALIGN_TOL_MM:.0f}mm —— 定位完成!")
    else:
        print(f"[r{round_no}] {src}: X右{dy_right:+.1f} Y前{dx_fwd:+.1f}mm "
              f"(残差 {residual:.1f}mm)")
        tx = frame(Mode.MOVE_V, round(dy_right), round(dx_fwd), 0)
        print(f"  纠偏帧: {tx.hex(' ').upper()}")
        print("  发送: python3 -c \"import serial; "
              "serial.Serial('/dev/ttyUSB0',115200,timeout=0.1)"
              f".write(bytes.fromhex('{tx.hex()}'))\"")
    return render(frame_img, [obj], obj)


def main():
    ap = argparse.ArgumentParser(description="纠偏手动执行测试(回车触发单轮)")
    ap.add_argument("--cam", type=int, default=2)
    ap.add_argument("--model", default=C.MODEL_PATH)
    ap.add_argument("--conf", type=float, default=0.5)
    args = ap.parse_args()

    from camera import Camera
    from detect import YOLOBackend

    cam = Camera(args.cam, "align_manual")
    cam.open()
    print(f"加载模型 {args.model} ...")
    backend = YOLOBackend(args.model)
    if finder._CALIB is not None:
        c = finder._CALIB
        print(f"内参: 主点=({c.cx:.1f},{c.cy:.1f}) | 比例尺 {C.PX_PER_MM} px/mm | "
              f"死区 {C.ALIGN_TOL_MM:.0f}mm")
    print("流程: 终端回车=纠偏一轮 → 发输出的 12 字节帧 → 车停 → 再回车; "
          "q+回车 或 窗口内按 q 退出")

    round_no = 0
    while True:
        frame = cam.read_latest()
        if frame is None:
            continue
        # 每帧实时检测叠加(YOLO框+霍夫顶面圆+主点十字)
        mats = finder.find_materials(backend, frame, conf=args.conf,
                                     px_per_mm=C.PX_PER_MM)
        m = finder.nearest_to_axis(mats)
        if m is not None:
            finder.refine_material_stages(
                frame, m, C.PX_PER_MM, r_lo=finder.CAP_R_LO,
                r_hi=finder.CAP_R_HI, param2=HOUGH_P2_SOFT)
        cv2.imshow("align_manual", render(frame, mats, m))
        # 非阻塞监听终端回车(触发一轮带数字输出的纠偏判定)
        if select.select([sys.stdin], [], [], 0)[0]:
            line = sys.stdin.readline().strip()
            if line.lower() == "q":
                break
            round_no += 1
            one_round(backend, cam, args.conf, round_no)
        k = cv2.waitKey(30) & 0xFF
        if k == ord("q"):
            break
    cam.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
