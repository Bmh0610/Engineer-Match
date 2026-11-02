"""车道走廊(竖条)可视化标定: 实时画面画出"横向位置门控"区域——画面中间 1/2 宽、从上到下贯通。

背景: 避障判定"挡没挡路"的横向口径=检出框落在这个竖条内才算挡路(条外=车身两侧, 车能绕开, 忽略)。
本工具把条画在实时画面上边看边调, 调好按 s 打印。obstacle.py 侧对应新门控参数(建议口径:
"框水平中心 x/画面宽"落条内=挡路), 区域定稿后加进 obstacle.py——现 LANE_BAND 是纵向 y 带
(近低远高按距离), 与本条是两个维度, 以后可并用(条内 AND 带内)。

用法: python3 tests/test_lane_band.py [--cam 0] [--band 0.25 0.75]
按键: u/j=左沿 左移/右移   i/k=右沿 左移/右移 (各 1% 画面宽)
      s=打印当前值   q/ESC=退出
前置: Nano 上跑窗口走 NoMachine 桌面: DISPLAY=:1001 python3 ...(号变了 ls /tmp/.X11-unix/ 查);
      前相机设备号 ls /dev/video* 核对(或 udev 别名 /dev/cam_*); 笔记本上也可直接跑(--cam 0=内置)。
"""
import argparse
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import cv2  # noqa: E402

from vision.camera import Camera  # noqa: E402

STEP = 0.01   # 每次按键移动 1% 画面宽


def main():
    ap = argparse.ArgumentParser(description="车道走廊(竖条)可视化标定")
    ap.add_argument("--cam", default="0",
                    help="相机设备号或 udev 别名(默认 0; 前相机编号 ls /dev/video* 核对)")
    ap.add_argument("--band", nargs=2, type=float, default=[0.25, 0.75],
                    metavar=("左沿", "右沿"), help="竖条左右沿(归一化 0~1), 默认=中间 1/2")
    args = ap.parse_args()

    lo, hi = sorted(args.band)
    cam = Camera(args.cam, "front")
    cam.open()
    print(f"  相机 {args.cam} 已打开; 初始 竖条 x=({lo:.2f},{hi:.2f}) (画面中间 1/2)")
    print("  按键: u/j=左沿 左移/右移  i/k=右沿 左移/右移  s=打印  q/ESC=退出")

    warned, t0 = False, time.time()
    try:
        while True:
            frame = cam.read_latest(timeout=0.2)
            if frame is not None:
                h, w = frame.shape[:2]
                x1, x2 = int(lo * w), int(hi * w)
                overlay = frame.copy()
                cv2.rectangle(overlay, (x1, 0), (x2 - 1, h - 1), (0, 200, 0), -1)
                cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)   # 条内半透明罩绿
                cv2.line(frame, (x1, 0), (x1, h - 1), (0, 255, 0), 1)
                cv2.line(frame, (x2, 0), (x2, h - 1), (0, 255, 0), 1)
                cv2.putText(frame, f"LANE_COL=({lo:.2f},{hi:.2f})  x=[{x1},{x2}]px / W={w}",
                            (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(frame, "u/j=left edge  i/k=right edge  s=print  q=quit",
                            (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow("lane_band", frame)
            elif time.time() - t0 > 3 and not warned:
                warned = True
                print("  ⚠ 3s 未出图: 核对 --cam 编号(ls /dev/video*)/是否被别的程序占用")

            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("u"):
                lo -= STEP          # 左沿左移
            elif key == ord("j"):
                lo += STEP          # 左沿右移
            elif key == ord("i"):
                hi -= STEP          # 右沿左移
            elif key == ord("k"):
                hi += STEP          # 右沿右移
            elif key == ord("s"):
                print(f"  LANE_COL = ({lo:.2f}, {hi:.2f})   # 竖条左右沿(x/画面宽), 框中心落条内=挡路")
                continue
            lo = min(max(lo, 0.0), hi - STEP)
            hi = max(min(hi, 1.0), lo + STEP)
    finally:
        cam.close()
        cv2.destroyAllWindows()
    print(f"  退出, 最终 LANE_COL = ({lo:.2f}, {hi:.2f})")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n中止")
