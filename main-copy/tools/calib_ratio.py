"""像素比例尺实测(手动两点测距版): 1mm = 多少像素, 不依赖角点检测。

用法: python3 tools/calib_ratio.py --cam 0
操作:
  ① 空格冻结一帧 → 鼠标沿"已知实际长度的线段"两端各点一下
  ② 到终端输入该线段实际长度 mm → 记一次样本, 画面恢复实时
  ③ 多测几处(画面不同位置/不同方向), q 结束打印均值 → 填入 config.PX_PER_MM
测什么: 棋盘格方格边、直尺刻度段、物料俯视圆的直径(最贴业务——物料放在
工作面上量, 高度天然正确)…… 任何长度已知的线段都行。
精度: 两点已用内参去畸变; 尽量点在画面中部(标定覆盖区内), 线段拉长些
(同样的点击误差, 线越长相对误差越小)。
交叉验证: px_per_mm ≈ fx / 相机离面高度mm(内参法, fx 见 vision/calib_end.yaml),
两法差 >5% 说明高度量错或标定有问题。
"""
import argparse
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import cv2
import numpy as np

from vision.camera import Camera

import config as _C
import calib as _calib_mod
# 与 finder 同口径: 无畸变镜头(USE_INTRINSICS=False)不加载内参/不去畸变
_CAL = _calib_mod.load() if _C.USE_INTRINSICS else None


class State:
    def __init__(self):
        self.frozen = None   # None=实时; 否则为冻结帧
        self.pts = []        # 冻结帧上已点击的端点


def on_mouse(event, x, y, flags, st):
    if event == cv2.EVENT_LBUTTONDOWN and st.frozen is not None and len(st.pts) < 2:
        st.pts.append((float(x), float(y)))


def main():
    ap = argparse.ArgumentParser(description="像素比例尺实测(px/mm, 手动两点测距)")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--ring", action="store_true",
                    help="环面标定模式: 打印 PX_PER_MM_RING(在环观测位姿下跑)")
    args = ap.parse_args()

    cam = Camera(args.cam, "ratio")
    cam.open()
    cv2.namedWindow("ratio")
    st = State()
    cv2.setMouseCallback("ratio", on_mouse, st)
    samples = []
    print("== 比例尺实测(两点测距) ==")
    print("空格冻结画面 → 鼠标点已知长度线段两端 → 终端输入实际 mm → 自动恢复实时")
    print("多测几处, q 结束打印均值 → 填入 config.PX_PER_MM | 提示: 直接量物料直径最贴业务")

    def draw(frame):
        disp = frame.copy()
        for i, p in enumerate(st.pts):
            cv2.circle(disp, (int(p[0]), int(p[1])), 4, (0, 0, 255), -1)
        if len(st.pts) == 2:
            cv2.line(disp, tuple(map(int, st.pts[0])), tuple(map(int, st.pts[1])),
                     (0, 255, 0), 2)
            cv2.putText(disp, f"{np.hypot(*(np.subtract(*st.pts))):.1f} px "
                              f"→ 去终端输入实际 mm",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            tip = ("已冻结: 点已知长度线段的两个端点" if st.frozen is not None
                   else "实时 | 空格=冻结采样  q=结束")
            cv2.putText(disp, tip + f"  (n={len(samples)})", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        return disp

    try:
        while True:
            frame = st.frozen if st.frozen is not None else cam.read_latest()
            if frame is None:
                continue
            cv2.imshow("ratio", draw(frame))
            key = cv2.waitKey(30) & 0xFF

            if st.frozen is not None and len(st.pts) == 2:
                p1, p2 = st.pts
                if _CAL is not None:
                    p1, p2 = _CAL.undistort_point(*p1), _CAL.undistort_point(*p2)
                px = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
                cv2.imshow("ratio", draw(st.frozen))
                cv2.waitKey(300)
                raw = input(f"  线段 {px:.1f}px, 实际长度 mm (回车取消): ").strip()
                try:
                    mm = float(raw)
                    if mm <= 0:
                        raise ValueError
                except ValueError:
                    print("  已取消本次, 画面恢复实时")
                else:
                    ppm = px / mm
                    samples.append(ppm)
                    print(f"  采样 {ppm:.3f} px/mm (n={len(samples)}, "
                          f"当前均值 {np.mean(samples):.3f})")
                st.frozen, st.pts = None, []
                continue

            if key == ord(" ") and st.frozen is None:
                st.frozen = cam.read_latest()
                st.pts = []
            elif key in (ord("q"), 27):
                break
    except KeyboardInterrupt:
        print()
    finally:
        cam.close()
        cv2.destroyAllWindows()

    if samples:
        mean, std = float(np.mean(samples)), float(np.std(samples))
        print(f"\n== 结果 ==\npx_per_mm = {mean:.3f} ± {std:.3f}  (n={len(samples)})")
        if std / mean > 0.05:
            print("!! 离散度 >5%: 各次测量差异大(点的位置/长度输入有误或高度变了), 建议重测")
        name = "PX_PER_MM_RING" if args.ring else "PX_PER_MM"
        where = "config 环放置段" if args.ring else "config 标定段"
        print(f"填入 {where}:  {name} = {mean:.3f}")
        if _CAL is not None:
            print(f"交叉验证: 相机离面高度mm 应 ≈ fx / {mean:.3f} = {_CAL.fx / mean:.0f}")
    else:
        print("未采样, 退出")


if __name__ == "__main__":
    main()
