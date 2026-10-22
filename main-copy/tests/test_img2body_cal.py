"""IMG2BODY 标定向导: 两次已知移动自动求解 图像偏差→车体系 的 2x2 映射。

前置: 物料放相机视野内(全程不拿走、霍夫稳定命中顶面), 车周边清场。
流程(跟着终端提示):
  回车1 采基准圆心 p0
  发 move_v(0,100,0)  前进100mm → 停稳 → 回车2 采 p1
  发 move_v(100,0,0)  右移100mm → 停稳 → 回车3 采 p2
求解: 像素位移 = M @ 车体位移(两次位移解出 M 的两列) → IMG2BODY = M⁻¹。
输出 config 格式的 IMG2BODY 四元组(直接复制进 config.py), 并带自检。

用法: python3 tests/test_img2body_cal.py --cam 2
"""
import argparse
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import config as C    # noqa: E402
import finder         # noqa: E402
from protocol import Mode, frame   # noqa: E402

HOUGH_P2_SOFT = 20    # 实拍顶面边缘弱, 累加器门槛放宽
STEP = 100.0          # 标定移动步长 mm


def solve_img2body(d_right_pix, d_fwd_pix, step=STEP):
    """两次已知车体位移的像素响应 → (IMG2BODY 四元组, 自检残差)。

    d_right_pix: 右移 step 引起的像素位移 (ix, iy)
    d_fwd_pix:   前进 step 引起的像素位移 (ix, iy)
    模型: 像素位移 = M @ 车体位移; IMG2BODY = M⁻¹ (像素→车体系)。
    """
    M = np.column_stack([np.asarray(d_right_pix, float) / step,
                         np.asarray(d_fwd_pix, float) / step])
    Minv = np.linalg.inv(M)
    # 自检: 反解车体位移应还原已知输入
    chk_r = Minv @ np.asarray(d_right_pix, float)
    chk_f = Minv @ np.asarray(d_fwd_pix, float)
    res = max(abs(chk_r[0] - step), abs(chk_r[1]), abs(chk_f[0]), abs(chk_f[1] - step))
    a, b = Minv[0]
    c, d = Minv[1]
    return (a, b, c, d), res


def grab_center(backend, cam, conf):
    """稳定帧 + 霍夫顶面精化 → (cx, cy); 检不出返回 None。"""
    obj, frame = finder.find_stable(
        lambda f: finder.nearest_to_axis(finder.find_materials(
            backend, f, conf=conf, px_per_mm=C.PX_PER_MM)),
        cam, timeout=2.0)
    if obj is None:
        return None
    finder.refine_material_stages(frame, obj, C.PX_PER_MM,
                                  r_lo=finder.CAP_R_LO, r_hi=finder.CAP_R_HI,
                                  param2=HOUGH_P2_SOFT)
    import cv2
    cv2.imshow("img2body_cal", frame)
    return obj.cx, obj.cy


def main():
    ap = argparse.ArgumentParser(description="IMG2BODY 标定向导")
    ap.add_argument("--cam", type=int, default=2)
    ap.add_argument("--model", default=C.MODEL_PATH)
    ap.add_argument("--conf", type=float, default=0.5)
    args = ap.parse_args()

    from camera import Camera
    from detect import YOLOBackend

    cam = Camera(args.cam, "cal")
    cam.open()
    print(f"加载模型 {args.model} ...")
    backend = YOLOBackend(args.model)
    print("把物料放在视野中央(全程不动)。跟着提示: 发车控命令 → 停稳 → 回车采样")

    pts = []
    steps = [(0.0, float(STEP), 0.0, f"前进{STEP:.0f}"),
             (float(STEP), 0.0, 0.0, f"右移{STEP:.0f}")]
    prompts = ["回车1: 采基准圆心 p0",
               "第2步: 发『前进100』帧, 停稳后回车2 采 p1",
               "第3步: 发『右移100』帧, 停稳后回车3 采 p2"]
    for (x, y, z, label), pr in zip(steps, prompts[1:]):
        tx = frame(Mode.MOVE_V, int(x), int(y), int(z))
        print(f"\n{pr}")
        print(f"  协议帧: {tx.hex(' ').upper()}")
        print("  发送: python3 -c \"import serial; "
              "serial.Serial('/dev/ttyUSB0',115200,timeout=0.1)"
              f".write(bytes.fromhex('{tx.hex()}'))\"")
        input("  车停稳后回车采样 ...")
        c = grab_center(backend, cam, args.conf)
        if c is None:
            print("!! 未检出稳定物料, 本步骤重来")
            return
        pts.append(c)
        print(f"   圆心 = ({c[0]:.1f}, {c[1]:.1f})")
    c0, c1, c2 = pts
    d_fwd = (c1[0] - c0[0], c1[1] - c0[1])
    d_right = (c2[0] - c1[0], c2[1] - c1[1])
    print(f"前进{STEP:.0f}mm 像素位移: ({d_fwd[0]:.1f}, {d_fwd[1]:.1f})")
    print(f"右移{STEP:.0f}mm 像素位移: ({d_right[0]:.1f}, {d_right[1]:.1f})")

    (a, b, c, d), res = solve_img2body(d_right, d_fwd, STEP)
    print(f"\n自检最大残差 {res:.2f}mm (应 <2mm; 大=采样时车没停稳/霍夫没命中)")
    print(f"\n复制进 config.py:")
    print(f"IMG2BODY = ({a:.4f}, {b:.4f}, {c:.4f}, {d:.4f})")


if __name__ == "__main__":
    main()
