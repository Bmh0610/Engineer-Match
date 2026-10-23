"""IMG2BODY 标定向导(实时预览版): 窗口常驻显示检测画面, 终端回车采样。

三次采样: 回车采基准 p0 → 发『前进100』帧 → 停稳回车采 p1 →
发『右移100』帧 → 停稳回车采 p2 → 自动解算 IMG2BODY(config 格式+自检)。
跳变保护: 采样离上次圆心 >350px 判定抓错目标, 拒绝并要求重试。
每步的协议帧在采样前打印(整段可复制到 Nano); 窗口标题栏同步显示当前步骤。

用法: python3 test_img2body_cal.py --cam 2    (窗口内 q 退出)
"""
import argparse
import select
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import cv2   # noqa: E402

import config as C    # noqa: E402
import finder         # noqa: E402
from protocol import Mode, frame   # noqa: E402
from test_yolo_live import render   # noqa: E402

HOUGH_P2_SOFT = 20    # 实拍顶面边缘弱, 累加器门槛放宽
STEP = 100.0          # 标定移动步长 mm
JUMP_MAX = 350.0      # 采样跳变保护(px)


def solve_img2body(d_right_pix, d_fwd_pix, step=STEP):
    """两次已知车体位移的像素响应 → (IMG2BODY 四元组, 自检残差)。

    模型: 像素位移 = M @ 车体位移; IMG2BODY = M⁻¹ (像素→车体系)。
    """
    M = np.column_stack([np.asarray(d_right_pix, float) / step,
                         np.asarray(d_fwd_pix, float) / step])
    Minv = np.linalg.inv(M)
    chk_r = Minv @ np.asarray(d_right_pix, float)
    chk_f = Minv @ np.asarray(d_fwd_pix, float)
    res = max(abs(chk_r[0] - step), abs(chk_r[1]), abs(chk_f[0]), abs(chk_f[1] - step))
    a, b = Minv[0]
    c, d = Minv[1]
    return (a, b, c, d), res


def grab_center(backend, cam, conf, expect):
    """稳定帧 + 霍夫顶面精化 → (cx, cy); 检不出/跳变过大返回 None。"""
    obj, frame_img = finder.find_stable(
        lambda f: finder.nearest_to_axis(finder.find_materials(
            backend, f, conf=conf, px_per_mm=C.PX_PER_MM)),
        cam, timeout=2.0)
    if obj is None:
        print("  !! 未检出稳定物料, 再回车重试")
        return None
    if expect is not None:
        import math
        jump = math.hypot(obj.cx - expect[0], obj.cy - expect[1])
        if jump > JUMP_MAX:
            print(f"  !! 采样跳变 {jump:.0f}px: 抓到的不是同一个物料"
                  f"({obj.cx:.0f},{obj.cy:.0f}), 调整后回车重试")
            return None
    finder.refine_material_stages(frame_img, obj, C.PX_PER_MM,
                                  r_lo=finder.CAP_R_LO, r_hi=finder.CAP_R_HI,
                                  param2=HOUGH_P2_SOFT)
    print(f"  ✓ 圆心 = ({obj.cx:.1f}, {obj.cy:.1f}) "
          f"({'霍夫顶面' if obj.hough else 'bbox中心'})")
    return obj.cx, obj.cy


def main():
    ap = argparse.ArgumentParser(description="IMG2BODY 标定向导(实时预览)")
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
    print("物料放视野中央全程不动。三步采样:")
    print("  p0      : 不动, 直接回车采基准")
    tx1 = frame(Mode.MOVE_V, 0, int(STEP), 0)
    print(f"  p1 之前 : 发帧 {tx1.hex(' ').upper()}  (前进{STEP:.0f})")
    tx2 = frame(Mode.MOVE_V, int(STEP), 0, 0)
    print(f"  p2 之前 : 发帧 {tx2.hex(' ').upper()}  (右移{STEP:.0f})")
    print("  发送命令模板(把 <帧hex> 换成上面对应帧, 去空格):")
    print('  python3 -c "import serial; serial.Serial(\'/dev/ttyUSB0\',115200,'
          "timeout=0.1).write(bytes.fromhex('<帧hex>'))\"")
    print("  (窗口标题栏也会同步提示当前步骤)")

    steps = [(0, int(STEP), 0, "前进100"), (int(STEP), 0, 0, "右移100")]
    pts = []          # (cx, cy) ×3
    round_no = 0
    while len(pts) < 3:
        frame = cam.read_latest()
        if frame is None:
            continue
        mats = finder.find_materials(backend, frame, conf=args.conf,
                                     px_per_mm=C.PX_PER_MM)
        m = finder.nearest_to_axis(mats)
        if m is not None:
            finder.refine_material_stages(
                frame, m, C.PX_PER_MM, r_lo=finder.CAP_R_LO,
                r_hi=finder.CAP_R_HI, param2=HOUGH_P2_SOFT)
        cv2.imshow("img2body_cal", render(frame, mats, m))
        # 步进提示写窗口标题栏(终端输出不会刷掉它)
        if len(pts) == 0:
            hint = "回车采 p0(基准)"
        else:
            x, y, z, label = steps[len(pts) - 1]
            tx = frame(Mode.MOVE_V, x, y, z)
            hint = (f"发帧: {tx.hex(' ').upper()} ({label}), 停稳后回车采 p{len(pts)}")
        cv2.setWindowTitle("img2body_cal", f"img2body_cal | {hint}")
        # 非阻塞监听终端回车
        if select.select([sys.stdin], [], [], 0)[0]:
            line = sys.stdin.readline().strip()
            if line.lower() == "q":
                break
            expect = pts[-1] if pts else None
            round_no += 1
            if 1 <= len(pts) <= 2:
                x, y, z, label = steps[len(pts) - 1]
                txc = frame(Mode.MOVE_V, x, y, z)
                print(f"  本步先发帧: {txc.hex(' ').upper()} ({label}), 停稳后再回车采样")
                continue
            c = grab_center(backend, cam, args.conf, expect)
            if c is not None:
                pts.append(c)
        k = cv2.waitKey(30) & 0xFF
        if k == ord("q"):
            break

    if len(pts) == 3:
        c0, c1, c2 = pts
        d_fwd = (c1[0] - c0[0], c1[1] - c0[1])
        d_right = (c2[0] - c1[0], c2[1] - c1[1])
        print(f"\n前进{STEP:.0f}mm 像素位移: ({d_fwd[0]:.1f}, {d_fwd[1]:.1f})")
        print(f"右移{STEP:.0f}mm 像素位移: ({d_right[0]:.1f}, {d_right[1]:.1f})")
        quad, res = solve_img2body(d_right, d_fwd, STEP)
        print(f"自检最大残差 {res:.2f}mm (应 <2mm; 大=采样时车没停稳/抓错目标)")
        a, b, c, d = quad
        print(f"\n复制进 config.py:")
        print(f"IMG2BODY = ({a:.4f}, {b:.4f}, {c:.4f}, {d:.4f})")
    else:
        print("采样不足 3 点, 未解算")
    cam.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
