"""IMG2BODY 标定(手动平移点击版): 开摄像头→点参考点→手动物料→再点→自动换算。

不需要串口/车动: 物料由你手动平移(尺量距离)。
流程(物料放相机视野内, 全程同一平面):
  ① 点『挪之前』物料上的参考点(顶面圆心或任一固定特征)
  ② 沿『车头方向』平移 D mm(默认 100, 尺量) → 点同一参考点 → 回车确认
     → 得前向响应
  ③ 沿『右方向』(车头右手侧) 平移 D mm → 点同一参考点 → 回车确认
     → 得右向响应
  → IMG2BODY = M⁻¹(config 格式 + 自检), 复制进 config.py。
方向参照: 车头=+y, 车右(臂侧)=+x; 挪动尽量贴着车身轴向量(偏几度影响不大,
挪得越远相对角度误差越小)。点击: 画面窗口左键; 回车=确认本组; q 退出。

用法: python3 test_img2body_cal.py --cam 2 --dist 100
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
from test_yolo_live import render   # noqa: E402


def solve_img2body(d_right_pix, d_fwd_pix, dist):
    """两次已知车体位移的像素响应 → (IMG2BODY 四元组, 自检残差)。

    模型: 像素位移 = M @ 车体位移; IMG2BODY = M⁻¹ (像素→车体系)。
    """
    M = np.column_stack([np.asarray(d_right_pix, float) / dist,
                         np.asarray(d_fwd_pix, float) / dist])
    Minv = np.linalg.inv(M)
    chk_r = Minv @ np.asarray(d_right_pix, float)
    chk_f = Minv @ np.asarray(d_fwd_pix, float)
    res = max(abs(chk_r[0] - dist), abs(chk_r[1]), abs(chk_f[0]), abs(chk_f[1] - dist))
    a, b = Minv[0]
    c, d = Minv[1]
    return (a, b, c, d), res


def main():
    ap = argparse.ArgumentParser(description="IMG2BODY 标定(手动平移点击版)")
    ap.add_argument("--cam", type=int, default=C.CAMERA_ID_END)   # 默认=config; 换设备号用 --cam N
    ap.add_argument("--model", default=C.MODEL_PATH)
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--dist", type=float, default=100.0,
                    help="每组手动平移的距离 mm(尺量)")
    args = ap.parse_args()
    D = args.dist

    from camera import Camera
    from detect import YOLOBackend

    cam = Camera(args.cam, "cal")
    cam.open()
    print(f"加载模型 {args.model} ...")
    backend = YOLOBackend(args.model)

    st = {"clicks": [],       # 当前组已点击的坐标
          "phase": 0,         # 0=前向组 1=右向组
          "done": {"fwd": None, "right": None}}
    names = [("fwd", "车头方向"), ("right", "右方向(臂侧)")]

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(st["clicks"]) < 2:
            st["clicks"].append((float(x), float(y)))
            print(f"  点{len(st['clicks'])}: ({x}, {y})", flush=True)

    cv2.namedWindow("img2body_cal")
    cv2.setMouseCallback("img2body_cal", on_mouse)

    print(f"\n== 采样组 1: 任意方向平移 {D}mm ==")
    print("点『挪之前』物料上的参考点(顶面圆心/固定特征)...")

    while True:
        frame = cam.read_latest()
        if frame is not None:
            disp = frame.copy()
            for i, p in enumerate(st["clicks"], 1):
                cv2.circle(disp, (int(p[0]), int(p[1])), 5, (0, 0, 255), -1)
                cv2.putText(disp, str(i), (int(p[0]) + 6, int(p[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            if len(st["clicks"]) == 2:
                cv2.line(disp, tuple(map(int, st["clicks"][0])),
                         tuple(map(int, st["clicks"][1])), (0, 255, 0), 1)
            cv2.setWindowTitle(
                "img2body_cal",
                "img2body_cal | 组%d/2 (%s) 已点%d/2 | q 退出"
                % (st["phase"] + 1, names[st["phase"]][1], len(st["clicks"])))
            cv2.imshow("img2body_cal", disp)
        # 非阻塞监听终端回车(确认本组)
        if select.select([sys.stdin], [], [], 0)[0]:
            line = sys.stdin.readline().strip()
            if line.lower() == "q":
                break
            if len(st["clicks"]) == 2:
                d = np.array(st["clicks"][1]) - np.array(st["clicks"][0])
                key = line.lower() if line.lower() in ("f", "r") else None
                if key is None:
                    print("  请输入本组实际挪动方向: f=车头(+y) / r=右(+x), 防标反")
                    continue
                st["done"][key] = d
                print(f"  ✓ {key} 组像素位移: ({d[0]:.1f}, {d[1]:.1f})")
                st["clicks"] = []
                st["phase"] += 1
                if st["phase"] == 1:
                    print(f"\n== 第 2 组: 另一个方向平移 {D}mm ==")
                    print("点『挪之前』的参考点...")
                else:
                    break
            else:
                print("  本组点满 2 个, 回车时输入方向: f=车头(+y) / r=右(+x)")
        k = cv2.waitKey(30) & 0xFF
        if k == ord("q"):
            break

    if st["done"]["fwd"] is None or st["done"]["right"] is None:
        print("采样不足, 未解算")
        return
    (a, b, c, d), res = solve_img2body(st["done"]["right"], st["done"]["fwd"], D)
    print(f"\n自检最大残差 {res:.2f}mm (应 <2mm; 大=点的位置/挪动距离不准)")
    print("\n复制进 config.py:")
    print(f"IMG2BODY = ({a:.4f}, {b:.4f}, {c:.4f}, {d:.4f})")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
