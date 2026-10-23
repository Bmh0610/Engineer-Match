"""YOLO 实时画面测试台: 每帧显示检出物料的颜色/置信度/圆心偏差(px+mm)。

用法:
  python3 tests/test_yolo_live.py --cam 2      # 实时; q 退出
换设备号: --cam 0/1/2/3 (编号随 USB 口变); --conf 改检出门槛; p 键开关跨帧后处理。

画面标注:
  物料框 = 颜色近似色; 圆心点 = 定位圆心(圆盘拟合, 已精化);
  实线框 = conf≥0.85(过抓取颜色确认门槛), 虚线框 = 不过;
  加粗框 = 离车最近物料; [axis] = 主点正下方(垂直视角, 定位基准);
  白圈 = 圆盘定位命中(最小外接圆); 绿十字 = 画面中心;
  橙框 = 抓取窗(config.GRAB_WINDOW, 目标进窗才出手)。
偏差文字 = 车移动坐标系 X右(臂侧) Y前(车头) mm + 图像 px, 与 protocol.move_v
参数同口径(纠偏发它的负值), 供纠偏量级预判。
终端每 0.5s 打一行汇总(*=离车最近), 不刷屏。
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
from test_yolo_live import render   # noqa: E402  复用主画面渲染(同目录)


def main():
    ap = argparse.ArgumentParser(description="YOLO 实时画面测试台(颜色+偏差)")
    ap.add_argument("--cam", type=int, default=2)
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--model", default=C.MODEL_PATH)
    args = ap.parse_args()

    from camera import Camera
    from detect import YOLOBackend

    cam = Camera(args.cam, "yolo_live")
    cam.open()
    print(f"加载模型 {args.model} ...")
    backend = YOLOBackend(args.model)
    print("标注: 实线=过颜色确认(conf≥0.85) 虚线=不过 | 加粗=离车最近 | "
          "绿十字=画面中心 | 橙框=抓取窗 | 白圈=圆盘定位命中")
    print("按键: 空格=定格 | p=跨帧后处理开关 | q 退出")

    tracker = finder.MaterialTracker()   # 跨帧后处理(消歧+确认+低通+保持)
    tracker_on = True
    frozen = None
    last_log = 0.0

    while True:
        frame = frozen if frozen is not None else cam.read_latest()
        if frame is None:
            continue
        mats = finder.find_materials(backend, frame, conf=args.conf,
                                     px_per_mm=C.PX_PER_MM)
        if tracker_on:
            mats = finder.disambiguate_pairs(mats)   # 易混对消歧
        nearest_m = finder.nearest(mats)
        cv2.imshow("yolo_live", render(frame, mats, nearest_m))
        if tracker_on:
            display_mats = tracker.update(mats, frame.shape, C.PX_PER_MM)
        else:
            display_mats = mats
        now = time.time()
        if display_mats and now - last_log > 0.5:
            last_log = now
            parts = [f"{'*' if m is nearest_m else ''}{m.color}"
                     f"({m.conf:.2f}, px {m.dx:+.0f},{m.dy:+.0f})"
                     f"{' [hold]' if m.holding else ' [new]' if not m.confirmed else ''}"
                     for m in display_mats]
            tag = "[定格] " if frozen is not None else ""
            print(tag + "检出: " + "  ".join(parts))
        key = cv2.waitKey(30) & 0xFF
        if key == ord(" "):
            frozen = None if frozen is None else frame.copy()
            print("定格" if frozen is not None else "恢复实时")
        elif key == ord("p"):
            tracker_on = not tracker_on
            tracker.reset()
            print(f"跨帧后处理: {'开' if tracker_on else '关'}")
        elif key == ord("q"):
            break
    cam.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
