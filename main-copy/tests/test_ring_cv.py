"""OpenCV 圆环检测调参/验证台(vision/ring_cv.py, 替代 YOLO 环检测的实车调参主力)。

用法:
  实时:   python3 tests/test_ring_cv.py --gui    # 末端相机实时+检出叠加(调参主力)
  批量:   python3 tests/test_ring_cv.py --offline <照片目录> [--out <目录>]
  耗时:   python3 tests/test_ring_cv.py --bench
参数覆盖(透传 ring_cv.DEFAULTS): --major-lo/--major-hi/--min-rings/--group-dist/...
窗口: 绿椭圆=检出环 中心十字+环号 灰椭圆=未成组候选(--show-all); q 退出。
⚠️ ring_check1 默认参数调自旧下视相机照片, 末端相机实车视野必须重调——
先 --gui 调到稳定检出三环+数字, 再跑 test_nav_paotu 放环段。
"""
import argparse
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import cv2   # noqa: E402
import numpy as np   # noqa: E402

import config as C   # noqa: E402
from vision import ring_cv   # noqa: E402


def draw(frame, results, show_all=False, all_ellipses=None):
    disp = frame.copy()
    if show_all and all_ellipses:
        for (cx, cy), (major, minor, angle), _, _ in all_ellipses:
            cv2.ellipse(disp, (int(cx), int(cy)), (int(major / 2),
                        int(minor / 2)), angle, 0, 360, (160, 160, 160), 1)
    for digit, cx, cy, outer, box, conf in results:
        major, minor, angle = outer
        cv2.ellipse(disp, (int(cx), int(cy)), (int(major / 2),
                    int(minor / 2)), angle, 0, 360, (0, 255, 0), 2)
        cv2.drawMarker(disp, (int(cx), int(cy)), (0, 0, 255),
                       cv2.MARKER_CROSS, 18, 2)
        label = f"ring{digit} ({int(cx)},{int(cy)})" + (
            f" conf={conf:.2f}" if conf else " [order]")
        cv2.putText(disp, label, (int(cx) - 60,
                    int(cy) - int(major / 2) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if box:
            cv2.rectangle(disp, box[:2],
                          (box[0] + box[2], box[1] + box[3]), (255, 0, 0), 1)
    # 相机中心(蓝十字, 纠偏基准=画面中心) ↔ 圆环中心(红叉): 连线+纠偏修正量
    ch_x, ch_y = disp.shape[1] // 2, disp.shape[0] // 2
    cv2.drawMarker(disp, (ch_x, ch_y), (255, 0, 0), cv2.MARKER_CROSS, 26, 2)
    cv2.putText(disp, "camera center", (ch_x + 10, ch_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    a, b, cc, d = C.IMG2BODY_RING
    for digit, cx, cy, outer, box, conf in results:
        dx, dy = cx - ch_x, cy - ch_y
        cv2.line(disp, (ch_x, ch_y), (int(cx), int(cy)), (0, 255, 255), 1)
        dy_right = a * dx + b * dy
        dx_fwd = cc * dx + d * dy
        cv2.putText(disp, f"右{dy_right:+.1f} 前{dx_fwd:+.1f}mm",
                    (int(cx) - 50, int(cy) + int(outer[1] / 2) + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return disp


def main():
    ap = argparse.ArgumentParser(description="OpenCV 圆环检测调参/验证台")
    ap.add_argument("--cam", default=C.CAMERA_ID_END)
    ap.add_argument("--gui", action="store_true", help="末端相机实时+检出叠加")
    ap.add_argument("--offline", default=None, help="批量验证照片目录")
    ap.add_argument("--out", default="ring_cv_results", help="批量标注输出目录")
    ap.add_argument("--bench", action="store_true", help="单帧耗时测试")
    ap.add_argument("--show-all", action="store_true", help="画未成组候选椭圆(灰)")
    ap.add_argument("--major-lo", type=float, default=None)
    ap.add_argument("--major-hi", type=float, default=None)
    ap.add_argument("--min-rings", type=int, default=None)
    ap.add_argument("--group-dist", type=float, default=None)
    args = ap.parse_args()

    overrides = {k: v for k, v in (("major_lo", args.major_lo),
                                   ("major_hi", args.major_hi),
                                   ("min_rings", args.min_rings),
                                   ("group_dist", args.group_dist))
                 if v is not None}

    def find(frame):
        return ring_cv.find(frame, **overrides)

    if args.bench:
        import numpy as np
        img = np.full((720, 1280, 3), 235, np.uint8)
        for cx in (320, 640, 960):
            for r in (80, 60, 40):
                cv2.circle(img, (cx, 360), r, (30, 30, 30), 5)
        t0 = time.time()
        n = 20
        for _ in range(n):
            find(img, **overrides)
        dt = (time.time() - t0) / n * 1000
        print(f"--bench: 平均 {dt:.1f} ms/帧 ({n} 次, 合成环图)")
        return

    from vision.camera import Camera

    if args.offline:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        photos = sorted(Path(args.offline).glob("*.jpg"))
        photos += sorted(Path(args.offline).glob("*.png"))
        print(f"批量: {len(photos)} 张 → {out}")
        for p in photos:
            frame = cv2.imread(str(p))
            if frame is None:
                continue
            t0 = time.time()
            results = find(frame, **overrides)
            disp = draw(frame, results)
            cv2.imwrite(str(out / p.name), disp)
            seg = " ".join(f"ring{r[0]}@({r[1]:.0f},{r[2]:.0f})"
                           for r in results) or "(无检出)"
            print(f"  {p.name}: {len(results)} 个 [{seg}] "
                  f"({time.time() - t0:.0f}ms)")
        return

    cam = Camera(args.cam, "ring_cv")
    cam.open()
    print(f"实时检测({args.cam}) | 绿椭圆=检出环 中心十字+环号 | "
          f"灰椭圆=候选(--show-all) | q 退出")
    last_log = 0.0
    while True:
        frame = cam.read_latest()
        if frame is None:
            continue
        t0 = time.time()
        results = find(frame, **overrides)
        seg = " ".join(f"ring{r[0]}@({r[1]:.0f},{r[2]:.0f})"
                       for r in results) or "(无检出)"
        if time.time() - last_log > 0.5:
            last_log = time.time()
            print(f"  检出 {len(results)}: {seg} ({(time.time() - t0) * 1000:.0f}ms)")
        if args.gui:
            disp = draw(frame, results)
            cv2.imshow("ring_cv", disp)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            time.sleep(0.2)
    cam.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n退出")
