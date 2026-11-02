"""YOLO 双画面测试台 + render() 定义处(align_manual/img2body_cal 复用)。

左画面=末端相机 × wuliao 引擎(物料 6 色, 原 render 全标注); 右画面=前视避障相机 ×
obstacle 引擎(障碍单类, 车道 ROI 门控: 红粗框=过门控会封边/灰细框=被滤, 绿矩形=LANE_ROI
中间2/3×下面5/6)。两画面合成一个窗口并排显示。

用法: python3 tests/test_yolo_live.py    # 默认 cam=cam_end + cam_front, 双引擎
  --cam-end/--cam-front 换设备 | --model-material/--model-obstacle 换权重 | --conf 共用门限
标注(左): 实线=conf≥0.85 虚线=不过 | 加粗=离车最近 | 白圈=圆盘定位 | 橙框=抓取窗
      | 绿十字=画面中心(纠偏基准) | 偏差文字=X右/Y前 mm(车系, move_v 口径)。
按键(共用): 空格=定格 u=去畸变(左) p=跨帧后处理 q=退出。
render() 签名未动——test_align_manual/img2body_cal 仍复用。
"""
import argparse
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import cv2   # noqa: E402

import config as C    # noqa: E402
from vision import finder
import vision.obstacle as obs          # 轻量(仅 threading/time); render_front 用其 LANE_ROI/_gate
from vision.obstacle import est_dist   # noqa: F402 (render_front 标注用)

BOX_BGR = {"red": (0, 0, 255), "green": (0, 255, 0), "blue": (255, 0, 0),
           "yellow": (0, 255, 255), "light_blue": (255, 255, 0),
           "black": (160, 160, 160)}
MAIN_PT, CENTER_PT, WINDOW_C = (0, 200, 0), (255, 0, 0), (0, 165, 255)


def cross(img, x, y, size, color, thickness=1):
    cv2.line(img, (x - size, y), (x + size, y), color, thickness)
    cv2.line(img, (x, y - size), (x, y + size), color, thickness)


def dash_rect(img, x, y, w, h, color, thickness=1, dash=8):
    for p1, p2 in [((x, y), (x + w, y)), ((x + w, y), (x + w, y + h)),
                   ((x + w, y + h), (x, y + h)), ((x, y + h), (x, y))]:
        n = max(int(cv2.norm((p2[0] - p1[0], p2[1] - p1[1])) // dash), 1)
        for i in range(0, n, 2):
            s, e = i / n, min((i + 1) / n, 1.0)
            cv2.line(img, (int(p1[0] + (p2[0] - p1[0]) * s),
                           int(p1[1] + (p2[1] - p1[1]) * s)),
                     (int(p1[0] + (p2[0] - p1[0]) * e),
                      int(p1[1] + (p2[1] - p1[1]) * e)), color, thickness)


def render(frame, mats, nearest_m, undistort_view=False, tracker_on=False):
    """检测结果叠加到帧副本(主画面)。被 test_align_manual/img2body_cal 复用。"""
    c = finder._CALIB
    if undistort_view and c is not None:
        disp = cv2.undistort(frame, c._K, c.dist)

        def mp(x, y):
            return c.undistort_point(x, y)
    else:
        disp = frame.copy()

        def mp(x, y):
            return float(x), float(y)
    h, w = disp.shape[:2]
    x0, x1, y0, y1 = C.GRAB_WINDOW
    wx, wy = int(x0 * w), int(y0 * h)
    cv2.rectangle(disp, (wx, wy),
                  (wx + int((x1 - x0) * w), wy + int((y1 - y0) * h)),
                  WINDOW_C, 1)
    cv2.putText(disp, "grab window", (wx + 4, wy + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, WINDOW_C, 1)
    ppx, ppy = mp(w / 2, h / 2)
    cross(disp, int(ppx), int(ppy), 12, MAIN_PT)   # 绿十字=画面中心(纠偏基准)
    for m in mats:
        color = BOX_BGR.get(m.color, (255, 255, 255))
        bx1, by1, bx2, by2 = map(int, m.box)
        solid = m.conf >= C.CONFIRM_CONF and not (tracker_on and not m.confirmed)
        if solid:
            cv2.rectangle(disp, (bx1, by1), (bx2, by2), color,
                          3 if m is nearest_m else 2)
        else:
            dash_rect(disp, bx1, by1, bx2 - bx1, by2 - by1, color)
        ccx, ccy = mp(m.cx, m.cy)
        cv2.circle(disp, (int(ccx), int(ccy)), 4, color, -1)
        if getattr(m, "ellipse", None):   # 圆盘定位命中: 白圈
            (ecx, ecy), (MA, ma), ang = m.ellipse
            exx, eyy = mp(ecx, ecy)
            cv2.ellipse(disp, (int(exx), int(eyy)),
                        (max(int(MA / 2), 3), max(int(ma / 2), 3)), ang,
                        0, 360, (255, 255, 255), 1)
        dy_right, dx_fwd = finder.to_body(m)
        status = (" [hold]" if m.holding else
                  " [new]" if tracker_on and not m.confirmed else "")
        label = (f"{m.color} {m.conf:.2f} | X{dy_right:+.1f} Y{dx_fwd:+.1f}mm "
                 f"(px {m.dx:+.0f},{m.dy:+.0f})"
                 + (" [near]" if m is nearest_m else "")
                 + (" E" if getattr(m, "ellipse", None) else "") + status)
        cv2.putText(disp, label, (bx1, max(14, by1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.putText(disp, "space=定格 u=去畸变 p=后处理 q=退出", (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return disp


def render_front(frame, dets):
    """右画面: 障碍检测结果叠加(ROI 绿矩形 + 过门控红粗框/被滤灰细框 + 状态大字)。"""
    import vision.obstacle as obs
    from vision.obstacle import est_dist
    h, w = frame.shape[:2]
    disp = frame.copy()
    if obs.LANE_ROI:   # 车道 ROI: 绿色半透明矩形(框底边中心落在内才算挡本车道)
        x0, x1, y0, y1 = obs.LANE_ROI
        overlay = disp.copy()
        cv2.rectangle(overlay, (int(x0 * w), int(y0 * h)),
                      (int(x1 * w), int(y1 * h)), (0, 255, 0), -1)
        cv2.addWeighted(overlay, 0.12, disp, 0.88, 0, disp)
        cv2.rectangle(disp, (int(x0 * w), int(y0 * h)),
                      (int(x1 * w), int(y1 * h)), (0, 255, 0), 1)
    gate = obs._gate(dets, disp.shape)
    for d in dets:
        on = d is gate
        x1, y1, x2, y2 = (int(v) for v in d.xyxy)
        color = (0, 0, 255) if on else (160, 160, 160)
        cv2.rectangle(disp, (x1, y1), (x2, y2), color, 3 if on else 1)
        cv2.putText(disp, f"{d.cls} {d.conf:.2f} "
                    f"~{est_dist(d.xyxy[3] - d.xyxy[1]):.0f}mm",
                    (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, color, 2)
    msg, color = (("OBSTACLE!", (0, 0, 255)) if gate else
                  ("detected (gated out)", (0, 200, 255)) if dets else
                  ("clear", (0, 255, 0)))
    cv2.putText(disp, msg, (8, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.putText(disp, "front/obstacle", (8, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return disp


def hstack_same_h(left, right):
    """两画面等高并排(高度取小者, 等比缩放)。"""
    h = min(left.shape[0], right.shape[0])

    def rs(img):
        s = h / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * s), h)) if s < 1 else img
    return cv2.hconcat([rs(left), rs(right)])


def main():
    ap = argparse.ArgumentParser(description="YOLO 双画面测试台(末端物料+前视障碍)")
    ap.add_argument("--cam-end", default=C.CAMERA_ID_END)
    ap.add_argument("--cam-front", default=C.CAMERA_ID_FRONT)
    ap.add_argument("--model-material", default=C.MODEL_PATH,
                    help="物料引擎(默认 wuliao-best.engine)")
    ap.add_argument("--model-obstacle", default=C.MODEL_PATH_OBSTACLE,
                    help="障碍引擎(默认 obstacle-best.engine)")
    ap.add_argument("--conf", type=float, default=0.5)
    args = ap.parse_args()

    import time

    from vision.camera import Camera
    from vision.detect import YOLOBackend

    end_cam = Camera(args.cam_end, "yolo_live/end")
    end_cam.open()
    front_cam = Camera(args.cam_front, "yolo_live/front")
    front_cam.open()
    print(f"物料引擎 {args.model_material} / 障碍引擎 {args.model_obstacle} ...")
    mat_backend = YOLOBackend(args.model_material)
    obs_backend = YOLOBackend(args.model_obstacle, names={0: "obstacle"})
    obs.WATCH_CONF = args.conf   # 右画面门限与左侧共用
    print("(左)实线=过颜色确认 虚线=不过 | 加粗=离车最近 | 白圈=圆盘定位 | "
          "绿十字=画面中心 | 橙框=抓取窗")
    print("(右)红粗框=过ROI门控 灰细框=被滤 | 绿矩形=LANE_ROI")
    print("按键: 空格=定格 | u=去畸变预览(左) | p=跨帧后处理 | q 退出")

    tracker = finder.MaterialTracker()
    tracker_on = True
    undistort_view = False
    frozen = None
    frozen_front = None
    last_log = 0.0
    while True:
        frame = frozen if frozen is not None else end_cam.read_latest()
        ffront = frozen_front if frozen_front is not None else front_cam.read_latest()
        if frame is None or ffront is None:
            time.sleep(0.05)
            continue
        mats = finder.find_materials(mat_backend, frame, conf=args.conf,
                                     px_per_mm=C.PX_PER_MM)
        if tracker_on:
            mats = finder.disambiguate_pairs(mats)
        nearest_m = finder.nearest(mats)
        display_mats = (tracker.update(mats, frame.shape, C.PX_PER_MM)
                        if tracker_on else mats)
        left = render(frame, display_mats, nearest_m, undistort_view, tracker_on)
        dets = obs_backend.infer(ffront)
        right = render_front(ffront, dets)
        gate = obs._gate(dets, ffront.shape)
        cv2.imshow("yolo_live", hstack_same_h(left, right))
        if time.time() - last_log > 0.5:
            last_log = time.time()
            parts = [f"{'*' if m is nearest_m else ''}{m.color}"
                     f"({m.conf:.2f}, px {m.dx:+.0f},{m.dy:+.0f})"
                     + (" [hold]" if m.holding else
                        " [new]" if not m.confirmed else "")
                     for m in display_mats]
            print(("[定格] " if frozen is not None else "") + "物料: "
                  + ("  ".join(parts) or "无")
                  + f" | 障碍: {'OBSTACLE!' if gate else 'clear'}")
        key = cv2.waitKey(30) & 0xFF
        if key == ord(" "):
            frozen = None if frozen is not None else frame.copy()
            frozen_front = (None if frozen_front is not None
                            else ffront.copy())
            print("定格" if frozen is not None else "恢复实时")
        elif key == ord("u"):
            undistort_view = not undistort_view
        elif key == ord("p"):
            tracker_on = not tracker_on
            tracker.reset()
            print(f"跨帧后处理: {'开' if tracker_on else '关'}")
        elif key == ord("q"):
            break
    end_cam.close()
    front_cam.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
