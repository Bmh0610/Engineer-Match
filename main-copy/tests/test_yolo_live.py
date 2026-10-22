"""YOLO 实时画面测试台: 每帧显示检出物料的颜色/置信度/圆心偏差(px+mm)。

用法:
  python3 tests/test_yolo_live.py --cam 2      # 实时; q 退出
换设备号: --cam 0/1/2/3 (2/3=USB 相机成对, 0/1=内置); --conf 改检出门槛。

窗口:
  yolo_live      主画面: YOLO 叠加(框/颜色/偏差/抓取窗/主点/霍夫白圈)
  hough_pipeline 霍夫流水线面板: 跟踪物料的 ROI→灰度→模糊→二值化→Canny→霍夫
                 六格中间图, 顶部常显当前参数与候选数。

按键:
  空格   定格/恢复(冻结图像, YOLO+霍夫在定格帧上持续重算——调参用它)
  w/s    霍夫 param2 ±5(累加器门槛: 误检多加, 漏检减; 范围 10~100)
  a/d    霍夫 param1 ±10(Canny 高阈值; 范围 50~300)
  鼠标点击画面中某物料框   面板切换跟踪该物料(默认跟踪离车最近的)
  u      去畸变预览切换;  h  霍夫精化开关;  q  退出
"""
import argparse
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import os
os.environ.setdefault("QT_LOGGING_RULES", "*.debug=false;*.warning=false")

import cv2
import numpy as np

import config as C       # noqa: E402
import finder            # noqa: E402

BOX_BGR = {"red": (0, 0, 255), "green": (0, 255, 0), "blue": (255, 0, 0),
           "yellow": (0, 255, 255), "light_blue": (255, 255, 0),
           "black": (160, 160, 160)}
MAIN_PT, CENTER_PT, WINDOW_C = (0, 200, 0), (255, 0, 0), (0, 165, 255)


def dash_line(img, p1, p2, color, thickness=1, dash=8):
    x1, y1 = p1
    x2, y2 = p2
    n = max(int(np.hypot(x2 - x1, y2 - y1) // dash), 1)
    for i in range(0, n, 2):
        s, e = i / n, min((i + 1) / n, 1.0)
        cv2.line(img, (int(x1 + (x2 - x1) * s), int(y1 + (y2 - y1) * s)),
                 (int(x1 + (x2 - x1) * e), int(y1 + (y2 - y1) * e)),
                 color, thickness)


def dash_rect(img, x, y, w, h, color, thickness=1, dash=8):
    dash_line(img, (x, y), (x + w, y), color, thickness, dash)
    dash_line(img, (x + w, y), (x + w, y + h), color, thickness, dash)
    dash_line(img, (x + w, y + h), (x, y + h), color, thickness, dash)
    dash_line(img, (x, y + h), (x, y), color, thickness, dash)


def cross(img, x, y, size, color, thickness=1):
    cv2.line(img, (x - size, y), (x + size, y), color, thickness)
    cv2.line(img, (x, y - size), (x, y + size), color, thickness)


def to_body_str(m):
    """图像偏差 → 车体系文字。X=车系x(右,臂侧) Y=车系y(前,车头),
    与 protocol.move_v(x, y) 参数一一对应(纠偏发的是它的负值)。"""
    dy_right = m.dx_mm * C.IMG2BODY[0]
    dx_fwd = m.dy_mm * C.IMG2BODY[1]
    return f"X右{dy_right:+.1f} Y前{dx_fwd:+.1f}mm"


def render(frame, mats, nearest_m, undistort_view=False, tracker_on=False):
    """检测结果叠加到帧副本(主画面)。tracker_on 时未确认=虚线+[new]、保持中=[hold]。"""
    c = finder._CALIB
    if undistort_view and c is not None:
        disp = cv2.undistort(frame, c._K, c.dist)

        def mp(x, y):   # 原图坐标 → 去畸变坐标(与整帧 remap 同一变换)
            return c.undistort_point(x, y)
    else:
        disp = frame.copy()

        def mp(x, y):
            return float(x), float(y)
    h, w = disp.shape[:2]
    x0, x1, y0, y1 = C.GRAB_WINDOW
    wx, wy = int(x0 * w), int(y0 * h)
    ww, wh = int((x1 - x0) * w), int((y1 - y0) * h)
    cv2.rectangle(disp, (wx, wy), (wx + ww, wy + wh), WINDOW_C, 1)
    cv2.putText(disp, "grab window", (wx + 4, wy + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, WINDOW_C, 1)
    if c is not None:
        px, py = c.principal_point(w, h)
    else:
        px, py = w / 2, h / 2
    ppx, ppy = mp(px, py)
    cross(disp, int(ppx), int(ppy), 12, MAIN_PT)        # 绿十字=标定主点
    pcx, pcy = mp(w / 2, h / 2)
    cross(disp, int(pcx), int(pcy), 12, CENTER_PT)      # 蓝十字=画面中心
    for m in mats:
        color = BOX_BGR.get(m.color, (255, 255, 255))
        bx1, by1, bx2, by2 = map(int, m.box)
        if undistort_view and c is not None:
            corners = [mp(bx1, by1), mp(bx2, by1), mp(bx2, by2), mp(bx1, by2)]
            xs = [p[0] for p in corners]
            ys = [p[1] for p in corners]
            bx1, by1, bx2, by2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
        solid = m.conf >= C.CONFIRM_CONF and not (tracker_on and not m.confirmed)
        if solid:
            cv2.rectangle(disp, (bx1, by1), (bx2, by2), color,
                          3 if m is nearest_m else 2)
        else:
            dash_rect(disp, bx1, by1, bx2 - bx1, by2 - by1, color)
        ccx, ccy = mp(m.cx, m.cy)
        cv2.circle(disp, (int(ccx), int(ccy)), 4, color, -1)
        if m.hough is not None:   # 霍夫精化命中: 白圈=传统 CV 找到的圆环
            hx, hy, hr = m.hough
            hxx, hyy = mp(hx, hy)
            cv2.circle(disp, (int(hxx), int(hyy)), max(int(hr), 3), (255, 255, 255), 1)
        dy_right = m.dx_mm * C.IMG2BODY[0]
        dx_fwd = m.dy_mm * C.IMG2BODY[1]
        # 画面标注 ASCII: X=车系x(右,臂侧) Y=车系y(前,车头) = move_v 参数口径
        status = (" [hold]" if m.holding else
                  " [new]" if tracker_on and not m.confirmed else "")
        label = (f"{m.color} {m.conf:.2f} | X{dy_right:+.1f} Y{dx_fwd:+.1f}mm "
                 f"(px {m.dx:+.0f},{m.dy:+.0f})"
                 + (" [near]" if m is nearest_m else "")
                 + (" H" if m.hough is not None else "") + status)
        cv2.putText(disp, label, (bx1, max(14, by1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    tag = "UNDISTORTED VIEW (u 切回原图)" if undistort_view else \
          "RAW (space=定格  u=去畸变  h=霍夫  p=后处理  q=退出)"
    cv2.putText(disp, tag, (10, 700), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1)
    return disp


CELL = 320   # 流水线面板单格边长


def _cell(img, label, size=CELL):
    if img is None:
        c = np.full((size, size, 3), 30, np.uint8)
        cv2.putText(c, "N/A", (size // 2 - 30, size // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 120, 120), 1)
    else:
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        c = cv2.resize(img, (size, size))
    cv2.putText(c, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 255, 255), 1)
    return c


def render_panel(stages, hit, param1, param2, r_lo, r_hi, pick):
    """霍夫流水线六格面板: ROI/灰度/模糊/二值化/Canny/霍夫结果。"""
    roi = stages.get("roi")
    cands = stages.get("candidates")
    res = None if roi is None else roi.copy()
    if res is not None and cands is not None:
        for cd in cands:   # 全部候选=绿细圈
            cv2.circle(res, (int(cd[0]), int(cd[1])), int(cd[2]), (0, 255, 0), 1)
        best = stages.get("best_roi")
        if best is not None:   # 选中=红粗圈+圆心十字
            cv2.circle(res, (int(best[0]), int(best[1])), int(best[2]), (0, 0, 255), 2)
            cv2.drawMarker(res, (int(best[0]), int(best[1])), (0, 0, 255),
                           cv2.MARKER_CROSS, 14, 1)
    n_cand = 0 if cands is None else len(cands)
    info = (f"p1={param1} p2={param2} r=[{r_lo:.2f},{r_hi:.2f}] pick={pick} "
            f"cands={n_cand} hit={'Y' if hit else 'N'} | 1-4:r  c:pick")
    top = np.full((26, CELL * 3, 3), 30, np.uint8)
    cv2.putText(top, info, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 255, 255), 1)
    row1 = np.hstack([_cell(stages.get("roi"), "1 ROI"),
                      _cell(stages.get("gray"), "2 gray"),
                      _cell(stages.get("blur"), "3 blur")])
    row2 = np.hstack([_cell(stages.get("binary"), "4 binary(otsu)"),
                      _cell(stages.get("edges"), "5 canny"),
                      _cell(res, "6 hough")])
    return np.vstack([top, row1, row2])


def _pick(mats, pt):
    """点击点命中的物料框; 无则 None(回退 nearest)。"""
    if pt is None:
        return None
    for m in mats:
        x1, y1, x2, y2 = m.box
        if x1 <= pt[0] <= x2 and y1 <= pt[1] <= y2:
            return m
    return None


def main():
    ap = argparse.ArgumentParser(description="YOLO 实时画面测试台(颜色+偏差+霍夫流水线)")
    ap.add_argument("--cam", type=int, default=None, help="相机设备号(默认 config.CAMERA_ID_END)")
    ap.add_argument("--conf", type=float, default=0.7, help="检出门槛")
    ap.add_argument("--model", default=C.MODEL_PATH, help="YOLO 权重路径")
    ap.add_argument("--no-hough", action="store_true", help="关闭霍夫圆精化(默认开)")
    args = ap.parse_args()

    from camera import Camera
    from detect import YOLOBackend

    cam = Camera(C.CAMERA_ID_END if args.cam is None else args.cam, "yolo_live")
    cam.open()
    print(f"加载模型 {args.model} ...")
    backend = YOLOBackend(args.model)
    if finder._CALIB is not None:
        c = finder._CALIB
        print(f"内参已加载: fx={c.fx:.1f} 主点=({c.cx:.1f},{c.cy:.1f}) rms={c.rms:.2f}"
              f" —— 偏差数值已去畸变; 画面原图(训练口径), 按 u 看去畸变预览")
    else:
        print("!! 内参未加载(vision/calib_end.yaml 缺失): 偏差按画面中心+带畸变计算 !!")
    print("yolo_live 窗口: 实线=过颜色确认 虚线=不过 | 加粗=离车最近 | 绿十字=主点 "
          "蓝十字=画面中心 | 橙框=抓取窗 | 白圈=霍夫圆")
    print("按键: 空格=定格(冻结帧上调参) | w/s=param2±5  a/d=param1±10 | "
          "1/2=r下限  3/4=r上限±0.05 | t=整体/顶面半径组  c=候选策略 | "
          "p=跨帧后处理(消歧/确认/低通/保持) | 点击物料框=面板跟踪它 | "
          "u=去畸变预览 h=霍夫开关 q=退出")

    # 会话状态
    frozen = None          # None=实时; 否则定格帧
    click_pt = None        # 点击选中的画面坐标(定格/实时均可点)
    param1, param2 = finder.HOUGH_PARAM1, finder.HOUGH_PARAM2
    r_lo, r_hi = finder.HOUGH_R_LO, finder.HOUGH_R_HI
    pick = "nearest"       # 多候选选择: nearest=离框心最近 / min_r=半径最小优先
    undistort_view = False
    hough_on = not args.no_hough
    tracker = finder.MaterialTracker()   # 跨帧后处理(消歧+确认+低通+保持)
    tracker_on = True
    last_stages, last_hit = None, False
    last_log = 0.0

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            nonlocal_click.append((x, y))

    nonlocal_click = []
    cv2.namedWindow("yolo_live")
    cv2.setMouseCallback("yolo_live", on_mouse)

    while True:
        frame = frozen if frozen is not None else cam.read_latest()
        if frame is None:
            continue
        if nonlocal_click:   # 取最新点击
            click_pt = nonlocal_click.pop()
        mats = finder.find_materials(backend, frame, conf=args.conf,
                                     px_per_mm=C.PX_PER_MM)
        if tracker_on:
            mats = finder.disambiguate_pairs(mats)   # 易混对消歧(霍夫前做)
        target = _pick(mats, click_pt) or finder.nearest(mats)
        if hough_on:
            for m in mats:
                if m is target:   # 跟踪物料: 用当前参数并产出中间图
                    last_hit, last_stages = finder.refine_material_stages(
                        frame, m, C.PX_PER_MM, param1=param1, param2=param2,
                        pick=pick)
                else:
                    finder.refine_material(frame, m, px_per_mm=C.PX_PER_MM)
        else:
            last_stages = None
        if tracker_on:
            display_mats = tracker.update(mats, frame.shape, C.PX_PER_MM)
        else:
            display_mats = mats
        nearest_m = finder.nearest(display_mats)
        cv2.imshow("yolo_live", render(frame, display_mats, nearest_m,
                                       undistort_view, tracker_on))
        if hough_on and last_stages is not None:
            cv2.imshow("hough_pipeline", render_panel(last_stages, last_hit,
                                                      param1, param2, r_lo, r_hi,
                                                      pick))
        now = time.time()
        if display_mats and now - last_log > 0.5:
            last_log = now
            parts = [f"{'*' if m is nearest_m else ''}{m.color}"
                     f"({m.conf:.2f}, {to_body_str(m)}, px {m.dx:+.0f},{m.dy:+.0f})"
                     f"{' H' if m.hough is not None else ' bbox'}"
                     f"{' [hold]' if m.holding else ' [new]' if not m.confirmed else ''}"
                     for m in display_mats]
            tag = "[定格] " if frozen is not None else ""
            print(tag + "检出: " + "  ".join(parts))

        key = cv2.waitKeyEx(1)
        k = key & 0xFF if key != -1 else -1
        if k == ord(" "):
            if frozen is None:
                frozen = frame.copy()
                print("已定格: w/s=p2  a/d=p1  1/2/3/4=半径范围  t=顶面组  c=策略, "
                      "点击选物料; 空格恢复实时")
            else:
                frozen = None
                print("恢复实时")
        elif k in (ord("w"), 82) or key in (2490368, 0x260000):
            param2 = min(param2 + 5, 100)
            print(f"param2={param2}")
        elif k in (ord("s"), 84) or key in (2621440, 0x280000):
            param2 = max(param2 - 5, 10)
            print(f"param2={param2}")
        elif k in (ord("a"), 81) or key in (2424832, 0x250000):
            param1 = max(param1 - 10, 50)
            print(f"param1={param1}")
        elif k in (ord("d"), 83) or key in (2555904, 0x270000):
            param1 = min(param1 + 10, 300)
            print(f"param1={param1}")
        elif k == ord("u"):
            undistort_view = not undistort_view
        elif k == ord("h"):
            hough_on = not hough_on
            print(f"霍夫圆精化: {'开' if hough_on else '关'}")
        elif k == ord("1"):
            r_lo = max(r_lo - 0.05, 0.10)
            print(f"r=[{r_lo:.2f},{r_hi:.2f}]")
        elif k == ord("2"):
            r_lo = min(r_lo + 0.05, r_hi - 0.05)
            print(f"r=[{r_lo:.2f},{r_hi:.2f}]")
        elif k == ord("3"):
            r_hi = max(r_hi - 0.05, r_lo + 0.05)
            print(f"r=[{r_lo:.2f},{r_hi:.2f}]")
        elif k == ord("4"):
            r_hi = min(r_hi + 0.05, 1.35)
            print(f"r=[{r_lo:.2f},{r_hi:.2f}]")
        elif k == ord("c"):
            pick = "min_r" if pick == "nearest" else "nearest"
            print(f"候选策略: {pick}({'半径最小优先' if pick == 'min_r' else '离框心最近'})")
        elif k == ord("t"):
            if (r_lo, r_hi) == (finder.CAP_R_LO, finder.CAP_R_HI):
                r_lo, r_hi = finder.HOUGH_R_LO, finder.HOUGH_R_HI
                mode = "整体立体"
            else:
                r_lo, r_hi = finder.CAP_R_LO, finder.CAP_R_HI
                mode = "顶面小圆"
            print(f"半径组[{mode}]: r=[{r_lo:.2f},{r_hi:.2f}]")
        elif k == ord("p"):
            tracker_on = not tracker_on
            tracker.reset()
            print(f"跨帧后处理(消歧+确认+低通+保持): {'开' if tracker_on else '关'}")
        elif k == ord("q"):
            break
    cam.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
