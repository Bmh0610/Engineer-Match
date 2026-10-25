"""张正友相机内参标定三合一工具(搬自 main_match/test/test_calib.py, 输出改到 main-copy/vision)。

棋盘格可用屏幕显示: 平面屏+等比全屏+方格毫米实测。
采图: python3 tools/calib_intr.py --capture --cam 0 --dir /tmp/calib
      空格存图(角点找全才存), 采 15-20 张: 棋盘占画面 1/3-2/3、±30-45° 倾斜
      姿态多样、平移覆盖四角(带倾斜也避摩尔纹), 每姿态停稳再拍。
标定: python3 tools/calib_intr.py --calibrate --dir /tmp/calib --square 20.0
      RMS<0.8 存 tools/calib_end.yaml; 0.8-1.5 存并警告; >1.5 拒存
      (可 --force 强制保存); fx 前后半稳定性差异>40% 同样拒存。
验证: python3 tools/calib_intr.py --verify --cam 0
      画面中心(蓝) vs 标定主点(红)双十字; 右图里的直线应该真的直。
默认按配套 checkerboard_9x6.png 的 9x6 内角点检测(半分辨率节流保证窗口流畅);
用别的棋盘时 --w/--h 指定。分辨率与实战一致(1280x720)。
"""
import argparse
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import cv2
import numpy as np
import yaml

from vision.camera import Camera

COMMON_PATTERNS = [(9, 6), (8, 6), (10, 7), (7, 5), (11, 8), (6, 5)]
VISION_DIR = BASE / "tools"


def detect(gray, w, h):
    ok, corners = cv2.findChessboardCornersSB(gray, (w, h))
    return corners if ok else None


def auto_detect(gray):
    for w, h in COMMON_PATTERNS:
        c = detect(gray, w, h)
        if c is not None:
            return w, h, c
    return None


def run_capture(args):
    Path(args.dir).mkdir(parents=True, exist_ok=True)
    imgs = sorted(Path(args.dir).glob("calib_*.jpg"))
    saved = len(imgs)
    idx = max((int(p.stem.split("_")[1]) for p in imgs
               if p.stem.split("_")[-1].isdigit()), default=-1) + 1
    cam = Camera(args.cam, "calib")
    cam.open()
    mode = "纯终端自动存图" if args.nogui else "窗口+自动存图"
    print(f"已有 {saved} 张; {mode}; Ctrl+C 退出。")
    pattern = (args.w, args.h) if args.w and args.h else None
    corners, last_t, last_save = None, 0.0, 0.0
    try:
        while True:
            frame = cam.read_latest()
            if frame is None:
                continue
            disp = None if args.nogui else frame.copy()
            if time.time() - last_t > 0.5:   # 检测节流: 半分辨率, ~2 次/秒
                last_t = time.time()
                gray = cv2.cvtColor(cv2.resize(frame, None, fx=0.5, fy=0.5),
                                    cv2.COLOR_BGR2GRAY)
                if pattern:
                    c = detect(gray, *pattern)
                else:
                    r = auto_detect(gray)
                    if r:
                        pattern, c = (r[0], r[1]), r[2]
                        print(f"自动识别内角点规格: {pattern[0]}x{pattern[1]}")
                    else:
                        c = None
                corners = c
                if c is not None and time.time() - last_save > 2.0:
                    p = Path(args.dir) / f"calib_{idx:03d}.jpg"
                    cv2.imwrite(str(p), frame)
                    idx += 1
                    saved += 1
                    last_save = time.time()
                    print(f"\n已存 {p.name} [{saved}/15] — 换一个倾斜角度!")
                    if saved >= 15:
                        print("已够 15 张! Ctrl+C 退出, 接着跑 --calibrate")
                if args.nogui:
                    if c is not None:
                        pts = c.reshape(-1, 2)
                        cov = (pts[:, 0].max() - pts[:, 0].min()) \
                            / gray.shape[1] * 100
                        tip = ("凑近些" if cov < 30
                               else "稍远些" if cov > 90 else "距离合适")
                        print(f"\r[{saved}/15] 角点OK 板占画面宽{cov:.0f}% {tip}"
                              f"          ", end="", flush=True)
                    else:
                        print(f"\r[{saved}/15] 未检出: 对准屏幕棋盘/凑近/停稳"
                              f"          ", end="", flush=True)
            if args.nogui:
                time.sleep(0.05)
                continue
            if corners is not None:
                cv2.drawChessboardCorners(disp, pattern, corners * 2.0, True)
                hint = f"[{saved}/15] auto-save ON ({pattern[0]}x{pattern[1]})  Ctrl+C to quit"
            else:
                hint = f"[{saved}/15] no corners - closer / hold still"
            cv2.putText(disp, hint, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 255), 2)
            cv2.imshow("capture", disp)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                if corners is not None:
                    p = Path(args.dir) / f"calib_{idx:03d}.jpg"
                    cv2.imwrite(str(p), frame)
                    idx += 1
                    saved += 1
                    last_save = time.time()
                    print(f"已存 {p.name} [{saved}/15]")
                else:
                    print("角点未找全, 调整姿态/距离后重试")
            elif key in (ord("q"), 27):
                break
    except KeyboardInterrupt:
        print()
    finally:
        cam.close()
        if not args.nogui:
            cv2.destroyAllWindows()


def _solve(imgs, square, pattern):
    """一组图跑角点+标定, 返回 (有效图数, rms, K, dist); 有效<4 返回 None。"""
    obj, pts, size = [], [], None
    for p in imgs:
        gray = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2GRAY)
        c = detect(gray, *pattern)
        if c is None:
            continue
        w, h = pattern
        size = gray.shape[::-1]
        obj.append(np.array([[i * square, j * square, 0.0]
                             for j in range(h) for i in range(w)], np.float32))
        pts.append(c)
    if len(obj) < 4:
        return None
    rms, K, dist, _, _ = cv2.calibrateCamera(obj, pts, size, None, None)
    return len(obj), rms, K, dist


def run_calibrate(args):
    imgs = sorted(Path(args.dir).glob("*.jpg"))
    if len(imgs) < 8:
        sys.exit(f"仅 {len(imgs)} 张, 至少 8 张(建议 15-20)")
    first = cv2.cvtColor(cv2.imread(str(imgs[0])), cv2.COLOR_BGR2GRAY)
    if args.w and args.h:
        pattern = (args.w, args.h)
    else:
        r = auto_detect(first)
        if not r:
            sys.exit("第一张图未识别到棋盘, 用 --w/--h 指定规格")
        pattern = (r[0], r[1])
        print(f"自动识别内角点规格: {pattern[0]}x{pattern[1]}")
    full = _solve(imgs, args.square, pattern)
    if not full or full[0] < 8:
        sys.exit("有效图不足 8 张, 重拍")
    n, rms, K, dist = full
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    w, h = first.shape[::-1]
    print(f"有效 {n}/{len(imgs)} 张 | RMS={rms:.3f}px")
    print(f"fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}"
          f"  (画面中心 {w // 2},{h // 2})")
    print(f"dist={np.round(dist.ravel(), 5).tolist()}")
    if abs(cx - w / 2) > 0.05 * w or abs(cy - h / 2) > 0.05 * h:
        print("!! 主点偏离画面中心 >5%: 通常是倾斜姿态不够, 补 30-45° 倾斜图重标")
    rejects = []
    if rms > 1.5:
        rejects.append(f"RMS={rms:.2f}>1.5: 查棋盘平整/图片模糊/白格过曝")
    # 稳定性自检: 前后两半分别标定, fx 应接近(差异大=姿态多样性不足, 解不可信)
    a = _solve(imgs[:len(imgs) // 2], args.square, pattern)
    b = _solve(imgs[len(imgs) // 2:], args.square, pattern)
    if a and b:
        f1, f2 = a[2][0, 0], b[2][0, 0]
        diff = abs(f1 - f2) / max(abs(f1), abs(f2)) * 100
        print(f"稳定性自检: 前半 fx={f1:.0f} 后半 fx={f2:.0f} 差异 {diff:.0f}%"
              f"  (1280x720 正常相机 fx 约 400-2000)")
        if not (200 <= fx <= 4000):
            rejects.append(f"fx={fx:.0f} 超出正常相机范围, 解不可信")
        if diff > 40:
            rejects.append(f"两组 fx 差异 {diff:.0f}%>40%: 姿态多样性不足, 解不可信")
        elif diff > 15:
            print("!! 两组 fx 偏差偏大: 建议补大倾角图重标")
    if rejects and not args.force:
        sys.exit("拒绝保存: " + "; ".join(rejects)
                 + "\n重拍要领: 近(板占画面一半以上)、斜(至少 6-8 张 45° 大倾角"
                   "梯形图, 短边≈长边 2/3)、覆盖四角; 或加 --force 强制保存")
    for r in rejects:
        print(f"!! --force 强制保存, 注意: {r}")
    out = {"fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy),
           "dist": [float(v) for v in dist.ravel()], "rms": float(rms)}
    VISION_DIR.mkdir(parents=True, exist_ok=True)
    path = VISION_DIR / "calib_end.yaml"
    path.write_text(yaml.safe_dump(out, sort_keys=False), encoding="utf-8")
    print(f"已保存 {path}  [{'合格' if rms < 0.8 else 'RMS 偏大, 建议重拍'}]")


def run_verify(args):
    sys.path.insert(0, str(BASE))
    import calib
    c = calib.load()
    if c is None:
        sys.exit("无 tools/calib_end.yaml, 先 --calibrate")
    cam = Camera(args.cam, "calib")
    cam.open()
    print(f"RMS={c.rms:.2f}px | 左=原图 右=去畸变 | 蓝十字=画面中心 "
          f"红十字=标定主点 偏移=({c.cx - 640:+.1f}, {c.cy - 360:+.1f})px"
          f" | 判据: 右图里的直线(门框/桌边/瓷砖缝)应该真的直 | 窗口 q 退出")
    try:
        while True:
            frame = cam.read_latest()
            if frame is None:
                continue
            und = cv2.undistort(frame, c._K, c.dist)
            panels = []
            for img in (frame, und):
                p = cv2.resize(img, (640, 360))
                cv2.drawMarker(p, (320, 180), (255, 0, 0), cv2.MARKER_CROSS, 24, 1)
                cv2.drawMarker(p, (int(c.cx / 2), int(c.cy / 2)), (0, 0, 255),
                               cv2.MARKER_CROSS, 24, 2)
                panels.append(p)
            view = np.hstack(panels)
            cv2.putText(view, "raw | undistorted", (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("verify", view)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break
    except KeyboardInterrupt:
        print()
    finally:
        cam.close()
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--capture", action="store_true")
    g.add_argument("--calibrate", action="store_true")
    g.add_argument("--verify", action="store_true")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--dir", default="/tmp/calib")
    ap.add_argument("--w", type=int, default=8, help="内角点列数(默认 8, 匹配 9×6 方格板)")
    ap.add_argument("--h", type=int, default=5, help="内角点行数(默认 5)")
    ap.add_argument("--square", type=float, help="方格边长 mm(尺量 10 格总宽/10)")
    ap.add_argument("--nogui", action="store_true",
                    help="采图纯终端模式: 无窗口, 状态打在终端, 自动存图")
    ap.add_argument("--force", action="store_true",
                    help="验收不过也强制保存(打印警告)")
    args = ap.parse_args()
    if args.calibrate and not args.square:
        ap.error("--calibrate 需要 --square(实测方格毫米值)")
    if args.capture:
        run_capture(args)
    elif args.calibrate:
        run_calibrate(args)
    else:
        run_verify(args)


if __name__ == "__main__":
    main()
