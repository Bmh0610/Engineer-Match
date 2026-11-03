"""抓料链路诊断(2026-11-03): 定位等色三关哪一关挂了。

背景: 实车等色三连 60s 超时+纠偏"未检出目标"——卡死在"检出"关的可能性最大。
本脚本循环取帧打印统计, 不发任何串口帧, 纯视觉诊断:
  每秒一行: 检出颗数 | 各物料色名+位置(是否进 GRAB_WINDOW 抓取窗)
  每 10s 一行汇总: 检出率(有物料帧/总帧)、进窗率、连续稳定可达次数

用法(实车, 末端相机对着转盘, 转盘在转):
  python3 tests/test_grab_diagnose.py             # 默认 30s
  python3 tests/test_grab_diagnose.py --secs 60   # 转盘多转几圈观察停歇
判读:
  检出颗数恒 0      -> 视野里没有转盘/物料(车位置错) 或 运动模糊到检不出 或 模型问题
  有检出但"进窗"恒 0 -> GRAB_WINDOW 参数/车对转盘的朝向偏
  进窗有但稳定恒 0   -> 停歇太短撞不上 confirm(需要调 STABLE_TOL_PX/CONFIRM_FRAMES)
"""
import sys
import time
from collections import Counter
from pathlib import Path

BASE = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, BASE)

import config as C  # noqa: E402
from vision import finder  # noqa: E402


def main():
    import argparse
    ap = argparse.ArgumentParser(description="抓料链路视觉诊断(纯视觉, 不发串口)")
    ap.add_argument("--secs", type=float, default=30.0, help="诊断时长(秒)")
    ap.add_argument("--model", default=C.MODEL_PATH)
    args = ap.parse_args()

    from vision.camera import Camera
    from vision.detect import YOLOBackend

    cam = Camera(C.CAMERA_ID_END, "end")
    cam.open()
    print(f"加载模型 {args.model} ...")
    backend = YOLOBackend(args.model)
    w0, w1, h0, h1 = C.GRAB_WINDOW
    print(f"GRAB_WINDOW={C.GRAB_WINDOW} | STABLE_TOL_PX={C.STABLE_TOL_PX} "
          f"CONFIRM_FRAMES={C.CONFIRM_FRAMES} | 诊断 {args.secs:.0f}s, Ctrl+C 停\n")

    t_end = time.time() + args.secs
    n_frame = n_hit = n_inwin = n_stable_try = 0
    win_hits = Counter()
    t_line = t_sum = time.time()
    while time.time() < t_end:
        f = cam.read_latest()
        if f is None:
            time.sleep(0.03)
            continue
        h, w = f.shape[:2]
        mats = finder.find_materials(backend, f, conf=0.5, px_per_mm=C.PX_PER_MM)
        n_frame += 1
        if mats:
            n_hit += 1
        seg = []
        for m in mats:
            inwin = (w0 * w <= m.cx <= w1 * w and h0 * h <= m.cy <= h1 * h)
            if inwin:
                n_inwin += 1
                win_hits[m.color] += 1
            seg.append(f"{m.color}({m.cx / w:.2f},{m.cy / h:.2f})"
                       f"{'✓窗' if inwin else ''}")
        now = time.time()
        if now - t_line >= 1.0:   # 每秒一行
            print(f"[{args.secs - (t_end - now):4.0f}s] 检出{len(mats)}颗: "
                  f"{'  '.join(seg) if seg else '(无)'}")
            t_line = now
        if now - t_sum >= 10.0:   # 每 10s 汇总
            print(f"  -- 汇总: 帧{n_frame} 有检出{n_hit}({100 * n_hit / max(1, n_frame):.0f}%) "
                  f"进窗帧{n_inwin} 窗内色分布{dict(win_hits)} --\n")
            t_sum = now
    print(f"\n== 诊断结束: 帧{n_frame} 有检出{n_hit}({100 * n_hit / max(1, n_frame):.0f}%) "
          f"进窗帧{n_inwin}({100 * n_inwin / max(1, n_frame):.0f}%) 窗内色分布{dict(win_hits)}")
    print("判读: 检出0%=视野/模糊/模型问题; 检出有但进窗0%=窗参数或朝向; "
          "进窗有但等色仍超时=停歇稳定撞不上(调 STABLE_TOL_PX/CONFIRM_FRAMES)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n中止")
