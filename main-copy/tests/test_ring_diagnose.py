"""环检测诊断(vision/ring_cv 版): 环检出率/位置/耗时——现场验证 OpenCV 环检测。

前身(2026-11-03 YOLO 类别表诊断)已随 YOLO 环检测退役归档(git 历史); 本版用
OpenCV ring_cv(黑线同心圆环+中心数字模板匹配+排列恢复, 无模型)做同款检出率诊断。

纯视觉不发串口, 循环取帧打印:
  每帧: 检出(几号环/位置) + 单帧耗时
  每 10s 汇总: 帧数/有环帧率/环号分布/平均耗时
用法: python3 tests/test_ring_diagnose.py --secs 30
判读: 有环帧率高(≥90%)+环号稳定正确 = 链路健康; 零检出 = 视野没环或参数不适配
(先 test_ring_cv --gui 调参)。
"""
import sys
import time
from collections import Counter
from pathlib import Path

BASE = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, BASE)

import config as C  # noqa: E402
from task.ring import find_rings  # noqa: E402


def main():
    import argparse
    ap = argparse.ArgumentParser(description="环检测诊断(纯视觉, 不发串口)")
    ap.add_argument("--secs", type=float, default=30.0, help="诊断时长(秒)")
    args = ap.parse_args()

    from vision.camera import Camera

    cam = Camera(C.CAMERA_ID_END, "end")
    cam.open()
    print(f"诊断 {args.secs:.0f}s, 末端相机对环区(臂在观测位), Ctrl+C 偆\n")

    t_end = time.time() + args.secs
    n_frame = n_ring = 0
    ring_digits = Counter()
    times = []
    t_line = time.time()
    while time.time() < t_end:
        f = cam.read_latest()
        if f is None:
            time.sleep(0.03)
            continue
        n_frame += 1
        t0 = time.time()
        rings = find_rings(f)
        times.append(time.time() - t0)
        n_ring += bool(rings)
        for r in rings:
            ring_digits[r.digit] += 1
        now = time.time()
        if now - t_line >= 1.0:
            seg = [f"{r.digit}号({r.cx / f.shape[1]:.2f},{r.cy / f.shape[0]:.2f})"
                   for r in rings]
            print(f"[{args.secs - (t_end - now):4.0f}s] 检出{len(rings)}: "
                  f"{' '.join(seg) if seg else '(无)'} "
                  f"| {times[-1] * 1000:.0f}ms")
            t_line = now
    avg = (sum(times) / len(times) * 1000) if times else 0.0
    print(f"\n== 诊断结束: 帧{n_frame} 有环帧{n_ring}"
          f"({100 * n_ring / max(1, n_frame):.0f}%) "
          f"环号分布{dict(ring_digits)} | 平均耗时 {avg:.0f}ms/帧")
    print("判读: 有环帧率高且环号稳定 = 链路健康; 零检出 = 视野没环或参数不适配"
          "(先 test_ring_cv --gui 调参)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n中止")
