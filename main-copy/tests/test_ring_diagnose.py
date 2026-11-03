"""环检测诊断(2026-11-03): 定位"未见 2 号环"——判环检不出的一锤定音工具。

背景: 放环开场 _seek_mid_ring 35s 内零检出(连"当前对准X号环"都没打印过)。
头号嫌疑: best.engine(原统一权重)的类别表与 RING_CLS={1:7,2:8,3:9} 不对齐
(7/8/9=circle1/2/3 是 wuliao 重训时的编排, 原权重从未实车验证过)。

本脚本纯视觉不发串口, 循环取帧打印:
  每秒一行: find_rings 检出(几号环/位置) + backend 原始检出类 id 分布(cls:conf)
  每 10s 汇总: 环检出率、各类 id 出现次数
用法: python3 tests/test_ring_diagnose.py --secs 30
判读:
  原始检出全在 0~6 类、从无 7/8/9 -> best.engine 类别表不对齐, 环检测换回
    wuliao 引擎(MODEL_PATH_RING 改指 wuliao-best.engine)即可修
  有 7/8/9 但 find_rings 空 -> conf 门槛/类映射问题(看 RING_CONF=0.4)
  原始检出全空 -> 视野里没环(车位置/朝向) 或 距离角度导致数字太小
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
    ap = argparse.ArgumentParser(description="环检测视觉诊断(纯视觉, 不发串口)")
    ap.add_argument("--secs", type=float, default=30.0, help="诊断时长(秒)")
    ap.add_argument("--model-ring", default=C.MODEL_PATH_RING,
                    help="环检测引擎(默认=best.engine 原统一权重)")
    args = ap.parse_args()

    from vision.camera import Camera
    from vision.detect import YOLOBackend

    cam = Camera(C.CAMERA_ID_END, "end")
    cam.open()
    print(f"加载环模型 {args.model_ring} ...")
    backend = YOLOBackend(args.model_ring)
    print(f"RING_CLS={C.RING_CLS} RING_CONF={C.RING_CONF} | "
          f"诊断 {args.secs:.0f}s, 末端相机对环区, Ctrl+C 停\n")

    t_end = time.time() + args.secs
    n_frame = n_ring = 0
    cls_cnt = Counter()
    ring_digits = Counter()
    t_line = t_sum = time.time()
    while time.time() < t_end:
        f = cam.read_latest()
        if f is None:
            time.sleep(0.03)
            continue
        n_frame += 1
        rings = find_rings(backend, f)
        n_ring += bool(rings)
        raw = backend.infer(f)   # 原始全类检出(不过滤类/阈值)
        for d in raw:
            cls_cnt[d.cls_id] += 1
        for r in rings:
            ring_digits[r.digit] += 1
        seg = [f"{r.digit}号({r.cx / f.shape[1]:.2f},{r.cy / f.shape[0]:.2f})"
               for r in rings]
        now = time.time()
        if now - t_line >= 1.0:
            raws = " ".join(f"cls{d.cls_id}:{d.conf:.2f}" for d in raw[:6])
            print(f"[{args.secs - (t_end - now):4.0f}s] 环检出{len(rings)}: "
                  f"{' '.join(seg) if seg else '(无)'} | 原始类: {raws or '(空)'}")
            t_line = now
        if now - t_sum >= 10.0:
            print(f"  -- 汇总: 帧{n_frame} 有环帧{n_ring}({100 * n_ring / max(1, n_frame):.0f}%) "
                  f"环号分布{dict(ring_digits)} 类id出现{dict(cls_cnt)} --\n")
            t_sum = now
    print(f"\n== 诊断结束: 帧{n_frame} 有环帧{n_ring}({100 * n_ring / max(1, n_frame):.0f}%) "
          f"环号分布{dict(ring_digits)} | 类id总分布{dict(sorted(cls_cnt.items()))}")
    if not cls_cnt:
        print("判读: 原始检出全空 -> 视野里没环(车位置/朝向)或数字太小")
    elif not ({7, 8, 9} & set(cls_cnt)):
        print("判读: 原始检出有类但无 7/8/9 -> best.engine 类别表与 RING_CLS 不对齐!"
              " 修法=MODEL_PATH_RING 改指 wuliao-best.engine(circle1/2/3 在其 7/8/9)")
    else:
        print("判读: 有 7/8/9 -> 类别对齐; 查 RING_CONF 门槛与检出率")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n中止")
