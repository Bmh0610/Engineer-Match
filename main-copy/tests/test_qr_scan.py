"""扫码专项测试: 开摄像头实时找二维码, 解出任务码后完整打印解析结果。

用法: python3 tests/test_qr_scan.py [--cam 0] [--timeout 30] [--no-gui]
  --cam   默认跟 config.CAMERA_ID_SCAN(专用扫码相机; 新插的设备号可能不同,
          ls /dev/video* 后 --cam N 逐个试)
  实时窗口: 每帧画二维码框+解码内容, 看得见识别过程; --no-gui 纯终端。
  q 退出; 解出合法任务码自动结束。

检出的三种结局:
  合法任务码 → 完整打印(原始码/两组颜色序/两组环号/码垛对应)后退出
  检出但格式非法 → 告警内容并继续等
  超时 → 明确报未扫到
"""
import sys
import time
from pathlib import Path

BASE = str(Path(__file__).resolve().parent.parent)   # main-copy 根(双端通用)
sys.path.insert(0, BASE)

import cv2   # noqa: E402

import config as C  # noqa: E402
from vision.qr import QrScanner, parse_task  # noqa: E402
from vision.qr import RULE_DIGIT2COLOR  # noqa: E402  (1红2黄3蓝4绿5黑6浅蓝)

TASK_PATTERN_TIP = "格式=第1组颜色序(1-6)x3 + 环号(1-5)x3 + 第2组颜色序 + 环号, 共 123+123+123+123 形态"


def report_task(task):
    """合法任务码 → 完整展开打印。"""
    print("\n" + "=" * 30 + " 扫码成功 " + "=" * 30)
    print(f"  原始码        : {task['raw']}")
    print(f"  第1组 颜色顺序: {task['raw'][0:3]}  → "
          + ", ".join(f"{j+1}.{c}" for j, c in enumerate(task['batch1_colors'])))
    print(f"  第2组 环号    : {task['raw'][4:7]}  → "
          + ", ".join(f"{j+1}.{c}放{r}号环" for j, (c, r) in enumerate(
              zip(task['batch1_colors'], task['batch1_rings']))))
    print(f"  第3组 颜色顺序: {task['raw'][8:11]} → {task['batch2_colors']}")
    print(f"  第4组 环号    : {task['raw'][12:15]} → {task['batch2_rings']}"
          f" (第2批放粗加工区)")
    print(f"  暂存区码垛    : "
          + ", ".join(f"{c}叠{k}号环" for c, k in task['stack_plan'].items()))
    print("=" * 72 + "\n")


def draw(frame, dets, ok_data):
    """二维码框+解码内容叠加。"""
    for data, (x, y, w, h) in dets:
        color = (0, 255, 0) if data == ok_data else (0, 255, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, data[:40], (x, max(y - 8, 16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame


def main():
    import argparse
    ap = argparse.ArgumentParser(description="扫码专项测试(实时框+完整结果回显)")
    ap.add_argument("--cam", default=str(C.CAMERA_ID_SCAN),
                    help="扫码相机: udev 别名(/dev/cam_scan)或编号 N——默认跟 config")
    ap.add_argument("--timeout", type=float, default=30.0,
                    help="扫码总时长(秒, 默认30)")
    ap.add_argument("--no-gui", action="store_true", help="纯终端, 不开预览窗口")
    args = ap.parse_args()

    from vision.camera import Camera

    dev = int(args.cam) if str(args.cam).isdigit() else args.cam
    cam = Camera(dev, "qr")
    cam.open()
    scanner = QrScanner()
    print(f"== 扫码测试: cam{args.cam} | 时长≤{args.timeout:.0f}s | {TASK_PATTERN_TIP}")
    print("   q 退出; 解出合法任务码自动结束\n")

    ok_task = None
    seen_invalid = set()
    deadline = time.time() + args.timeout
    try:
        while time.time() < deadline:
            frame = cam.read_latest(timeout=0.2)
            if frame is None:
                continue
            dets = scanner.detect(frame)
            for data, _box in dets:
                task = parse_task(data)
                if task is not None:
                    ok_task = task
                    ok_data = data
                    if not args.no_gui:
                        cv2.imshow("qr_scan", draw(frame, dets, ok_data))
                        cv2.waitKey(800)   # 停一拍让你看到绿框
                    report_task(task)
                    break
                if data not in seen_invalid:
                    seen_invalid.add(data)
                    print(f"  !! 检出二维码但格式非法: {data!r} —— 继续等合法码")
            else:
                if not args.no_gui and frame is not None:
                    cv2.imshow("qr_scan", draw(frame, dets, None))
            if ok_task is not None:
                break
            if not args.no_gui and cv2.waitKey(1) & 0xFF == ord("q"):
                print("  (q 退出)")
                break
    except KeyboardInterrupt:
        print("\n中止")
    finally:
        if not args.no_gui:
            cv2.destroyAllWindows()
        cam.close()

    if ok_task is None and time.time() >= deadline:
        print(f"\n== 超时未扫到合法任务码 (可能: 相机不对(--cam N 试)/距离角度/光照) ==")
    elif ok_task is None:
        print("\n== 手动退出, 未扫到 ==")
    return ok_task


if __name__ == "__main__":
    main()
