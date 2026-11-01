"""障碍物检测专项测试: 任意相机加载统一 10 类模型, 验证 obstacle_black 检出链路。

背景(2026-10-31): 前向避障相机未装, 先借扫码相机(/dev/cam_scan)验证
"模型加载→推理→门控→check_edge"链路——把障碍物放进相机视野看检出。
⚠️ 车道过滤已升级为 LANE_ROI 矩形(2026-11-01 默认=水平中间2/3×垂直下面4/5, 滤掉
广角画面里的相邻边/前方交叉边): 默认生效; --lane-roi 覆盖 / --no-lane 关闭。
扫码相机借测时 ROI 按画面大致居中即可(验证"认得出"), 装前相机后按实视野调。

模式:
  默认      实时打印: 每帧类0检出(置信度/框/估算距离)+门控结果, 0.5s 一条
  --check   节点检测语义: ObstacleWatcher.check_edge() 5帧≥3帧判定 True/False
  --gui     开窗口画框(NoMachine/DISPLAY 下用, q 键退窗; 默认纯终端)
Ctrl+C 退出(终端模式); --check 模式输入 q 回车退出。

用法: python3 tests/test_obstacle_live.py [--cam /dev/cam_scan] [--model best.engine]
                [--check] [--gui] [--conf 0.5] [--lane-band 0.4,1.0]
"""
import argparse
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import config as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="障碍物检测专项(任意相机)")
    ap.add_argument("--cam", default="/dev/cam_scan",
                    help="相机设备(默认扫码相机; 装前相机后换其别名)")
    ap.add_argument("--model", default=C.MODEL_PATH)
    ap.add_argument("--check", action="store_true",
                    help="check_edge 模式: 5帧≥3帧判定(其余时间待机)")
    ap.add_argument("--gui", action="store_true", help="开窗口画框(默认纯终端)")
    ap.add_argument("--conf", type=float, default=0.5, help="帧内门限(默认0.5)")
    ap.add_argument("--lane-roi", default=None,
                    help="覆盖车道ROI x0,x1,y0,y1 (默认用 obstacle.LANE_ROI 模块值)")
    ap.add_argument("--no-lane", action="store_true", help="关闭车道过滤(全画面)")
    args = ap.parse_args()

    import cv2

    import vision.obstacle as obs
    from vision.camera import Camera
    from vision.detect import CLASSES, YOLOBackend
    from vision.obstacle import ObstacleWatcher, est_dist

    if args.no_lane:
        obs.LANE_ROI, obs.LANE_BAND = None, None
    elif args.lane_roi:
        obs.LANE_ROI = tuple(float(v) for v in args.lane_roi.split(","))
    obs.WATCH_CONF = args.conf
    print(f"车道ROI: {obs.LANE_ROI} | 门限 conf≥{args.conf} | 模型 {args.model}")

    print(f"打开相机 {args.cam} ...")
    cam = Camera(args.cam, "obstacle-test")
    cam.open()
    print(f"加载模型(首帧推理含预热, 稍等)...")
    backend = YOLOBackend(args.model)
    watcher = ObstacleWatcher(backend, cam)

    def draw(frame, dets):
        h, w = frame.shape[:2]
        if obs.LANE_ROI:    # 车道ROI(若开): 绿色半透明矩形
            x0, x1, y0, y1 = obs.LANE_ROI
            overlay = frame.copy()
            cv2.rectangle(overlay, (int(x0 * w), int(y0 * h)),
                          (int(x1 * w), int(y1 * h)), (0, 255, 0), -1)
            cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)
            cv2.rectangle(frame, (int(x0 * w), int(y0 * h)),
                          (int(x1 * w), int(y1 * h)), (0, 255, 0), 1)
        gate = obs._gate(dets, frame.shape)
        for d in dets:      # 过门控=红粗框; 被滤=灰细框
            on = d is gate
            x1, y1, x2, y2 = (int(v) for v in d.xyxy)
            cv2.rectangle(frame, (x1, y1), (x2, y2),
                          (0, 0, 255) if on else (160, 160, 160), 3 if on else 1)
            cv2.putText(frame, f"{d.cls} {d.conf:.2f} "
                        f"~{est_dist(d.xyxy[3] - d.xyxy[1]):.0f}mm",
                        (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 0, 255) if on else (160, 160, 160), 2)
        msg, color = (("OBSTACLE!", (0, 0, 255)) if gate else
                      ("detected (gated out)", (0, 200, 255)) if dets else
                      ("clear", (0, 255, 0)))
        cv2.putText(frame, msg, (8, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.putText(frame, f"conf>={args.conf}  LANE_ROI={obs.LANE_ROI}",
                    (8, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow("obstacle-test", frame)

    try:
        if args.check:
            input("把障碍物放进视野后回车: 执行 check_edge(5帧≥3帧)...")
            while True:
                t0 = time.time()
                blocked = watcher.check_edge()
                print(f"check_edge → {'⚠ 有障碍(封边语义)' if blocked else '✓ 无障碍'}"
                      f" ({time.time() - t0:.1f}s)  [回车重测/q退出]")
                if args.gui:
                    f = cam.read_latest()
                    if f is not None:
                        draw(f, backend.infer(f, classes=obs.OBSTACLE_CLS))
                        cv2.waitKey(1)
                if input().strip().lower() == "q":
                    break
        else:
            print("实时检测中(0.5s 一条; 类0=obstacle_black; q 退出)")
            last = 0.0
            while True:
                f = cam.read_latest()
                if f is not None:
                    dets = backend.infer(f, classes=obs.OBSTACLE_CLS)
                    if args.gui:
                        draw(f, dets)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                    if time.time() - last >= 0.5:
                        last = time.time()
                        gate = obs._gate(dets, f.shape)
                        if dets:
                            for d in dets:
                                h = d.xyxy[3] - d.xyxy[1]
                                print(f"  检出 {d.cls} conf={d.conf:.2f} "
                                      f"框y2/H={d.xyxy[3] / f.shape[0]:.2f} "
                                      f"估算距离≈{est_dist(h):.0f}mm"
                                      f"{' → 过门控' if d is gate else ' (被门控滤)'}")
                        else:
                            print("  无检出")
                time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n退出")
    finally:
        if args.gui:
            cv2.destroyAllWindows()
        cam.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n退出")
