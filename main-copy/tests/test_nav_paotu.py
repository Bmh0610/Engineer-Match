"""节点式跑图测试(路径规划版): 启1 → 码 → 物料盘 → 粗加工 → 暂存 → 物料盘 → 粗加工 → 暂存 → 启1。

与 tests/test_paotu.py(硬编码腿)的区别: 每段走 NavDriver.goto——A* 按代价模型选路
(边长+转向+路口费+边费), 工作点(物料盘/粗加工/暂存)到达后自动对齐停车朝向
(WORK_HEADING, 臂朝区域); 每段前回车门控(--auto 连跑)。本测试只跑图不动作:
到达工作点打印"此处应执行动作组"提示后继续下一段。

节点检测(--check, 需前相机): 转向对准后 check_edge 扫下一条边, 命中=封边重规划
(封边超 config.MAX_OBSTACLES 硬失败); 默认纯跑图不检测。

手动障碍测避障(不用相机, 两种):
  ① 预封边: --block 1,6 --block 2,5 (可多次)——发车前把边封死, 验证 A* 改道
    (如 --block 1,6 时 启1→物料盘 改走 1→2→5→6);
  ② 门控封边: 逐段回车处输入 'a,b' 回车=模拟"节点检测命中"先把边封掉再执行该段
    (goto 自动绕行, 车不驶入封锁边); 直接回车=正常执行; q=退出。
    连续封边超 config.MAX_OBSTACLES=2 会硬失败(看门狗)。

路线(节点号=map.yaml): 1 → 2 → 6 → 4 → 8 → 6 → 4 → 8 → 1
  ⚠ 码(2)↔物料盘(6) 无直连边, 由 A* 自选 2→1→6(外圈) 或 2→5→6(内十字, 含窄道边费)。

用法: python3 tests/test_nav_paotu.py [--port /dev/ttyUSB0] [--start 1|3]
               [--check --cam 2 --model best.engine] [--auto]
               [--block 1,6 --block 2,5]
前置: 整车上电 → 启停区摆车(车头朝码) → 回车发车对位 → 逐段回车执行。
"""
import argparse
import math
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import config as C  # noqa: E402

ROUTE = (2, 6, 4, 8, 6, 4, 8, 1)
ZONE = {1: "启停区1", 2: "二维码板", 3: "启停区2",
        4: "粗加工区", 5: "中心", 6: "物料盘", 7: "角(启1行)",
        8: "暂存区", 9: "角(启2行)"}
WORK = (6, 4, 8)   # 工作点: goto 自动对齐臂朝(WORK_HEADING), 到达后此处应执行动作组


def main():
    ap = argparse.ArgumentParser(description="节点式跑图(路径规划版)")
    ap.add_argument("--port", default=C.SERIAL_PORT)
    ap.add_argument("--start", type=int, choices=(1, 3), default=1,
                    help="出发点节点: 1=启停区1(默认), 3=启停区2")
    ap.add_argument("--check", action="store_true",
                    help="启用节点检测(需前相机; 默认纯跑图)")
    ap.add_argument("--cam", default=2,
                    help="前部相机设备号(待实测; 仅 --check 时打开)")
    ap.add_argument("--model", default=C.MODEL_PATH)
    ap.add_argument("--auto", action="store_true", help="跳过逐段回车门控(联调后期用)")
    ap.add_argument("--block", action="append", default=[],
                    help="预封边(模拟障碍), 可多次: --block 1,6 --block 2,5")
    args = ap.parse_args()

    from communication.protocol import Link
    from task.nav_drive import WORK_APPROACH_MM, NavDriver

    watcher = None
    if args.check:
        from vision.camera import Camera
        from vision.detect import YOLOBackend
        from vision.obstacle import ObstacleWatcher
        print(f"加载模型 {args.model} ...")
        backend = YOLOBackend(args.model)
        front = Camera(args.cam, "front")
        front.open()
        watcher = ObstacleWatcher(backend, front)

    lk = Link(args.port)
    start_heading = math.pi / 2 if args.start == 1 else -math.pi / 2   # 摆车约定: 车头朝码
    driver = NavDriver(lk, watcher, "navigation/map.yaml", args.start, start_heading)

    for spec in args.block:   # 预封边: 发车前模拟已知障碍, goto 的 A* 自动绕行
        try:
            a, b = (int(v) for v in spec.replace("，", ",").split(","))
        except ValueError:
            sys.exit(f"!! --block 格式: a,b (如 --block 1,6), 收到 {spec!r}")
        if a not in ZONE or b not in ZONE:
            sys.exit(f"!! --block {spec}: 节点号须在 1~9")
        driver.nav.map.block_edge(a, b)
        print(f"  ⚠ 预封边 {a}↔{b} (模拟障碍)")

    def gate(msg):
        """回车门控; 输入 'a,b'=先封边(模拟节点检测命中)再继续, q=退出, 直接回车=执行。"""
        if args.auto:
            return
        while True:
            s = input(msg + " [回车=执行 | a,b=封边 | q=退出] ").strip().replace("，", ",")
            if not s:
                return
            if s.lower() in ("q", "quit", "exit"):
                raise KeyboardInterrupt
            try:
                a, b = (int(v) for v in s.split(","))
            except ValueError:
                print("  !! 格式 a,b (如 1,6); 直接回车=执行")
                continue
            if a not in ZONE or b not in ZONE:
                print("  !! 节点号须在 1~9")
                continue
            driver.nav.map.block_edge(a, b)
            print(f"  ⚠ 已封边 {a}↔{b} (模拟障碍), 该段将绕行")

    gate(f"确认: 已上电 + 节点{args.start}({ZONE[args.start]})摆车(车头朝码)。回车发车对位...")
    driver.launch()
    t0 = time.time()
    for i, target in enumerate(ROUTE, 1):
        gate(f"[{i}/{len(ROUTE)}] 回车: goto 节点{target}({ZONE[target]})...")
        driver.goto(target)
        print(f"  ✓ 到达 节点{target}({ZONE[target]}) [{time.time() - t0:.0f}s]")
        if target in WORK:
            mm = WORK_APPROACH_MM.get(target, 0)
            if mm:
                gate(f"    回车: 工作点进给{mm:.0f}mm(纠偏前)")
                driver.approach(mm)
            print("  · 此处应执行动作组(纠偏+抓放; 本测试只跑图不动作)")
            if mm:
                gate(f"    回车: 工作点退回{mm:.0f}mm(退回节点再转向)")
                driver.retreat(mm)
    gate("回车: 收车对位(恢复发车朝向+退回摆车位)...")
    driver.finish()
    print(f"== 跑图完成, 总耗时 {time.time() - t0:.0f}s ==")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n中止")
