"""main-copy 主状态机: 一键启动(长/短按判出发区) → 导航到码→扫码 → 导航到物料盘 →
纠偏 → 抓3(进给/退回) → [导航到粗加工→环放置] → 统计。

State 模式(gcxl 同款): State.execute(ctx) 推进 ctx.current_state。
行程 = NavDriver 节点式导航(goto, 2026-11-01 起): 每条边"转向对准 → check_edge 摄像头判
下一条边 → 命中封边重规划(重规划后的新边同样转向后再判) → 前进"; 工作点自动对齐臂朝
(WORK_HEADING), 物料盘进给/退回(WORK_APPROACH_MM); 旧 drive() 硬编码腿保留备用。

用法:
  实车:   python3 main.py --skip-start    # 电控 0x2201/0x2202 未实现, 默认直跑
  待回执: python3 main.py                 # 等一键启动: 短按(0x2201)=启1, 长按(0x2202)=启2
  无前相机: --no-check                    # 关闭节点检测(纯跑图)
  环放置: python3 main.py --skip-start --place-ring
          # 抓3后导航到粗加工区自动放环(未见环/标定未填自动跳过, 料留盘上)
"""
import argparse
import math
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

import config as C  # noqa: E402


# ---- 出发点(一键启动长短按, 2026-10-29 电控口径: 短按<2s=0x2201→启1, 长按≥2s=0x2202→启2) ----
# 摆车约定=车头朝二维码板: 启1 出发车头朝 +y(π/2), 启2 出发朝 -y(-π/2)。
# start_node/start_heading 本步只记账+打印, 路线消费等导航层(nav_drive)落地后接。
ZONE_NAME = {1: "启停区1", 3: "启停区2"}    # 节点号=map.yaml 编号(1~9: 启1=1, 启2=3)
START_HEADING = {1: math.pi / 2, 3: -math.pi / 2}


# ---- 行程函数(备用: 行程已改 NavDriver 节点式导航, 见 Step1/2) ----

def drive(lk, moves):
    """逐条 move_v(参数原值 mm/°) 等完成码, 条间 2s。"""
    from communication.protocol import Mode
    for i, (x, y, z, label) in enumerate(moves, 1):
        print(f"-> {label}")
        lk.send_and_wait(lambda x=x, y=y, z=z: lk.move_v(x, y, z),
                         Mode.ACK_CHASSIS, C.CMD_TIMEOUT, label)
        print(f"   ✓ ({i}/{len(moves)})")
        if i < len(moves):
            time.sleep(2)


# ---- 状态机 ----

class State:
    def execute(self, ctx):
        raise NotImplementedError


class Step0_WaitStart(State):
    def execute(self, ctx):
        if ctx.skip_start:
            if ctx.start_node is None:
                ctx.start_node = 1
            ctx.start_heading = START_HEADING[ctx.start_node]
            print(f"Step0 跳过一键启动(--skip-start), 出发点={ZONE_NAME[ctx.start_node]}"
                  f" (--start-node 可改)")
            ctx.current_state = Step1_ScanLeg()
            return
        print("Step0 等待一键启动: 短按(0x2201)=启停区1 / 长按(0x2202)=启停区2 ...")
        while True:
            for f in ctx.lk.poll():
                if f.mode == Mode.BUTTON_START:
                    ctx.start_node = 1
                    print("  收到一键启动: 短按 → 启停区1")
                elif f.mode == Mode.BUTTON_START_2:
                    ctx.start_node = 3
                    print("  收到一键启动: 长按 → 启停区2")
                else:
                    continue
                ctx.start_heading = START_HEADING[ctx.start_node]
                ctx.current_state = Step1_ScanLeg()
                return
            time.sleep(0.05)


class Step1_ScanLeg(State):
    """行程①(导航版): 发车对位 → goto 节点2(自动对齐 车头朝启2/臂朝暂存) → 扫码。

    沿途逐边: 转向对准 → check_edge 摄像头判下一条边 → 命中封边重规划(新边再判)。"""

    def execute(self, ctx):
        print("== Step1 行程①(导航→扫码) ==")
        from task.nav_drive import NavDriver
        ctx.nav = NavDriver(ctx.lk, ctx.watcher, "navigation/map.yaml",
                            ctx.start_node, ctx.start_heading)
        ctx.nav.launch()
        ctx.nav.goto(2)
        task = ctx.scanner.scan_from_camera(ctx.scan_cam, C.SCAN_TIMEOUT)
        if task is None:
            from vision.qr import parse_task
            task = parse_task(C.DEFAULT_TASK_CODE)
            print(f"  !! 扫码超时, 用兜底码 {C.DEFAULT_TASK_CODE}")
        ctx.task = task
        print(f"  任务码: {task['raw']} | 批1: {task['batch1_colors']}"
              f" 环{task['batch1_rings']}")
        ctx.current_state = Step2_ApproachLeg()


class Step2_ApproachLeg(State):
    """行程②(导航版): goto 节点6 物料盘(自动对齐臂朝转盘)。"""

    def execute(self, ctx):
        print("== Step2 行程②(导航→物料盘) ==")
        ctx.nav.goto(6)
        ctx.current_state = Step3_Align()


class Step3_Align(State):
    """纠偏(定一): 目标=离主点最近的物料, 等停歇(稳定帧内置)。"""

    def execute(self, ctx):
        print("== Step3 连续自动纠偏 ==")
        ctx.grab.align()
        ctx.current_state = Step4_Grab3()


class Step4_Grab3(State):
    """抓3: 工作点进给(x+100) → 每停歇窗口重判角色定点抓放(总回执 23) → 退回节点。"""

    def execute(self, ctx):
        print("== Step4 抓3(判色位→固定点位抓放, 计数23) ==")
        from task.nav_drive import WORK_APPROACH_MM
        mm = WORK_APPROACH_MM.get(6, 0)
        if mm:
            ctx.nav.approach(mm)     # 纠偏前进给(臂侧横移, 图外不入图)
        ctx.placed_colors = ctx.grab.run_grab3(ctx.task)
        if mm:
            ctx.nav.retreat(mm)      # 抓完退回节点位, 再转向去下一段
        ctx.current_state = Step5_PlaceRing() if ctx.place_ring else Step6_Done()


class Step5_PlaceRing(State):
    """环放置(导航到粗加工区, 定一式): goto 节点4(臂朝环区) → 对画面中间环纠偏 → 查位放3。

    未见环/标定未填/环号超3都跳过, 料留盘上。
    """

    def execute(self, ctx):
        print("== Step5 环放置(导航→粗加工区, 定一+查表, 计数25) ==")
        ctx.nav.goto(4)
        ctx.place.run_place3(ctx.task, ctx.placed_colors)
        ctx.current_state = Step6_Done()


class Step6_Done(State):
    def execute(self, ctx):
        s, p = ctx.grab.stats, ctx.place.stats
        extra = f" 放 {p['place_ok']}/{p['place_total']}" if ctx.place_ring else ""
        print(f"== 完成: 抓 {s['grab_ok']}/{s['grab_total']}{extra} ==")
        ctx.current_state = None


class Context:
    def __init__(self, lk, grab, place, scanner, scan_cam, watcher,
                 skip_start=False, place_ring=False, start_node=None):
        self.lk, self.grab, self.place = lk, grab, place
        self.scanner, self.scan_cam = scanner, scan_cam
        self.watcher = watcher          # ObstacleWatcher(前相机+障碍引擎); None=纯跑图
        self.nav = None                 # Step1 构建 NavDriver(须先 launch 再 goto)
        self.skip_start, self.place_ring = skip_start, place_ring
        self.start_node = start_node   # None=未定(等按键); 1=启1, 3=启2 (map.yaml 节点号)
        self.start_heading = None      # Step0 定出发点后按 START_HEADING 填
        self.task = None
        self.placed_colors = None
        self.current_state = Step0_WaitStart()

    def run(self):
        while self.current_state is not None:
            self.current_state.execute(self)


def main():
    ap = argparse.ArgumentParser(description="智能搬运主状态机")
    ap.add_argument("--port", default=C.SERIAL_PORT)
    ap.add_argument("--model", default=C.MODEL_PATH)
    ap.add_argument("--skip-start", action="store_true",
                    help="跳过一键启动等待(电控 0x2201/0x2202 未实现时用)")
    ap.add_argument("--start-node", type=int, choices=(1, 3), default=None,
                    help="出发点: 1=启停区1, 3=启停区2; --skip-start 时生效(默认1), "
                         "否则以一键启动长短按为准")
    ap.add_argument("--place-ring", action="store_true",
                    help="抓3后导航到粗加工区接环放置")
    ap.add_argument("--no-check", action="store_true",
                    help="关闭节点检测(无前相机/纯跑图用; 默认开启——节点上摄像头判下一条边)")
    ap.add_argument("--model-ring", default=C.MODEL_PATH_RING,
                    help="环检测模型(与物料模型可拆分)")
    args = ap.parse_args()

    from vision.camera import Camera
    from vision.detect import YOLOBackend
    from task.grab import StepGrab
    from task.ring import StepRing
    from communication.protocol import Link, Mode
    from vision.qr import QrScanner

    scan_cam = Camera(C.CAMERA_ID_SCAN, "scan")
    scan_cam.open()
    end_cam = Camera(C.CAMERA_ID_END, "end")
    end_cam.open()
    watcher = None
    if not args.no_check:
        front_cam = Camera(C.CAMERA_ID_FRONT, "front")
        front_cam.open()
        print(f"加载避障模型 {C.MODEL_PATH_OBSTACLE} ...")
        obstacle_backend = YOLOBackend(C.MODEL_PATH_OBSTACLE, names={0: "obstacle"})
        from vision.obstacle import ObstacleWatcher
        calib = None
        if C.USE_INTRINSICS_FRONT:
            from tools.calib import load as calib_load
            calib = calib_load(BASE / "tools/calib_front.yaml")
            if calib is None:
                print("  ⚠ 内参缺(tools/calib_front.yaml): 去畸变跳过, 先跑 "
                      "tools/calib_intr.py --calibrate --out calib_front.yaml")
        watcher = ObstacleWatcher(obstacle_backend, front_cam, calib)
    else:
        print("节点检测已关闭(--no-check): 纯跑图")
    print(f"加载模型 {args.model} ...")
    backend = YOLOBackend(args.model)
    backend_ring = YOLOBackend(args.model_ring) if args.place_ring else None
    lk = Link(args.port)

    grab = StepGrab(lk, end_cam, backend)
    place = StepRing(lk, end_cam, backend_ring)
    ctx = Context(lk, grab, place, QrScanner(), scan_cam, watcher,
                  skip_start=args.skip_start, place_ring=args.place_ring,
                  start_node=args.start_node)
    try:
        ctx.run()
    except RuntimeError as e:   # 硬失败: 通信静默(不重发策略)
        print(f"!! 中止(硬失败): {e}")


if __name__ == "__main__":
    main()
