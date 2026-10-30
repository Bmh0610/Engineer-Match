"""main-copy 主状态机: 一键启动 → 行程①(扫码) → 行程② → 纠偏 → 抓3 → [环放置] → 统计。

State 模式(gcxl 同款): State.execute(ctx) 推进 ctx.current_state。
行程 = drive(moves) 通用函数(动作序列=函数默认参数, 条间 2s);
避障后续挂 drive 层(封边重规划), 见记忆"行程打包避障"。

用法:
  实车:   python3 main.py --skip-start    # 电控 0x2201/0x2202 未实现, 默认直跑
  待回执: python3 main.py                 # 等一键启动: 短按(0x2201)=启1, 长按(0x2202)=启2
  环放置: python3 main.py --skip-start --place-ring
          # 抓3后接环放置(⚠️车须已在粗加工区; 运动段行程接入前的联调用;
          # 环面标定未填 config 时自动跳过, 料留盘上)
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


# ---- 行程函数(动作序列=默认参数; 避障后挂这层) ----

def drive(lk, moves):
    """逐条 move_v(参数原值 mm/°) 等完成码, 条间 2s。"""
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
    """行程①: 前进100 → 扫码(左侧相机, 兜底码) → 顺转90°。"""

    def execute(self, ctx):
        print("== Step1 行程①(扫码) ==")
        drive(ctx.lk, ((0, 100, 0, "前进100mm"),))
        task = ctx.scanner.scan_from_camera(ctx.scan_cam, C.SCAN_TIMEOUT)
        if task is None:
            from vision.qr import parse_task
            task = parse_task(C.DEFAULT_TASK_CODE)
            print(f"  !! 扫码超时, 用兜底码 {C.DEFAULT_TASK_CODE}")
        ctx.task = task
        print(f"  任务码: {task['raw']} | 批1: {task['batch1_colors']}"
              f" 环{task['batch1_rings']}")
        drive(ctx.lk, ((0, 0, -90, "顺时针转90°"),))
        ctx.current_state = Step2_ApproachLeg()


class Step2_ApproachLeg(State):
    """行程②: 前进100 → 右移100, 到原料区纠偏位。"""

    def execute(self, ctx):
        print("== Step2 行程②(就位) ==")
        drive(ctx.lk, ((0, 100, 0, "前进100mm"),
                       (100, 0, 0, "右移100mm")))
        ctx.current_state = Step3_Align()


class Step3_Align(State):
    """纠偏(定一): 目标=离主点最近的物料, 等停歇(稳定帧内置)。"""

    def execute(self, ctx):
        print("== Step3 连续自动纠偏 ==")
        ctx.grab.align()
        ctx.current_state = Step4_Grab3()


class Step4_Grab3(State):
    """抓3: 每停歇窗口重判角色, 任务码颜色序定点抓放(总回执 23 判结束)。"""

    def execute(self, ctx):
        print("== Step4 抓3(先到判色位→固定点位抓放, 计数23) ==")
        ctx.placed_colors = ctx.grab.run_grab3(ctx.task)
        ctx.current_state = Step5_PlaceRing() if ctx.place_ring else Step6_Done()


class Step5_PlaceRing(State):
    """环放置(粗加工区, 定一式): 对画面中间环整车纠偏 → 数字查位 盘→环 放3。

    ⚠️ 前提=车已到粗加工区环前; 运动段行程接入前, --place-ring 只在
    车被人工摆到位(或实车联调)时用。未见环/标定未填/环号超3都跳过, 料留盘上。
    """

    def execute(self, ctx):
        print("== Step5 环放置(粗加工区, 定一+查表, 计数25) ==")
        ctx.place.run_place3(ctx.task, ctx.placed_colors)
        ctx.current_state = Step6_Done()


class Step6_Done(State):
    def execute(self, ctx):
        s, p = ctx.grab.stats, ctx.place.stats
        extra = f" 放 {p['place_ok']}/{p['place_total']}" if ctx.place_ring else ""
        print(f"== 完成: 抓 {s['grab_ok']}/{s['grab_total']}{extra} ==")
        ctx.current_state = None


class Context:
    def __init__(self, lk, grab, place, scanner, scan_cam,
                 skip_start=False, place_ring=False, start_node=None):
        self.lk, self.grab, self.place = lk, grab, place
        self.scanner, self.scan_cam = scanner, scan_cam
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
                    help="抓3后接环放置(前提: 车已到粗加工区)")
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
    print(f"加载模型 {args.model} ...")
    backend = YOLOBackend(args.model)
    backend_ring = YOLOBackend(args.model_ring) if args.place_ring else None
    lk = Link(args.port)

    grab = StepGrab(lk, end_cam, backend)
    place = StepRing(lk, end_cam, backend_ring)
    ctx = Context(lk, grab, place, QrScanner(), scan_cam,
                  skip_start=args.skip_start, place_ring=args.place_ring,
                  start_node=args.start_node)
    try:
        ctx.run()
    except RuntimeError as e:   # 硬失败: 通信静默(不重发策略)
        print(f"!! 中止(硬失败): {e}")


if __name__ == "__main__":
    main()
