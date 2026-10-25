"""main-copy 主状态机: 一键启动 → 行程①(扫码) → 行程② → 纠偏 → 抓3 → 统计。

State 模式(gcxl 同款): State.execute(ctx) 推进 ctx.current_state。
行程 = drive(moves) 通用函数(动作序列=函数默认参数, 条间 2s);
避障后续挂 drive 层(封边重规划), 见记忆"行程打包避障"。

用法:
  实车:   python3 main.py --skip-start    # 电控 0x0020 未实现, 默认直跑
  待回执: python3 main.py                 # 等一键启动回执
"""
import argparse
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

import config as C  # noqa: E402


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
            print("Step0 跳过一键启动(--skip-start)")
            ctx.current_state = Step1_ScanLeg()
            return
        print("Step0 等待一键启动(0x0020)...")
        while True:
            for f in ctx.lk.poll():
                if f.mode == Mode.BUTTON_START:
                    print("  收到一键启动!")
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
            from qr import parse_task
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
        ctx.grab.run_grab3(ctx.task)
        ctx.current_state = Step5_Done()


class Step5_Done(State):
    def execute(self, ctx):
        s = ctx.grab.stats
        print(f"== 完成: 抓 {s['grab_ok']}/{s['grab_total']} ==")
        ctx.current_state = None


class Context:
    def __init__(self, lk, grab, scanner, scan_cam, skip_start=False):
        self.lk, self.grab = lk, grab
        self.scanner, self.scan_cam = scanner, scan_cam
        self.skip_start = skip_start
        self.task = None
        self.current_state = Step0_WaitStart()

    def run(self):
        while self.current_state is not None:
            self.current_state.execute(self)


def main():
    ap = argparse.ArgumentParser(description="智能搬运主状态机")
    ap.add_argument("--port", default=C.SERIAL_PORT)
    ap.add_argument("--model", default="/home/ubuntu/main-copy/best.engine")
    ap.add_argument("--skip-start", action="store_true",
                    help="跳过一键启动等待(电控 0x0020 未实现时用)")
    args = ap.parse_args()

    from camera import Camera
    from detect import YOLOBackend
    from grab import StepGrab
    from protocol import Link, Mode
    from qr import QrScanner

    scan_cam = Camera(C.CAMERA_ID_SCAN, "scan")
    scan_cam.open()
    end_cam = Camera(C.CAMERA_ID_END, "end")
    end_cam.open()
    print(f"加载模型 {args.model} ...")
    backend = YOLOBackend(args.model)
    lk = Link(args.port)

    grab = StepGrab(lk, end_cam, backend)
    ctx = Context(lk, grab, QrScanner(), scan_cam, skip_start=args.skip_start)
    try:
        ctx.run()
    except RuntimeError as e:   # 硬失败: 通信静默(不重发策略)
        print(f"!! 中止(硬失败): {e}")


if __name__ == "__main__":
    main()
