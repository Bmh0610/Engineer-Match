"""main-copy 主状态机(两圈比赛流程, MODE5 成套动作版, 2026-11-02 定稿):
一键启动(长/短按判出发区) → goto 扫码 → goto 物料盘守株抓料(批1) → goto 粗加工区
放环(G2)+抓环(倒序恒等) → goto 暂存区放环(恒等) → goto 物料盘(返回) → 守株抓料(批2)
→ goto 粗加工区放环(G4)+抓环 → goto 暂存区码垛(视觉同色叠放) → goto 启停区 → 统计。

State 模式(gcxl 同款)。臂动作全走 task/mission.py(MODE5: 上位机只发组号+位号,
成套动作下位机执行, 等 0x0015); 底盘全走 NavDriver 节点式导航(goto/封边重规划)。
节点(map.yaml): 1/3=启停区 2=扫码位 6=物料盘 4=粗加工区 8=暂存区。
视觉引擎分工: 物料/码垛判色=wuliao(MODEL_PATH), 避障=obstacle(MODEL_PATH_OBSTACLE);
圆环检测=OpenCV(vision/ring_cv, 无模型——YOLO 环检测已退役)。

用法:
  实车:   python3 main.py --skip-start     # 电控 0x2201/0x2202 未实现时默认直跑
  待回执: python3 main.py                  # 短按(0x2201)=启1, 长按(0x2202)=启2
  无前相机: --no-check                     # 关闭节点检测(纯跑图)
  码垛推演: --no-vision                    # 码垛不开相机, 按任务码推演
  段调试:   --skip-to rough1|buffer1|rough2|stack|home   # 人工摆车+兜底码
"""
import argparse
import math
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

import config as C  # noqa: E402
import task.mission as M  # noqa: E402
from communication.protocol import Mode  # noqa: E402
from vision.qr import parse_task  # noqa: E402
from display import create_display  # noqa: E402
from task.nav_drive import WORK_APPROACH_MM  # noqa: E402



# ---- 出发点(一键启动长短按, 2026-10-29 电控口径: 短按<2s=0x2201→启1, 长按≥2s=0x2202→启2) ----
# 摆车约定=车头朝二维码板: 启1 出发车头朝 +y(π/2), 启2 出发朝 -y(-π/2)。
ZONE_NAME = {1: "启停区1", 3: "启停区2"}    # 节点号=map.yaml 编号(1~9)
START_HEADING = {1: math.pi / 2, 3: -math.pi / 2}

# 比赛流程节点(map.yaml)
NODE_SCAN, NODE_MATERIAL, NODE_ROUGH, NODE_BUFFER = 2, 6, 4, 8


# ---- 状态机 ----

class State:
    def execute(self, ctx):
        raise NotImplementedError


def _goto(ctx, node, label):
    """导航到节点; skip-to 调试时 nav=None(人工摆车)只打印跳过。"""
    if ctx.nav is None:
        print(f"== [skip-to] 人工摆车, 跳过导航 → {label} (节点{node}) ==")
        return
    print(f"== 导航 → {label} (节点{node}) ==")
    ctx.nav.goto(node)


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
    """行程①: goto 节点2 扫码。扫码超时可手动输任务码, 再不行用兜底码。"""

    def execute(self, ctx):
        print("== Step1 行程①(导航→扫码) ==")
        from task.nav_drive import NavDriver
        ctx.nav = NavDriver(ctx.lk, ctx.watcher, "navigation/map.yaml",
                            ctx.start_node, ctx.start_heading)
        ctx.nav.launch()
        ctx.nav.goto(NODE_SCAN)
        task = ctx.scanner.scan_from_camera(ctx.scan_cam, C.SCAN_TIMEOUT)
        if task is None:
            try:
                code = input(f"  !! 扫码超时; 手动输任务码"
                             f"(回车=兜底 {C.DEFAULT_TASK_CODE}): ").strip()
            except EOFError:
                code = ""
            task = parse_task(code) if code else None
            if task is None:
                task = parse_task(C.DEFAULT_TASK_CODE)
                print(f"  用兜底码 {C.DEFAULT_TASK_CODE}")
        ctx.task = task
        print(f"  任务码: {task['raw']} | 批1 {task['batch1_colors']} 环{task['batch1_rings']}"
              f" | 批2 {task['batch2_colors']} 环{task['batch2_rings']}")
        if ctx.display:
            c1 = ' → '.join(task["batch1_colors"])
            r1 = ','.join(str(r) for r in task["batch1_rings"])
            c2 = ' → '.join(task["batch2_colors"])
            r2 = ','.join(str(r) for r in task["batch2_rings"])
            ctx.display.set_task(task['raw'],
                                 f"批1: {c1} | 环号 {r1}",
                                 f"批2: {c2} | 环号 {r2}")
        ctx.current_state = Step2_Leg1Grab()


class Step2_Leg1Grab(State):
    """批1 抓料: goto 节点6 → 进给100(臂侧横移, 守株抓取位) → 守株(定一纠偏建锚+等色)
    →组2 → 退回100 → 前进30(臂不在中心补偿: 退回后车体在节点-30, 前推回节点位)
    → 之后按地图节点跑。"""

    def execute(self, ctx):
        _goto(ctx, NODE_MATERIAL, "物料盘")
        mm = WORK_APPROACH_MM.get(NODE_MATERIAL, 0)
        if mm:
            ctx.nav.approach(mm)
        if ctx.display:
            ctx.display.set_step("物料盘→守株抓料")
        ok, tot = M.grab_materials(ctx.lk, ctx.end_cam, ctx.backend,
                                   ctx.task["batch1_colors"], ctx.grab_wait, "批1")
        if mm:
            ctx.nav.retreat(mm)
        ctx.nav.approach(C.ARM_CENTER_COMP_MM, axis="y")   # 臂不在中心: 沿y+前推30, 修正后车体=图上节点
        ctx.stats.append(("批1抓料", ok, tot))
        ctx.current_state = Step3_Leg1Rough()


class Step3_Leg1Rough(State):
    """批1 粗加工区: goto 节点4 → 放环(G2 序)+抓环(倒序恒等)。"""

    def execute(self, ctx):
        _goto(ctx, NODE_ROUGH, "粗加工区")
        ok1, t1 = M.place_rings(ctx.lk, ctx.end_cam, ctx.backend_ring,
                                ctx.task["batch1_rings"], C.ACTION_TIMEOUT, mode="rough")
        ok2, t2 = M.grab_rings(ctx.lk, ctx.task["batch1_rings"], C.ACTION_TIMEOUT)
        ctx.stats += [("批1放环", ok1, t1), ("批1抓环", ok2, t2)]
        ctx.current_state = Step4_Leg1Buffer()


class Step4_Leg1Buffer(State):
    """批1 暂存区: goto 节点8 → 放环(恒等, 盘格i→位i, 零账本)。"""

    def execute(self, ctx):
        _goto(ctx, NODE_BUFFER, "暂存区")
        ok, tot = M.place_rings(ctx.lk, ctx.end_cam, ctx.backend_ring,
                                ctx.task["batch1_rings"], C.ACTION_TIMEOUT, mode="buffer")
        ctx.stats.append(("批1暂存区放环", ok, tot))
        ctx.current_state = Step5_Leg2Grab()


class Step5_Leg2Grab(State):
    """批2: goto 节点6 返回物料盘 → 进给100 → 守株抓料(重新定一纠偏) → 退回100
    → 前进30(臂不在中心补偿) → 按图跑。"""

    def execute(self, ctx):
        _goto(ctx, NODE_MATERIAL, "物料盘(返回)")
        mm = WORK_APPROACH_MM.get(NODE_MATERIAL, 0)
        if mm:
            ctx.nav.approach(mm)
        ok, tot = M.grab_materials(ctx.lk, ctx.end_cam, ctx.backend,
                                   ctx.task["batch2_colors"], ctx.grab_wait, "批2")
        if mm:
            ctx.nav.retreat(mm)
        ctx.nav.approach(C.ARM_CENTER_COMP_MM, axis="y")
        ctx.stats.append(("批2抓料", ok, tot))
        ctx.current_state = Step6_Leg2Rough()


class Step6_Leg2Rough(State):
    """批2 粗加工区: goto 节点4 → 放环(G4 序)+抓环(倒序恒等)。"""

    def execute(self, ctx):
        _goto(ctx, NODE_ROUGH, "粗加工区")
        ok1, t1 = M.place_rings(ctx.lk, ctx.end_cam, ctx.backend_ring,
                                ctx.task["batch2_rings"], C.ACTION_TIMEOUT, mode="rough")
        ok2, t2 = M.grab_rings(ctx.lk, ctx.task["batch2_rings"], C.ACTION_TIMEOUT)
        ctx.stats += [("批2放环", ok1, t1), ("批2抓环", ok2, t2)]
        ctx.current_state = Step7_Leg2Stack()


class Step7_Leg2Stack(State):
    """批2 暂存区码垛: goto 节点8 → 视觉识别环上物料色+盘上账本定格, 同色叠批1 上方。
    --no-vision 时按任务码推演(不开相机不纠偏)。"""

    def execute(self, ctx):
        _goto(ctx, NODE_BUFFER, "暂存区")
        ok, tot = M.stack_rings(ctx.lk, ctx.end_cam, ctx.backend, ctx.task,
                                C.ACTION_TIMEOUT, use_vision=not ctx.no_vision)
        ctx.stats.append(("批2码垛", ok, tot))
        ctx.current_state = Step8_ReturnHome()


class Step8_ReturnHome(State):
    """回启停区: goto 出发节点(启1=1 / 启2=3)。"""

    def execute(self, ctx):
        _goto(ctx, ctx.start_node, f"{ZONE_NAME[ctx.start_node]}(回程)")
        # 入库微调(2026-11-05 用户定): 启1 到节点1后不转, 场地系 y-200→x-200
        # 回摆车位(车头朝西: 场地y-=车右移/场地x-=车前进)。仅全程模式(nav 存在);
        # 路径 [8,7,6,1] 末边朝西固定, 若因封边改道则朝向不保证——跳过微调防错位。
        if ctx.start_node == 1 and ctx.nav is not None:
            ctx.nav._send_move(-200, 0, 0, "入库: 场地y-200(车右移)")
            ctx.nav._send_move(0, 200, 0, "入库: 场地x-200(车前进)")
        ctx.current_state = Step9_Done()


class Step9_Done(State):
    def execute(self, ctx):
        print("== 两圈完成, 统计 ==")
        for name, ok, tot in ctx.stats:
            print(f"  {name}: {ok}/{tot}")
        ctx.current_state = None


class Context:
    SKIP = {"rough1": Step3_Leg1Rough, "buffer1": Step4_Leg1Buffer,
            "rough2": Step6_Leg2Rough, "stack": Step7_Leg2Stack,
            "home": Step8_ReturnHome}

    def __init__(self, lk, scanner, scan_cam, end_cam, backend, backend_ring,
                 display, watcher, skip_start=False, start_node=None,
                 grab_wait=60.0, no_vision=False, skip_to=None):
        self.lk, self.scanner, self.scan_cam = lk, scanner, scan_cam
        self.end_cam, self.backend = end_cam, backend
        self.backend_ring = backend_ring   # 圆环检测(YOLO, 2026-11-05 换回); None=判环跳过
        self.display = display           # StatusDisplay 或 None(无 DISPLAY 环境)
        self.watcher = watcher          # ObstacleWatcher(前相机+障碍引擎); None=纯跑图
        self.nav = None                 # Step1 构建 NavDriver(须先 launch 再 goto)
        self.skip_start, self.start_node = skip_start, start_node
        self.start_heading = None
        self.grab_wait, self.no_vision = grab_wait, no_vision
        self.task = None
        self.stats = []                 # [(名称, ok, total)]
        if skip_to and skip_to != "start":
            self.start_node = start_node or 1      # 段调试默认按启1 摆车(回程 goto 用)
            self.start_heading = START_HEADING[self.start_node]
            self.task = parse_task(C.DEFAULT_TASK_CODE)   # 段调试: 人工摆车+兜底码
            print(f"[skip-to {skip_to}] 出发点={ZONE_NAME[self.start_node]} "
                  f"兜底码 {C.DEFAULT_TASK_CODE}"
                  f" (批1 {self.task['batch1_colors']} 环{self.task['batch1_rings']})")
            self.current_state = self.SKIP[skip_to]()
        else:
            self.current_state = Step0_WaitStart()

    def run(self):
        while self.current_state is not None:
            self.current_state.execute(self)


def main():
    ap = argparse.ArgumentParser(description="智能搬运主状态机(两圈 MODE5 版)")
    ap.add_argument("--port", default=C.SERIAL_PORT)
    ap.add_argument("--model", default=C.MODEL_PATH)
    ap.add_argument("--model-ring", default=C.MODEL_PATH_RING,
                    help="圆环检测引擎(YOLO, 2026-11-05 换回; 过渡期=best.engine)")
    ap.add_argument("--skip-start", action="store_true",
                    help="跳过一键启动等待(电控 0x2201/0x2202 未实现时用)")
    ap.add_argument("--start-node", type=int, choices=(1, 3), default=None,
                    help="出发点: 1=启停区1, 3=启停区2; --skip-start 时生效(默认1), "
                         "否则以一键启动长短按为准")
    ap.add_argument("--no-check", action="store_true",
                    help="关闭节点检测(无前相机/纯跑图用; 默认开启——节点上摄像头判下一条边)")
    ap.add_argument("--no-vision", action="store_true",
                    help="码垛不开相机, 按任务码推演(未接末端相机/联调)")
    ap.add_argument("--wait", type=float, default=60.0,
                    help="守株抓料单色等待超时(秒, 默认60; 比赛值按转盘一圈周期调)")
    ap.add_argument("--skip-to", choices=("start", "rough1", "buffer1",
                                          "rough2", "stack", "home"),
                    default="start",
                    help="start=全程(默认); 其余=人工摆车从该段动作组起跑(兜底码)")
    args = ap.parse_args()

    from vision.camera import Camera
    from vision.detect import YOLOBackend
    from communication.protocol import Link
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
    print(f"加载物料模型 {args.model} / 圆环模型 {args.model_ring} ...")
    backend = YOLOBackend(args.model)              # 物料/码垛判色(wuliao 引擎)
    try:
        backend_ring = YOLOBackend(args.model_ring)   # 圆环(YOLO 换回; 缺文件→None)
    except Exception as e:
        print(f"  ⚠ 圆环引擎加载失败({e}): 判环将跳过(保完赛)")
        backend_ring = None
    lk = Link(args.port)
    display = create_display()   # HDMI 物理屏全屏状态窗; 无 DISPLAY 时 None(静默跳过)

    ctx = Context(lk, QrScanner(), scan_cam, end_cam, backend, backend_ring,
                  display,
                  watcher, skip_start=args.skip_start, start_node=args.start_node,
                  grab_wait=args.wait, no_vision=args.no_vision,
                  skip_to=args.skip_to)
    try:
        ctx.run()
    except RuntimeError as e:   # 硬失败: 通信静默(不重发策略)
        print(f"!! 中止(硬失败): {e}")


if __name__ == "__main__":
    main()
