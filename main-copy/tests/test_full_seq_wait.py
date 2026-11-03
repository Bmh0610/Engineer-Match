"""整场流程联调·守株待兔抓取版(2026-10-27): 与 test_full_seq.py 唯一区别=抓料换笨办法。

抓料(对号入座, 2026-11-03 定): 定一纠偏(复位位, 建锚)之后车一动不动, 臂停复位位
盯夹爪正下方——**任意色**进抓取窗(GRAB_WINDOW)且连帧稳定(转盘停歇,
finder.wait_any_color)即抓, 盘格=该色在任务码色序中的位次(色 c 第 k 位→盘格 k,
如 213: 检出红→盘2、黄→盘1、蓝→盘3) → 发 MODE5 组2 → 等 0x0015(35s)。
不再等特定色顺序, 命中不挑色; 识别失败的料不抓等下轮; 单窗超时剩余色跳过(保完赛)。

其余(腿/扫码/放环 MODE5 组1 车上格z→地面y / 抓环 MODE5 组3 倒序恒等 环r→盘格r /
判环找中环/对中环纠偏建锚)与 test_full_seq.py 完全同款, 全文见那份。

流程: 右移150→前进1000→扫码→顺转90→前进850→顺转90→前进875→逆转90→【守株抓料】
→逆转90→前进920→前进850→逆转90→【放环】【抓环】(粗加工区)→后退820→顺转90
→后退800(暂存区)→【放环】【抓环】→后退850→顺转90→后退850→后退800→逆转90
→左移50→后退50→完成(回启停区)。

用法: python3 tests/test_full_seq_wait.py [--skip-to start|rough|buffer|home]
      [--wait 60]
全自动连跑(2026-11-03 用户定: 全文件无人工门控, 到位即纠偏/动作), 打印帧 hex
+回执, 条间 2s; Ctrl+C 退出。
"""
import math
import sys
import time
from pathlib import Path

BASE = str(Path(__file__).resolve().parent.parent)   # main-copy 根(双端通用)
sys.path.insert(0, BASE)

import config as C  # noqa: E402
from communication.protocol import Mode, frame  # noqa: E402
from vision import finder  # noqa: E402
from vision.qr import parse_task  # noqa: E402
from task.ring import find_rings  # noqa: E402

ACK_NAME = {Mode.ACK_CHASSIS: "0x0011底盘", Mode.ACK_ARM: "0x0012臂",
            Mode.ACK_GRIPPER: "0x0013夹爪", Mode.ACK_ACTION: "0x0015成套动作"}

_STEP = [0]


def run_step(lk, label, tx, ack, timeout=None, gap=1.5):
    """发帧等回执(全自动, 无门控)。
    timeout=None 用 C.CMD_TIMEOUT; MODE5 成套动作传 ACTION_TIMEOUT——只等一次,
    调用方不得再包一层等待(曾双等: run_step 消费回执后 run_action 又空等 35s,
    每条动作白等一个超时窗, 2026-11-02 实车暴露)。
    gap=完成后条间(默认1.5s=底盘腿间; 臂/MODE5 调用方传 0=零间隔衔接,
    2026-11-03 用户定: 到位立刻纠偏/组动作完成立刻发下一条)。"""
    _STEP[0] += 1
    print(f"[{_STEP[0]}] {label}")
    lk.flush()
    lk.ser.write(tx)
    wait_s = C.CMD_TIMEOUT if timeout is None else timeout
    t0 = time.time()
    got = None
    while time.time() - t0 < wait_s:
        for f in lk.poll():
            print(f"    RX {f}")
            if f.mode == ack and got is None:
                got = f
        if got:
            break
        time.sleep(0.03)
    if got:
        print(f"    ✓ {ACK_NAME[ack]}完成 ({time.time()-t0:.1f}s)")
    else:
        print(f"    !! {wait_s:.0f}s 无 {ACK_NAME[ack]} 完成回执")
    time.sleep(gap)


def arm(lk, label, pose, gap=0):
    run_step(lk, f"{label} {tuple(pose)}", frame(Mode.MOVE_A, *pose),
             Mode.ACK_ARM, gap=gap)


def leg_move(lk, dx, dy, dz, label):
    """行程腿 move_v(不入账——腿间移动, 非工作区微调)。"""
    run_step(lk, f"[腿] {label} (右{dx:+.0f} 前{dy:+.0f} 转{dz:+.0f})",
             frame(Mode.MOVE_V, dx, dy, dz), Mode.ACK_CHASSIS)


# ---- 纠偏(看-动-再看) ----

def _send_move_v(lk, dy_right, dx_fwd):
    lk.flush()
    lk.ser.write(frame(Mode.MOVE_V, round(dy_right), round(dx_fwd), 0))
    t0 = time.time()
    while time.time() - t0 < C.CMD_TIMEOUT:
        for f in lk.poll():
            if f.mode == Mode.ACK_CHASSIS:
                return True
        time.sleep(0.03)
    print("  !! 纠偏 move_v 无回执(继续)")
    return False


def _align(lk, cam, find_fn, matrix, label, timeout):
    """整车纠偏: 看-动-再看, 死区3mm ≤5轮, 超时带误差继续(保完赛)。
    无门控全自动(2026-11-03 用户定: 到地方直接纠偏)。"""
    deadline = time.time() + C.ALIGN_TIMEOUT
    for i in range(1, C.ALIGN_MAX_ITER + 1):
        obj, _fr = finder.find_stable(find_fn, cam,
                                      timeout=max(0.5, deadline - time.time()))
        if obj is None:
            print(f"  [纠偏] 第{i}轮未检出目标, 放弃(保完赛继续)")
            return
        a, b, c, d = matrix
        dy_right = a * obj.dx + b * obj.dy
        dx_fwd = c * obj.dx + d * obj.dy
        residual = math.hypot(dx_fwd, dy_right)
        if residual <= C.ALIGN_TOL_MM:
            print(f"  ✓ 纠偏结束 (残差 {residual:.1f}mm, {i} 轮)")
            return
        if time.time() > deadline:
            break
        print(f"  [纠偏] 第{i}轮: 前{dx_fwd:+.1f} 右{dy_right:+.1f} "
              f"(残差 {residual:.1f}mm) → move_v")
        _send_move_v(lk, dy_right, dx_fwd)
        time.sleep(C.ALIGN_SETTLE)
    print("  纠偏结束(带误差, 保完赛继续)")


def find_material_fn(backend):
    def find(f):
        mats = finder.find_materials(backend, f, conf=0.5, px_per_mm=C.PX_PER_MM)
        return finder.nearest_to_axis(mats, f.shape)
    return find


def find_ring_mid_fn(backend):
    """中环=离画面中心最近的数字环(建锚用; 建锚时环空数字可见)。"""
    def find(f):
        rings = find_rings(backend, f)
        return min(rings, key=lambda r: r.dx ** 2 + r.dy ** 2) if rings else None
    return find


def _seek_mid_ring(lk, cam, backend, timeout):
    """纠偏前判环找中环: 相机对着几号环, 1号(左)→y-80mm, 3号(右)→y+80mm,
    直到对准2号环(中环)再纠偏; 未见环直接交给纠偏兜底。放环开场环空数字可见。
    机械臂不在车中心, 到环区可能对着侧环——每次纠偏前判环纠正(用户 2026-10-27)。"""
    deadline = time.time() + timeout
    moved = 0
    while time.time() < deadline and moved < 3:
        f = cam.read_latest(timeout=0.2)
        if f is None:
            time.sleep(0.2)
            continue
        rings = find_rings(backend, f)
        if not rings:
            time.sleep(0.2)
            continue
        hit = min(rings, key=lambda r: r.dx ** 2 + r.dy ** 2)
        if hit.digit == 2:
            print("  ✓ 已对准 2 号环(中环), 直接纠偏")
            return
        dy = -C.RING_SEEK_STEP if hit.digit == 1 else C.RING_SEEK_STEP
        print(f"  · 当前对准 {hit.digit} 号环, 沿 y{dy:+.0f} 移动找 2 号环")
        leg_move(lk, 0, dy, 0, f"判环送位(y{dy:+.0f}, 找中环)")
        moved += 1
        time.sleep(0.5)   # 挪后沉降再探
    print("  !! 未见 2 号环(限次/限时), 直接纠偏(保完赛)")


# ---- MODE 5 成套动作(三段搬运由下位机执行, 上位机只调用+等 0x0015) ----

def run_action(lk, group, outer, car, timeout, gap=0):
    """发 MODE5 成套动作, 单次等 0x0015(timeout 秒, 等待在 run_step 内);
    超时告警跳过(不盲目重发——文档警告重发会让动作再做一遍)。
    gap=0: 0x0015 即动作完成(协议口径), 立即发下一条/开始下一轮等色。"""
    run_step(lk, f"成套动作 组{group} (车外位{outer}, 车上格{car})",
             frame(Mode.ACTION, int(group), int(outer), int(car)),
             Mode.ACK_ACTION, timeout=timeout, gap=gap)


def action_grab_materials(lk, cam, backend, task, wait):
    """抓料×3轮(守株待兔·对号入座): 定一(复位位, 建锚)后车一动不动——
    **任意色**进窗+停歇稳定即抓(不等特定色), 盘格=该色在任务码色序中的位次
    → MODE5 组2 (2, 2, k)。识别失败的料不抓等下轮, 单窗超时剩余色跳过(保完赛)。"""
    print("\n===== 动作组①: 抓料 (守株·对号入座: 任意色停歇即抓, 车全程不动) =====")
    arm(lk, "缩回复位(定一纠偏位)", C.RESET)
    _align(lk, cam, find_material_fn(backend), C.IMG2BODY,
           "定一纠偏(建锚)", wait)
    order = list(task["batch1_colors"])
    grabbed = set()   # 已处理的色(无论回执成败——超时不重发, 宁可盘格空置不死循环)
    skip = set()      # 色序外/识别存疑的色, 本次不再处理(防反复检出死循环)
    while len(grabbed) < len(order):
        print(f"\n<等任意色进窗+停歇即抓, 色序{order} 已抓{sorted(grabbed)} "
              f"跳过{sorted(skip)}, ≤{wait:.0f}s>")
        mat, _fr = finder.wait_any_color(backend, cam, px_per_mm=C.PX_PER_MM,
                                         timeout=max(0.5, wait),
                                         exclude=grabbed | skip)
        if mat is None:
            print(f"  !! {wait:.0f}s 未等到新色稳定进窗, 剩余色跳过(保完赛, 盘格空置)")
            break
        if mat.color not in order:
            print(f"  !! 检出 {mat.color} 不在任务码色序(识别存疑), 本次不再处理该色")
            skip.add(mat.color)
            continue
        k = order.index(mat.color) + 1
        a, b, c, d = C.IMG2BODY
        print(f"  ✓ 看到: {mat.color} (偏差 右{a*mat.dx+b*mat.dy:+.1f} "
              f"前{c*mat.dx+d*mat.dy:+.1f}mm) → 任务码第{k}位 → 组2 放盘格{k}")
        run_action(lk, 2, 2, k, C.ACTION_TIMEOUT)   # Y=2 取料工位, Z=色序位k
        grabbed.add(mat.color)   # 成败都记账: 超时重发会让动作再做一遍(文档警告)
    print("\n== 抓料完成 (车全程在锚点位, 账本为零) ==")


# ---- 动作组②: 放环(盘→环, MODE5 组1: 车→地面) ----

def action_place_rings(lk, cam, backend, task, timeout, mode="rough"):
    """放环×3轮: 判环找中环+对中环纠偏建锚(唯一底盘动作) → MODE5 组1(车→地面)。
    rough(粗加工区): (1, rings[i], 盘格i) 按任务码环号序; buffer(暂存区, 第一圈):
    (1, i, i) 恒等正序——抓环恒等后盘格i=环i的料, 恰满足 G2 两区同位(零账本)。"""
    print(f"\n===== 动作组②: 放环 (MODE5 组1: 车→地面, "
          f"{'任务码环号序' if mode == 'rough' else '恒等'}) =====")
    arm(lk, "回复位", C.RESET)
    _seek_mid_ring(lk, cam, backend, timeout)   # 判环: 非2号沿y±80找中环
    _align(lk, cam, find_ring_mid_fn(backend), C.IMG2BODY_RING,
           "对中环纠偏(建锚)", timeout)
    rings = task["batch1_rings"]
    if mode == "rough":
        seq = [(ring_no, i, task["batch1_colors"][i - 1])
               for i, ring_no in enumerate(rings, 1)]
    else:
        seq = [(i, i, task["batch1_colors"][i - 1]) for i in range(1, len(rings) + 1)]
    for y, z, color in seq:
        pos_name = {1: "左", 2: "中", 3: "右"}[y]
        print(f"\n----- 放 {color} → 环{y}({pos_name}位), 取盘格{z} -----")
        run_action(lk, 1, y, z, C.ACTION_TIMEOUT)   # 组1: 车上格z → 地面y号位
    print("\n== 放环完成 (车在中环锚点位) ==")


# ---- 动作组③: 抓环(环→盘, MODE5 组3: 地面→车) ----

def action_grab_rings(lk, task, timeout):
    """抓环×3轮(纯记忆不开视觉): 倒序恒等 MODE5 组3——环 r → 盘格 r,
    按环号从大到小取(环3→盘3、环2→盘2、环1→盘1)。"""
    print("\n===== 动作组③: 抓环 (MODE5 组3: 地面→车, 倒序恒等) =====")
    for r in sorted(set(task["batch1_rings"]), reverse=True):
        pos_name = {1: "左", 2: "中", 3: "右"}[r]
        print(f"\n----- 抓 {r} 号环({pos_name}位) → 车上格{r} -----")
        run_action(lk, 3, r, r, C.ACTION_TIMEOUT)   # 组3: 地面r号位→车上格r
    print("\n== 抓环完成 (车在中环锚点位, 纯记忆) ==")


# ---- 扫码 ----

def scan_step(cam):
    """扫码: 末端相机尝试 SCAN_TIMEOUT(扫码相机未接), 超时自动用兜底码; 无回车门控。"""
    from vision.qr import QrScanner
    print(f"<扫码: ≤{C.SCAN_TIMEOUT:.0f}s, 超时自动用兜底码>")
    task = QrScanner().scan_from_camera(cam, C.SCAN_TIMEOUT)
    if task is None:
        task = parse_task(C.DEFAULT_TASK_CODE)
        print(f"  !! 未扫到, 用兜底码 {C.DEFAULT_TASK_CODE}")
    print(f"  任务码: {task['raw']} | 色序 {task['batch1_colors']} "
          f"环号 {task['batch1_rings']}")
    return task


# ---- 主流程 ----

def main():
    import argparse
    ap = argparse.ArgumentParser(description="整场流程联调·守株待兔抓取版")
    ap.add_argument("--port", default=C.SERIAL_PORT)
    ap.add_argument("--model", default=C.MODEL_PATH)
    ap.add_argument("--model-ring", default=C.MODEL_PATH_RING,
                    help="环检测引擎(默认=原统一 best.engine——用户定: 圆环用原权重)")
    ap.add_argument("--skip-to", choices=("start", "rough", "buffer", "home"),
                    default="start",
                    help="start=全流程(默认); rough/buffer=从该区动作组起跑"
                         "(车人工摆位); home=只跑回程腿")
    ap.add_argument("--timeout", type=float, default=5.0,
                    help="纠偏单轮等稳定帧超时(秒)")
    ap.add_argument("--wait", type=float, default=60.0,
                    help="单色守株等待超时(秒, 默认60; 比赛值按转盘一圈周期调) "
                         "-- 门控可回车续等")
    args = ap.parse_args()

    from vision.camera import Camera
    from vision.detect import YOLOBackend
    from communication.protocol import Link

    cam = Camera(C.CAMERA_ID_END, "end")
    cam.open()
    print(f"加载物料模型 {args.model} / 环模型 {args.model_ring} ...")
    backend = YOLOBackend(args.model)           # 物料(wuliao 引擎)
    backend_ring = YOLOBackend(args.model_ring)  # 圆环(原统一 best.engine)
    lk = Link(args.port)
    print(f"口径: 观察{C.OBSERVE} 复位{C.RESET} 最大伸{C.ARM_REACH_MAX} "
          f"行距{C.TURN_STEP_MM} 放/抓环高{C.H_PLACE_RING}/{C.H_GRAB_RING} "
          f"| Ctrl+C 退出")

    task = None
    if args.skip_to == "start":
        leg_move(lk, 150, 0, 0, "右移150(起步对位)")
        leg_move(lk, 0, 1000, 0, "前进1000(到扫码位)")
        task = scan_step(cam)
        leg_move(lk, 0, 0, -90, "顺时针转90")
        leg_move(lk, 0, 850, 0, "前进850")
        leg_move(lk, 0, 0, -90, "顺时针转90")
        leg_move(lk, 0, 875, 0, "前进875")
        leg_move(lk, 0, 0, 90, "逆时针转90(机械臂朝物料盘)")
        action_grab_materials(lk, cam, backend, task, args.wait)
        leg_move(lk, 0, 0, 90, "逆时针转90")
        leg_move(lk, 0, 920, 0, "前进920")
        leg_move(lk, 0, 850, 0, "前进850")
        leg_move(lk, 0, 0, 90, "逆时针转90")
    if args.skip_to in ("start", "rough"):
        if task is None:
            task = parse_task(C.DEFAULT_TASK_CODE)
        action_place_rings(lk, cam, backend_ring, task, args.timeout)
        action_grab_rings(lk, task, args.timeout)
        leg_move(lk, 0, -820, 0, "后退820")
        leg_move(lk, 0, 0, -90, "顺时针转90")
        leg_move(lk, 0, -800, 0, "后退800(到暂存区)")
    if args.skip_to in ("start", "rough", "buffer"):
        if task is None:
            task = parse_task(C.DEFAULT_TASK_CODE)
        action_place_rings(lk, cam, backend_ring, task, args.timeout)
        action_grab_rings(lk, task, args.timeout)
    if args.skip_to in ("start", "rough", "buffer", "home"):
        leg_move(lk, 0, -850, 0, "后退850")
        leg_move(lk, 0, 0, -90, "顺时针转90")
        leg_move(lk, 0, -850, 0, "后退850")
        leg_move(lk, 0, -800, 0, "后退800")
        leg_move(lk, 0, 0, 90, "逆时针转90")
        leg_move(lk, -50, 0, 0, "左移50")
        leg_move(lk, 0, -50, 0, "后退50(回启停区)")
    print("\n== 全部完成 (账本为零=车在锚点位) ==")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n中止")
