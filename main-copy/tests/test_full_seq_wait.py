"""整场流程联调·守株待兔抓取版(2026-10-27): 与 test_full_seq.py 唯一区别=抓料换笨办法。

抓料(按任务码顺序逐色等待, 2026-11-05 定——抓取顺序=评分项): 定一纠偏
(复位位, 建锚)之后车一动不动, 臂停复位位盯夹爪正下方——**任务码第1组顺序**
的下一色进抓取窗(GRAB_WINDOW)且连帧稳定(转盘停歇, finder.wait_for_color)即抓
→ 发 MODE5 组2 → 等 0x0015(35s)。单色超时跳过(保完赛)。

其余(腿/扫码/放环 MODE5 组1 车上格z→地面y / 抓环 MODE5 组3 正序恒等 环r→盘格r /
判环找中环/对中环纠偏建锚)与 test_full_seq.py 完全同款, 全文见那份。

两圈比赛流程(2026-11-04 定, 对齐 main.py 产线语义; 行车=固定腿距非 NavDriver):
  第一圈: 右移150→前进1000→扫码→顺转90→前进850→顺转90→前进875→逆转90
          →【守株抓料批1】→逆转90→前进920→前进850→逆转90→【放环G2】【抓环】
          →后退820→顺转90→后退800(暂存区)→【放环·恒等】(物料留位, 不抓环!)
  回料:   后退850→顺转90→后退820→顺转90→前进120→逆转90(摆回批1抓料位, 820/120 待实车校准)
  第二圈: 【守株抓料批2】→逆转90→前进920→前进850→逆转90→【放环G4】【抓环】
          →后退820→顺转90→后退800(暂存区)→【码垛·组4 叠批1上】
  回程:   后退850→顺转90→后退850→后退800→逆转90→左移50→后退50→完成(回启停区)。

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

# ---- 网页状态显示(Flask, 浏览器访问 Nano_IP:5000) ----
from flask import Flask

_web_app = Flask(__name__)
_web = {"html": """<html><head><meta charset="utf-8"><meta http-equiv="refresh" content="5">
<style>body{font-family:monospace;background:#1a1a2e;color:#e0e0e0;padding:30px;font-size:1.1em}
h2{color:#64ffda}.box{background:#16213e;padding:12px 16px;border-radius:6px;margin:8px 0}
.label{color:#8892b0;font-size:0.9em}</style></head><body>
<h2>⏳ 等待扫码...</h2></body></html>"""}

_web_app = Flask(__name__)


@_web_app.route('/')
def _web_page():
    return _web["html"]


def _web_start(port=5000):
    from threading import Thread
    Thread(target=lambda: _web_app.run(host='0.0.0.0', port=port, debug=False),
           daemon=True, name="web-display").start()
    print(f"  网页状态面板 → http://0.0.0.0:{port} (局域网内浏览器打开 Nano_IP:{port})")


def _web_task(task):
    """扫码成功后更新网页: 任务码+两批色序/环号。"""
    c1 = ' → '.join(task["batch1_colors"])
    r1 = ','.join(str(r) for r in task["batch1_rings"])
    c2 = ' → '.join(task["batch2_colors"])
    r2 = ','.join(str(r) for r in task["batch2_rings"])
    _web["html"] = f"""
    <html><head><meta charset="utf-8"><meta http-equiv="refresh" content="5">
    <style>body{{font-family:monospace;background:#1a1a2e;color:#e0e0e0;padding:30px;font-size:1.1em}}
    h2{{color:#64ffda}} h3{{color:#bd93f9}} .box{{background:#16213e;padding:12px 16px;
    border-radius:6px;margin:8px 0;border-left:4px solid #64ffda}}
    .label{{color:#8892b0;font-size:0.9em}} .tag{{display:inline-block;background:#0f3460;
    padding:2px 10px;border-radius:4px;margin:2px}}</style></head><body>
    <h2>📋 任务信息</h2>
    <div class="box"><span class="label">任务码原文:</span><br>
    <span style="font-size:1.5em;color:#64ffda">{task['raw']}</span></div>
    <div class="box"><b>批1</b> (粗加工区放环):<br>
    <span class="tag">{c1}</span> 环号 <span class="tag">{r1}</span></div>
    <div class="box"><b>批2</b> (暂存区码垛):<br>
    <span class="tag">{c2}</span> 环号 <span class="tag">{r2}</span></div>
    </body></html>"""


_STEP = [0]


def run_step(lk, label, tx, ack, timeout=None, gap=1.0):
    """发帧等回执(全自动, 无门控)。
    timeout=None 用 C.CMD_TIMEOUT; MODE5 成套动作传 ACTION_TIMEOUT——只等一次,
    调用方不得再包一层等待(曾双等: run_step 消费回执后 run_action 又空等 35s,
    每条动作白等一个超时窗, 2026-11-02 实车暴露)。
    gap=完成后条间(默认1.0s=底盘腿间, 2026-11-05 用户定 1.5→1; 臂/MODE5 调用方传 0=零间隔衔接,
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
    """整车纠偏: 看-动-再看, 死区3mm ≤ALIGN_MAX_ITER轮(10), 超时带误差继续(保完赛)。
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
        if residual <= C.ALIGN_TOL_MM_RING:
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


def find_ring_mid_fn(backend=None):
    """中环=离画面中心最近的数字环(建锚用; 建锚时环空数字可见)。"""
    def find(f):
        rings = find_rings(f, backend=backend)
        return min(rings, key=lambda r: r.dx ** 2 + r.dy ** 2) if rings else None
    return find


def _seek_mid_ring(lk, cam, backend, timeout):
    """纠偏前判环找中环: 相机对着几号环, 1号(左)→y-80mm, 3号(右)→y+80mm,
    直到对准2号环(中环)再纠偏; 未见环直接交给纠偏兜底。放环开场环空数字可见。
    机械臂不在车中心, 到环区可能对着侧环——每次纠偏前判环纠正(用户 2026-10-27)。"""
    deadline = time.time() + timeout
    moved = 0
    n_none = 0
    t_line = time.time()
    while time.time() < deadline and moved < 3:
        f = cam.read_latest(timeout=0.2)
        if f is None:
            n_none += 1
            if time.time() - t_line >= 1.0:
                print(f"  [判环] 无帧(read_latest None×{n_none})——相机/链路问题?")
                t_line = time.time()
            time.sleep(0.2)
            continue
        rings = find_rings(f, backend=backend)
        now = time.time()
        if now - t_line >= 1.0:   # 每秒诊断行
            seg = [f"{r.digit}号({r.cx / f.shape[1]:.2f})" for r in rings]
            print(f"  [判环] 环检出{len(rings)}: {' '.join(seg) or '(无)'}")
            t_line = now
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


def action_grab_materials(lk, cam, backend, task, wait, colors=None):
    """抓料×3轮(守株待兔·按任务码顺序逐色等待, 抓取顺序=评分项):
    定一(复位位, 建锚)后车一动不动——逐色等**任务码第1组顺序**的目标色进窗+
    停歇稳定即抓 → MODE5 组2 (2, 2, i)。单色超时跳过(保完赛, 盘格空置)。"""
    print("\n===== 动作组①: 抓料 (守株: 按任务码顺序逐色等待, 车全程不动) =====")
    arm(lk, "缩回复位(定一纠偏位)", C.RESET)
    _align(lk, cam, find_material_fn(backend), C.IMG2BODY,
           "定一纠偏(建锚)", wait)
    for i, color in enumerate(colors or task["batch1_colors"], 1):
        print(f"\n<等 {color}(第{i}个抓): 转进窗+停歇即发 MODE5, ≤{wait:.0f}s>")
        mat, _fr = finder.wait_for_color(backend, cam, color,
                                         px_per_mm=C.PX_PER_MM,
                                         timeout=max(0.5, wait))
        if mat is None:
            print(f"  !! {wait:.0f}s 未等到 {color}, 跳过该色(保完赛, 盘格空置)")
            continue
        a, b, c, d = C.IMG2BODY
        print(f"  ✓ 看到: {mat.color} (偏差 右{a*mat.dx+b*mat.dy:+.1f} "
              f"前{c*mat.dx+d*mat.dy:+.1f}mm) → 组2 抓取上盘格{i}")
        run_action(lk, 2, 2, i, C.ACTION_TIMEOUT)   # Y=2 取料工位, Z=顺序格i
    print("\n== 抓料完成 (车全程在锚点位, 账本为零) ==")


# ---- 动作组②: 放环(盘→环, MODE5 组1: 车→地面) ----

def action_place_rings(lk, cam, backend, task, timeout, mode="rough", rings=None, colors=None):
    """放环×3轮: 判环找中环+对中环纠偏建锚(唯一底盘动作) → MODE5 组1(车→地面)。
    rough(粗加工区): (1, rings[i], 盘格i) 按任务码环号序; buffer(暂存区, 第一圈):
    (1, i, i) 恒等正序——抓环恒等后盘格i=环i的料, 恰满足 G2 两区同位(零账本)。"""
    print(f"\n===== 动作组②: 放环 (MODE5 组1: 车→地面, "
          f"{'任务码环号序' if mode == 'rough' else '恒等'}) =====")
    arm(lk, "回复位", C.RESET)
    arm(lk, "圆环纠偏位(前伸32mm)", C.RING_ALIGN)
    _seek_mid_ring(lk, cam, backend, timeout)   # 判环: 非2号沿y±80找中环
    _align(lk, cam, find_ring_mid_fn(backend), C.IMG2BODY_RING,
           "对中环纠偏(建锚)", timeout)
    rings = rings or task["batch1_rings"]
    colors = colors or task["batch1_colors"]
    if mode == "rough":
        seq = [(ring_no, i, colors[i - 1]) for i, ring_no in enumerate(rings, 1)]
    else:
        seq = [(i, i, colors[i - 1]) for i in range(1, len(rings) + 1)]
    for y, z, color in seq:
        pos_name = {1: "左", 2: "中", 3: "右"}[y]
        print(f"\n----- 放 {color} → 环{y}({pos_name}位), 取盘格{z} -----")
        run_action(lk, 1, y, z, C.ACTION_TIMEOUT)   # 组1: 车上格z → 地面y号位
    print("\n== 放环完成 (车在中环锚点位) ==")


# ---- 动作组③: 抓环(环→盘, MODE5 组3: 地面→车) ----

def action_grab_rings(lk, task, timeout, rings=None):
    """抓环×3轮(纯记忆不开视觉): 正序恒等 MODE5 组3——环 r → 盘格 r,
    按环号从小到大取(环1→盘1、环2→盘2、环3→盘3; 2026-11-03 用户定, 原倒序)。"""
    print("\n===== 动作组③: 抓环 (MODE5 组3: 地面→车, 正序恒等) =====")
    for r in sorted(set(rings or task["batch1_rings"])):
        pos_name = {1: "左", 2: "中", 3: "右"}[r]
        print(f"\n----- 抓 {r} 号环({pos_name}位) → 车上格{r} -----")
        run_action(lk, 3, r, r, C.ACTION_TIMEOUT)   # 组3: 地面r号位→车上格r
    print("\n== 抓环完成 (车在中环锚点位, 纯记忆) ==")


# ---- 动作组④: 码垛(批2 暂存区, 组4 叠批1 上; 复用 mission 库) ----

def action_stack_rings(lk, cam, backend, task, timeout):
    """码垛: 批2 同色叠批1 正上方(MODE5 组4, 视觉识别环上物料定 Y+账本定 Z)。
    组4 内部放置高度 1000 自理; 前提=地面该位已有批1 物料(批1 暂存放环保证)。"""
    import task.mission as M
    print("\n===== 动作组④: 码垛 (MODE5 组4: 车→地面第二层) =====")
    ok, tot = M.stack_rings(lk, cam, backend, task, timeout, use_vision=True)
    print(f"\n== 码垛完成 ({ok}/{tot}) ==")


# ---- 扫码 ----

def scan_step(cam, preview=False):
    """扫码: 专用扫码相机(/dev/cam_scan, 2026-11-03 修——此前误用末端相机扫码),
    超时自动用兜底码; 无回车门控。preview=True 实时开扫码画面(检测框,
    无 DISPLAY 自动降级)。"""
    from vision.qr import QrScanner
    print(f"<扫码: ≤{C.SCAN_TIMEOUT:.0f}s, 超时自动用兜底码>")
    task = QrScanner().scan_from_camera(cam, C.SCAN_TIMEOUT, preview=preview)
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
                    help="圆环检测引擎(YOLO 换回; 过渡期=best.engine circle7/8/9)")
    ap.add_argument("--skip-to", choices=("start", "rough", "buffer", "grab2", "home"),
                    default="start",
                    help="start=两圈全程(默认); rough/buffer/grab2=从该段起跑"
                         "(车人工摆位, 从该段跑到底); home=只跑回程腿")
    ap.add_argument("--timeout", type=float, default=5.0,
                    help="纠偏单轮等稳定帧超时(秒)")
    ap.add_argument("--wait", type=float, default=60.0,
                    help="单色守株等待超时(秒, 默认60; 比赛值按转盘一圈周期调) "
                         "-- 门控可回车续等")
    ap.add_argument("--preview", action="store_true",
                    help="扫码时开实时预览窗口(画面+检测框; 无 DISPLAY 自动降级)")
    args = ap.parse_args()

    from vision.camera import Camera
    from vision.detect import YOLOBackend
    from communication.protocol import Link

    scan_cam = Camera(C.CAMERA_ID_SCAN, "scan")   # 扫码专用相机(/dev/cam_scan)
    scan_cam.open()
    _web_start()
    try:
        import webbrowser
        webbrowser.open(f"http://localhost:{5000}")
    except Exception:
        print(f"  状态页面: http://localhost:5000 (浏览器手动打开)")
    cam = Camera(C.CAMERA_ID_END, "end")          # 末端相机: 守株等色/纠偏/判环
    cam.open()
    print(f"加载物料模型 {args.model} / 圆环模型 {args.model_ring} ...")
    backend = YOLOBackend(args.model)           # 物料(wuliao 引擎)
    try:
        backend_ring = YOLOBackend(args.model_ring)   # 圆环(YOLO 换回; 缺文件→None)
    except Exception as e:
        print(f"  ⚠ 圆环引擎加载失败({e}): 判环将跳过(保完赛)")
        backend_ring = None
    lk = Link(args.port)
    from display import create_display
    display = create_display()   # HDMI 物理屏全屏状态窗; 无 DISPLAY 时 None(静默跳过)
    print(f"口径: 观察{C.OBSERVE} 复位{C.RESET} 最大伸{C.ARM_REACH_MAX} "
          f"行距{C.TURN_STEP_MM} 放/抓环高{C.H_PLACE_RING}/{C.H_GRAB_RING} "
          f"| Ctrl+C 退出")

    task = None
    if args.skip_to == "start":
        leg_move(lk, 150, 0, 0, "右移150(起步对位)")
        leg_move(lk, 0, 1000, 0, "前进1000(到扫码位)")
        task = scan_step(scan_cam, preview=args.preview)
        _web_task(task)
        if display:
            c1 = ' → '.join(task["batch1_colors"])
            r1 = ','.join(str(r) for r in task["batch1_rings"])
            c2 = ' → '.join(task["batch2_colors"])
            r2 = ','.join(str(r) for r in task["batch2_rings"])
            display.set_task(task['raw'],
                             f"批1: {c1} | 环号 {r1}",
                             f"批2: {c2} | 环号 {r2}")
        leg_move(lk, 0, 0, -90, "顺时针转90")
        leg_move(lk, 0, 850, 0, "前进850")
        leg_move(lk, 0, 0, -90, "顺时针转90")
        leg_move(lk, 0, 875, 0, "前进875")
        leg_move(lk, 0, 0, 90, "逆时针转90(机械臂朝物料盘)")
        if display:
            display.set_step("守株抓料(批1)")
        action_grab_materials(lk, cam, backend, task, args.wait)   # G2
        leg_move(lk, 0, 0, 90, "逆时针转90")
        leg_move(lk, 0, 920, 0, "前进920")
        leg_move(lk, 0, 850, 0, "前进850")
        leg_move(lk, 0, 0, 90, "逆时针转90")
    if args.skip_to in ("start", "rough"):
        if task is None:
            task = parse_task(C.DEFAULT_TASK_CODE)
        if display:
            display.set_step("粗加工区→放环(批1)")
        action_place_rings(lk, cam, backend, task, args.timeout)                    # G2
        action_grab_rings(lk, task, args.timeout)
        leg_move(lk, 0, -820, 0, "后退820")
        leg_move(lk, 0, 0, -90, "顺时针转90")
        leg_move(lk, 0, -800, 0, "后退800(到暂存区)")
    if args.skip_to in ("start", "rough", "buffer"):
        if task is None:
            task = parse_task(C.DEFAULT_TASK_CODE)
        if display:
            display.set_step("暂存区→放环(批1, 物料留位等码垛)")
        action_place_rings(lk, cam, backend, task, args.timeout, mode="buffer")    # 恒等, 只放环
        # 批1 暂存区不抓环——物料留在暂存位, 批2 码垛叠其上(比赛规则)

    # ---- 回物料盘(批2): 摆回批1抓料结束位(位置+朝向), 腿距待实车校准 ----
    if args.skip_to in ("start", "rough", "buffer", "grab2"):
        if task is None:
            task = parse_task(C.DEFAULT_TASK_CODE)
        if args.skip_to == "grab2":
            print("== [grab2] 人工摆物料盘(批1抓料位, 车头朝东), 直接跑批2 ==")
        else:
            if display:
                display.set_step("回物料盘(取批2)")
            leg_move(lk, 0, -850, 0, "后退850(南行至南车道)")
            leg_move(lk, 0, 0, -90, "顺时针转90(朝东)")
            leg_move(lk, 0, -820, 0, "后退820(西行至物料盘经线)")
            leg_move(lk, 0, 0, -90, "顺时针转90(朝南)")
            leg_move(lk, 0, 120, 0, "前进120(对齐抓料位)")
            leg_move(lk, 0, 0, 90, "逆时针转90(朝东=批1抓料朝向)")
        if display:
            display.set_step("守株抓料(批2)")
        action_grab_materials(lk, cam, backend, task, args.wait, colors=task["batch2_colors"])
        leg_move(lk, 0, 0, 90, "逆时针转90")
        leg_move(lk, 0, 920, 0, "前进920")
        leg_move(lk, 0, 850, 0, "前进850")
        leg_move(lk, 0, 0, 90, "逆时针转90")
        if display:
            display.set_step("粗加工区→放环(批2)")
        action_place_rings(lk, cam, backend, task, args.timeout,
                           rings=task["batch2_rings"], colors=task["batch2_colors"])
        action_grab_rings(lk, task, args.timeout, rings=task["batch2_rings"])
        leg_move(lk, 0, -820, 0, "后退820")
        leg_move(lk, 0, 0, -90, "顺时针转90")
        leg_move(lk, 0, -800, 0, "后退800(到暂存区)")
        if display:
            display.set_step("暂存区→码垛(批2)")
        action_stack_rings(lk, cam, backend, task, args.timeout)
    if args.skip_to in ("start", "rough", "buffer", "grab2", "home"):
        leg_move(lk, 0, -850, 0, "后退850")
        leg_move(lk, 0, 0, -90, "顺时针转90")
        leg_move(lk, 0, -850, 0, "后退850")
        leg_move(lk, 0, -800, 0, "后退800")
        leg_move(lk, 0, 0, 90, "逆时针转90")
        leg_move(lk, -50, 0, 0, "左移50")
        leg_move(lk, 0, -50, 0, "后退50(回启停区)")
    print("\n== 全部完成 (账本为零=车在锚点位) ==")
    if display:
        display.set_step("全部完成")
        time.sleep(3)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n中止")
