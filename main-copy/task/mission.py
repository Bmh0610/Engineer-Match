"""MODE5 成套动作业务库(2026-11-02 用户定稿口径): 抓料/放环/抓环/码垛 + 判环纠偏。

口径(456+321 例, 详见 tests/test_arm_seq.py docstring 验算表):
- 抓料(组2): 守株待兔·对号入座(2026-11-03 定)——臂缩回复位→定一纠偏建锚→车全程
  不动, **任意色**进窗+停歇稳定即发 (2, 2, k); Y=2 恒定=取料工位, Z=k=该色在
  任务码色序中的位次(识别对=色必落自己的位次格, 顺序账自动成立)。批1/批2 复用。
- 放环(组1): 臂回复位→判环找中环→对中环纠偏建锚(唯一底盘动作), 然后逐轮发:
  rough 模式(粗加工区): (1, rings[i], 盘格i)——取盘格 i 的料放 rings[i] 号环;
  buffer 模式(暂存区第一圈): (1, i, i) 恒等正序——盘格 i(=环 i 的料)放暂存位 i,
  恰满足 G2 两区同位(零账本)。
- 抓环(组3): 纯记忆不开视觉(放环后环上数字被物料挡), 正序恒等(2026-11-03 定)——
  放过的环号从小到大 (3, r, r): 环 r → 盘格 r。
- 码垛(组1, 第二圈暂存区): 批2 同色叠批1 正上方。账本=批2 抓环后盘格 z 上的颜色
  是 G3[G4.index(z)]; 视觉版拍帧 find_materials 识别环上物料(画面左/中/右=位 1/2/3,
  朝向实车核对)定 Y、账本定 Z; --no-vision 推演版按 (1, stack_plan[色], G4[i])。
  层高(叠批1 物料顶+76mm): MODE5 组1 无层参数, 下探由下位机自理(移交电控)。
"""
import math
import time

import config as C
from communication.protocol import Mode, frame

ACTION_LABEL = {1: "放环(车→地面)", 2: "抓料(原料区→车)", 3: "抓环(地面→车)"}


def action_label(group, outer, car):
    """一条 MODE5 的中文说明(打印/门控共用)。"""
    if group == 2:
        return f"抓料(原料区取料工位{outer}→车): 上盘格{car} ({group},{outer},{car})"
    name = ACTION_LABEL.get(group, f"组{group}")
    return f"{name}: 车外位{outer} ↔ 车上格{car} ({group},{outer},{car})"


def run_action(lk, group, outer, car, timeout=None, gated=False):
    """发 MODE5 成套动作, 等 0x0015 完成回执; gated=True 回车门控(测试用)。
    超时告警跳过不重发(文档警告重发会让动作再做一遍)。返回是否收到回执。"""
    timeout = C.ACTION_TIMEOUT if timeout is None else timeout
    if gated:
        try:
            input(f"<- 回车: {action_label(group, outer, car)}")
        except EOFError:
            pass
    tx = frame(Mode.ACTION, int(group), int(outer), int(car))
    print(f"  TX {action_label(group, outer, car)}\n     hex={tx.hex(' ').upper()}")
    lk.flush()
    lk.ser.write(tx)
    t0 = time.time()
    while time.time() - t0 < timeout:
        for f in lk.poll():
            if f.mode == Mode.ACK_ACTION:
                print(f"  ✓ 成套动作完成 ({time.time()-t0:.1f}s)")
                return True
        time.sleep(0.05)
    print(f"  !! {timeout:.0f}s 未收到 0x0015 —— 告警跳过(不盲目重发)")
    time.sleep(1)
    return False


def arm_move(lk, pose, label="臂", timeout=None):
    """臂单步(0x0002 等 0x0012): 码放动作组前后的收/放臂辅助。"""
    timeout = C.CMD_TIMEOUT if timeout is None else timeout
    tx = frame(Mode.MOVE_A, *pose)
    print(f"  TX 臂 {label} {tuple(pose)} hex={tx.hex(' ').upper()}")
    lk.flush()
    lk.ser.write(tx)
    t0 = time.time()
    while time.time() - t0 < timeout:
        for f in lk.poll():
            if f.mode == Mode.ACK_ARM:
                print(f"  ✓ 臂到位 ({time.time()-t0:.1f}s)")
                return True
        time.sleep(0.03)
    print(f"  !! {timeout:.0f}s 无 0x0012 臂回执(继续)")
    return False


# ---- 纠偏原语(看-动-再看; 自 test_full_seq_wait 提炼, 直发 move_v 不经 NavDriver) ----

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


def align(lk, cam, find_fn, matrix, label, timeout, auto=True,
          offset=(0.0, 0.0)):
    """整车纠偏: 看-动-再看, 死区 ≤ALIGN_TOL_MM ≤ALIGN_MAX_ITER 轮, 超时带误差继续(保完赛)。
    auto=True 不弹门控直接跑(建锚用); False 回车触发/s 跳过(人工兜底)。
    offset=(右,前) mm: 纠偏结束后补走的固定基准补偿(相机基准点漂移, 2026-11-03)。"""
    if not auto:
        try:
            ans = input(f"\n<- 回车={label}(看-动-再看) / s=跳过: ").strip().lower()
        except EOFError:
            return
        if ans == "s":
            print("  !! 跳过纠偏(人工兜底)")
            return
    from vision import finder
    deadline = time.time() + C.ALIGN_TIMEOUT
    done = False
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
            done = True
            break
        if time.time() > deadline:
            break
        print(f"  [纠偏] 第{i}轮: 前{dx_fwd:+.1f} 右{dy_right:+.1f} (残差 {residual:.1f}mm)")
        _send_move_v(lk, dy_right, dx_fwd)
        time.sleep(C.ALIGN_SETTLE)
    print("  纠偏结束(带误差, 保完赛继续)")
    if offset[0] or offset[1]:
        _send_move_v(lk, offset[0], offset[1])
        print(f"  [纠偏] 基准补偿: 右{offset[0]:+.1f} 前{offset[1]:+.1f}mm")


def find_material_fn(backend):
    """物料检测闭包: 离画面中心最近的一颗(定一/码垛纠偏用)。"""
    from vision import finder

    def find(f):
        mats = finder.find_materials(backend, f, conf=0.5, px_per_mm=C.PX_PER_MM)
        return finder.nearest_to_axis(mats, f.shape)
    return find


def find_ring_mid_fn():
    """中环检测闭包: 离画面中心最近的数字环(建锚用; 仅环空数字可见时可靠)。"""
    from task.ring import find_rings

    def find(f):
        rings = find_rings(f)
        return min(rings, key=lambda r: r.dx ** 2 + r.dy ** 2) if rings else None
    return find


def seek_mid_ring(lk, cam, timeout):
    """纠偏前判环找中环: 对准 1 号环 y-RING_SEEK_STEP、3 号 y+RING_SEEK_STEP,
    直到对准 2 号环再纠偏; 未见环直接纠偏兜底。仅放环开场(环空数字可见)用;
    码垛段数字被批1物料挡, 不做判环(直接对物料纠偏)。"""
    from task.ring import find_rings
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
        rings = find_rings(f)
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
        print(f"  · 当前对准 {hit.digit} 号环, 沿 y{dy:+.0f} 找 2 号环")
        _send_move_v(lk, dy, 0)
        moved += 1
        time.sleep(0.5)
    print("  !! 未见 2 号环(限次/限时), 直接纠偏(保完赛)")


# ---- 动作组①: 抓料(守株待兔, 批1/批2 通用) ----

def grab_materials(lk, cam, backend, colors, wait, batch="批1"):
    """守株抓料·对号入座版(2026-11-03 用户定): 臂缩回复位→定一纠偏建锚→车全程
    不动——**任意色**进窗+停歇稳定即抓(不再等特定色, 命中率不挑色), 盘格
    Z=该色在任务码色序中的位次(色 c 在第 k 位 → 组2 (2,2,k) 放盘格 k)。
    Y=2 恒定=原料区取料工位。识别失败/色序外的料不抓(等下轮停歇再看),
    单窗超时剩余色跳过(保完赛)。返回 (ok, total)。"""
    from vision import finder
    print(f"\n===== {batch} 抓料 (守株·对号入座: 任意色停歇即抓, Z=色序位) =====")
    arm_move(lk, C.RESET, "缩回复位(定一纠偏位)")
    align(lk, cam, find_material_fn(backend), C.IMG2BODY,
          "定一纠偏(建锚)", wait, auto=True)
    order = list(colors)
    grabbed = set()   # 已处理的色(无论回执成败——超时不重发, 宁可盘格空置不死循环)
    skip = set()      # 色序外/识别存疑的色, 本次不再处理(防反复检出死循环)
    ok = 0
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
        print(f"  ✓ 看到: {mat.color} (右{a*mat.dx+b*mat.dy:+.1f} "
              f"前{c*mat.dx+d*mat.dy:+.1f}mm) → 任务码第{k}位 → 组2 放盘格{k}")
        if run_action(lk, 2, 2, k, C.ACTION_TIMEOUT):
            ok += 1
        grabbed.add(mat.color)   # 成败都记账: 超时重发会让动作再做一遍(文档警告)
    print(f"\n== {batch} 抓料完成 ({ok}/{len(order)}, 车在锚点位) ==")
    return ok, len(order)


# ---- 动作组②③: 放环 + 抓环 ----

def place_rings(lk, cam, rings, timeout, mode="rough", colors=None):
    """放环(组1): 回复位→判环找中环→对中环纠偏建锚→逐轮发。
    mode='rough': (1, rings[i], 盘格i) 按任务码环号序(粗加工区, G2/G4);
    mode='buffer': (1, i, i) 恒等正序(暂存区第一圈, 零账本)。返回 (ok, total)。"""
    from vision import finder
    print(f"\n===== 放环 ({'粗加工区' if mode == 'rough' else '暂存区'}) =====")
    arm_move(lk, C.RESET, "回复位")
    seek_mid_ring(lk, cam, timeout)
    align(lk, cam, find_ring_mid_fn(), C.IMG2BODY_RING,
          "对中环纠偏(建锚)", timeout, auto=True, offset=C.RING_ALIGN_OFFSET)
    if mode == "rough":
        seq = [(r, i) for i, r in enumerate(rings, 1)]
    else:
        seq = [(i, i) for i in range(1, len(rings) + 1)]
    ok = 0
    for y, z in seq:
        if run_action(lk, 1, y, z, timeout):
            ok += 1
    print(f"== 放环完成 ({ok}/{len(seq)}, 车在中环锚点位) ==")
    return ok, len(seq)


def grab_rings(lk, rings, timeout):
    """抓环(组3) 纯记忆正序恒等(2026-11-03 用户定, 原倒序): 放过的环号从小到大
    (3, r, r), 环 r → 盘格 r——恒等映射与次序无关, 统一取料侧 123 顺序。
    返回 (ok, total)。"""
    print("\n===== 抓环 (纯记忆, 正序恒等: 环r→盘格r) =====")
    order = sorted(set(rings))
    ok = 0
    for r in order:
        if run_action(lk, 3, r, r, timeout):
            ok += 1
    print(f"== 抓环完成 ({ok}/{len(order)}, 盘格=环号恒等) ==")
    return ok, len(order)


# ---- 动作组④: 码垛(第二圈暂存区, 同色叠放) ----

def stack_plan_of(task):
    """批2 盘上账本: {颜色: 盘格}——批2 抓环后盘格 z 上的颜色 = G3[G4.index(z)]。"""
    return {c: task["batch2_rings"][k]
            for k, c in enumerate(task["batch2_colors"])}


def stack_rings(lk, cam, backend, task, timeout, use_vision=True, gated=False):
    """码垛(第二圈暂存区): 批2 同色叠批1 正上方(组1, 层高下位机自理)。
    vision: 对物料纠偏(数字被挡不判环)→拍帧检出环上物料按画面 x 排序(左/中/右=位
    1/2/3, 朝向实车核对)→每颗颜色查盘上账本定 Z→(1, 位, 盘格)。
    no-vision: 按 (1, stack_plan[色], G4[k]) 推演(cam/backend 传 None)。gated 逐条回车。
    返回 (ok, total)。"""
    print(f"\n===== 码垛 (暂存区, {'视觉识别' if use_vision else '任务码推演'}) =====")
    z_of_color = stack_plan_of(task)
    sp = task["stack_plan"]          # {批1颜色: 环号} = 批2 同色的目标位
    colors2 = task["batch2_colors"]
    jobs = []                        # (位Y, 盘格Z, 颜色)
    if use_vision:
        arm_move(lk, C.RESET, "抬臂至标定高度(码垛纠偏前, RESET Y=1810=标定高度, 2026-11-03 用户改)")
        matrix = C.IMG2BODY_STACK if C.IMG2BODY_STACK is not None else C.IMG2BODY_RING
        if C.IMG2BODY_STACK is None:
            print("  ⚠ IMG2BODY_STACK 未标定: 回退圆环矩阵 IMG2BODY_RING"
                  "(同场景几何最接近; 标定: test_img2body_cal.py --stack 后填 config)")
        align(lk, cam, find_material_fn(backend), matrix,
              "码垛纠偏(对物料建锚)", timeout, auto=True,
              offset=C.RING_ALIGN_OFFSET)
        from vision import finder
        f = cam.read_latest(timeout=1.0)
        mats = [] if f is None else finder.find_materials(
            backend, f, conf=0.5, px_per_mm=C.PX_PER_MM)
        if len(mats) < 3:
            print(f"  !! 仅检出 {len(mats)}/3 颗(限时/视野), 只码检出的(保完赛)")
        mats.sort(key=lambda m: m.dx)      # dx 小=画面左=位1 (朝向实车核对!)
        for pos, m in enumerate(mats[:3], 1):
            z = z_of_color.get(m.color)
            if z is None:
                print(f"  !! 位{pos} 检出 {m.color} 不在盘上账本, 跳过")
                continue
            print(f"  · 位{pos}={m.color} → 盘格{z} 同色")
            jobs.append((pos, z, m.color))
    else:
        for k, c in enumerate(colors2):
            pos = sp.get(c)
            if pos is None:
                print(f"  !! {c} 不在批1 (异色码), 跳过")
                continue
            jobs.append((pos, task["batch2_rings"][k], c))
    ok = 0
    for y, z, c in jobs:
        print(f"  · 码 {c}: 位{y} ← 盘格{z}")
        if run_action(lk, 1, y, z, timeout, gated=gated):
            ok += 1
    print(f"== 码垛完成 ({ok}/{len(colors2)}) ==")
    return ok, len(colors2)
