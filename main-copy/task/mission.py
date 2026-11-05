"""MODE5 成套动作业务库(2026-11-02 用户定稿口径): 抓料/放环/抓环/码垛 + 判环纠偏。

口径(456+321 例, 详见 tests/test_arm_seq.py docstring 验算表):
- 抓料(组2): 守株待兔·对号入座(2026-11-03 定)——臂缩回复位→定一纠偏
  (align_material_turntable: 转动中暂停不发帧/停歇后框内优先选料, 2026-11-04)→车全程
  不动, **任意色**进窗+停歇稳定即发 (2, 2, k); Y=2 恒定=取料工位, Z=k=该色在
  任务码色序中的位次(识别对=色必落自己的位次格, 顺序账自动成立)。批1/批2 复用。
- 放环(组1): 臂回复位→判环找中环→对中环纠偏建锚(唯一底盘动作), 然后逐轮发:
  rough 模式(粗加工区): (1, rings[i], 盘格i)——取盘格 i 的料放 rings[i] 号环;
  buffer 模式(暂存区第一圈): (1, i, i) 恒等正序——盘格 i(=环 i 的料)放暂存位 i,
  恰满足 G2 两区同位(零账本)。
- 抓环(组3): 纯记忆不开视觉(放环后环上数字被物料挡), 正序恒等(2026-11-03 定)——
  放过的环号从小到大 (3, r, r): 环 r → 盘格 r。
- 码垛(组4, 第二圈暂存区): 批2 同色叠批1 正上方(放置高度1000 组4 内部自理)。
  账本=批2 抓环后盘格 z 上的颜色
  是 G3[G4.index(z)]; 视觉版拍帧 find_materials 识别环上物料(画面左/中/右=位 1/2/3,
  朝向实车核对)定 Y、账本定 Z; --no-vision 推演版按 (1, stack_plan[色], G4[i])。
  层高(叠批1 物料顶+76mm): MODE5 组1 无层参数, 下探由下位机自理(移交电控)。
"""
import math
import time

import config as C
from communication.protocol import Mode, frame

ACTION_LABEL = {1: "放环(车→地面)", 2: "抓料(原料区→车)", 3: "抓环(地面→车)",
                4: "码垛(车→地面第二层)"}


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
          offset=(0.0, 0.0), phase_switch_mm=None):
    """整车纠偏(圆环区/码垛): 看-动-再看, 死区 ≤ALIGN_TOL_MM_RING(1mm) ≤ALIGN_MAX_ITER 轮, 超时带误差继续(保完赛)。
    auto=True 不弹门控直接跑(建锚用); False 回车触发/s 跳过(人工兜底)。
    offset=(右,前) mm: 纠偏结束后补走的固定基准补偿(相机基准点漂移, 2026-11-03)。
    phase_switch_mm(2026-11-05 用户定两阶段): 非 None 时粗/精两段——
      粗段(bbox 中心, 对模糊帧稳)残差 <switch(mm) → 锁存切精段(几何中心),
      粗段死区=switch(10)、精段死区=ALIGN_TOL_MM_RING(1); 单向不回粗(防两基准
      差几 px 来回横跳)。对象无 bbox 中心时粗=精(天然退化单阶段)。"""
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
    phase_fine = False
    for i in range(1, C.ALIGN_MAX_ITER + 1):
        obj, _fr = finder.find_stable(find_fn, cam,
                                      timeout=max(0.5, deadline - time.time()))
        if obj is None:
            print(f"  [纠偏] 第{i}轮未检出目标, 放弃(保完赛继续)")
            return
        a, b, c, d = matrix
        has_bbox = phase_switch_mm is not None and getattr(obj, "bbox_cx", None) is not None
        fine_dx = a * obj.dx + b * obj.dy
        fine_dy = c * obj.dx + d * obj.dy
        fine_res = math.hypot(fine_dx, fine_dy)
        if has_bbox and not phase_fine:
            cdy = a * obj.bbox_cx + b * obj.bbox_cy
            cdx = c * obj.bbox_cx + d * obj.bbox_cy
            cres = math.hypot(cdx, cdy)
            if cres < phase_switch_mm:
                phase_fine = True   # 单向锁存切精段
        if phase_fine or not has_bbox:
            dy_right, dx_fwd, residual = fine_dx, fine_dy, fine_res
            tol = C.ALIGN_TOL_MM_RING
            tag = "精段(几何)"
        else:
            dy_right, dx_fwd, residual = cdy, cdx, cres
            tol = phase_switch_mm
            tag = "粗段(bbox)"
        if residual <= tol:
            print(f"  ✓ 纠偏结束 ({tag}, 残差 {residual:.1f}mm, {i} 轮)")
            done = True
            break
        if time.time() > deadline:
            break
        print(f"  [纠偏] 第{i}轮[{tag}]: 前{dx_fwd:+.1f} 右{dy_right:+.1f} (残差 {residual:.1f}mm)")
        _send_move_v(lk, dy_right, dx_fwd)
        time.sleep(C.ALIGN_SETTLE)
    print("  纠偏结束(带误差, 保完赛继续)")
    if offset[0] or offset[1]:
        _send_move_v(lk, offset[0], offset[1])
        print(f"  [纠偏] 基准补偿: 右{offset[0]:+.1f} 前{offset[1]:+.1f}mm")


def _last_frame(cam):
    """取一帧用于存证(读不到返回 None)。"""
    try:
        return cam.read_latest(timeout=0.2)
    except Exception:
        return None


def _dump_miss_frame(frame, path, tag):
    """0 检出存证: 现场帧覆盖写(失败静默, 不干扰主流程)。"""
    if frame is None:
        return
    try:
        import cv2
        cv2.imwrite(path, frame)
        print(f"  [存证] {tag} 现场帧 → {path}")
    except Exception:
        pass


def align_angle(lk, cam, centers_fn, timeout, offset=(0.0, 0.0)):
    """角度纠偏(2026-11-05 用户定): 三心(环心/物料心)连线平行化——连线倾角
    θ(相对画面水平线)≈车身偏航角, 转向 move_v(0,0,±θ) 让连线回平。
    centers_fn(f) → 中心列表[(cx, cy), ...]; 检出降级梯度: ≥2 点即拟合
    (2 点无共线校验, 噪声+20%); <2 点等待, 持续 ANGLE_GIVEUP_S → 放弃返回
    (调用方进平移纠偏兜底, 保完赛)。收敛/超时/放弃均正常返回。"""
    from vision import finder
    deadline = time.time() + timeout
    giveup_deadline = time.time() + C.ANGLE_GIVEUP_S
    n_round = 0
    while time.time() < deadline and n_round < C.ALIGN_MAX_ITER:
        f = cam.read_latest(timeout=0.2)
        centers = [] if f is None else centers_fn(f)
        if len(centers) < 2:
            if time.time() >= giveup_deadline:
                print("  [角度] 可见中心持续<2, 放弃角度纠偏(平移纠偏兜底)")
                return
            time.sleep(0.2)
            continue
        ang, res = finder.line_angle_deg(centers)
        if ang is None:
            print(f"  [角度] 三心不共线(残差 {res:.0f}px), 等待重试...")
            time.sleep(0.3)
            continue
        ang = ang * C.ANGLE_SIGN
        if abs(ang) <= C.ANGLE_TOL_DEG:
            print(f"  ✓ 角度纠偏结束 (倾角 {ang:+.1f}°, {n_round} 轮)")
            if abs(offset[0]) > 1e-6 or abs(offset[1]) > 1e-6:
                _send_move_v(lk, offset[0], offset[1])
            return
        step = max(-C.ANGLE_MAX_STEP, min(C.ANGLE_MAX_STEP, ang)) * C.ANGLE_SIGN
        n_round += 1
        print(f"  [角度] 第{n_round}轮: 连线倾角 {ang:+.1f}° → 转向 {step:+.1f}°")
        lk.move_v(0, 0, round(step * 10))   # move_v z 单位=度(底盘协议 1°/单位)
        t0 = time.time()
        while time.time() - t0 < C.CMD_TIMEOUT:
            for f in lk.poll():
                if f.mode == Mode.ACK_CHASSIS:
                    break
            else:
                time.sleep(0.03)
                continue
            break
        time.sleep(0.3)


def align_angle_rings(lk, cam, backend, timeout):
    """圆环区角度纠偏: 三环心连线平行化(place_rings 平移纠偏完成后调用)。"""
    centers_fn = make_ring_centers_fn(backend)
    align_angle(lk, cam, centers_fn, timeout)


def align_angle_materials(lk, cam, backend, timeout):
    """码垛区角度纠偏: 三物料心连线平行化(stack_rings 平移纠偏完成后调用)。"""
    centers_fn = make_material_centers_fn(backend)
    align_angle(lk, cam, centers_fn, timeout)


def make_ring_centers_fn(backend):
    """三环心检出闭包: find_rings ≥2 环 → [(cx, cy), ...](按 cx 排序)。"""
    from task.ring import find_rings

    def centers(f):
        rings = find_rings(f, backend=backend)
        return sorted([(r.cx, r.cy) for r in rings])[:3]
    return centers


def make_material_centers_fn(backend):
    """三物料心检出闭包: find_materials ≥2 颗 → [(cx, cy), ...](按 cx 排序)。"""
    from vision import finder

    def centers(f):
        mats = finder.find_materials(backend, f, conf=0.5, px_per_mm=C.PX_PER_MM)
        return sorted([(m.cx, m.cy) for m in mats])[:3]
    return centers


def align_material_turntable(lk, cam, backend, timeout, offset=(0.0, 0.0)):
    """转盘定一纠偏(2026-11-04 用户定, 三态): 转动中/无料时**不发任何底盘帧**
    (防底盘追着转动的物料跑), 停歇后才看-动-再看。
    死区=ALIGN_TOL_MM_MATERIAL(3mm, 用户定: 转盘场景大步快收)。
    目标选择: ALIGN_BOX 框内离画面中心最近的物料优先(对中后恰在抓取窗, 守株
    立即可抓); 框内无料 → 画面最底下物料(离车最近, 对中后进框)。
    0 颗不放弃(转动中模糊会瞬时闪失, 总超时兜底)。终态口径不变=物料对画面中心
    (下位机抓取点示教依据)。offset 同 align()——纠偏结束后补走固定基准补偿。
    返回纠偏目标物料(跳过首颗抓取用, 2026-11-05 用户定: 纠偏料不抓, 从下一个
    进窗的料开始——车刚 move_v 完的余震窗口避开), 超时/未检出返回 None。"""
    from vision import finder
    deadline = time.time() + timeout
    x0, x1, y0, y1 = C.ALIGN_BOX
    round_no = 0
    while time.time() < deadline:
        f1 = cam.read_latest(timeout=0.2)
        m1 = [] if f1 is None else finder.find_materials(
            backend, f1, conf=0.5, px_per_mm=C.PX_PER_MM)
        time.sleep(0.2)   # 判稳基线: 两帧隔 0.2s——转盘最慢 36°/s 物料位移约 10px+
                          # (远超 4px 阈值), 慢转/启动段必检出(短窗口曾漏检致追料)
        f2 = cam.read_latest(timeout=0.2)
        m2 = [] if f2 is None else finder.find_materials(
            backend, f2, conf=0.5, px_per_mm=C.PX_PER_MM)
        if not m1 or not m2:
            print("  [纠偏] 暂未检出物料, 等待(不发帧)...")
            _dump_miss_frame(f1, "/tmp/mat_miss.jpg", "物料定一")
            time.sleep(0.3)
            continue
        # 停歇判定: 双向最近邻全 ≤ 阈值(2026-11-05 修: 单向会被物料交错换位欺骗——
        # f1 的 A 在 f2 里近邻是 B 时位移虚小, 实际转盘在动)
        tol = C.STABLE_TOL_PX * 2
        pairs_f12 = [min(math.hypot(a.cx - b.cx, a.cy - b.cy) for b in m2) for a in m1]
        pairs_f21 = [min(math.hypot(b.cx - a.cx, b.cy - a.cy) for a in m1) for b in m2]
        if max(pairs_f12 + pairs_f21) > tol:
            print(f"  [纠偏] 转盘转动中, 暂停纠偏(不发帧, 最大位移 "
                  f"{max(pairs_f12 + pairs_f21):.0f}px)")
            time.sleep(0.3)
            continue
        # 停歇: 目标=框内离画面中心最近, 无则画面最底下(cy 最大)——用 f2(最新帧,
        # 2026-11-05 修: 原用 f1 是 0.2s+ 前旧帧, 暂停期间转盘累积位移在放行瞬间
        # 一次性污染测量, 实车第2轮残差 137mm 即此)
        h, w = f2.shape[:2]
        in_box = [m for m in m2
                  if x0 * w <= m.cx <= x1 * w and y0 * h <= m.cy <= y1 * h]
        if in_box:
            tgt = min(in_box, key=lambda m: m.dx ** 2 + m.dy ** 2)
            src_tag = "框内"
        else:
            tgt = max(m2, key=lambda m: m.cy)
            src_tag = "最底下"
        # 放行前再确认一次(2026-11-05 修): f3 与 f2 比对, 堵"判定通过瞬间转盘启动"
        # 的最后窗口; 不过关则不发帧, 重新判停
        time.sleep(0.2)
        f3 = cam.read_latest(timeout=0.2)
        m3 = [] if f3 is None else finder.find_materials(
            backend, f3, conf=0.5, px_per_mm=C.PX_PER_MM)
        if not m3:
            continue
        pairs_23 = [min(math.hypot(b.cx - cc.cx, b.cy - cc.cy) for cc in m3)
                    for b in m2]
        pairs_32 = [min(math.hypot(cc.cx - b.cx, cc.cy - b.cy) for b in m2)
                    for cc in m3]
        if max(pairs_23 + pairs_32) > tol:
            print(f"  [纠偏] 放行前复检: 转盘又动了(最大位移 "
                  f"{max(pairs_23 + pairs_32):.0f}px), 不发帧重判")
            continue
        round_no += 1
        a, b, c, d = C.IMG2BODY
        dy_right = a * tgt.dx + b * tgt.dy
        dx_fwd = c * tgt.dx + d * tgt.dy
        residual = math.hypot(dx_fwd, dy_right)
        if residual <= C.ALIGN_TOL_MM_MATERIAL:
            print(f"  ✓ 定一纠偏结束 ({src_tag}料, 残差 {residual:.1f}mm, {round_no} 轮)")
            if abs(offset[0]) > 1e-6 or abs(offset[1]) > 1e-6:
                print(f"  [纠偏] 补基准补偿 前{offset[1]:+.1f} 右{offset[0]:+.1f}")
                _send_move_v(lk, offset[0], offset[1])
            return tgt   # 返回纠偏目标物料(调用方跳过首颗抓取, 2026-11-05 用户定)
        if time.time() > deadline:
            break
        print(f"  [纠偏] 第{round_no}轮({src_tag}料): 前{dx_fwd:+.1f} 右{dy_right:+.1f} "
              f"(残差 {residual:.1f}mm)")
        _send_move_v(lk, dy_right, dx_fwd)
        time.sleep(C.ALIGN_SETTLE)
    print("  [纠偏] 超时, 带误差继续(保完赛)")
    if abs(offset[0]) > 1e-6 or abs(offset[1]) > 1e-6:
        _send_move_v(lk, offset[0], offset[1])
    return tgt if round_no else None   # 超时也带最后目标(有轮次才有 tgt)


def find_material_fn(backend):
    """物料检测闭包: 离画面中心最近的一颗(定一用)。"""
    from vision import finder

    def find(f):
        mats = finder.find_materials(backend, f, conf=0.5, px_per_mm=C.PX_PER_MM)
        return finder.nearest_to_axis(mats, f.shape)
    return find


def find_stack_material_fn(backend):
    """码垛纠偏闭包: 只取**中环带**内物料(锚点=中环上那颗, 与圆环纠偏锁 digit==2
    同思路——2026-11-04 用户定)。带=|dx| ≤ RING_PITCH_PX/2≈130px(画面中心±130px,
    环距 150mm×环比例尺已知); 左右位物料结构性排除。带内无物料 → None → 放弃纠偏。"""
    from vision import finder
    half = C.RING_PITCH_PX / 2   # ≈130px

    def find(f):
        mats = finder.find_materials(backend, f, conf=0.5,
                                     px_per_mm=C.PX_PER_MM)
        mid = [m for m in mats if abs(m.dx) <= half]
        return min(mid, key=lambda m: m.dx ** 2 + m.dy ** 2) if mid else None
    return find


def find_ring_mid_fn(backend=None):
    """中环检测闭包: 只返回 digit==2 的环(建锚用; 仅环空数字可见时可靠)。
    未检出 2 号环 → None(调用方放弃纠偏, 保完赛)。"""
    from task.ring import find_rings

    def find(f):
        rings = find_rings(f, backend=backend)
        mid = [r for r in rings if r.digit == 2]
        return min(mid, key=lambda r: r.dx ** 2 + r.dy ** 2) if mid else None
    return find


def seek_mid_ring(lk, cam, backend, timeout):
    """纠偏前判环找中环(2026-11-05 用户定三规则):
    ① 看到 1 号环 → y+80；看到 3 号环 → y−80(2026-11-05 方向更正, 原反);
    ② 一个环都没看到 → x+10 步进扫描(直到看到环);
    ③ 对准 2 号环 → 直接纠偏。仅放环开场(环空数字可见)用;
    码垛段数字被批1物料挡, 不做判环(直接对物料纠偏)。"""
    from task.ring import find_rings
    deadline = time.time() + timeout
    moved = 0
    scanned = 0
    t_line = time.time()
    while time.time() < deadline and moved < 3:
        f = cam.read_latest(timeout=0.2)
        if f is None:
            n_none = getattr(seek_mid_ring, "_n_none", 0) + 1
            seek_mid_ring._n_none = n_none
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
            # ② 无环: x 步进扫描(2026-11-06 用户定 x+10→x-10, 2026-11-05 原地等待改横移找环)
            _send_move_v(lk, C.RING_SEEK_STEP_X, 0)
            scanned += 1
            print(f"  · 无环, x{C.RING_SEEK_STEP_X:+.0f} 扫描(第{scanned}步)")
            time.sleep(0.3)
            continue
        hit = min(rings, key=lambda r: r.dx ** 2 + r.dy ** 2)
        if hit.digit == 2:
            print("  ✓ 已对准 2 号环(中环), 直接纠偏")
            return
        # ① 方向(2026-11-05 用户更正): 看到 1 号 → y+80; 看到 3 号 → y−80
        dy = C.RING_SEEK_STEP if hit.digit == 1 else -C.RING_SEEK_STEP
        print(f"  · 当前对准 {hit.digit} 号环, 沿 y{dy:+.0f} 找 2 号环")
        _send_move_v(lk, dy, 0)
        moved += 1
        time.sleep(0.5)
    print("  !! 未见 2 号环(限次/限时), 直接纠偏(保完赛)")


# ---- 动作组①: 抓料(守株待兔, 批1/批2 通用) ----

def grab_materials(lk, cam, backend, colors, wait, batch="批1"):
    """守株抓料·按任务码顺序逐色等待(2026-11-05 用户定, 抓取顺序=评分项):
    臂缩回复位→定一纠偏建锚→车全程不动——逐色等**任务码第1组顺序**的目标色
    进窗+停歇稳定即发组2 (2, 2, i)。Y=2 恒定=取料工位; Z=i=顺序格(第 i 色上
    盘格 i)。单色超时跳过(保完赛, 盘格空置)。返回 (ok, total)。"""
    from vision import finder
    print(f"\n===== {batch} 抓料 (守株: 按任务码顺序逐色等待) =====")
    arm_move(lk, C.RESET, "缩回复位(定一纠偏位)")
    align_material_turntable(lk, cam, backend, wait)   # 三态: 转动暂停/停歇框内优先
    ok = 0
    for i, color in enumerate(colors, 1):
        print(f"\n<等 {color}(第{i}个抓): 转进窗+停歇即发 MODE5, ≤{wait:.0f}s>")
        mat, _fr = finder.wait_for_color(backend, cam, color,
                                         px_per_mm=C.PX_PER_MM,
                                         timeout=max(0.5, wait))
        if mat is None:
            print(f"  !! {wait:.0f}s 未等到 {color}, 跳过该色(保完赛, 盘格空置)")
            continue
        a, b, c, d = C.IMG2BODY
        print(f"  ✓ 看到: {mat.color} (右{a*mat.dx+b*mat.dy:+.1f} "
              f"前{c*mat.dx+d*mat.dy:+.1f}mm) → 组2 抓取上盘格{i}")
        if run_action(lk, 2, 2, i, C.ACTION_TIMEOUT):
            ok += 1
    print(f"\n== {batch} 抓料完成 ({ok}/{len(colors)}, 车在锚点位) ==")
    return ok, len(colors)


def place_rings(lk, cam, backend, rings, timeout, mode="rough", colors=None):
    """放环(组1): 回复位→判环找中环→对中环纠偏建锚→逐轮发。
    mode='rough': (1, rings[i], 盘格i) 按任务码环号序(粗加工区, G2/G4);
    mode='buffer': (1, i, i) 恒等正序(暂存区第一圈, 零账本)。返回 (ok, total)。"""
    from vision import finder
    print(f"\n===== 放环 ({'粗加工区' if mode == 'rough' else '暂存区'}) =====")
    arm_move(lk, C.RESET, "回复位")
    arm_move(lk, C.RING_ALIGN, "圆环纠偏位(前伸32mm)")
    seek_mid_ring(lk, cam, backend, timeout)
    align(lk, cam, find_ring_mid_fn(backend), C.IMG2BODY_RING,
          "对中环纠偏(建锚)", timeout, auto=True, phase_switch_mm=10.0)
    align_angle_rings(lk, cam, backend, timeout)   # 角度纠偏: 平移后三心连线平行化
    if mode == "rough":
        seq = [(r, i) for i, r in enumerate(rings, 1)]
    else:
        seq = [(i, i) for i in range(1, len(rings) + 1)]
    ok = 0
    for y, z in seq:
        if run_action(lk, 1, y, z, timeout):
            ok += 1
    if mode == "buffer":   # 2026-11-05 用户定: 暂存区放环后回节点位20; 粗区取消
        _send_move_v(lk, 0, -20)   # (粗区走30会让同区抓环偏30抓空)
        print("  · 放环后 y-20 回节点位")
    print(f"== 放环完成 ({ok}/{len(seq)}, 车在中环锚点位) ==")
    return ok, len(seq)


def grab_rings(lk, rings, timeout):
    """抓环(组3) 纯记忆正序恒等(2026-11-03 用户定, 原倒序): 放过的环号从小到大
    环 r → 盘格 r——恒等映射与次序无关, 统一取料侧 123 顺序。
    抓完 y−20 回节点位(2026-11-05 用户定 30→20)。返回 (ok, total)。"""
    print("\n===== 抓环 (纯记忆, 正序恒等: 环r→盘格r) =====")
    order = sorted(set(rings))
    ok = 0
    for r in order:
        if run_action(lk, 3, r, r, timeout):
            ok += 1
    _send_move_v(lk, 0, 20)    # 2026-11-05 用户定: 抓环后 y+20(方向反转)
    print("  · 抓环后 y+20")
    print(f"== 抓环完成 ({ok}/{len(order)}, 盘格=环号恒等) ==")
    return ok, len(order)


# ---- 动作组④: 码垛(第二圈暂存区, 同色叠放) ----

def stack_plan_of(task):
    """批2 盘上账本: {颜色: 盘格}——批2 抓环后盘格 z 上的颜色 = G3[G4.index(z)]。"""
    return {c: task["batch2_rings"][k]
            for k, c in enumerate(task["batch2_colors"])}


def stack_rings(lk, cam, backend, task, timeout, use_vision=True, gated=False):
    """码垛(第二圈暂存区): 批2 同色叠批1 正上方——**动作组4**(叠放第二层,
    放置高度1000 由下位机自理; 前提地面该位已有批1 物料=流程保证)。
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
        arm_move(lk, C.RING_ALIGN, "圆环纠偏位(前伸32mm)")
        matrix = C.IMG2BODY_STACK if C.IMG2BODY_STACK is not None else C.IMG2BODY_RING
        if C.IMG2BODY_STACK is None:
            print("  ⚠ IMG2BODY_STACK 未标定: 回退圆环矩阵 IMG2BODY_RING"
                  "(同场景几何最接近; 标定: test_img2body_cal.py --stack 后填 config)")
        align_material_turntable(lk, cam, backend, timeout)   # 三态: 等清晰帧+框内优先
        align_angle_materials(lk, cam, backend, timeout)   # 码垛角度纠偏: 三物料心连线平行化
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
        if run_action(lk, 4, y, z, timeout, gated=gated):
            ok += 1
    print(f"== 码垛完成 ({ok}/{len(colors2)}) ==")
    return ok, len(colors2)
