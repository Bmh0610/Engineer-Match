"""MODE5 成套动作两圈全序列测试(2026-11-02 定稿口径)。

任务码执行前交互输入一次(空回车=默认): 4 组完整码=两圈全跑, 2 组简写=只跑第一圈。

序列(456+321 例, 全程零账本——环位是固定靶位, 放环后每环内容即任务码推演的确定结果):
  ①抓料   组2 (2,i,i) i=1..3    视觉判序, 第 i 个抓到的上盘格 i
  ②粗区放环 组1 (1,G2[i],i)     盘格i → 环G2[i]
  ③粗区抓环 组3 (3,r,r) r=3,2,1 倒序恒等: 环r→盘格r(纯记忆, 数字被物料挡)
  ④暂区放环 组1 (1,i,i) i=1..3  盘格i→暂存位i(恒等恰=G2 两区同位, 零账本)
  ⑤~⑦批2  同①~③(放环用 G4)
  ⑧码垛   组1: 开末端相机识别环上物料色(左/中/右=位1/2/3)定 Y, 盘上账本
                (盘格z 色=G3[G4.index(z)])定 Z, 同色叠批1上方;
                --no-vision 时按 (1, stack_plan[色], G4[k]) 纯推演。

用法: python3 tests/test_arm_seq.py [--code 213+321] [--no-vision] [--timeout 35]
213+321 → 12 条; 213+321+312+231 → 24 条(码垛须批1/批2 同三色, 规则约束)。
段间打印【实车此处需行车】提示(本文件不发底盘帧, 行车走 main.py/NavDriver)。
每条回车执行等 0x0015; 超时告警跳过不重发(重发会让动作再做一遍); Ctrl+C 退出。
"""
import sys
from pathlib import Path

BASE = str(Path(__file__).resolve().parent.parent)   # main-copy 根(双端通用)
sys.path.insert(0, BASE)

import config as C  # noqa: E402
import task.mission as M  # noqa: E402
from vision.qr import parse_test_code  # noqa: E402  (见 qr.py: 兼容 2 组简写)

_STEP = [0]


def seg(title):
    """段标题 + 步号清零。"""
    _STEP[0] = 0
    print(f"\n{'='*8} {title} {'='*8}")


def run(lk, group, outer, car, timeout):
    """带步号的逐条门控 MODE5。"""
    _STEP[0] += 1
    try:
        input(f"[{_STEP[0]}] <- 回车: {M.action_label(group, outer, car)}")
    except EOFError:
        pass
    return M.run_action(lk, group, outer, car, timeout)


def grab_seg(lk, args, colors, batch="①"):
    """抓料段: 组2 (2, i, i)。"""
    seg(f"{batch} 抓料 色序={colors}")
    for i in range(1, len(colors) + 1):
        run(lk, 2, i, i, args.timeout)


def place_seg(lk, args, rings, batch="②"):
    """粗区放环段: 组1 (1, rings[i], 盘格i)。"""
    seg(f"{batch} 粗区放环 环号={rings}")
    for i, r in enumerate(rings, 1):
        run(lk, 1, r, i, args.timeout)


def grab_rings_seg(lk, args, rings, batch="③"):
    """粗区抓环段: 组3 倒序恒等 (3, r, r)。"""
    order = sorted(set(rings), reverse=True)
    seg(f"{batch} 粗区抓环(倒序恒等) 环{'→'.join(map(str, order))}")
    for r in order:
        run(lk, 3, r, r, args.timeout)


def buffer_seg(lk, args, rings, batch="④"):
    """暂存区放环段(第一圈): 组1 恒等正序 (1, i, i)。"""
    seg(f"{batch} 暂存区放环(恒等) 盘i→位i")
    for i in range(1, len(rings) + 1):
        run(lk, 1, i, i, args.timeout)


def stack_seg(lk, args, task):
    """码垛段(第二圈): 组1 同色叠放, 视觉/推演两版。"""
    seg(f"⑧ 暂存区码垛 (批2={task['batch2_colors']}, "
        f"{'任务码推演' if args.no_vision else '视觉识别'})")
    if args.no_vision:
        M.stack_rings(lk, None, None, task, args.timeout,
                      use_vision=False, gated=True)
        return
    from vision.camera import Camera
    from vision.detect import YOLOBackend
    cam = Camera(C.CAMERA_ID_END, "end")
    cam.open()
    print(f"加载模型 {args.model} ...")
    backend = YOLOBackend(args.model)
    M.stack_rings(lk, cam, backend, task, args.timeout, use_vision=True, gated=True)


def drive_hint(txt):
    print(f"\n>>>>> 【实车此处需行车】{txt} (本测试不发底盘帧) <<<<<\n")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="MODE5 两圈成套动作序列测试")
    ap.add_argument("--port", default=C.SERIAL_PORT)
    ap.add_argument("--code", default="213+321",
                    help="默认任务码(执行前交互输入可覆盖, 空回车=用此默认)")
    ap.add_argument("--timeout", type=float, default=35.0,
                    help="单条 MODE5 等完成回执超时(秒)")
    ap.add_argument("--no-vision", action="store_true",
                    help="码垛段不开相机, 按任务码推演(无车冒烟/未接相机)")
    ap.add_argument("--model", default=C.MODEL_PATH, help="物料模型(码垛视觉用)")
    args = ap.parse_args()

    from communication.protocol import Link

    # 执行前交互输入一次任务码(空回车=默认)
    try:
        code = input(f"任务码[{args.code}]: ").strip()
    except EOFError:
        code = ""
    if code:
        args.code = code

    task = parse_test_code(args.code)
    if task is None:
        print(f"!! 任务码 {args.code} 格式非法, 退出")
        return
    lk = Link(args.port)
    c1, r1 = task["batch1_colors"], task["batch1_rings"]
    two = bool(task["batch2_colors"])
    print(f"任务码 {task['raw']} | 批1 色{c1} 环{r1}"
          + (f" | 批2 色{task['batch2_colors']} 环{task['batch2_rings']}" if two else ""))
    print("口径: 抓料按序上盘i / 放环(1,环号,盘格i) / 抓环倒序(3,r,r) / "
          "暂区(1,i,i) / 码垛同色查账 | Ctrl+C 退出")

    seg("第一圈 (批1)")
    grab_seg(lk, args, c1, "①")
    drive_hint("物料区 → 粗加工区")
    place_seg(lk, args, r1, "②")
    grab_rings_seg(lk, args, r1, "③")
    drive_hint("粗加工区 → 暂存区")
    buffer_seg(lk, args, r1, "④")

    if not two:
        print("\n== 第一圈完成 (2 组简写码, 不跑批2) ==")
        return

    seg("第二圈 (批2)")
    drive_hint("暂存区 → 返回原料区")
    grab_seg(lk, args, task["batch2_colors"], "⑤")
    drive_hint("物料区 → 粗加工区")
    place_seg(lk, args, task["batch2_rings"], "⑥")
    grab_rings_seg(lk, args, task["batch2_rings"], "⑦")
    drive_hint("粗加工区 → 暂存区")
    stack_seg(lk, args, task)
    print("\n== 两圈全部完成 ==")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n中止")
