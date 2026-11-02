"""MODE5 成套动作序列测试: 解析任务码 → 按色序抓料上盘 → 按环号放环。

任务码兼容两种形态:
  4组完整码 213+321+456+123: 两批料(第1/3组=抓料色序, 第2/4组=放环环号), 走 qr.parse_task;
  2组简写   213+321:        只跑第一批料(第1组=抓料色序, 第2组=放环环号), 本地解析。
例 213+321: 抓料 黄→盘1、红→盘2、蓝→盘3; 放环 盘1→环3、盘2→环2、盘3→环1。

用法: python3 tests/test_arm_seq.py [--code 213+321] [--timeout 35]
每条 MODE5 逐条回车执行(下位机执行整套臂动作), 等 0x0015 完成回执;
超时告警跳过该条(不盲目重发——重发会让动作再做一遍); Ctrl+C 退出。
"""
import re
import sys
import time
from pathlib import Path

BASE = str(Path(__file__).resolve().parent.parent)   # main-copy 根(双端通用)
sys.path.insert(0, BASE)

import config as C  # noqa: E402
from communication.protocol import Mode, frame  # noqa: E402
from vision.qr import RULE_DIGIT2COLOR, parse_task  # noqa: E402

SIMPLE_PATTERN = re.compile(r"^([1-6]{3})\+([1-5]{3})$")   # 2组简写: 色序+环号


def parse_test_code(raw):
    """测试码解析: 4组完整码走 parse_task; 2组简写补 batch2 空占位; 非法 None。"""
    task = parse_task(raw)
    if task:
        return task
    m = SIMPLE_PATTERN.match(raw.strip()) if raw else None
    if not m:
        return None
    colors = [RULE_DIGIT2COLOR[d] for d in m.group(1)]
    return {"raw": raw.strip(),
            "batch1_colors": colors, "batch1_rings": [int(d) for d in m.group(2)],
            "batch2_colors": [], "batch2_rings": [],
            "stack_plan": dict(zip(colors, [int(d) for d in m.group(2)]))}


ACK_NAME = {Mode.ACK_CHASSIS: "0x0011底盘", Mode.ACK_ARM: "0x0012臂",
            Mode.ACK_GRIPPER: "0x0013夹爪", Mode.ACK_ACTION: "0x0015成套动作"}

_STEP = [0]


def confirm(msg):
    try:
        input(f"\n[{_STEP[0]}] <- 回车: {msg}")
    except EOFError:
        pass


def label_debug(group, outer, car):
    """打印用说明: 按动作组给一句中文。"""
    return {2: f"抓料: 原料区位{outer} → 车上格{car}",
            1: f"放环: 车上格{car} → 环{outer}",
            3: f"抓环: 环{outer} → 车上格{car}"}.get(
        group, f"组{group}: 外位{outer} 车上格{car}")


def run_action(lk, group, outer, car, timeout):
    """发 MODE5 成套动作, 等 0x0015 完成回执; 超时告警跳过(不盲目重发——
    文档警告重发会让动作再做一遍)。"""
    _STEP[0] += 1
    tx = frame(Mode.ACTION, int(group), int(outer), int(car))
    confirm(f"{label_debug(group, outer, car)}\n  hex={tx.hex(' ').upper()}")
    lk.flush()
    lk.ser.write(tx)
    t0 = time.time()
    got = None
    while time.time() - t0 < timeout:
        for f in lk.poll():
            print(f"    RX {f}")
            if f.mode == Mode.ACK_ACTION and got is None:
                got = f
        if got:
            break
        time.sleep(0.05)
    if got:
        print(f"    ✓ 成套动作完成 ({time.time()-t0:.1f}s)")
    else:
        print(f"    !! {timeout:.0f}s 未收到 0x0015 完成回执 —— 告警跳过"
              f"(不盲目重发: 若指令实际在执行, 重发会重复抓放)")
    time.sleep(2)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="MODE5 成套动作序列测试")
    ap.add_argument("--port", default=C.SERIAL_PORT)
    ap.add_argument("--code", default="213+321",
                    help="任务码: 4组完整码或2组简写(色序+环号)")
    ap.add_argument("--timeout", type=float, default=35.0,
                    help="单条 MODE5 等完成回执超时(秒)")
    args = ap.parse_args()

    from communication.protocol import Link

    task = parse_test_code(args.code)
    if task is None:
        print(f"!! 任务码 {args.code} 格式非法, 退出")
        return
    lk = Link(args.port)
    colors, rings = task["batch1_colors"], task["batch1_rings"]
    print(f"任务码 {task['raw']} | 抓料色序 {colors} | 放环环号 {rings}")
    print("口径: 原料区位 i 的料 → 盘格 i → 环 任务码环号位 | 每条回车, Ctrl+C 退出\n")

    _STEP[0] = 0
    # 抓料 ×3: 第 i 色从原料区位 i 抓到盘格 i (组2, Y=色序 Z=顺序格)
    for i in range(1, len(colors) + 1):
        run_action(lk, 2, i, i, args.timeout)
    # 放环 ×3: 盘格 i 的料 → 环 任务码环号[i] (组1, Y=环号 Z=顺序格)
    for i, ring_no in enumerate(rings, 1):
        run_action(lk, 1, ring_no, i, args.timeout)
    print("\n== 序列完成 ==")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n中止")
