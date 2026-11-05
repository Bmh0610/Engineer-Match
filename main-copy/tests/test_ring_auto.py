"""环纠偏自动动车测试(=main.py 放环纠偏段, 去掉 MODE5): 臂到位 → 判环找中环
→ 对 2 号环自动看-动-再看, 车自己动。

⚠ 与 test_ring_align.py(手动协议式, 脚本不动车)的区别: 本测试接串口, 纠偏帧
自动发送——摆车回车后车会自己动, 回车前确保车前区域安全。

链路=正式链路逐字复用(task/mission.py, 与 place_rings 纠偏段一致):
  arm_move(RESET) → arm_move(RING_ALIGN 圆环纠偏位)
  → seek_mid_ring(判环: 1号→y+80 / 3号→y−80 / 无环→x+10 扫描, ≤3 步)
  → align(find_ring_mid_fn 锁 2 号环, IMG2BODY_RING, 粗段(bbox)残差<10 切
    精段(几何), 精段死区 ALIGN_TOL_MM_RING=1mm, ≤ALIGN_MAX_ITER 轮,
    超时带误差继续保完赛)
  → 收敛帧打印 数字→位置 映射(顺便核对 1/2/3 检出分档)。

用法: python3 tests/test_ring_auto.py [--port /dev/ttyUSB0]
       [--model-ring <engine>] [--timeout 35] [--rounds N]
回车=跑一轮(判环+自动纠偏); q 退出; --rounds N=回车一次连跑 N 轮(看残差分布)。
臂只在开局摆一次(循环内不再动臂); 退出臂留在 RING_ALIGN 位。
"""
import sys
from pathlib import Path

BASE = str(Path(__file__).resolve().parent.parent)   # main-copy 根(双端通用)
sys.path.insert(0, BASE)

import config as C  # noqa: E402
from task import mission as M  # noqa: E402   (模块级: run_round 也要用)


def digit_map(f, backend):
    """收敛帧 数字→位置 {1/2/3: 'L'/'M'/'R'}: 同 ring.StepRing._digit_map 口径
    (画面中心=M 基准, 半间距阈值 RING_PITCH_PX/2)。"""
    from task.ring import find_rings
    pos = {}
    for r in find_rings(f, backend=backend):
        pos[r.digit] = ("L" if r.dx < -C.RING_PITCH_PX / 2
                        else "R" if r.dx > C.RING_PITCH_PX / 2 else "M")
    return pos


def run_round(lk, cam, backend, timeout):
    """一轮=place_rings 纠偏段(找环+纠偏一体, on_miss 丢标找回, 取消放弃);
    收敛后打印数字→位置。"""
    M.align(lk, cam, M.find_ring_mid_fn(backend), C.IMG2BODY_RING,
            "对中环纠偏(建锚)", timeout, auto=True, phase_switch_mm=10.0,
            on_miss=lambda dl: M._reseek_ring2(lk, cam, backend, dl))
    f = cam.read_latest(timeout=0.5)
    if f is not None:
        print(f"  数字→位置: {digit_map(f, backend)}")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="环纠偏自动动车测试(=place_rings 纠偏段)")
    ap.add_argument("--port", default=C.SERIAL_PORT)
    ap.add_argument("--model-ring", default=C.MODEL_PATH_RING,
                    help="圆环检测引擎(YOLO 定身份; 缺文件=判环全空, 测试无意义)")
    ap.add_argument("--timeout", type=float, default=C.RING_ALIGN_TOTAL,
                    help="单轮找环+纠偏一体总预算(秒; 默认 RING_ALIGN_TOTAL)")
    ap.add_argument("--rounds", type=int, default=0,
                    help="回车一次连跑轮数(0=每轮手动回车)")
    args = ap.parse_args()

    from communication.protocol import Link
    from vision.camera import Camera
    from vision.detect import YOLOBackend

    cam = Camera(C.CAMERA_ID_END, "end")
    cam.open()
    print(f"加载圆环模型 {args.model_ring} ...")
    try:
        backend = YOLOBackend(args.model_ring)   # 缺文件→None; find_rings 返回空
    except Exception as e:
        print(f"!! 圆环引擎加载失败({e}): 判环会跳过, 自动纠偏无从谈起, 退出")
        return
    lk = Link(args.port)

    print("== 臂到位(main.py 同款): RESET → RING_ALIGN ==")
    M.arm_move(lk, C.RESET, "回复位")
    M.arm_move(lk, C.RING_ALIGN, "圆环纠偏位(前伸32mm)")

    print("\n⚠ 自动动车模式: 回车后判环+纠偏期间车会自己动, 保证车前区域安全")
    print("回车=跑一轮(判环找中环→自动纠偏 2 号环); q 退出")
    if args.rounds:
        print(f"--rounds {args.rounds}: 每次回车连跑 {args.rounds} 轮\n")
    rnd = 0
    while True:
        try:
            cmd = input(f"\n[第{rnd + 1}轮] 回车自动纠偏 / q 退出: ").strip().lower()
        except EOFError:
            break
        if cmd == "q":
            break
        for _ in range(max(1, args.rounds)):
            rnd += 1
            print(f"\n—— 第{rnd}轮 ——" if args.rounds > 1 else "")
            run_round(lk, cam, backend, args.timeout)
    print("\n退出(臂留在 RING_ALIGN 纠偏位)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n退出")
