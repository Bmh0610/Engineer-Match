"""GPIO 输入/输出/边沿回环自检: 拔掉传感器两根信号线, 用一根杜邦线短接两个脚, 验证 Jetson 侧链路。

用法: python3 tests/test_gpio_loopback.py [--out 29] [--pin 31]
判读: 电平命中 10/10 + 边沿捕获 ✓ = Jetson 侧输出→输入→边沿事件全通, 问题在传感器/接线;
      电平 0/10 = 短接线没插好或脚号不对(按板上丝印重数, pin1 角有三角标记)。
"""
import argparse
import threading
import time
import warnings

warnings.filterwarnings("ignore", message="Could not open /dev/mem")   # pinmux 检查缺 /dev/mem 权限, 无害跳过

import Jetson.GPIO as GPIO


def main():
    ap = argparse.ArgumentParser(description="GPIO 回环自检(杜邦线短接两脚)")
    ap.add_argument("--out", type=int, default=29, help="输出脚(BOARD)")
    ap.add_argument("--pin", type=int, default=31, help="输入脚(BOARD)")
    args = ap.parse_args()

    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(args.out, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(args.pin, GPIO.IN)
    print(f"  短接 pin{args.out} ↔ pin{args.pin}, 开始自检")

    # ① 电平直读: 输出拉高后输入应跟着高, 10 次
    ok = 0
    for _ in range(10):
        GPIO.output(args.out, True)
        time.sleep(0.00005)
        if GPIO.input(args.pin):
            ok += 1
        GPIO.output(args.out, False)
        time.sleep(0.05)
    print(f"  ① 电平直读: {ok}/10 {'✓' if ok == 10 else '⚠(短接线/脚号)'}")

    # ② 边沿事件(信息性, 不影响判定): 子线程 30ms 后拉高制造边沿, 主线程挂 RISING 监听。
    #    ⚠ 本板 R39.2 Jetson.GPIO timeout 只收整数且单位=秒(30→30.03s 实测, float→TypeError),
    #    故 timeout=1(=1s 兜底); 测距主链路已改纯轮询, 此项仅探明边沿事件机制是否可用。
    GPIO.output(args.out, False)
    time.sleep(0.05)
    threading.Timer(0.03, lambda: GPIO.output(args.out, True)).start()
    ev = GPIO.wait_for_edge(args.pin, GPIO.RISING, timeout=1)
    GPIO.output(args.out, False)
    print(f"  ② 边沿事件: {'✓ 捕获 RISING' if ev is not None else '⚠ 未捕获(仅信息性)'}")

    GPIO.cleanup()
    if ok == 10 and ev is not None:
        print("  == Jetson 侧链路全通, 无回波问题在传感器/接线(VCC 5V? TRIG/ECHO 接反?) ==")
    elif ok == 10:
        print("  == 电平通但边沿未捕获, 把输出发给助手 ==")
    else:
        print("  == 电平不通: 重插短接线/核对脚号(拿不准万用表通断挡量) ==")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n中止")
