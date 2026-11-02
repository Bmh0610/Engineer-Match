"""超声波 HC-SR04 测距自测: 持续打印前向探头距离(中位数), 验证接线/读数/抖动。无车可跑。

接线(BOARD 物理脚号, 2026-10-31 已接): VCC=pin2(5V) GND=pin6 TRIG=pin29 ECHO=pin31;
ECHO 实测 3.3V 兼容版直连(串 1kΩ 对读数无影响)。⚠ 经典 5V 版 ECHO 直连会打坏引脚,
换模块前先量 ECHO 电平, 是 5V 就加分压(1k+2k)。

用法: python3 tests/test_ultrasonic.py [--trig 29] [--echo 31] [--threshold 60]
                [--n 3] [--interval 0.3]
前置: Jetson.GPIO 已装(缺: pip3 install Jetson.GPIO 或 sudo apt install python3-jetson-gpio)
      + 用户在 gpio 组(权限报错: sudo usermod -aG gpio $USER 重登录, 或临时 sudo python3)。
⚠ 本板 L4T R39.2 的 Jetson.GPIO wait_for_edge(timeout) 只收整数且单位=秒(2026-10-31 探针:
  30→30.03s, float→TypeError)——毫秒级超时无法表达, 故测距用纯轮询双沿, 不依赖边沿事件。
退出: 输入 q 回车, 或 Ctrl+C。
验收: 纸箱放 20/40/60/100cm, 读数与卷尺吻合 ±3cm、中位数后抖动 ±2~3cm 内
      → 链路通过, 再进 nav_drive 集成; 读不出/乱跳按脚本提示排查接线。
"""
import argparse
import sys
import threading
import time
import warnings

warnings.filterwarnings("ignore", message="Could not open /dev/mem")   # pinmux 检查缺 /dev/mem 权限, 无害跳过

GPIO = None   # main() 里 _load_gpio() 加载: 延迟导入, 没装库的机器上 --help 也能用


def _load_gpio():
    global GPIO
    try:
        import Jetson.GPIO as _GPIO
    except ImportError:
        sys.exit("!! 未装 Jetson.GPIO: pip3 install Jetson.GPIO "
                 "(或 sudo apt install python3-jetson-gpio)")
    GPIO = _GPIO


def measure(trig, echo, timeout_s=0.03):
    """触发一次测距 → 距离 cm; 无回波(超量程/未接/丢沿)返回 None。

    距离 = 回波脉宽 µs / 58 (声速 343m/s 往返折半); timeout_s=30ms 对应 ~5m 上限(HC-SR04 标称 4m)。
    测双沿用纯轮询而非 wait_for_edge: 本板 Jetson.GPIO timeout 只收整数且单位=秒(探针实测),
    毫秒级超时无法表达、事件机制未经真边沿验证; 轮询粒度 ~5µs≈0.1mm 精度更好,
    代价=测距窗口单核占满(≤30ms, 对节点问询频率无影响)。
    """
    deadline = time.monotonic() + timeout_s
    if GPIO.input(echo):   # 上次残留高电平: 等落完再触发, 防"上升沿"永远等不到
        while GPIO.input(echo) and time.monotonic() < deadline:
            pass
        if GPIO.input(echo):
            return None
    GPIO.output(trig, GPIO.LOW)
    time.sleep(0.00001)    # sleep 粒度实际 ~60µs, 触发 >10µs 即满足, 模块不挑剔
    GPIO.output(trig, GPIO.HIGH)
    time.sleep(0.00001)
    GPIO.output(trig, GPIO.LOW)
    while not GPIO.input(echo):              # 等上升沿(超时=无回波)
        if time.monotonic() > deadline:
            return None
    t0 = time.monotonic_ns()                 # 双沿各滞后一个轮询粒度(~5µs), 相减基本抵消
    while GPIO.input(echo):                  # 等下降沿(400cm 对应 23ms < 30ms 上限)
        if time.monotonic() > deadline:
            return None
    return (time.monotonic_ns() - t0) / 1000.0 / 58.0


def read_median(trig, echo, n=3, timeout_s=0.03):
    """连测 n 次取中位数, 滤单次丢沿/毛刺; 全部无回波返回 None。"""
    vals = sorted(v for v in (measure(trig, echo, timeout_s) for _ in range(n))
                  if v is not None)
    return vals[len(vals) // 2] if vals else None


def start_quit_watcher():
    """后台线程等 'q' 回车退出; stdin 关闭(无终端)时静默让位给 Ctrl+C。"""
    stop = threading.Event()

    def _watch():
        try:
            while not stop.is_set():
                if input().strip().lower() in ("q", "quit", "exit"):
                    stop.set()
        except EOFError:
            pass

    threading.Thread(target=_watch, daemon=True).start()
    return stop


def main():
    ap = argparse.ArgumentParser(description="HC-SR04 超声波测距自测(无车)")
    ap.add_argument("--trig", type=int, default=29, help="TRIG 板脚(BOARD)")
    ap.add_argument("--echo", type=int, default=31, help="ECHO 板脚(BOARD)")
    ap.add_argument("--threshold", type=float, default=60.0,
                    help="警示阈值 cm(集成时即 config.US_THRESHOLD_CM)")
    ap.add_argument("--n", type=int, default=3, help="每次打印的中位数采样数")
    ap.add_argument("--interval", type=float, default=0.3, help="打印间隔秒")
    args = ap.parse_args()

    _load_gpio()
    sys.stdout.reconfigure(line_buffering=True)   # 流式输出: 管道/SSH 场景实时可见(板端实测块缓冲+硬杀会吞行)
    try:
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(args.trig, GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(args.echo, GPIO.IN)
    except (PermissionError, OSError) as e:
        sys.exit(f"!! GPIO 打不开({e}): sudo usermod -aG gpio $USER 后重新登录, "
                 f"或临时 sudo python3 运行")
    time.sleep(0.1)   # 模块上电稳定

    print(f"  Jetson.GPIO OK: {GPIO.JETSON_INFO.get('TYPE', '未知型号')}")
    print(f"  TRIG=pin{args.trig} ECHO=pin{args.echo} (BOARD), "
          f"阈值 {args.threshold}cm, q 或 Ctrl+C 退出")
    if GPIO.input(args.echo):
        print("  ⚠ ECHO 静态电平=1(应 0): 疑似 TRIG/ECHO 接反或未共地, 读数多半异常")

    none_run, hint_shown = 0, False
    try:
        stop = start_quit_watcher()
        while not stop.is_set():
            d = read_median(args.trig, args.echo, n=args.n)
            if d is None:
                none_run += 1
                print(f"  front  --- cm (无回波)")
                if none_run == 8 and not hint_shown:
                    hint_shown = True
                    print("  !! 连续无回波, 排查: ① TRIG/ECHO 是否接反"
                          "(模块丝印 VCC·Trig·Echo·Gnd) ② VCC 是否上 5V "
                          "③ 目标是否在 2~400cm 内 ④ 经典 5V 版 ECHO 须分压")
            else:
                none_run = 0
                mark = "  ⚠<阈值" if d < args.threshold else ""
                print(f"  front {d:6.1f} cm{mark}")
            time.sleep(args.interval)
        print("\n退出")
    finally:
        GPIO.cleanup()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n中止")
