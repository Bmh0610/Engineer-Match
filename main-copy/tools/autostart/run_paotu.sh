#!/bin/bash
# 开机自启包装(2026-11-06): GNOME 会话内跑 main.py --no-check, 崩溃自动拉起。
# 安装: 本文件由 ~/.config/autostart/paotu.desktop 拉起(需 GDM 自动登录, 见同目录 README.md)。
# 停止: touch ~/STOP_PAOTU && pkill -f run_paotu.sh   (调试前必做, 防占 ttyUSB0)
# 恢复: rm ~/STOP_PAOTU (下次开机生效) 或手动 cd ~/main-copy && python3 main.py --no-check
set -u
DIR=/home/popcorn-car/main-copy          # 比赛跑的版本目录(复件版改这行)
LOG=/home/popcorn-car/main_boot.log
STOP=/home/popcorn-car/STOP_PAOTU
cd "$DIR" || exit 1
say() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

say "=== run_paotu 启动, 等设备就绪 ==="
for i in $(seq 1 15); do
    [ -e /dev/ttyUSB0 ] && [ -e /dev/cam_end ] && [ -e /dev/cam_front ] && break
    sleep 1
done
[ -e /dev/ttyUSB0 ] && [ -e /dev/cam_end ] && [ -e /dev/cam_front ] \
    || say "!! 15s 内设备未齐(ttyUSB0/cam_end/cam_front), 仍尝试启动(main.py 报错会循环重试)"
sleep 3   # udev 别名/桌面稳定余量

while true; do
    [ -f "$STOP" ] && { say "哨兵 $STOP 在, 不启动, 退出"; exit 0; }
    say "--- main.py --no-check 启动 ---"
    PYTHONUNBUFFERED=1 python3 main.py --no-check >> "$LOG" 2>&1
    rc=$?
    if [ "$rc" -eq 0 ]; then
        # rc=0 = 正常跑完, 或运行期硬失败(串口断线等, main.py 捕获后打印"中止(硬失败)")
        #   ——两种都不该再自动开新一场(硬失败=人工介入, 项目纪律), 跑完即歇
        say "--- main.py 退出 rc=0 (跑完/硬失败人工介入), 不再重启 ---"
        exit 0
    fi
    say "--- main.py 崩溃 rc=$rc (开机设备未就绪等), 8s 后重试 ---"
    # 日志超 10MB 滚动(防比赛日写爆 SD 卡)
    if [ -f "$LOG" ] && [ "$(stat -c%s "$LOG")" -gt 10485760 ]; then
        mv "$LOG" "$LOG.old"
    fi
    sleep 8
done
