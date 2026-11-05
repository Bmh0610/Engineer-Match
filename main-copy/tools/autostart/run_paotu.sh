#!/bin/bash
# 开机自启包装(2026-11-06): GNOME 会话内跑 main.py --no-check, 每次开机只跑一遍。
# 安装: 本文件由 ~/.config/autostart/paotu.desktop 拉起(需 GDM 自动登录, 见同目录 README.md)。
# 停止: touch ~/STOP_PAOTU; pkill -f run_paotu.sh; pkill -f "main.py --no-check"
# 恢复: rm ~/STOP_PAOTU (下次开机生效) 或手动 cd ~/main-copy && python3 main.py --no-check
set -u
DIR=/home/popcorn-car/main-copy          # 比赛跑的版本目录(复件版改这行)
LOG=/home/popcorn-car/main_boot.log
STOP=/home/popcorn-car/STOP_PAOTU
cd "$DIR" || exit 1
say() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

[ -f "$STOP" ] && { say "哨兵 $STOP 在, 不启动, 退出"; exit 0; }

say "=== run_paotu 启动, 等设备就绪 ==="
for i in $(seq 1 15); do
    [ -e /dev/ttyUSB0 ] && [ -e /dev/cam_end ] && [ -e /dev/cam_front ] && break
    sleep 1
done
[ -e /dev/ttyUSB0 ] && [ -e /dev/cam_end ] && [ -e /dev/cam_front ] \
    || say "!! 15s 内设备未齐(ttyUSB0/cam_end/cam_front), 仍尝试启动(main.py 报错会留在日志)"
sleep 3   # udev 别名/桌面稳定余量

# 日志超 10MB 滚动(放启动前: 跑完 tail 看到的就是本次运行日志)
if [ -f "$LOG" ] && [ "$(stat -c%s "$LOG")" -gt 10485760 ]; then
    mv "$LOG" "$LOG.old"
fi

say "--- main.py --no-check 启动(每次开机仅此一遍) ---"
PYTHONUNBUFFERED=1 python3 main.py --no-check >> "$LOG" 2>&1
rc=$?
# 单次运行定案(2026-11-06 用户两次反馈后定): 不看退出码、不重启——
# ① rc 分流不可靠: main.py 退出时 cv2/TRT 线程 abort 核心转储(实锤过) rc≠0 但结果无损;
# ② 用户要"跑完即歇"。再跑一遍 = reboot 或手动 cd ~/main-copy && python3 main.py --no-check
say "--- main.py 退出 rc=$rc, 单次运行结束, 不自动重启 ---"
