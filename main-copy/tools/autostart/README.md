# Nano 开机自启 main.py --no-check（2026-11-06）

比赛现场流程：整车上电 → 自动登录进桌面 → 本目录 `paotu.desktop` 自动拉起
`run_paotu.sh` → 等设备就绪(≤15s) → `main.py --no-check` 停在 Step0 等一键启动
→ HDMI 屏显示任务窗。全程不用 SSH。

## 已安装位置（Nano 板上）

| 文件 | 位置 |
|---|---|
| 自启入口 | `~/.config/autostart/paotu.desktop` |
| 包装脚本 | `~/main-copy/tools/autostart/run_paotu.sh` |
| 运行日志 | `~/main_boot.log`（超 10MB 滚动 .old） |
| GDM 自动登录 | `/etc/gdm3/custom.conf`（备份 `custom.conf.bak-paotu`） |

## 现场速查

```bash
# 看程序跑到哪了
tail -f ~/main_boot.log

# 停止(调试/手动跑测试前必做——自启实例占着 ttyUSB0): 先杀循环再杀主程序
touch ~/STOP_PAOTU; pkill -f run_paotu.sh; pkill -f "main.py --no-check"

# 恢复自启(下次开机生效)
rm ~/STOP_PAOTU
```

## 彻底拆除

```bash
rm ~/.config/autostart/paotu.desktop
sudo cp /etc/gdm3/custom.conf.bak-paotu /etc/gdm3/custom.conf
```

## 注意

- **比赛务必接 HDMI 屏再上电**：无屏时 GNOME 会话可能起不来，自启不生效。
- 跑的是 `~/main-copy`（逐色等色版主线）；要用复件版改 `run_paotu.sh` 里 `DIR=` 一行。
- **每次开机只跑一遍**（2026-11-06 用户定终版）：跑完/崩溃都不自动重启（不看退出码——main.py 退出时 cv2/TRT abort 核心转储 rc≠0 但结果无损，按 rc 分流不可靠）。再跑一遍 = `sudo reboot` 或手动 `cd ~/main-copy && python3 main.py --no-check`。
- ⚠ 固件一键启动(0x2201/0x2202)未上车前，程序会一直停在等启动（属预期）。
