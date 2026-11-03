"""比赛用状态显示(tkinter): Nano HDMI 物理屏幕直显, 无网无浏览器依赖。

规则硬性要求(工创赛): 车身上部/亮光/字高≥12mm/不遮挡/显示任务码+完成统计。
单窗口垂直布局(2026-11-04 用户澄清: "2+2"指四组任务码上排两组/下排两组,
不是四个格子——曾误解成 2×2 四格已回退); 大字体满足字高要求。
可退出(2026-11-04 用户定): 按 Esc/q 或点 × 关窗——**只关显示不影响主进程**
(set_* 调用在关窗后自动变 no-op); 全屏置顶会盖住 NoMachine 终端, 退窗即见终端。
无 DISPLAY 环境(SSH 纯终端)自动降级为 None, 调用方静默跳过。

用法:
    from display import StatusDisplay
    disp = StatusDisplay()           # 无 DISPLAY 时返回 None
    disp.set_task("213+321", "黄→红→蓝", "红→蓝→黄")
    disp.set_step("导航→物料盘")
    disp.set_stats("批1抓料 3/3")
    ...
    disp.root.update()               # 每次操作后调, 刷新 GUI 事件
"""
import tkinter as tk

BG = '#0a0a1a'
FG_DIM = '#8892b0'
FG_MAIN = '#e0e0e0'
FG_ACCENT = '#64ffda'
FG_WARN = '#ff6b6b'


class StatusDisplay:
    """2×2 状态窗口: 任务码+批次信息 / 当前步骤+完成统计; Esc/q 可退不影响进程。"""

    def __init__(self, title="智能搬运"):
        self._alive = True
        self.root = tk.Tk()
        self.root.title(title)
        self.root.attributes('-fullscreen', True)
        self.root.attributes('-topmost', True)
        self.root.configure(bg=BG)

        self._title = tk.Label(self.root, text=title
                               + "    (Esc/q 退显示, 不影响运行)",
                               font=('monospace', 18, 'bold'),
                               fg=FG_ACCENT, bg=BG)
        self._title.pack(pady=(40, 5))

        # 任务码: 四组码 2+2 拆行(上排两组/下排两组, 第二行带 + 前缀), 简写整行
        self._code = tk.Label(self.root, text="待扫码...",
                              font=('monospace', 64, 'bold'),
                              fg='#ffffff', bg=BG, justify='center')
        self._code.pack(pady=15)

        self._batch = tk.Label(self.root, text="",
                               font=('monospace', 26),
                               fg=FG_DIM, bg=BG, justify='left')
        self._batch.pack(pady=8)

        sep = tk.Frame(self.root, bg=FG_ACCENT, height=3, width=600)
        sep.pack(pady=10)

        self._step = tk.Label(self.root, text="待机",
                              font=('monospace', 40, 'bold'),
                              fg=FG_WARN, bg=BG)
        self._step.pack(pady=25)

        self._stats = tk.Label(self.root, text="",
                               font=('monospace', 30, 'bold'),
                               fg=FG_ACCENT, bg=BG)
        self._stats.pack(pady=15)

        # 可退出: Esc/q/× 关窗(只关显示, 主进程照跑)
        for key in ('<Escape>', '<q>'):
            self.root.bind(key, lambda e: self.close())
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.update()

    def _guard(fn):
        """关窗后 set_*/update 调用变 no-op(不影响主进程)。"""
        def wrap(self, *a, **k):
            if not self._alive:
                return
            try:
                return fn(self, *a, **k)
            except tk.TclError:
                self._alive = False
        return wrap

    @_guard
    def set_task(self, code, batch1, batch2):
        """扫码后设任务信息: 四组码 2+2 拆两行(上排第1+2组/下排第3+4组,
        下行带 + 前缀); 2 组简写整行。batch1/batch2=色序+环号文字。"""
        groups = code.split("+")
        if len(groups) >= 4:
            shown = "+".join(groups[:2]) + "\n+" + "+".join(groups[2:])
        else:
            shown = code
        self._code.config(text=shown)
        self._batch.config(text=batch1 + "\n" + batch2)
        self.root.update()

    @_guard
    def set_step(self, text):
        """更新当前步骤显示。"""
        self._step.config(text=text)
        self.root.update()

    @_guard
    def set_stats(self, text):
        """更新完成统计。"""
        self._stats.config(text=text)
        self.root.update()

    def close(self):
        """关窗(只关显示): 主进程后续 set_* 调用自动 no-op。"""
        self._alive = False
        try:
            self.root.destroy()
        except tk.TclError:
            pass


def create_display(title="智能搬运"):
    """创建状态窗口; 无 DISPLAY 环境返回 None(调用方静默跳过)。"""
    try:
        return StatusDisplay(title)
    except Exception:
        return None
