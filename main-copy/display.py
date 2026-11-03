"""比赛用状态显示(tkinter): Nano HDMI 物理屏幕直显, 无网无浏览器依赖。

规则硬性要求(工创赛): 车身上部/亮光/字高≥12mm/不遮挡/显示任务码+完成统计。
布局 2×2(2026-11-04 用户定): 上行=任务码+批次信息, 下行=当前步骤+完成统计,
大字体满足字高要求。
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

        top = tk.Frame(self.root, bg=BG)
        top.pack(fill='x', pady=(8, 0))
        tk.Label(top, text=title, font=('monospace', 18, 'bold'),
                 fg=FG_ACCENT, bg=BG).pack(side='left', padx=20)
        tk.Label(top, text="Esc/q 退显示(不影响运行)", font=('monospace', 12),
                 fg=FG_DIM, bg=BG).pack(side='right', padx=20)

        # 2×2 网格: 上=任务码|批次, 下=当前步骤|统计
        grid = tk.Frame(self.root, bg=BG)
        grid.pack(expand=True, fill='both')
        for c in (0, 1):
            grid.columnconfigure(c, weight=1, uniform='cell')
        for r in (0, 1):
            grid.rowconfigure(r, weight=1)

        def cell(r, c, cap):
            box = tk.Frame(grid, bg=BG, highlightthickness=2,
                           highlightbackground='#1e2a3a')
            box.grid(row=r, column=c, sticky='nsew', padx=6, pady=6)
            tk.Label(box, text=cap, font=('monospace', 16),
                     fg=FG_DIM, bg=BG).pack(pady=(14, 2))
            return box

        box_code = cell(0, 0, "任务码")
        self._code = tk.Label(box_code, text="待扫码...",
                              font=('monospace', 54, 'bold'),
                              fg='#ffffff', bg=BG, justify='center')
        self._code.pack(expand=True)

        box_batch = cell(0, 1, "批次 (色序 | 环号)")
        self._batch = tk.Label(box_batch, text="待扫码...",
                               font=('monospace', 24),
                               fg=FG_MAIN, bg=BG, justify='left')
        self._batch.pack(expand=True)

        box_step = cell(1, 0, "当前步骤")
        self._step = tk.Label(box_step, text="待机",
                              font=('monospace', 36, 'bold'),
                              fg=FG_WARN, bg=BG, wraplength=500)
        self._step.pack(expand=True)

        box_stats = cell(1, 1, "完成统计")
        self._stats = tk.Label(box_stats, text="",
                               font=('monospace', 28, 'bold'),
                               fg=FG_ACCENT, bg=BG, justify='left')
        self._stats.pack(expand=True)

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
        """扫码后设任务信息: code=原文(任务码分两行, 4组码按+拆两行/2组简写整行),
        batch1/batch2=色序+环号文字。"""
        if "+" in code:
            line1, line2 = code.split("+", 1)
            shown = f"{line1}\n+{line2}"
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
