"""比赛用全屏状态显示(tkinter): Nano HDMI 物理屏幕直显, 无网无浏览器依赖。

规则硬性要求(工创赛): 车身上部/亮光/字高≥12mm/不遮挡/显示任务码+完成统计。
本窗口全屏置顶暗底亮字, 大字体满足字高要求; 全程不关闭不最小化。
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
    """全屏状态窗口: 任务码+批次信息+当前步骤+完成统计。"""

    def __init__(self, title="智能搬运"):
        self.root = tk.Tk()
        self.root.title(title)
        self.root.attributes('-fullscreen', True)
        self.root.attributes('-topmost', True)
        self.root.configure(bg=BG)
        self.root.protocol("WM_DELETE_WINDOW", lambda: None)  # 禁关

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        frame = tk.Frame(self.root, bg=BG)
        frame.grid(row=0, column=0, sticky='nsew')

        self._title = tk.Label(frame, text=title,
                               font=('monospace', 22, 'bold'),
                               fg=FG_ACCENT, bg=BG)
        self._title.pack(pady=(50, 5))

        self._code = tk.Label(frame, text="待扫码...",
                              font=('monospace', 72, 'bold'),
                              fg='#ffffff', bg=BG)
        self._code.pack(pady=15)

        self._batch = tk.Label(frame, text="",
                               font=('monospace', 26),
                               fg=FG_DIM, bg=BG, justify='left')
        self._batch.pack(pady=8)

        sep = tk.Frame(frame, bg=FG_ACCENT, height=3, width=600)
        sep.pack(pady=10)

        self._step = tk.Label(frame, text="待机",
                              font=('monospace', 40, 'bold'),
                              fg=FG_WARN, bg=BG)
        self._step.pack(pady=25)

        self._stats = tk.Label(frame, text="",
                               font=('monospace', 30, 'bold'),
                               fg=FG_ACCENT, bg=BG)
        self._stats.pack(pady=15)

        self.root.update()

    def set_task(self, code, batch1, batch2):
        """扫码后设任务信息: code=原文, batch1/batch2=色序+环号文字。"""
        self._code.config(text=code)
        self._batch.config(text=batch1 + "\n" + batch2)
        self.root.update()

    def set_step(self, text):
        """更新当前步骤显示。"""
        self._step.config(text=text)
        self.root.update()

    def set_stats(self, text):
        """更新完成统计。"""
        self._stats.config(text=text)
        self.root.update()

    def close(self):
        self.root.destroy()


def create_display(title="智能搬运"):
    """创建全屏状态窗口; 无 DISPLAY 环境返回 None(调用方静默跳过)。"""
    try:
        return StatusDisplay(title)
    except Exception:
        return None
