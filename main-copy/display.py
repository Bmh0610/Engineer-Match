"""比赛用状态显示(tkinter): Nano HDMI 物理屏幕直显, 无网无浏览器依赖。

屏幕内容**只有任务码**(2026-11-05 用户定: 只能是数字和加号, 别的无任何)——
四组码 2+2 拆两行(上排第1+2组/下排第3+4组, 下行带 + 前缀), 大字号满足
规则"字高≥12mm"要求。无 DISPLAY 环境(SSH 纯终端)自动降级为 None。

可退出: Esc/q 关窗——只关显示不影响主进程(set_* 关窗后自动 no-op)。

用法:
    from display import create_display
    disp = create_display()          # 无 DISPLAY 时返回 None
    disp.set_task("213+321+312+231") # 屏幕只刷新任务码
    ...
    disp.root.update()               # 每次操作后调, 刷新 GUI 事件
"""
import tkinter as tk

BG = '#000000'
FG = '#ffffff'


class StatusDisplay:
    """全屏任务码窗: 屏幕上只有数字和加号(2+2 两行); Esc/q 可退不影响进程。"""

    def __init__(self, title="智能搬运"):
        self._alive = True
        self.root = tk.Tk()
        self.root.title(title)
        self.root.attributes('-fullscreen', True)
        self.root.attributes('-topmost', True)
        self.root.configure(bg=BG)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        self._code = tk.Label(self.root, text="",
                              font=('monospace', 96, 'bold'),
                              fg=FG, bg=BG, justify='center')
        self._code.grid(row=0, column=0, sticky='nsew')

        # 可退出: Esc/q 关窗(只关显示, 主进程照跑); 屏幕上无任何提示文字
        for key in ('<Escape>', '<q>'):
            self.root.bind(key, lambda e: self.close())
        self.root.update()

    def set_task(self, code, batch1=None, batch2=None):
        """扫码后设任务码(屏幕只显示数字和加号; batch 参数忽略, 兼容调用方)。
        四组码 2+2 拆两行(下行带 + 前缀); 2 组简写整行。"""
        groups = code.split("+")
        if len(groups) >= 4:
            shown = "+".join(groups[:2]) + "\n+" + "+".join(groups[2:])
        else:
            shown = code
        self._code.config(text=shown)
        self.root.update()

    def set_step(self, text):
        """兼容接口: 屏幕只有任务码, 步骤不再显示(no-op)。"""
        pass

    def set_stats(self, text):
        """兼容接口: 屏幕只有任务码, 统计不再显示(no-op)。"""
        pass

    def close(self):
        """关窗(只关显示): 主进程后续 set_* 调用自动 no-op。"""
        self._alive = False
        try:
            self.root.destroy()
        except tk.TclError:
            pass


def create_display(title="智能搬运"):
    """创建任务码全屏窗; 无 DISPLAY 环境返回 None(调用方静默跳过)。"""
    try:
        return StatusDisplay(title)
    except Exception:
        return None
