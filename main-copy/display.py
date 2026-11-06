"""比赛用状态显示(tkinter): Nano HDMI 物理屏幕直显, 无网无浏览器依赖。

屏幕内容**只有任务码**(2026-11-05 用户定: 只能是数字和加号, 别的无任何)——
四组码 2+2 拆两行(上排第1+2组/下排第3+4组, 下行带 + 前缀), 大字号满足
规则"字高≥12mm"要求。无 DISPLAY 环境(SSH 纯终端)自动降级为 None。

可退出: Esc/q 关窗——只关显示不影响主进程(set_* 关窗后自动 no-op)。
2026-11-06 重构(用户报 Esc 无效): Tk 挪进独立显示线程跑 mainloop——旧版靠
调用方"每次操作后 update()"泵事件, main.py 实际从不调 → 事件循环从未运行,
按键事件堆队列无人处理(顺带修掉窗口只渲染一次的冻结); 主线程经指令队列
投递更新(所有 tk 调用都在显示线程, 无 Tcl 跨线程问题), focus_force 抢键盘焦点。
已知噪声: 进程退出时可能仍见 Tcl_AsyncDelete 核心转储(Tcl 终结化在主线程执行
的 CPython 限制, 与 cv2/TRT 退出噪声同类)——发生在全部结果输出之后, 无损。

用法:
    from display import create_display
    disp = create_display()          # 无 DISPLAY 时返回 None
    disp.set_task("213+321+312+231") # 屏幕只刷新任务码(线程安全, 随时可调)
    ...
    disp.close()                     # 程序退出前可选; Esc/q 同效
"""
import os
import queue
import threading
import tkinter as tk

BG = '#000000'
FG = '#ffffff'


class StatusDisplay:
    """全屏任务码窗(独立显示线程): 屏幕上只有数字和加号(2+2 两行);
    Esc/q 可退不影响进程; set_* 线程安全随时可调。"""

    def __init__(self, title="智能搬运"):
        self._alive = True
        self._q = queue.Queue()          # 指令队列: ("task", text) / ("close",)
        self._ready = threading.Event()  # 显示线程建窗完成(或失败)
        self._t = threading.Thread(target=self._run, args=(title,), daemon=True)
        self._t.start()

    def _run(self, title):
        """显示线程主体: 建窗 → 50ms 轮询指令队列 → mainloop(常驻泵事件)。"""
        try:
            self.root = tk.Tk()
        except Exception:
            self._alive = False
            self._ready.set()
            return
        self.root.title(title)
        self.root.attributes('-fullscreen', True)
        self.root.attributes('-topmost', True)
        self.root.configure(bg=BG)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        self._code = tk.Label(self.root, text="",
                              font=('monospace', 64, 'bold'),   # ≈25mm 字高(2026-11-05 用户定 140 太大)
                              fg=FG, bg=BG, justify='center')
        self._code.grid(row=0, column=0, sticky='nsew')

        # 可退出: Esc/q 关窗(只关显示, 主进程照跑); bind_all+focus_force——
        # 全屏置顶窗从 autostart/SSH 拉起时键盘焦点常留在别的窗口, 不抢焦点收不到键
        for key in ('<Escape>', '<q>'):
            self.root.bind_all(key, lambda e: self._close_in_thread())
        self.root.focus_force()

        self._ready.set()
        self._poll()
        self.root.mainloop()

    def _poll(self):
        """50ms 轮询指令队列(跨线程安全: 所有 tk 调用都在显示线程内)。"""
        try:
            while True:
                op = self._q.get_nowait()
                if op[0] == "task":
                    self._code.config(text=op[1])
                elif op[0] == "close":
                    self._close_in_thread()
                    return   # 已销毁, 不再续轮询
        except queue.Empty:
            pass
        if self._alive:
            self.root.after(50, self._poll)

    def _close_in_thread(self):
        """销毁窗(仅显示线程内调): mainloop 随之退出, 线程结束。
        root 引用就地置 None——Tk 在本线程创建也要在本线程回收, 否则主线程
        退出时 GC 会踩 Tcl_AsyncDelete: async handler deleted by the wrong thread。"""
        self._alive = False
        try:
            self.root.destroy()
        except tk.TclError:
            pass
        self.root = None

    def set_task(self, code, batch1=None, batch2=None):
        """扫码后设任务码(屏幕只显示数字和加号; batch 参数忽略, 兼容调用方)。
        四组码 2+2 拆两行(下行带 + 前缀); 2 组简写整行。关窗后 no-op。"""
        if not self._alive:
            return
        groups = code.split("+")
        if len(groups) >= 4:
            shown = "+".join(groups[:2]) + "\n+" + "+".join(groups[2:])
        else:
            shown = code
        self._q.put(("task", shown))

    def set_step(self, text):
        """兼容接口: 屏幕只有任务码, 步骤不再显示(no-op)。"""
        pass

    def set_stats(self, text):
        """兼容接口: 屏幕只有任务码, 统计不再显示(no-op)。"""
        pass

    def close(self):
        """请求关窗(线程安全, 走指令队列): 只关显示, 主进程照跑。
        join 显示线程(≤2s)——主进程退出前收干净, 防 Tcl 退出 abort。"""
        if self._alive:
            self._q.put(("close",))
        self._t.join(timeout=2.0)


def create_display(title="智能搬运"):
    """创建任务码全屏窗; 无 DISPLAY 或建窗失败返回 None(调用方静默跳过)。"""
    if not os.environ.get("DISPLAY"):
        return None
    try:
        d = StatusDisplay(title)
        if not d._ready.wait(timeout=3.0) or not d._alive:
            return None   # 建窗失败/超时(线程内已置 _alive=False)
        return d
    except Exception:
        return None
