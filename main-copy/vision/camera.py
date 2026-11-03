"""末端相机: 采集线程常开, 后台维护"最新帧"(移植自 main_match/vision/cameras.py)。

纠偏是"看-动-再看": 车动后的旧帧没有价值, 消费方一律 read_latest() 拿最新一帧。
摄像头编号持久化: dev 可传 udev 别名("/dev/cam_scan"), open 时解析成
当前实际编号再打开(V4L2 后端只认数字)——重插后 udev 更新指向, 解析自动跟随。
曝光锁定: 自动曝光/白平衡是色漂主因(亮蓝→浅蓝, 亮绿→黄), 联调时按
"最亮物料直方图不贴 255"标定曝光值, 填 EXPOSURE 并把 LOCK_EXPOSURE
改 True —— 两件事一起做, 只锁不调值会更糟。

自愈(2026-11-04, 实车暴露): 行车振动后 USB 冻结=cap.read() 连续失败或
连续吐同一帧(冻帧)——旧版无限空转永不恢复(批2 全瞎 30~60s 根因)。现版:
连续失败/冻帧/全黑 达阈值即 release+重开(udev 别名重新解析), 重开期间
_frame 置 None(消费方 read_latest=None → 三态纠偏/等色走等待, 不拿旧帧动作)。
"""
import threading
import time

import cv2

W, H = 1280, 720
LOCK_EXPOSURE = False   # 联调标定后改 True(同时填 EXPOSURE)
EXPOSURE = None         # 锁定曝光值; V4L2 单位常为 100μs 级

READ_FAIL_LIMIT = 5     # cap.read() 连续失败 N 次 → 重开
FROZEN_LIMIT = 15       # 连续 N 帧逐字节相同(≈0.5s) → 疑设备挂起, 重开
BLACK_MEAN = 3.0        # 帧均值低于此=全黑(遮蔽/异常), 连续 FROZEN_LIMIT 帧同样重开
REOPEN_TRIES = 3        # 重开尝试次数(间隔 1s); 失败则采集停摆待人工


class Camera:
    def __init__(self, dev, name=""):
        self.dev, self.name = dev, name
        self.cap = None
        self._frame = None
        self._lock = threading.Lock()
        self._thread = None

    def _resolve_dev(self):
        """udev 别名 → 当前实际节点号(每次重开都重新解析, 跟随 udev 更新)。"""
        dev = self.dev
        if isinstance(dev, str) and not dev.isdigit():
            import os
            import re
            real = os.path.realpath(dev)
            m = re.search(r"video(\d+)$", real)
            if not m:
                raise RuntimeError(f"相机 {dev} 解析不到 video 节点: {real}")
            dev = int(m.group(1))
        return int(dev)

    def _open_capture(self):
        dev = self._resolve_dev()
        cap = cv2.VideoCapture(dev)
        if not cap.isOpened():   # 不存在/被占用/权限不足时显式报错, 别静默黑屏
            raise RuntimeError(
                f"相机 {self.dev} 打不开: 设备不存在/被占用/权限不足"
                f" (ls /dev/video* 核对编号; 被占用时关掉别的程序; 刚插 USB 等几秒)")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
        if LOCK_EXPOSURE:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)   # V4L2: 1=手动, 3=自动
            cap.set(cv2.CAP_PROP_EXPOSURE, EXPOSURE)
            cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        return cap

    def open(self):
        self.cap = self._open_capture()
        self._thread = threading.Thread(target=self._grab_loop, daemon=True)
        self._thread.start()

    def _reopen(self):
        """自愈重开: release 旧句柄 → 重新解析/打开(udev 别名跟随), 至多 3 次。
        期间 _frame 置 None——消费方 read_latest 拿 None 走等待, 不拿旧帧动作。"""
        with self._lock:
            self._frame = None
            cap, self.cap = self.cap, None
        try:
            if cap:
                cap.release()
        except Exception:
            pass
        for attempt in range(1, REOPEN_TRIES + 1):
            time.sleep(1.0)
            try:
                cap = self._open_capture()
            except Exception as e:
                print(f"[相机{self.name}] 重开失败({attempt}/{REOPEN_TRIES}): {e}")
                continue
            with self._lock:
                self.cap = cap
            print(f"[相机{self.name}] 重开成功(第{attempt}次)")
            return True
        print(f"[相机{self.name}] 重开 {REOPEN_TRIES} 次失败——采集停摆, 需人工检查 USB")
        return False

    def _grab_once(self, state):
        """采集一轮; state=dict(fail, last_bytes, same, black)。返回 False=线程应退出。"""
        cap = self.cap
        if cap is None:
            return False
        ok, frame = cap.read()
        if ok and frame is not None:
            state["fail"] = 0
            fb = frame.tobytes()
            if fb == state["last_bytes"]:
                state["same"] += 1
            else:
                state["same"] = 0
                state["last_bytes"] = fb
            black = float(frame.mean()) < BLACK_MEAN
            state["black"] = state["black"] + 1 if black else 0
            if state["same"] >= FROZEN_LIMIT:
                print(f"[相机{self.name}] 疑似冻帧(连续 {state['same']} 帧逐字节相同), 重开采集...")
                self._reopen()
                state.update(last_bytes=None, same=0, black=0)
            elif state["black"] >= FROZEN_LIMIT:
                print(f"[相机{self.name}] 连续全黑帧 ×{state['black']}, 重开采集...")
                self._reopen()
                state.update(last_bytes=None, same=0, black=0)
            else:
                with self._lock:
                    self._frame = frame
        else:
            state["fail"] += 1
            state.update(last_bytes=None, same=0, black=0)
            if state["fail"] >= READ_FAIL_LIMIT:
                print(f"[相机{self.name}] 采集连续失败 ×{state['fail']}, 重开采集...")
                self._reopen()
                state["fail"] = 0
            else:
                time.sleep(0.05)   # 失败时别空转烧 CPU
        return True

    def _grab_loop(self):
        state = {"fail": 0, "last_bytes": None, "same": 0, "black": 0}
        while self._grab_once(state):
            pass

    def read_latest(self, timeout=2.0):
        """取最新一帧; 刚 open 未出图/重开期间/超时返回 None(消费方走等待)。"""
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                if self._frame is not None:
                    return self._frame
            time.sleep(0.01)
        return None

    def close(self):
        cap, self.cap = self.cap, None
        if self._thread:
            self._thread.join(timeout=1)
        if cap:
            cap.release()
