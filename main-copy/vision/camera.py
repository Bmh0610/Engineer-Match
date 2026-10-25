"""末端相机: 采集线程常开, 后台维护"最新帧"(移植自 main_match/vision/cameras.py)。

纠偏是"看-动-再看": 车动后的旧帧没有价值, 消费方一律 read_latest() 拿最新一帧。
曝光锁定: 自动曝光/白平衡是色漂主因(亮蓝→浅蓝, 亮绿→黄), 联调时按
"最亮物料直方图不贴 255"标定曝光值, 填 EXPOSURE 并把 LOCK_EXPOSURE
改 True —— 两件事一起做, 只锁不调值会更糟。
"""
import threading
import time

import cv2

W, H = 1280, 720
LOCK_EXPOSURE = False   # 联调标定后改 True(同时填 EXPOSURE)
EXPOSURE = None         # 锁定曝光值; V4L2 单位常为 100μs 级


class Camera:
    def __init__(self, dev, name=""):
        self.dev, self.name = dev, name
        self.cap = None
        self._frame = None
        self._lock = threading.Lock()
        self._thread = None

    def open(self):
        self.cap = cv2.VideoCapture(self.dev)
        if not self.cap.isOpened():   # 不存在/被占用/权限不足时显式报错, 别静默黑屏
            raise RuntimeError(
                f"相机 /dev/video{self.dev} 打不开: 设备不存在/被占用/权限不足"
                f" (ls /dev/video* 核对编号; 被占用时关掉别的程序; 刚插 USB 等几秒)")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
        if LOCK_EXPOSURE:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)   # V4L2: 1=手动, 3=自动
            self.cap.set(cv2.CAP_PROP_EXPOSURE, EXPOSURE)
            self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        self._thread = threading.Thread(target=self._grab_loop, daemon=True)
        self._thread.start()

    def _grab_loop(self):
        while True:
            cap = self.cap
            if cap is None:
                break
            ok, frame = cap.read()
            if ok:
                with self._lock:
                    self._frame = frame

    def read_latest(self, timeout=2.0):
        """取最新一帧; 刚 open 未出图或超时返回 None。"""
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
