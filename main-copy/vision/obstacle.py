"""障碍物前端: 节点问一次(主路径) + 行驶监听(保险), 共用三道防线。

黑对黑防线(数据集仅 17 张同框, 部署端必须兜):
  ① 阶段门控: 只开类 0, 物料 6 类不出结果(反之工作点亦然) —— 主力
  ② 车道带校验: 联调标定的"车道在画面里的可见带"(近低远高一条带),
     判"挡没挡路"而非"像不像障碍" —— 最弱, 标不好可关(LANE_BAND=None)
  ③ 时序确认: 连续 N 帧才认, 单帧看走眼不报警 —— 主力
距离 = 像素高度→mm 标定表插值, 仅写日志(封边只需"有/无")。
消费接口对齐 navigator.block_and_replan(由调度层接 on_obstacle 回调)。
"""
import threading
import time

OBSTACLE_CLS = [0]
WATCH_CONF = 0.5        # 帧内检出门限
N_CONFIRM = 3           # 连续 N 帧才算(行驶监听)
CHECK_FRAMES = 5        # check_edge 取帧数
CHECK_NEED = 3          # check_edge 达标帧数
FRAME_GAP = 0.05

# 车道带(归一化): 框底边 y2/H 须落在带内; 近处障碍→带底, 远处→带上。
# 联调按前部相机倾角实测标定; 标不好置 None 关闭这道防线。
LANE_BAND = (0.4, 1.0)

# 距离标定表 [(框高px, 距离mm)]: 卷尺量几点填表, 线性插值。仅日志用。
DIST_TABLE = [(30, 1000), (60, 500), (120, 250)]


def est_dist(h_px):
    """框高 px → 距离 mm, 段内线性插值, 段外取端点。仅日志用。"""
    pts = sorted(DIST_TABLE)
    if h_px <= pts[0][0]:
        return pts[0][1]
    if h_px >= pts[-1][0]:
        return pts[-1][1]
    for (h1, d1), (h2, d2) in zip(pts, pts[1:]):
        if h1 <= h_px <= h2:
            return d1 + (d2 - d1) * (h_px - h1) / (h2 - h1)


def _gate(dets, frame_shape):
    """一帧的类 0 检出 → 通过帧内门限(conf + 车道带)的那条, 无则 None。"""
    h = frame_shape[0]
    for d in dets:
        if d.conf < WATCH_CONF:
            continue
        if LANE_BAND:
            y2 = d.xyxy[3] / h
            if not LANE_BAND[0] <= y2 <= LANE_BAND[1]:
                continue
        return d
    return None


class ObstacleWatcher:
    def __init__(self, backend, cam):
        self.backend, self.cam = backend, cam
        self._hit = 0
        self._alarm = False
        self._stop = threading.Event()
        self._thread = None
        self.on_obstacle = None   # 调度层塞: fn(det, dist_mm) — 停车/倒回/封边/重规划

    def _look(self):
        f = self.cam.read_latest()
        if f is None:
            return None
        return _gate(self.backend.infer(f, classes=OBSTACLE_CLS), f.shape)

    def check_edge(self, n_frames=CHECK_FRAMES, need=CHECK_NEED):
        """节点转向对准下一条边后问一次: n 帧里 ≥need 帧疑似 → True(有障碍)。
        重规划转向后再问一次即覆盖"再次检测"。"""
        hits = 0
        for _ in range(n_frames):
            if self._look():
                hits += 1
            time.sleep(FRAME_GAP)
        return hits >= need

    def start_watch(self, on_obstacle):
        """行驶中持续监听。连续 N_CONFIRM 帧命中 → 调一次 on_obstacle(det, dist_mm),
        之后进入"已报警"状态不再重复报, 调度处理完调 reset() 复位。"""
        self.on_obstacle = on_obstacle
        self.reset()
        self._stop.clear()
        self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()

    def _watch_loop(self):
        while not self._stop.is_set():
            d = self._look()
            if d and not self._alarm:
                self._hit += 1
                if self._hit >= N_CONFIRM:
                    self._alarm = True
                    h_px = d.xyxy[3] - d.xyxy[1]
                    self.on_obstacle(d, est_dist(h_px))
            elif not d:
                self._hit = 0
            time.sleep(FRAME_GAP)

    def reset(self):
        """重规划/转向后调用: 清计数与已报警状态。"""
        self._hit = 0
        self._alarm = False

    def stop_watch(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)
