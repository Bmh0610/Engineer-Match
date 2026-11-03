"""二维码扫码与任务码解析(检测核心移植自 gcxl2025-main, 解析按 2027 规则)。

任务码 = 4 组 3 位, "+" 连接, 如 156+123+516+231:
  第1组 颜色(1红2黄3蓝4绿5黑6浅蓝) = 第一批抓取顺序
  第2组 环号(1-5) = 第一批在粗加工区和暂存区的放置位置(两区同号)
  第3组 颜色 = 第二批抓取顺序
  第4组 环号 = 第二批在粗加工区的放置位置
暂存区码垛(只发生在暂存区): 批2颜色X 叠在批1同色X 的暂存区环号上
(每色全场2个; 粗加工区只平放, 第一批放完即搬走)。
格式非法一律返回 None(gcxl 同策略, 调用方兜底)。
"""
import os
import re
import time

import cv2

# 任务码数字 → 颜色名(与 config.CLS2COLOR 的值同名, 下游按名匹配)
RULE_DIGIT2COLOR = {"1": "red", "2": "yellow", "3": "blue",
                    "4": "green", "5": "black", "6": "light_blue"}

TASK_PATTERN = re.compile(r"^([1-6]{3})\+([1-5]{3})\+([1-6]{3})\+([1-5]{3})$")


def parse_task(raw):
    """任务码 → 解析字典; 格式非法返回 None(gcxl _validate_mission_code 同款)。"""
    m = TASK_PATTERN.match(raw.strip()) if raw else None
    if not m:
        return None
    g1, g2, g3, g4 = m.groups()
    b1c = [RULE_DIGIT2COLOR[d] for d in g1]
    b2c = [RULE_DIGIT2COLOR[d] for d in g3]
    return {"raw": raw.strip(),
            "batch1_colors": b1c, "batch1_rings": [int(d) for d in g2],
            "batch2_colors": b2c, "batch2_rings": [int(d) for d in g4],
            "stack_plan": dict(zip(b1c, [int(d) for d in g2]))}


SIMPLE_PATTERN = re.compile(r"^([1-6]{3})\+([1-5]{3})$")   # 2组简写: 色序+环号


def parse_test_code(raw):
    """测试码解析: 4 组完整码走 parse_task; 2 组简写(色序+环号, 只跑第一批)
    补 batch2 空占位; 格式非法返回 None。"""
    task = parse_task(raw)
    if task:
        return task
    m = SIMPLE_PATTERN.match(raw.strip()) if raw else None
    if not m:
        return None
    colors = [RULE_DIGIT2COLOR[d] for d in m.group(1)]
    rings = [int(d) for d in m.group(2)]
    return {"raw": raw.strip(),
            "batch1_colors": colors, "batch1_rings": rings,
            "batch2_colors": [], "batch2_rings": [],
            "stack_plan": dict(zip(colors, rings))}


def draw_dets(frame, dets, ok_data=None):
    """检测框+解码内容叠加(调试预览用: 合法任务码绿框, 非法码黄框)。"""
    for data, (x, y, w, h) in dets:
        color = (0, 255, 0) if data == ok_data else (0, 255, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, data[:40], (x, max(y - 8, 16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame


class QrScanner:
    """cv2 二维码检测(核心移植自 gcxl detect.py:692-800)。"""

    # 小码救回: 1×检不出时逐级放大再检(640×480下6.5cm码~97px, 低于~100px检测极限)
    SCALES = (1.0, 1.3, 2.0)

    def __init__(self):
        self.detector = cv2.QRCodeDetector()

    def _detect_scaled(self, img, scale):
        """单尺度检测, 框坐标按 1/scale 映射回原帧。"""
        ret_val, decoded_info, points, _ = self.detector.detectAndDecodeMulti(img)
        out = []
        if ret_val:
            for i, data in enumerate(decoded_info):
                if data:
                    pts = points[i]
                    x1, y1 = pts[:, 0].min() / scale, pts[:, 1].min() / scale
                    x2, y2 = pts[:, 0].max() / scale, pts[:, 1].max() / scale
                    out.append((data, (int(x1), int(y1),
                                       int(x2 - x1), int(y2 - y1))))
        return out

    def detect(self, frame):
        """一帧 → [(data, box), ...]; 解码失败返回空表。box=(x,y,w,h) 像素。
        逐级放大: 原帧1× → 1.3× → 2×, 小码(黑白格低于检测极限)靠放大救回。"""
        for scale in self.SCALES:
            if scale == 1.0:
                dets = self._detect_scaled(frame, 1.0)
            else:
                h, w = frame.shape[:2]
                big = cv2.resize(frame, (int(w * scale), int(h * scale)),
                                 interpolation=cv2.INTER_CUBIC)
                dets = self._detect_scaled(big, scale)
            if dets:
                return dets
        return []

    def scan_from_camera(self, cam, timeout=5.0, poll=0.05, preview=False):
        """循环取帧扫码, 首个合法任务码 → parse_task 结果; 超时返回 None。

        cam: Camera(read_latest 接口); 不开窗口(比赛无显示依赖)。
        preview=True 开调试窗口: 实时画面+检测框(黄=检出/绿=合法任务码),
        无 DISPLAY 自动降级纯终端; 按 q 可提前放弃扫码(返回 None)。
        """
        show = preview and bool(os.environ.get("DISPLAY"))
        if preview and not show:
            print("  !! 无 DISPLAY, 扫码预览不可用(走纯终端日志)")
        deadline = time.time() + timeout
        invalid_seen = set()
        try:
            while time.time() < deadline:
                frame = cam.read_latest(timeout=0.2)
                if frame is None:
                    continue
                dets = self.detect(frame)
                if show and not dets:
                    cv2.imshow("qr_scan", frame)
                for data, _box in dets:
                    parsed = parse_task(data)
                    if parsed:
                        if show:
                            cv2.imshow("qr_scan", draw_dets(frame, dets, data))
                            cv2.waitKey(800)   # 停一拍让你看到绿框
                        return parsed
                    if data not in invalid_seen:
                        invalid_seen.add(data)
                        print(f"  !! 检出二维码但格式非法: {data!r} —— 继续等合法码")
                if show and dets:
                    cv2.imshow("qr_scan", draw_dets(frame, dets, None))
                if show and cv2.waitKey(1) & 0xFF == ord("q"):
                    print("  (q: 放弃扫码)")
                    return None
                time.sleep(poll)
            return None
        finally:
            if show:
                cv2.destroyAllWindows()
