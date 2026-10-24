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


class QrScanner:
    """cv2 二维码检测(核心移植自 gcxl detect.py:692-800)。"""

    def __init__(self):
        self.detector = cv2.QRCodeDetector()

    def detect(self, frame):
        """一帧 → [(data, box), ...]; 解码失败返回空表。box=(x,y,w,h) 像素。"""
        ret_val, decoded_info, points, _ = self.detector.detectAndDecodeMulti(frame)
        out = []
        if ret_val:
            for i, data in enumerate(decoded_info):
                if data:
                    pts = points[i]
                    x1, y1 = pts[:, 0].min(), pts[:, 1].min()
                    x2, y2 = pts[:, 0].max(), pts[:, 1].max()
                    out.append((data, (int(x1), int(y1),
                                       int(x2 - x1), int(y2 - y1))))
        return out

    def scan_from_camera(self, cam, timeout=5.0, poll=0.05):
        """循环取帧扫码, 首个合法任务码 → parse_task 结果; 超时返回 None。

        cam: Camera(read_latest 接口); 不开窗口(比赛无显示依赖)。
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            frame = cam.read_latest(timeout=0.2)
            if frame is None:
                continue
            for data, _box in self.detect(frame):
                parsed = parse_task(data)
                if parsed:
                    return parsed
            time.sleep(poll)
        return None
