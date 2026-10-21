"""末端相机内参标定结果的载入与使用(张正友法, tools/calib_intr.py 生成)。

vision/calib_end.yaml: fx, fy, cx, cy, dist[k1,k2,p1,p2,k3], rms。用途:
  ① principal_point 主点替代"画面中心=光轴"假设(当前系统偏差源)
  ② undistort_point 检测点去畸变(边缘目标几像素→几毫米修正)
     —— YOLO 输入帧保持原图(同训练分布), 只修检测出的点
无标定文件时 load() 返回 None, 前端自动回退旧逻辑, 不破坏现状。
内参是镜头属性, 桌面标即可、装机前后通用; 重调焦/换镜头才需重标。
"""
import math
from pathlib import Path

import cv2
import numpy as np
import yaml

CALIB_DIR = Path(__file__).resolve().parent / "vision"


class Calib:
    _oob_warned = False   # 外推区告警只打一次

    def __init__(self, data):
        self.fx, self.fy = float(data["fx"]), float(data["fy"])
        self.cx, self.cy = float(data["cx"]), float(data["cy"])
        self.dist = np.array(data["dist"], np.float64)
        self.rms = float(data["rms"])
        self._K = np.array([[self.fx, 0, self.cx],
                            [0, self.fy, self.cy],
                            [0, 0, 1]], np.float64)

    def principal_point(self, w=0, h=0):
        """标定主点 (cx, cy); 参数仅为兼容无标定回退的调用签名。"""
        return self.cx, self.cy

    def undistort_point(self, x, y):
        """像素 → 无畸变像素(观测坐标 → 理想坐标)。

        注: 本机 OpenCV 5.0 实测 cv2.undistortPoints 对稀疏点恒等返回
        (最小用例复现, 畸变系数未被应用), 故自实现:
        本组参数 f(r²)=1+k1r²+k2r⁴+k3r⁶ 严格单调(判别式<0) → 逆唯一,
        沿入射射线对"畸变半径"二分(单调 100% 收敛, 大半径处不动点/牛顿
        迭代都会过冲发散, 已实测), 再二维牛顿精修 2 次消切向分量。
        """
        k1, k2, p1, p2, k3 = self.dist
        xn = (x - self.cx) / self.fx
        yn = (y - self.cy) / self.fy
        rd = math.hypot(xn, yn)
        if rd < 1e-9:
            return float(x), float(y)
        ux, uy = xn / rd, yn / rd

        def along(s):
            """射线上的理想点 (s·ux, s·uy) 经畸变后, 在射线方向的半径。"""
            r2 = s * s
            f = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 ** 3
            tx = 2 * p1 * (s * ux) * (s * uy) + p2 * (r2 + 2 * (s * ux) ** 2)
            ty = p1 * (r2 + 2 * (s * uy) ** 2) + 2 * p2 * (s * ux) * (s * uy)
            return s * f + tx * ux + ty * uy

        lo, hi = 0.0, 0.25
        prev = along(hi)
        while hi < 2.5:   # 动态探测 along(s) 单调段上端: 超出即标定外推区, 无解
            nxt = hi + 0.05
            v = along(nxt)
            if v <= prev:
                break
            prev, hi = v, nxt
        if rd > prev:
            if not Calib._oob_warned:
                Calib._oob_warned = True
                print(f"!! 去畸变: 点 ({x:.0f},{y:.0f}) 超出标定模型覆盖半径, "
                      f"原样返回(不外推); 边缘目标多时建议补 30-45° 倾斜图重标内参 !!")
            return float(x), float(y)
        for _ in range(60):   # along 单调递增段内, 二分必收敛
            mid = (lo + hi) / 2
            if along(mid) < rd:
                lo = mid
            else:
                hi = mid
        s = (lo + hi) / 2
        xu, yu = s * ux, s * uy

        def F(a, b):   # 全量残差(含切向垂直分量), 供牛顿精修
            r2 = a * a + b * b
            f = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 ** 3
            return (a * f + 2 * p1 * a * b + p2 * (r2 + 2 * a * a) - xn,
                    b * f + p1 * (r2 + 2 * b * b) + 2 * p2 * a * b - yn)

        for _ in range(2):   # 初值已在真解邻域, 2 次牛顿到 <1e-6 px
            fx, fy = F(xu, yu)
            e = 1e-7
            ax, ay = F(xu + e, yu)
            bx, by = F(xu, yu + e)
            j11, j12 = (ax - fx) / e, (ay - fy) / e
            j21, j22 = (bx - fx) / e, (by - fy) / e
            det = j11 * j22 - j12 * j21
            if abs(det) < 1e-14:
                break
            xu -= (j22 * fx - j12 * fy) / det
            yu -= (j11 * fy - j21 * fx) / det
        return xu * self.fx + self.cx, yu * self.fy + self.cy


def load(path=None):
    """读 calib_end.yaml; 文件不存在返回 None(前端回退画面中心)。"""
    p = Path(path) if path else CALIB_DIR / "calib_end.yaml"
    if not p.exists():
        return None
    return Calib(yaml.safe_load(p.read_text(encoding="utf-8")))
