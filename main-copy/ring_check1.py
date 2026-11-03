#!/usr/bin/env python3
"""离线验证同色数字圆环: 检测圆环几何中心并按内部数字识别 1/2/3。

目标 = 地面上的黑线虚线同心圆(中心印数字 1/2/3, 大小随距离变化)。
路线: Lab 软件补光 → 自适应阈值取黑线 → 闭运算桥接虚线 → 轮廓椭圆拟合 + 弧覆盖率过滤
→ 同心椭圆分组(>=min_rings 条嵌套椭圆 = 一个圆环标记, 尺寸天然自适应)
→ 提取中心数字轮廓，以严格 IoU + 形态学容差轮廓匹配判定 1/2/3。
椭圆细边线思路参考 src/vision.py fit_ring_ellipse_center 与旧版 finder.py。
不依赖 YOLO 模型、不开相机、不开串口。
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}

# 圆环尺寸随距离变化, 覆盖远近两档(原图 1280x720 实测): 远~30px, 近~280px
MAJOR_LO, MAJOR_HI = 30, 280          # 椭圆长轴范围
AXIS_RATIO_LO = 0.30                  # 短轴/长轴下限(透视压扁不超过此值)
MARKER_AXIS_RATIO_LO = 0.60           # 成组后的真实圆环下限；过滤文字/鞋边长弧
GROUP_DIST = 70.0                     # 同心判定: 椭圆中心距离阈值(px)
MIN_RINGS = 1                         # 数字复核会排除假圆，保留远处单轮廓圆环
COVER_LO, COVER_HI = 0.55, 1.35       # 轮廓弧长/椭圆周长比: 太小=断弧拟合,中心漂移
MORPH_KERNEL = 5                      # 闭运算核(桥接虚线断口, 过大会粘连相邻环线)
ADAPT_BLOCK = 51                      # 自适应阈值邻域(须为奇数, 覆盖环线间距)
ADAPT_C = 10                          # 自适应阈值偏移: 低于邻域均值该值的像素算黑线
DIGIT_AREA_MIN = 18                   # 数字轮廓最小面积(远处数字也要保留)
ANGLE_TOL = 30.0                      # 地面标记主轴允许偏离水平的角度


def parse_args():
    parser = argparse.ArgumentParser(
        description="离线检测并标注圆环标记",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=("示例: python3 tools/ring_check1.py --interactive\n"
                "      python3 tools/ring_check1.py 图片目录 --output 结果目录 "
                "--save-relight --digit-min-confidence 0.30"))
    parser.add_argument("inputs", nargs="*", type=Path,
                        default=[ROOT / "photos" / "down_camera_19700101_080710"],
                        help="图片文件或目录, 默认使用080710原始验证帧")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "ring_check1_results",
                        help="标注图输出目录, 默认 new_gc/ring_check1_results")
    parser.add_argument("--min-rings", type=int, default=MIN_RINGS,
                        help="同心椭圆至少几条才算圆环标记")
    parser.add_argument("--major-lo", type=float, default=MAJOR_LO)
    parser.add_argument("--major-hi", type=float, default=MAJOR_HI)
    parser.add_argument("--show-all", action="store_true",
                        help="额外画出未成组的单条椭圆(灰), 便于调参")
    parser.add_argument("--no-relight", action="store_true",
                        help="关闭 Lab 亮度场补偿(默认开启软件补光)")
    parser.add_argument("--save-relight", action="store_true",
                        help="把软件补光图保存到输出目录的 relight/ 子目录")
    parser.add_argument("--keep-unknown", action="store_true",
                        help="保留没有读出数字的圆形候选(默认只输出1/2/3)")
    parser.add_argument("--no-order-recovery", action="store_true",
                        help="关闭三圆环按空间排列恢复1/2/3身份(默认开启)")
    parser.add_argument("--interactive", action="store_true",
                        help="进入交互模式，逐项输入测试路径和检测参数")
    parser.add_argument("--axis-ratio-lo", type=float, default=AXIS_RATIO_LO,
                        help="单条候选椭圆的短轴/长轴下限")
    parser.add_argument("--marker-axis-ratio-lo", type=float,
                        default=MARKER_AXIS_RATIO_LO,
                        help="成组圆环的短轴/长轴下限")
    parser.add_argument("--group-dist", type=float, default=GROUP_DIST,
                        help="同心轮廓允许的最大中心距离(px)")
    parser.add_argument("--cover-lo", type=float, default=COVER_LO,
                        help="轮廓弧长/拟合椭圆周长下限")
    parser.add_argument("--cover-hi", type=float, default=COVER_HI,
                        help="轮廓弧长/拟合椭圆周长上限")
    parser.add_argument("--morph-kernel", type=int, default=MORPH_KERNEL,
                        help="闭运算核尺寸，必须为正奇数")
    parser.add_argument("--adapt-block", type=int, default=ADAPT_BLOCK,
                        help="自适应阈值邻域，必须为大于1的奇数")
    parser.add_argument("--adapt-c", type=float, default=ADAPT_C,
                        help="自适应阈值常数C")
    parser.add_argument("--digit-min-confidence", type=float, default=0.0,
                        help="数字最低置信度，低于该值按unknown处理")
    parser.add_argument("--soft-match-min", type=float,
                        default=SOFT_MATCH_MIN,
                        help="容差轮廓匹配最低得分")
    parser.add_argument("--soft-match-margin", type=float,
                        default=SOFT_MATCH_MARGIN,
                        help="容差匹配第一、第二名的最小分差")
    parser.add_argument("--jpeg-quality", type=int, default=95,
                        help="输出JPEG质量(1-100)")
    return parser.parse_args()


def _prompt_value(label, current, cast):
    """交互读取单个值；直接回车保留当前值。"""
    while True:
        raw = input(f"{label} [{current}]: ").strip()
        if not raw:
            return current
        try:
            return cast(raw)
        except (TypeError, ValueError):
            print(f"输入无效，请输入 {cast.__name__} 类型的值")


def _prompt_bool(label, current):
    """交互读取是/否；直接回车保留当前值。"""
    default_text = "Y" if current else "N"
    while True:
        raw = input(f"{label} (y/n) [{default_text}]: ").strip().lower()
        if not raw:
            return current
        if raw in {"y", "yes", "1", "是"}:
            return True
        if raw in {"n", "no", "0", "否"}:
            return False
        print("输入无效，请输入 y 或 n")


def configure_interactive(args):
    """让用户在终端输入测试集和常用检测参数。"""
    print("\n=== ring_check1 交互参数 ===")
    default_inputs = ",".join(str(path) for path in args.inputs)
    raw_inputs = input(
        f"测试图片/目录，多个路径用逗号分隔 [{default_inputs}]: ").strip()
    if raw_inputs:
        args.inputs = [Path(item.strip()) for item in raw_inputs.split(",")
                       if item.strip()]
    args.output = Path(_prompt_value("输出目录", str(args.output), str))
    args.min_rings = _prompt_value("最少同心轮廓数", args.min_rings, int)
    args.major_lo = _prompt_value("圆环长轴下限(px)", args.major_lo, float)
    args.major_hi = _prompt_value("圆环长轴上限(px)", args.major_hi, float)
    args.axis_ratio_lo = _prompt_value(
        "候选椭圆短轴/长轴下限", args.axis_ratio_lo, float)
    args.marker_axis_ratio_lo = _prompt_value(
        "成组圆环短轴/长轴下限", args.marker_axis_ratio_lo, float)
    args.group_dist = _prompt_value(
        "同心中心距离(px)", args.group_dist, float)
    args.cover_lo = _prompt_value("轮廓覆盖率下限", args.cover_lo, float)
    args.cover_hi = _prompt_value("轮廓覆盖率上限", args.cover_hi, float)
    args.morph_kernel = _prompt_value(
        "闭运算核尺寸(奇数)", args.morph_kernel, int)
    args.adapt_block = _prompt_value(
        "自适应阈值邻域(奇数)", args.adapt_block, int)
    args.adapt_c = _prompt_value("自适应阈值C", args.adapt_c, float)
    args.digit_min_confidence = _prompt_value(
        "数字最低置信度", args.digit_min_confidence, float)
    args.soft_match_min = _prompt_value(
        "容差轮廓匹配最低得分", args.soft_match_min, float)
    args.soft_match_margin = _prompt_value(
        "容差匹配最小分差", args.soft_match_margin, float)
    relight_enabled = _prompt_bool("启用软件补光", not args.no_relight)
    args.no_relight = not relight_enabled
    args.save_relight = _prompt_bool("保存补光图片", args.save_relight)
    args.keep_unknown = _prompt_bool("保留unknown候选", args.keep_unknown)
    order_recovery = _prompt_bool(
        "启用三圆环1-2-3排列恢复", not args.no_order_recovery)
    args.no_order_recovery = not order_recovery
    args.show_all = _prompt_bool("绘制所有候选椭圆", args.show_all)
    args.jpeg_quality = _prompt_value(
        "输出JPEG质量", args.jpeg_quality, int)
    print("=== 参数输入完成 ===\n")
    return args


def validate_args(args):
    """提前拒绝会导致 OpenCV 异常或无意义结果的参数。"""
    if not args.inputs:
        raise ValueError("至少需要一个输入图片或目录")
    if args.min_rings < 1:
        raise ValueError("--min-rings 必须 >= 1")
    if not 0 < args.major_lo < args.major_hi:
        raise ValueError("圆环尺寸必须满足 0 < major-lo < major-hi")
    for name, value in (("axis-ratio-lo", args.axis_ratio_lo),
                        ("marker-axis-ratio-lo", args.marker_axis_ratio_lo)):
        if not 0 < value <= 1:
            raise ValueError(f"--{name} 必须在 (0, 1] 范围")
    if args.group_dist <= 0:
        raise ValueError("--group-dist 必须 > 0")
    if not 0 < args.cover_lo < args.cover_hi:
        raise ValueError("覆盖率必须满足 0 < cover-lo < cover-hi")
    if args.morph_kernel < 1 or args.morph_kernel % 2 == 0:
        raise ValueError("--morph-kernel 必须为正奇数")
    if args.adapt_block <= 1 or args.adapt_block % 2 == 0:
        raise ValueError("--adapt-block 必须为大于1的奇数")
    if not 0 <= args.digit_min_confidence <= 1:
        raise ValueError("--digit-min-confidence 必须在 [0, 1] 范围")
    if not 0 <= args.soft_match_min <= 1:
        raise ValueError("--soft-match-min 必须在 [0, 1] 范围")
    if not 0 <= args.soft_match_margin <= 1:
        raise ValueError("--soft-match-margin 必须在 [0, 1] 范围")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality 必须在 [1, 100] 范围")


def software_relight(frame):
    """只校正 Lab 亮度通道, 均衡阴影且不改变物体颜色。"""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lightness, channel_a, channel_b = cv2.split(lab)
    sigma = max(15.0, min(frame.shape[:2]) * 0.06)
    illumination = cv2.GaussianBlur(lightness, (0, 0), sigmaX=sigma,
                                    sigmaY=sigma)
    reference = float(np.median(illumination))
    corrected = np.clip(lightness.astype(np.float32)
                        - illumination.astype(np.float32) + reference,
                        0, 255).astype(np.uint8)
    corrected = cv2.createCLAHE(clipLimit=2.0,
                                tileGridSize=(8, 8)).apply(corrected)
    return cv2.cvtColor(cv2.merge((corrected, channel_a, channel_b)),
                        cv2.COLOR_LAB2BGR)


def discover_images(inputs):
    images = []
    for raw in inputs:
        path = raw.expanduser().resolve()
        if path.is_file():
            images.append(path)
        elif path.is_dir():
            images.extend(sorted(
                p for p in path.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
            ))
        else:
            raise FileNotFoundError(f"输入不存在: {path}")
    if not images:
        raise ValueError("没有找到可处理的图片")
    return images


def fit_ellipses(frame, major_lo, major_hi, axis_ratio_lo=AXIS_RATIO_LO,
                 cover_lo=COVER_LO, cover_hi=COVER_HI,
                 morph_kernel=MORPH_KERNEL, adapt_block=ADAPT_BLOCK,
                 adapt_c=ADAPT_C):
    # 兼容旧接口名称：实际使用轮廓面积矩中心，不拟合椭圆。
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    dark=cv2.adaptiveThreshold(cv2.GaussianBlur(gray,(3,3),0),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,adapt_block,adapt_c)
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(morph_kernel,morph_kernel))
    closed=cv2.morphologyEx(dark,cv2.MORPH_CLOSE,kernel)
    contours,_=cv2.findContours(closed,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    height,width=frame.shape[:2];out=[]
    for contour in contours:
        if len(contour)<20:continue
        x,y,w,h=cv2.boundingRect(contour)
        major,minor=max(w,h),min(w,h)
        if not major_lo<=major<=major_hi or minor/major<max(axis_ratio_lo,0.60):continue
        if x<=0 or y<=0 or x+w>=width or y+h>=height:continue
        area=cv2.contourArea(contour);perimeter=cv2.arcLength(contour,True)
        if area<=0 or perimeter<=0:continue
        if 4*np.pi*area/(perimeter*perimeter)<0.55 or not 0.55<=area/(w*h)<=0.90:continue
        hull_area=cv2.contourArea(cv2.convexHull(contour))
        if hull_area<=0 or area/hull_area<0.85:continue
        moments=cv2.moments(contour)
        if moments['m00']==0:continue
        cx=moments['m10']/moments['m00'];cy=moments['m01']/moments['m00']
        if abs(cx-(x+(w-1)/2))>major*0.08 or abs(cy-(y+(h-1)/2))>major*0.08:continue
        # 包围框宽高仅用于尺寸筛选与显示，非拟合轴长。
        out.append(((cx,cy),(float(major),float(minor),0.0 if w>=h else 90.0),area,len(contour)))
    return out


def _robust_ring_center(ring_items, outer_major):
    """由多层同心轮廓估计中心，剔除残弧造成的偏心拟合。

    轮廓点数大致随周长增加，直接用点数加权会让最外层残缺弧主导中心。
    以最大外圈为中心基准做离群剔除，再用 点数/长轴 加权，使每一层圆环
    的权重大致相同。这样不会把远处数字2/3的横向轮廓误当成内环中心。
    只剩单层轮廓时退化为外圈椭圆中心。
    """
    centers = np.array([item[0] for item in ring_items], dtype=float)
    majors = np.array([item[1][0] for item in ring_items], dtype=float)
    point_counts = np.array([item[3] for item in ring_items], dtype=float)
    outer_index = int(np.argmax(majors))
    outer_center = centers[outer_index]
    distances = np.linalg.norm(centers - outer_center, axis=1)
    inliers = distances <= max(4.0, outer_major * 0.08)
    if np.count_nonzero(inliers) >= 1:
        centers = centers[inliers]
        majors = majors[inliers]
        point_counts = point_counts[inliers]
    weights = point_counts / np.maximum(majors, 1.0)
    return tuple(np.average(centers, axis=0, weights=weights))


def group_concentric(ellipses, min_rings, group_dist=GROUP_DIST):
    """把中心聚集的椭圆分组成圆环标记。

    返回 [(中心, 外圈椭圆, 内圈椭圆, 候选数, 支持点数)]; 数字小轮廓
    可以帮助形成候选组, 但中心只由尺寸不小于外圈 45% 的圆环轮廓计算。
    """
    remaining = sorted(ellipses, key=lambda e: -e[1][0])   # 外圈优先
    groups = []
    while remaining:
        head = remaining.pop(0)
        cx, cy = head[0]
        group = [head]
        rest = []
        for item in remaining:
            ix, iy = item[0]
            if (ix - cx) ** 2 + (iy - cy) ** 2 <= group_dist ** 2:
                group.append(item)
            else:
                rest.append(item)
        remaining = rest
        outer_major = max(e[1][0] for e in group)
        ring_items = [e for e in group if e[1][0] >= outer_major * 0.45]
        center = _robust_ring_center(ring_items, outer_major)
        outer = max(ring_items, key=lambda e: e[1][0])[1]
        groups.append([center, outer, group])

    merged = []
    for group in sorted(groups, key=lambda g: -g[1][0]):
        cx, cy = group[0]
        absorbed = False
        for target in merged:
            tx, ty = target[0]
            gap = ((tx - cx) ** 2 + (ty - cy) ** 2) ** 0.5
            if gap <= max(target[1][0], group[1][0]) * 0.45:
                target[2].extend(group[2])
                outer_major = max(e[1][0] for e in target[2])
                ring_items = [e for e in target[2]
                              if e[1][0] >= outer_major * 0.45]
                target[0] = _robust_ring_center(ring_items, outer_major)
                target[1] = max(ring_items, key=lambda e: e[1][0])[1]
                absorbed = True
                break
        if not absorbed:
            merged.append(group)

    markers = []
    for center, outer, items in merged:
        if len(items) < min_rings:
            continue
        outer_major = max(e[1][0] for e in items)
        ring_items = [e for e in items if e[1][0] >= outer_major * 0.45]
        center = _robust_ring_center(ring_items, outer_major)
        outer = max(ring_items, key=lambda e: e[1][0])[1]
        inner = min((e[1] for e in ring_items), key=lambda axes: axes[0])
        support_points = sum(e[3] for e in ring_items)
        markers.append((center, outer, inner, len(items), support_points))
    return markers


# ---- 数字识别: 实拍样本做模板(严格IoU + 形态学容差轮廓匹配) ----
# 模板取自本视频实拍数字(打印字体固定, 真样本远比 Hershey 通用字体准):
# 二值化→最大竖长轮廓→外接框裁剪并缩放到统一尺寸。

DIGIT_SIZE = (48, 72)                 # 归一化模板/待测尺寸(宽,高)
IOU_GOOD = 0.60                       # IoU 达到该值采信，否则做容差轮廓匹配
SOFT_MATCH_MIN = 0.78                 # 允许1像素轮廓偏移后的最低匹配度
SOFT_MATCH_MARGIN = 0.04              # 最优数字必须比次优数字至少高此值
EDGE_TOL = 0.30                       # 外圈椭圆越界超过半轴该比例 = 贴边假环剔除


def _load_sample_templates():
    """加载实拍模板，并补充多种 OpenCV 字体以适配粗细字体变化。"""
    template_dir = Path(__file__).resolve().parent / "ring_templates1"
    templates = {digit: [] for digit in "123"}
    if template_dir.is_dir():
        for digit in "123":
            for path in sorted(template_dir.glob(f"{digit}_*.png")):
                img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    templates[digit].append(img)
    # 实拍集是粗体字；three_circle 使用细线衬线字。始终加入多字体、多线宽
    # 合成样本，避免模板只适用于一次拍摄中的一种打印字体。
    fonts = (cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX,
             cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_TRIPLEX,
             cv2.FONT_HERSHEY_COMPLEX_SMALL)
    for digit in "123":
        for font in fonts:
            for thickness in (1, 2, 3):
                canvas = np.zeros((160, 160), np.uint8)
                scale = 4.0 if font != cv2.FONT_HERSHEY_COMPLEX_SMALL else 5.0
                (tw, th), _ = cv2.getTextSize(digit, font, scale, thickness)
                origin = ((160 - tw) // 2, (160 + th) // 2)
                cv2.putText(canvas, digit, origin, font, scale, 255,
                            thickness, cv2.LINE_AA)
                templates[digit].append(canvas)
    return templates


_TEMPLATES = _load_sample_templates()


def _normalize_digit(binary):
    """选中心竖长轮廓并等比例归一化; 无效返回 None。"""
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    height, width = binary.shape
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        if area < DIGIT_AREA_MIN or h < height * 0.30 or h == 0:
            continue
        if w / h > 0.72:              # 椭圆环线/横向残段不是数字
            continue
        fill_ratio = area / max(w * h, 1)
        if fill_ratio > 0.68:          # 闭合椭圆的包围面积明显大于数字笔画
            continue
        if x <= 0 or x + w >= width:
            continue                  # 左右边界通常是环线/背景连通块
        center_error = abs((x + w / 2) - width / 2) / max(width, 1)
        candidates.append(((1.0 - min(center_error, 0.9)) * 1000 + h, contour))
    if not candidates:
        return None
    contour = max(candidates, key=lambda item: item[0])[1]
    x, y, w, h = cv2.boundingRect(contour)
    digit_roi = binary[y:y + h, x:x + w]
    canvas = np.zeros((DIGIT_SIZE[1], DIGIT_SIZE[0]), np.uint8)
    scale = min((DIGIT_SIZE[0] - 8) / w, (DIGIT_SIZE[1] - 8) / h)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(digit_roi, (new_w, new_h),
                         interpolation=cv2.INTER_AREA)
    left, top = (DIGIT_SIZE[0] - new_w) // 2, (DIGIT_SIZE[1] - new_h) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas, (x, y, w, h)


def _templates_normalized():
    out = {digit: [] for digit in "123"}
    for digit, images in _TEMPLATES.items():
        for img in images:
            _, binary = cv2.threshold(img, 0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            item = _normalize_digit(binary)
            if item is not None:
                out[digit].append(item[0])
    return out


_TEMPLATES_NORM = _templates_normalized()


DIGIT_WINDOW_RADIUS = 0.24            # 中心正方形半径/外圈短轴，支持任意旋转


def _soft_shape_similarity(norm, template):
    """参考形态学梯度思路，允许轮廓有1像素位移或笔画粗细变化。"""
    foreground = (norm > 127).astype(np.uint8)
    template_foreground = (template > 127).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    foreground_near = cv2.dilate(foreground, kernel) > 0
    template_near = cv2.dilate(template_foreground, kernel) > 0
    foreground = foreground > 0
    template_foreground = template_foreground > 0
    precision = np.logical_and(foreground, template_near).sum()
    precision /= max(foreground.sum(), 1)
    recall = np.logical_and(template_foreground, foreground_near).sum()
    recall /= max(template_foreground.sum(), 1)
    return float(2 * precision * recall / max(precision + recall, 1e-9))


def _score_digit(norm, soft_match_min=SOFT_MATCH_MIN,
                 soft_match_margin=SOFT_MATCH_MARGIN):
    """严格IoU优先，失败后使用有轮廓容差的匹配，返回数字和置信度。"""
    scores = {}
    for digit, templates in _TEMPLATES_NORM.items():
        similarities = []
        for tmpl in templates:
            inter = np.logical_and(norm > 127, tmpl > 127).sum()
            union = np.logical_or(norm > 127, tmpl > 127).sum()
            similarities.append(inter / max(union, 1))
        if similarities:
            scores[digit] = max(similarities)
    if not scores:
        return None, 0.0
    best_iou = max(scores, key=scores.get)
    if scores[best_iou] >= IOU_GOOD:
        return best_iou, float(scores[best_iou])

    soft_scores = {}
    for digit, templates in _TEMPLATES_NORM.items():
        if templates:
            soft_scores[digit] = max(
                _soft_shape_similarity(norm, template)
                for template in templates)
    if len(soft_scores) < 2:
        return None, 0.0
    ranking = sorted(soft_scores, key=soft_scores.get, reverse=True)
    best_digit, second_digit = ranking[:2]
    best_score = soft_scores[best_digit]
    margin = best_score - soft_scores[second_digit]
    if best_score < soft_match_min or margin < soft_match_margin:
        return None, 0.0
    return best_digit, float(best_score)


def _classify_digit_window(frame, enhance, center, outer, axes,
                           allow_rotations, soft_match_min,
                           soft_match_margin):
    """在一个指定大小的中心窗内提取并分类数字。"""
    (cx, cy), (_, _, outer_angle) = center, outer
    mask = np.zeros(frame.shape[:2], np.uint8)
    # 数字窗按图像坐标放置，不跟随外圈椭圆主轴旋转。旧相机中的圆环
    # 因透视呈竖椭圆，但数字仍保持竖直；近圆形窗口则旋转与否都等价。
    mask_angle = 0 if allow_rotations else (outer_angle - 90.0) % 180.0
    cv2.ellipse(mask, (int(cx), int(cy)), axes, mask_angle,
                0, 360, 255, -1)
    x0, x1 = max(0, int(cx - axes[0])), min(frame.shape[1], int(cx + axes[0]))
    y0, y1 = max(0, int(cy - axes[1])), min(frame.shape[0], int(cy + axes[1]))
    inside = enhance[y0:y1, x0:x1].copy()
    inside_mask = mask[y0:y1, x0:x1]
    if inside.size == 0:
        return None, None, 0.0
    # 掩膜外必须保持白背景。旧代码 bitwise_and 后为 0，再做 BINARY_INV
    # 会变成巨大的白色前景，导致模板和识别对象都混入两侧环线。
    inside[inside_mask == 0] = 255
    _, dark = cv2.threshold(inside, 0, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dark[inside_mask == 0] = 0
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    # 数字可能随相机/图片方向旋转。逐一尝试 0/90/180/270 度，选择
    # 模板得分最高的方向；圆环中心仍使用原图轮廓，不受数字旋转影响。
    rotations = ((dark, cv2.rotate(dark, cv2.ROTATE_90_CLOCKWISE),
                  cv2.rotate(dark, cv2.ROTATE_180),
                  cv2.rotate(dark, cv2.ROTATE_90_COUNTERCLOCKWISE))
                 if allow_rotations else (dark,))
    best = None
    for rotation_index, candidate in enumerate(rotations):
        item = _normalize_digit(candidate)
        if item is None:
            continue
        norm, rect = item
        digit, confidence = _score_digit(
            norm, soft_match_min=soft_match_min,
            soft_match_margin=soft_match_margin)
        if digit is None:
            continue
        if best is None or confidence > best[1]:
            best = (digit, confidence, rotation_index, rect)
    if best is None:
        return None, None, 0.0
    digit, confidence, rotation_index, (x, y, w, h) = best
    if rotation_index == 0:
        box = (x0 + x, y0 + y, w, h)
    else:
        # 旋转后的外接框不直接对应原图坐标，标出实际参与识别的中心窗。
        box = (x0, y0, x1 - x0, y1 - y0)
    return digit, box, round(confidence, 3)


def classify_digit(frame, center, outer, soft_match_min=SOFT_MATCH_MIN,
                   soft_match_margin=SOFT_MATCH_MARGIN):
    """识别圆环内部数字；数字仅分类身份，不参与圆环中心计算。"""
    (_, _), (major, minor, _) = center, outer
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    enhance = cv2.createCLAHE(clipLimit=2.5,
                              tileGridSize=(8, 8)).apply(gray)
    near_circle = minor / major >= 0.85
    if near_circle:
        # three_circle 的数字方向可能旋转，用中心正方形避免裁掉数字两端。
        half_window = max(int(minor * DIGIT_WINDOW_RADIUS), 8)
        primary = (half_window, half_window)
        fallback_half = max(half_window, 32)
        fallback = (fallback_half, fallback_half)
    else:
        # 原 50 张透视椭圆使用窄高窗，保持原算法的有效提取范围。
        primary = (max(int(major / 2 * 0.36), 6),
                   max(int(minor / 2 * 0.80), 8))
        fallback = (max(primary[0], 32), max(primary[1], 48))
    result = _classify_digit_window(frame, enhance, center, outer, primary,
                                    allow_rotations=near_circle,
                                    soft_match_min=soft_match_min,
                                    soft_match_margin=soft_match_margin)
    if result[0] is not None:
        return result
    # 外圈只检出一部分时估计尺寸偏小，用较大的最小窗再尝试一次。
    if fallback != primary:
        return _classify_digit_window(frame, enhance, center, outer, fallback,
                                      allow_rotations=near_circle,
                                      soft_match_min=soft_match_min,
                                      soft_match_margin=soft_match_margin)
    return result


def _deduplicate_identities(results):
    """每个编号只保留一个最佳候选，unknown 候选原样保留。"""
    best_by_digit = {}
    unknown = []
    for result in results:
        digit = result[5]
        if digit is None:
            unknown.append(result)
            continue
        quality = (result[7], result[3], result[4])
        current = best_by_digit.get(digit)
        if current is None or quality > (current[7], current[3], current[4]):
            best_by_digit[digit] = result
    return list(best_by_digit.values()) + unknown


def _recover_weak_identity_by_spacing(results):
    """用两个可靠相邻编号预测第三个，仅接纳预测点附近的弱圆环候选。"""
    known = {item[5]: item for item in results if item[5] is not None}
    if len(known) != 2:
        return results, 0
    missing = ({"1", "2", "3"} - set(known)).pop()
    if missing == "1" and "2" in known and "3" in known:
        expected = 2 * np.array(known["2"][0]) - np.array(known["3"][0])
        reference = (known["2"], known["3"])
    elif missing == "2" and "1" in known and "3" in known:
        expected = (np.array(known["1"][0])
                    + np.array(known["3"][0])) / 2.0
        reference = (known["1"], known["3"])
    elif missing == "3" and "1" in known and "2" in known:
        expected = 2 * np.array(known["2"][0]) - np.array(known["1"][0])
        reference = (known["1"], known["2"])
    else:
        return results, 0

    reference_size = float(np.median([item[1][0] for item in reference]))
    weak = []
    for item in results:
        if item[5] is not None or item[3] < 3 or item[4] < 300:
            continue
        size_ratio = item[1][0] / max(reference_size, 1.0)
        distance = float(np.linalg.norm(np.array(item[0]) - expected))
        # 正常尺寸候选已由三环规则处理；这里只接纳因低对比度仅剩内层的候选。
        if (0.40 <= size_ratio <= 0.80
                and distance <= reference_size * 0.35):
            weak.append((distance, -item[3], -item[4], item))
    if not weak:
        return results, 0
    item = min(weak, key=lambda candidate: candidate[:3])[3]
    replacement = item[:5] + (missing, None, 0.0)
    return [replacement if candidate is item else candidate
            for candidate in results], 1


def _recover_ordered_identities(results, min_major):
    """从三个可靠圆环几何候选的空间顺序恢复 1/2/3。

    仅当候选尺寸接近、近似共线、相邻间距相近，并且不会与置信度较高
    的数字模板结果冲突时才启用。数字方向/纹理不参与这个兜底步骤。
    返回 (更新后的结果, 本次补回的身份数)。
    """
    if {item[5] for item in results if item[5]} == {"1", "2", "3"}:
        return results, 0
    eligible = [
        item for item in results
        if item[1][0] >= max(60.0, min_major)
        and (item[3] >= 3 or item[4] >= 300)
    ]
    if len(eligible) < 3:
        return results, 0

    strong_known = [item for item in results
                    if item[5] is not None and item[7] >= 0.60]
    best = None
    for combination in itertools.combinations(eligible, 3):
        points = np.array([item[0] for item in combination], dtype=float)
        dominant_axis = int(np.argmax(np.ptp(points, axis=0)))
        ordered = sorted(combination,
                         key=lambda item: item[0][dominant_axis])
        ordered_points = np.array([item[0] for item in ordered], dtype=float)
        sizes = np.array([item[1][0] for item in ordered], dtype=float)
        median_size = float(np.median(sizes))
        gaps = np.diff(ordered_points[:, dominant_axis])
        other_axis = 1 - dominant_axis
        middle_residual = abs(
            ordered_points[1, other_axis]
            - (ordered_points[0, other_axis]
               + ordered_points[2, other_axis]) / 2.0)

        # 贴近画面右边缘时外圈可能只提取出较内层椭圆，长轴会低估约30%。
        if sizes.max() / max(sizes.min(), 1.0) > 1.45:
            continue
        if (gaps.min() < median_size * 0.50
                or gaps.max() > median_size * 2.50
                or gaps.max() / max(gaps.min(), 1.0) > 1.70):
            continue
        if middle_residual > median_size * 0.40:
            continue
        if any(item not in combination for item in strong_known):
            continue
        if any(item[5] is not None and item[5] != str(index)
               and item[7] >= 0.60
               for index, item in enumerate(ordered, 1)):
            continue

        matched_digits = sum(item[5] == str(index)
                             for index, item in enumerate(ordered, 1))
        score = (middle_residual / max(median_size, 1.0)
                 + np.std(sizes) / max(median_size, 1.0)
                 + (gaps.max() / max(gaps.min(), 1.0) - 1.0) * 0.25
                 - matched_digits * 0.18
                 - sum(min(item[3], 5) for item in ordered) * 0.01)
        if best is None or score < best[0]:
            best = (float(score), ordered)

    # 无数字可用时要求几何得分也足够好，避免把三个背景圆弧强行编号。
    if best is None or best[0] > 0.32:
        return _recover_weak_identity_by_spacing(results)

    ordered = best[1]
    replacements = {}
    recovered = 0
    for index, item in enumerate(ordered, 1):
        expected = str(index)
        if item[5] != expected:
            # confidence=0 表示身份来自几何顺序，不冒充数字模板置信度。
            replacements[id(item)] = item[:5] + (expected, None, 0.0)
            recovered += 1
    updated = [replacements.get(id(item), item) for item in results]
    return updated, recovered


def draw(frame, results, all_ellipses, show_all):
    canvas = frame.copy()
    if show_all:
        for (cx, cy), (major, minor, angle), _, _ in all_ellipses:
            cv2.ellipse(canvas, (int(cx), int(cy)), (int(major / 2),
                        int(minor / 2)), angle, 0, 360, (160, 160, 160), 1)
    for center, outer, inner, count, points, digit, box, confidence in results:
        cx, cy = center
        major, minor, angle = outer
        cv2.ellipse(canvas, (int(cx), int(cy)), (int(major / 2),
                    int(minor / 2)), angle, 0, 360, (0, 255, 0), 2)
        cv2.drawMarker(canvas, (int(cx), int(cy)), (0, 0, 255),
                       cv2.MARKER_CROSS, 18, 2)
        source = "/order" if digit and confidence == 0.0 else ""
        label = (f"ring{digit}{source} ({int(cx)},{int(cy)}) pts={points}"
                 if digit else f"ring? ({int(cx)},{int(cy)}) pts={points}")
        cv2.putText(canvas, label,
                    (int(cx) - 60, int(cy) - int(major / 2) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if box:
            cv2.rectangle(canvas, box[:2], (box[0] + box[2], box[1] + box[3]),
                          (255, 0, 0), 1)
    cv2.putText(canvas, f"markers={len(results)}", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return canvas


def analyze(frame, args):
    analysis_frame = frame if args.no_relight else software_relight(frame)
    ellipses = fit_ellipses(
        analysis_frame, args.major_lo, args.major_hi,
        axis_ratio_lo=args.axis_ratio_lo,
        cover_lo=args.cover_lo, cover_hi=args.cover_hi,
        morph_kernel=args.morph_kernel,
        adapt_block=args.adapt_block, adapt_c=args.adapt_c)
    markers = group_concentric(ellipses, args.min_rings,
                               group_dist=args.group_dist)
    height, width = frame.shape[:2]
    candidates = []
    for center, outer, inner, count, support_points in markers:
        cx, cy = center
        major, minor, angle = outer
        if minor / major < args.marker_axis_ratio_lo:
            continue
        # 贴边剔除: 外圈椭圆转到正矩形后越界过多, 多为底盘/线缆弧线凑出的假组
        box = cv2.boxPoints(((cx, cy), (major, minor), angle))
        bx_lo, bx_hi = box[:, 0].min(), box[:, 0].max()
        by_lo, by_hi = box[:, 1].min(), box[:, 1].max()
        if (bx_lo < -major * EDGE_TOL or by_lo < -minor * EDGE_TOL
                or bx_hi > width + major * EDGE_TOL
                or by_hi > height + minor * EDGE_TOL):
            continue
        digit, box_digit, confidence = classify_digit(
            analysis_frame, center, outer,
            soft_match_min=args.soft_match_min,
            soft_match_margin=args.soft_match_margin)
        if digit is not None and confidence < args.digit_min_confidence:
            digit, box_digit, confidence = None, None, 0.0
        candidates.append((center, outer, inner, count, support_points,
                           digit, box_digit, confidence))
    # 先去除同编号重复检测，使可靠数字候选能约束后续空间排列恢复。
    candidates = _deduplicate_identities(candidates)
    if not args.no_order_recovery:
        candidates, _ = _recover_ordered_identities(
            candidates, min_major=args.major_lo)
    results = _deduplicate_identities(candidates)
    if not args.keep_unknown:
        results = [item for item in results if item[5] is not None]
    return results, ellipses, analysis_frame


def main():
    args = parse_args()
    if args.interactive:
        args = configure_interactive(args)
    validate_args(args)
    images = discover_images(args.inputs)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    print("运行参数: "
          f"图片={len(images)} 输出={output} 补光={not args.no_relight} "
          f"保存补光={args.save_relight} 长轴={args.major_lo:g}-{args.major_hi:g} "
          f"轴比>={args.marker_axis_ratio_lo:g} "
          f"数字置信度>={args.digit_min_confidence:g} "
          f"容差匹配>={args.soft_match_min:g}/分差>={args.soft_match_margin:g} "
          f"排列恢复={not args.no_order_recovery}")

    histogram = {}
    digit_frames = {"1": 0, "2": 0, "3": 0}
    order_recovered_frames = 0
    order_recovered_rings = 0
    records = []
    relight_dir = output / "relight"
    if args.save_relight and not args.no_relight:
        relight_dir.mkdir(parents=True, exist_ok=True)
    for path in images:
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"!! 读图失败: {path}", file=sys.stderr)
            continue
        results, ellipses, analysis_frame = analyze(frame, args)
        cv2.imwrite(str(output / path.name),
                    draw(frame, results, ellipses, args.show_all),
                    [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
        if args.save_relight and not args.no_relight:
            cv2.imwrite(str(relight_dir / path.name), analysis_frame,
                        [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
        key = min(len(results), 4)
        histogram[key] = histogram.get(key, 0) + 1
        found_digits = {r[5] for r in results if r[5]}
        recovered_in_frame = sum(1 for r in results
                                 if r[5] and r[7] == 0.0)
        if recovered_in_frame:
            order_recovered_frames += 1
            order_recovered_rings += recovered_in_frame
        for d in found_digits:
            digit_frames[d] += 1
        detail = " ".join(
            f"{'ring' + (r[5] or '?')}@({r[0][0]:.0f},{r[0][1]:.0f})"
            f"[pts={r[4]},conf={r[7]:.2f},"
            f"src={'order' if r[7] == 0.0 else 'digit'}]" if r[5]
            else f"?@({r[0][0]:.0f},{r[0][1]:.0f})[pts={r[4]}]"
            for r in results)
        print(f"{path.name}: {len(results)}个 [{detail}]")
        records.append({
            "input": str(path),
            "ring_count": len(results),
            "rings": [{
                "identity": f"circle{r[5]}" if r[5] else "unknown",
                "center_x": round(float(r[0][0]), 2),
                "center_y": round(float(r[0][1]), 2),
                "outer_major": round(float(r[1][0]), 2),
                "outer_minor": round(float(r[1][1]), 2),
                "inner_major": round(float(r[2][0]), 2),
                "inner_minor": round(float(r[2][1]), 2),
                "support_points": int(r[4]),
                "candidate_contours": int(r[3]),
                "digit_confidence": float(r[7]),
                "identity_source": ("order" if r[5] and r[7] == 0.0
                                    else "digit" if r[5] else "unknown"),
            } for r in results],
        })

    total = sum(histogram.values())
    summary = " ".join(f"{k}个:{v}张" for k, v in sorted(histogram.items()))
    print(f"全部完成: {total}张 -> {output}")
    print(f"每帧圆环数分布: {summary}")
    print(f"数字识别覆盖帧数: " +
          " ".join(f"含{d}的帧={c}" for d, c in digit_frames.items()))
    print(f"排列恢复: {order_recovered_frames}帧/{order_recovered_rings}个身份")
    (output / "summary.json").write_text(json.dumps({
        "input_count": total,
        "software_relight": not args.no_relight,
        "parameters": {
            "min_rings": args.min_rings,
            "major_lo": args.major_lo,
            "major_hi": args.major_hi,
            "axis_ratio_lo": args.axis_ratio_lo,
            "marker_axis_ratio_lo": args.marker_axis_ratio_lo,
            "group_dist": args.group_dist,
            "cover_lo": args.cover_lo,
            "cover_hi": args.cover_hi,
            "morph_kernel": args.morph_kernel,
            "adapt_block": args.adapt_block,
            "adapt_c": args.adapt_c,
            "digit_min_confidence": args.digit_min_confidence,
            "soft_match_min": args.soft_match_min,
            "soft_match_margin": args.soft_match_margin,
            "order_recovery": not args.no_order_recovery,
            "jpeg_quality": args.jpeg_quality,
        },
        "ring_count_histogram": histogram,
        "digit_frame_coverage": digit_frames,
        "order_recovered_frames": order_recovered_frames,
        "order_recovered_rings": order_recovered_rings,
        "images": records,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
