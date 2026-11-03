"""OpenCV 圆环检测运行时核心(替代 YOLO 环检测, 移植自离线验证器 ring_check1.py)。

检测地面上黑线虚线同心圆环(中心印数字 1/2/3): Lab 软件补光(纯图像处理, 无硬件补光)
→ 自适应阈值取黑线 → 闭运算桥接虚线 → 轮廓筛选 → 同心分组成环 → 中心数字识别
(实拍模板 ring_templates1/ 优先, 多字体合成模板兜底; IoU 严格匹配优先, 形态学
容差匹配次之) → 三环空间排列恢复 1/2/3 身份。纯 CPU, 无模型/相机/串口依赖。

接口: find(frame, keep_unknown=False, **参数覆盖)
  → [(center, outer, inner, n_candidates, support_points, digit, box, conf)]
digit 为 "1"/"2"/"3" 字符串(conf=0.0 表示身份来自排列恢复); 消费方(task/ring.py)
转 int 并组装 Ring。未知环默认丢弃(与 YOLO 版语义一致)。

⚠️ 参数默认值=旧下视相机 1280×720 离线照片调参口径, 换末端相机实车视野后
**必须重调**(tests/test_ring_cv.py 实时调参)。实拍数字模板 ring_templates1/
缺失时仅用合成字体模板兜底(置信度会降), 补拍模板放进该目录即自动加载。
"""
import itertools
from pathlib import Path

import cv2
import numpy as np

# 模板目录(实拍数字样本 {digit}_{n}.png); 缺失=仅合成字体模板兜底
TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "ring_templates1"

# 圆环尺寸随距离变化(1280x720 口径): 远~30px, 近~280px
MAJOR_LO, MAJOR_HI = 30, 280          # 椭圆长轴范围
AXIS_RATIO_LO = 0.30                  # 候选椭圆短轴/长轴下限(透视压扁)
MARKER_AXIS_RATIO_LO = 0.60           # 成组圆环短轴/长轴下限(滤文字/长弧)
GROUP_DIST = 70.0                     # 同心判定: 椭圆中心距离阈值(px)
MIN_RINGS = 1                         # 数字复核会排除假圆, 保留远处单轮廓圆环
COVER_LO, COVER_HI = 0.55, 1.35       # 轮廓弧长/椭圆周长比: 太小=断弧拟合
MORPH_KERNEL = 5                      # 闭运算核(桥接虚线断口)
ADAPT_BLOCK = 51                      # 自适应阈值邻域(奇数, 覆盖环线间距)
ADAPT_C = 10.0                        # 自适应阈值偏移C
DIGIT_AREA_MIN = 18                   # 数字轮廓最小面积
IOU_GOOD = 0.60                       # IoU 达标采信, 否则形态学容差匹配
SOFT_MATCH_MIN = 0.78                 # 容差匹配最低得分
SOFT_MATCH_MARGIN = 0.04              # 最优与次优最小分差
EDGE_TOL = 0.30                       # 外圈越界超半轴该比例=贴边假环剔除
DIGIT_WINDOW_RADIUS = 0.24            # 中心窗半径/外圈短轴(支持旋转数字)
DIGIT_SIZE = (48, 72)                 # 数字归一化尺寸(宽, 高)

DEFAULTS = {
    "min_rings": MIN_RINGS,
    "major_lo": MAJOR_LO, "major_hi": MAJOR_HI,
    "axis_ratio_lo": AXIS_RATIO_LO,
    "marker_axis_ratio_lo": MARKER_AXIS_RATIO_LO,
    "group_dist": GROUP_DIST,
    "cover_lo": COVER_LO, "cover_hi": COVER_HI,
    "morph_kernel": MORPH_KERNEL,
    "adapt_block": ADAPT_BLOCK, "adapt_c": ADAPT_C,
    "digit_min_confidence": 0.0,
    "soft_match_min": SOFT_MATCH_MIN, "soft_match_margin": SOFT_MATCH_MARGIN,
    "no_order_recovery": False, "keep_unknown": False,
    "no_relight": False,
}


def software_relight(frame):
    """只校正 Lab 亮度通道, 均衡阴影且不改变物体颜色(纯软件, 无硬件补光)。"""
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


def fit_ellipses(frame, major_lo, major_hi, axis_ratio_lo=AXIS_RATIO_LO,
                 cover_lo=COVER_LO, cover_hi=COVER_HI,
                 morph_kernel=MORPH_KERNEL, adapt_block=ADAPT_BLOCK,
                 adapt_c=ADAPT_C):
    """自适应阈值取黑线 → 闭运算桥接虚线 → 轮廓筛选(几何/圆度/凸性), 返回候选。"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dark = cv2.adaptiveThreshold(cv2.GaussianBlur(gray, (3, 3), 0), 255,
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, adapt_block, adapt_c)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (morph_kernel, morph_kernel))
    closed = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_NONE)
    height, width = frame.shape[:2]
    out = []
    for contour in contours:
        if len(contour) < 20:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        major, minor = max(w, h), min(w, h)
        if not major_lo <= major <= major_hi or minor / major < max(axis_ratio_lo, 0.60):
            continue
        if x <= 0 or y <= 0 or x + w >= width or y + h >= height:
            continue
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if area <= 0 or perimeter <= 0:
            continue
        if 4 * np.pi * area / (perimeter * perimeter) < 0.55 \
                or not 0.55 <= area / (w * h) <= 0.90:
            continue
        hull_area = cv2.contourArea(cv2.convexHull(contour))
        if hull_area <= 0 or area / hull_area < 0.85:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
        if abs(cx - (x + (w - 1) / 2)) > major * 0.08 \
                or abs(cy - (y + (h - 1) / 2)) > major * 0.08:
            continue
        out.append(((cx, cy), (float(major), float(minor),
                               0.0 if w >= h else 90.0), area, len(contour)))
    return out


def _robust_ring_center(ring_items, outer_major):
    """多层同心轮廓加权估中心(点数/长轴加权), 剔除残弧偏心拟合。"""
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
    """中心聚集的椭圆分组成圆环标记。"""
    remaining = sorted(ellipses, key=lambda e: -e[1][0])
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
        ring_items = [e for e in group
                      if e[1][0] >= max(e[1][0] for e in group) * 0.45]
        center = _robust_ring_center(ring_items,
                                     max(e[1][0] for e in group))
        groups.append([center, max(ring_items, key=lambda e: e[1][0])[1],
                       group])

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


# ---- 数字识别: 实拍模板优先+多字体合成兜底 (IoU + 形态学容差匹配) ----

DIGIT_SIZE = (48, 72)                 # 数字归一化尺寸(宽, 高)
DIGIT_AREA_MIN = 18                   # 数字轮廓最小面积
IOU_GOOD = 0.60                       # IoU 达标采信, 否则形态学容差匹配
SOFT_MATCH_MIN = 0.78                 # 容差匹配最低得分
SOFT_MATCH_MARGIN = 0.04              # 最优与次优最小分差
EDGE_TOL = 0.30                       # 外圈越界假环剔除比例
DIGIT_WINDOW_RADIUS = 0.24            # 中心窗半径/外圈短轴


def _load_sample_templates(template_dir=None):
    """加载实拍数字模板(ring_templates1/{digit}_{n}.png); 目录缺失跳过。"""
    folder = Path(template_dir) if template_dir else TEMPLATE_DIR
    templates = {digit: [] for digit in "123"}
    if folder.is_dir():
        for digit in "123":
            for path in sorted(folder.glob(f"{digit}_*.png")):
                img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    templates[digit].append(img)
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
    """选中心竖长轮廓并等比归一化; 无效返回 None。"""
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
        if w / h > 0.72:
            continue
        fill_ratio = area / max(w * h, 1)
        if fill_ratio > 0.68:
            continue
        if x <= 0 or x + w >= width:
            continue
        center_error = abs((x + w / 2) - width / 2) / max(width, 1)
        candidates.append(((1.0 - min(center_error, 0.9)) * 1000 + h, contour))
    if not candidates:
        return None
    contour = max(candidates, key=lambda item: item[0])[1]
    x, y, w, h = cv2.boundingRect(contour)
    digit_roi = binary[y:y + h, x:x + w]
    canvas = np.zeros((DIGIT_SIZE[1], DIGIT_SIZE[0]), np.uint8)
    scale = min((DIGIT_SIZE[0] - 8) / w, (DIGIT_SIZE[1] - 8) / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
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


def _soft_shape_similarity(norm, template):
    """形态学容差形状相似度(允许 1px 位移/笔画粗细变化)。"""
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
    """严格 IoU 优先, 失败后轮廓容差匹配; 返回 (数字, 置信度)。"""
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
                           allow_rotations, soft_match_min, soft_match_margin):
    """中心窗内提取并分类数字。"""
    (cx, cy), (_, _, outer_angle) = center, outer
    mask = np.zeros(frame.shape[:2], np.uint8)
    mask_angle = 0 if allow_rotations else (outer_angle - 90.0) % 180.0
    cv2.ellipse(mask, (int(cx), int(cy)), axes, mask_angle,
                0, 360, 255, -1)
    x0, x1 = max(0, int(cx - axes[0])), min(frame.shape[1], int(cx + axes[0]))
    y0, y1 = max(0, int(cy - axes[1])), min(frame.shape[0], int(cy + axes[1]))
    inside = enhance[y0:y1, x0:x1].copy()
    inside_mask = mask[y0:y1, x0:x1]
    if inside.size == 0:
        return None, None, 0.0
    inside[inside_mask == 0] = 255
    _, dark = cv2.threshold(inside, 0, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dark[inside_mask == 0] = 0
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
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
        box = (x0, y0, x1 - x0, y1 - y0)
    return digit, box, round(confidence, 3)


def classify_digit(frame, center, outer, soft_match_min=SOFT_MATCH_MIN,
                   soft_match_margin=SOFT_MATCH_MARGIN):
    """识别圆环内部数字(仅身份, 不参与中心计算)。"""
    (_, _), (major, minor, _) = center, outer
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    enhance = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
    near_circle = minor / major >= 0.85
    if near_circle:
        half_window = max(int(minor * DIGIT_WINDOW_RADIUS), 8)
        primary = (half_window, half_window)
        fallback_half = max(half_window, 32)
        fallback = (fallback_half, fallback_half)
    else:
        primary = (max(int(major / 2 * 0.36), 6),
                   max(int(minor / 2 * 0.80), 8))
        fallback = (max(primary[0], 32), max(primary[1], 48))
    result = _classify_digit_window(frame, enhance, center, outer, primary,
                                    allow_rotations=near_circle,
                                    soft_match_min=soft_match_min,
                                    soft_match_margin=soft_match_margin)
    if result[0] is not None:
        return result
    if fallback != primary:
        return _classify_digit_window(frame, enhance, center, outer, fallback,
                                      allow_rotations=near_circle,
                                      soft_match_min=soft_match_min,
                                      soft_match_margin=soft_match_margin)
    return result


def _deduplicate_identities(results):
    """每个编号只留最佳候选, unknown 原样保留。"""
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
    """两个可靠编号预测第三个, 只接纳预测点附近的弱候选。"""
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
    """三几何候选近似共线且间距相近时, 按空间顺序恢复 1/2/3。"""
    if {item[5] for item in results if item[5]} == {"1", "2", "3"}:
        return results, 0
    eligible = [item for item in results
                if item[1][0] >= max(60.0, min_major)
                and (item[3] >= 3 or item[4] >= 300)]
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
    if best is None or best[0] > 0.32:
        return _recover_weak_identity_by_spacing(results)
    ordered = best[1]
    replacements = {}
    recovered = 0
    for index, item in enumerate(ordered, 1):
        expected = str(index)
        if item[5] != expected:
            replacements[id(item)] = item[:5] + (expected, None, 0.0)
            recovered += 1
    updated = [replacements.get(id(item), item) for item in results]
    return updated, recovered


def analyze(frame, **overrides):
    """完整分析一帧: 参数=DEFAULTS 覆盖。返回 (results, analysis_frame)。"""
    p = {**DEFAULTS, **overrides}
    analysis_frame = frame if p["no_relight"] else software_relight(frame)
    ellipses = fit_ellipses(
        analysis_frame, p["major_lo"], p["major_hi"],
        axis_ratio_lo=p["axis_ratio_lo"],
        cover_lo=p["cover_lo"], cover_hi=p["cover_hi"],
        morph_kernel=p["morph_kernel"],
        adapt_block=p["adapt_block"], adapt_c=p["adapt_c"])
    markers = group_concentric(ellipses, p["min_rings"],
                               group_dist=p["group_dist"])
    height, width = frame.shape[:2]
    candidates = []
    for center, outer, inner, count, support_points in markers:
        cx, cy = center
        major, minor, angle = outer
        if minor / major < p["marker_axis_ratio_lo"]:
            continue
        box = cv2.boxPoints(((cx, cy), (major, minor), angle))
        bx_lo, bx_hi = box[:, 0].min(), box[:, 0].max()
        by_lo, by_hi = box[:, 1].min(), box[:, 1].max()
        if (bx_lo < -major * EDGE_TOL or by_lo < -minor * EDGE_TOL
                or bx_hi > width + major * EDGE_TOL
                or by_hi > height + minor * EDGE_TOL):
            continue
        digit, box_digit, confidence = classify_digit(
            analysis_frame, center, outer,
            soft_match_min=p["soft_match_min"],
            soft_match_margin=p["soft_match_margin"])
        if digit is not None and confidence < p["digit_min_confidence"]:
            digit, box_digit, confidence = None, None, 0.0
        candidates.append((center, outer, inner, count, support_points,
                           digit, box_digit, confidence))
    candidates = _deduplicate_identities(candidates)
    if not p["no_order_recovery"]:
        candidates, _ = _recover_ordered_identities(
            candidates, min_major=p["major_lo"])
    results = _deduplicate_identities(candidates)
    if not p["keep_unknown"]:
        results = [item for item in results if item[5] is not None]
    return results, analysis_frame


def find(frame, keep_unknown=False, **overrides):
    """便捷接口: 一帧 → [(digit:str, cx, cy, outer, box, conf)]。

    digit="1"/"2"/"3"(conf=0.0 表示身份来自排列恢复); 默认丢弃未知环
    (keep_unknown=True 保留 digit=None 候选)。"""
    results, _ = analyze(frame, keep_unknown=keep_unknown, **overrides)
    return [(item[5], item[0][0], item[0][1], item[1], item[6], item[7])
            for item in results]
