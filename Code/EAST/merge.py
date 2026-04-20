import cv2 as cv
import numpy as np
import math

def rr_to_pts(rr):
    return cv.boxPoints(rr).astype(np.float32)


def rr_center(rr):
    return np.array(rr[0], dtype=np.float32)


def rr_size(rr):
    w, h = rr[1]
    return float(w), float(h)


def rr_angle_deg(rr):
    """
    Normalize angle to the long-axis direction.
    """
    (cx, cy), (w, h), a = rr
    if h > w:
        a = a + 90.0
    while a < -90:
        a += 180
    while a >= 90:
        a -= 180
    return float(a)


def angle_diff_deg(a, b):
    d = abs(a - b) % 180.0
    if d > 90.0:
        d = 180.0 - d
    return d


def point_segment_distance(p, a, b):
    ab = b - a
    denom = np.dot(ab, ab)
    if denom < 1e-8:
        return np.linalg.norm(p - a)
    t = np.dot(p - a, ab) / denom
    t = np.clip(t, 0.0, 1.0)
    proj = a + t * ab
    return np.linalg.norm(p - proj)


def poly_min_edge_distance(poly1, poly2):
    min_d = float("inf")

    for p in poly1:
        for q in poly2:
            min_d = min(min_d, np.linalg.norm(p - q))

    edges1 = [(poly1[i], poly1[(i + 1) % 4]) for i in range(4)]
    edges2 = [(poly2[i], poly2[(i + 1) % 4]) for i in range(4)]

    for p in poly1:
        for a, b in edges2:
            min_d = min(min_d, point_segment_distance(p, a, b))

    for p in poly2:
        for a, b in edges1:
            min_d = min(min_d, point_segment_distance(p, a, b))

    return float(min_d)


def try_rotated_intersection(rr1, rr2):
    status, _ = cv.rotatedRectangleIntersection(rr1, rr2)
    return status != cv.INTERSECT_NONE


def box_line_features(rr):
    """
    Returns:
      center, long-axis unit vector u, short-axis unit vector v,
      long_len, short_len, angle_deg
    """
    c = rr_center(rr)
    w, h = rr_size(rr)
    a_deg = rr_angle_deg(rr)
    a = math.radians(a_deg)

    u = np.array([math.cos(a), math.sin(a)], dtype=np.float32)
    v = np.array([-u[1], u[0]], dtype=np.float32)

    long_len = max(w, h)
    short_len = min(w, h)
    return c, u, v, long_len, short_len, a_deg


def expand_rotated_rect_long_axis(rr, expand_ratio=0.25):
    """
    Expand only along the long axis for LINKING purposes.
    expand_ratio=0.25 means increase long dimension by 25%.
    """
    (cx, cy), (w, h), a = rr
    if w >= h:
        w = w * (1.0 + expand_ratio)
    else:
        h = h * (1.0 + expand_ratio)
    return ((cx, cy), (w, h), a)


def should_link_boxes(
    rr1,
    rr2,
    angle_thresh_deg=12.0,
    max_perp_factor=1.2,
    max_along_gap_factor=1.8,
    height_ratio_thresh=0.45,
    allow_intersection=True,
    allow_close=True,
    long_axis_expand_ratio=0.25,
):
    """
    Decide whether two rotated boxes likely belong to the same text line.

    Upgrade:
    - expand boxes along their long axis for overlap/closeness checks only
    - keep original boxes for geometry/merging
    """
    c1, u1, v1, L1, H1, a1 = box_line_features(rr1)
    c2, u2, v2, L2, H2, a2 = box_line_features(rr2)

    # 1) similar angle
    if angle_diff_deg(a1, a2) > angle_thresh_deg:
        return False

    # average line direction
    u = u1 + u2
    nu = np.linalg.norm(u)
    if nu < 1e-8:
        u = u1
    else:
        u = u / nu
    v = np.array([-u[1], u[0]], dtype=np.float32)

    d = c2 - c1
    along = abs(np.dot(d, u))
    perp = abs(np.dot(d, v))

    # 2) similar heights
    h_avg = 0.5 * (H1 + H2)
    if h_avg < 1e-6:
        return False
    if abs(H1 - H2) / h_avg > height_ratio_thresh:
        return False

    # 3) close in perpendicular direction
    if perp > max_perp_factor * h_avg:
        return False

    # 4) not too far apart along text direction
    raw_gap = along - 0.5 * (L1 + L2)
    raw_gap = max(0.0, raw_gap)
    if raw_gap > max_along_gap_factor * h_avg:
        return False

    # 5) expanded-box overlap / closeness checks
    rr1_exp = expand_rotated_rect_long_axis(rr1, long_axis_expand_ratio)
    rr2_exp = expand_rotated_rect_long_axis(rr2, long_axis_expand_ratio)

    if allow_intersection and try_rotated_intersection(rr1_exp, rr2_exp):
        return True

    if allow_close:
        p1 = rr_to_pts(rr1_exp)
        p2 = rr_to_pts(rr2_exp)
        mind = poly_min_edge_distance(p1, p2)
        if mind <= 0.6 * h_avg:
            return True

    return False


def connected_components_from_adj(adj):
    n = len(adj)
    seen = [False] * n
    comps = []

    for i in range(n):
        if seen[i]:
            continue
        stack = [i]
        seen[i] = True
        comp = []

        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nb in adj[cur]:
                if not seen[nb]:
                    seen[nb] = True
                    stack.append(nb)

        comps.append(sorted(comp))
    return comps


def merge_rotated_group(rects):
    """
    Merge using ORIGINAL boxes only.
    """
    all_pts = np.vstack([rr_to_pts(rr) for rr in rects]).astype(np.float32)
    merged = cv.minAreaRect(all_pts)
    return merged


def group_and_merge_rotated_rects(
    rects,
    confidences=None,
    angle_thresh_deg=12.0,
    max_perp_factor=1.2,
    max_along_gap_factor=1.8,
    height_ratio_thresh=0.45,
    long_axis_expand_ratio=0.25,
    min_group_size=1,
):
    n = len(rects)
    if n == 0:
        if confidences is None:
            return [], [], None
        return [], [], []

    adj = [set() for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            if should_link_boxes(
                rects[i],
                rects[j],
                angle_thresh_deg=angle_thresh_deg,
                max_perp_factor=max_perp_factor,
                max_along_gap_factor=max_along_gap_factor,
                height_ratio_thresh=height_ratio_thresh,
                long_axis_expand_ratio=long_axis_expand_ratio,
            ):
                adj[i].add(j)
                adj[j].add(i)

    groups = connected_components_from_adj(adj)
    groups = [g for g in groups if len(g) >= min_group_size]

    merged_rects = []
    merged_confidences = [] if confidences is not None else None

    for g in groups:
        merged_rects.append(merge_rotated_group([rects[k] for k in g]))
        if confidences is not None:
            merged_confidences.append(max(confidences[k] for k in g))

    return merged_rects, groups, merged_confidences


def draw_rotated_rects(image, rects, color=(0, 255, 0), thickness=2):
    out = image.copy()
    for rr in rects:
        pts = cv.boxPoints(rr).astype(np.int32)
        cv.polylines(out, [pts], True, color, thickness)
    return out