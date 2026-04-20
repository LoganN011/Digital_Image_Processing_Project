"""
EAST does not come with cv2 automatically. You must download the pretrained EAST model separately:

curl -L -o frozen_east_text_detection.pb \
  https://github.com/oyyd/frozen_east_text_detection.pb/raw/master/frozen_east_text_detection.pb

"""

import cv2 as cv
import numpy as np
import merge

from pathlib import Path

from unm.dip.transform import scale
script_dir = Path(__file__).parent

img_path = script_dir / "unm2.jpg"
east_path = script_dir / "frozen_east_text_detection.pb"

image = cv.imread(str(img_path))

h, w = image.shape[0], image.shape[1]
# Must be a multiple of 32 for EAST
h32, w32 = int(round(h / 32) * 32), int(round(w / 32) * 32)

orig = image.copy()

model = cv.dnn.TextDetectionModel_EAST(str(east_path))

model.setConfidenceThreshold(0.5)  

# Threshold for removing overlapping boxes.
model.setNMSThreshold(0.4) 

model.setInputParams(
    scale=1.0,                      # pixel intensity
    size=(w32, h32),                # input image size
    mean=(123.68, 116.78, 103.94),  # standard mean values
    swapRB=True,                    # BRG to RGB
    crop=False
)

rotated_rects, confidences = model.detectTextRectangles(image)

for rr in rotated_rects:
    pts = cv.boxPoints(rr)           # 4 corner points
    pts = np.array(pts, np.int32)
    cv.polylines(orig, [pts], True, (0, 255, 0), 2)

# --- post-processing ---

# Merge axis-aligned and overlapping boxes

merged_rects, groups, merged_scores = merge.group_and_merge_rotated_rects(
    rotated_rects,
    confidences=confidences,
    angle_thresh_deg=15.0,
    max_perp_factor=1.5,
    max_along_gap_factor=2.5,
    height_ratio_thresh=0.60,
    long_axis_expand_ratio=0.35,        # 0.2 = mild, 0.5 = aggressive
    min_group_size=1,
)

vis = merge.draw_rotated_rects(image, rotated_rects, color=(0, 255, 0), thickness=2)
vis2 = merge.draw_rotated_rects(image, merged_rects, color=(0, 0, 255), thickness=3)

cv.imwrite(str(script_dir / f"raw_{img_path.stem}.png"), vis)
cv.imwrite(str(script_dir / f"merged_{img_path.stem}.png"), vis2)