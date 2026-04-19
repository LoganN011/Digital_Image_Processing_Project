"""
EAST does not come with cv2 automatically. You must download the pretrained EAST model separately:

curl -L -o frozen_east_text_detection.pb \
  https://github.com/oyyd/frozen_east_text_detection.pb/raw/master/frozen_east_text_detection.pb

"""


# --- packages ---

from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import time
import cv2

from pathlib import Path
script_dir = Path(__file__).parent

# --- arguments ---

# --image: path to the input image
# --east: path to the input EAST text detector
# --min-confidence: probability threshold to determine text
# --width: resized image width (must be a multiple of 32)
# --height: resized image height (must be a multiple of 32)

# Default paths to the input image and EAST text detector
img_path = script_dir / "unm1.jpg"
east_path = script_dir / "frozen_east_text_detection.pb"

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str,
	help="path to input image", default=str(img_path))
ap.add_argument("-east", "--east", type=str,
	help="path to input EAST text detector", default=str(east_path))
ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
	help="minimum probability required to inspect a region")
ap.add_argument("-w", "--width", type=int, default=320,
	help="resized image width (should be multiple of 32)")
ap.add_argument("-e", "--height", type=int, default=320,
	help="resized image height (should be multiple of 32)")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])

# Resize the image to the desired dimensions
# image = cv2.resize(image, (args["width"], args["height"]))
orig = image.copy()
(H, W) = image.shape[:2]

# Ratio in change for width and height
(newW, newH) = (args["width"], args["height"])
rW = W / float(newW)
rH = H / float(newH)

# Resize the image, ignoring aspect ratio
image = cv2.resize(image, (newW, newH))
(H, W) = image.shape[:2]   


# --- Model ---

# Output layer
layerNames = [
    # The first layer is our output sigmoid activation which gives us the probability of a region containing text or not.
	"feature_fusion/Conv_7/Sigmoid",

    # The second layer can be used to derive the bounding box coordinates of text from the feature maps produced by the network.
	"feature_fusion/concat_3"
    ]

# Load the pre-trained EAST text detector
net = cv2.dnn.readNet(args["east"])

# Construct a blob from the image and perform a forward pass of the model to obtain the two output layer sets
blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
	(123.68, 116.78, 103.94), swapRB=True, crop=False)

start = time.time()
net.setInput(blob)
(scores, geometry) = net.forward(layerNames)
end = time.time()
print(f"Text detection took {(end - start):.6f} seconds")



# --- bounding box rectangles and confidence scores ---

rects = []
confidences = []

for y in range(0, scores.shape[2]):
	# Extract the scores (probabilities)
	scoresData = scores[0, 0, y]
	
    # Extract the geometrical data used for potential bounding box coordinates that surround text
	xData0 = geometry[0, 0, y]
	xData1 = geometry[0, 1, y]
	xData2 = geometry[0, 2, y]
	xData3 = geometry[0, 3, y]
	anglesData = geometry[0, 4, y]

	for x in range(0, scores.shape[3]):
		
		if scoresData[x] < args["min_confidence"]:
			continue
		
        # Need offset factor as the resulting feature maps will be 4x smaller than the input image
		(offsetX, offsetY) = (x * 4.0, y * 4.0)
		
		# Extract the rotation angle for the prediction and then compute the sine and cosine
		angle = anglesData[x]
		cos = np.cos(angle)
		sin = np.sin(angle)
		
		# Derive the width and height of the bounding box from the geometry volume
		h = xData0[x] + xData2[x]
		w = xData1[x] + xData3[x]
		
		# Starting and ending (x, y)-coordinates for the text prediction bounding box
		endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
		endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
		startX = int(endX - w)
		startY = int(endY - h)
		
		# Store bounding box coordinates and probability score
		rects.append((startX, startY, endX, endY))
		confidences.append(scoresData[x])
		
# Apply non-maxima suppression to suppress weak, overlapping bounding boxes
boxes = non_max_suppression(np.array(rects), probs=confidences, overlapThresh=0.3)

for (startX, startY, endX, endY) in boxes:
	# Scale the bounding box coordinates based on the respective ratios
	startX = int(startX * rW)
	startY = int(startY * rH)
	endX = int(endX * rW)
	endY = int(endY * rH)
	
	# Draw the bounding box
	cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)

# --- display image ---

cv2.imwrite(str(script_dir / f"text_detection_{Path(args['image']).stem}.jpg"), orig)

# cv2.imshow("Text Detection", orig)
# cv2.waitKey(0)
# cv2.destroyAllWindows()