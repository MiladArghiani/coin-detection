# Coin Detection & Counting

A comparison of three different approaches for detecting and counting coins in an image:
classical contour detection, the Hough Circle Transform, and a deep learning object
detector (YOLOv8n).

## Approaches
1. **Contour Detection** — OpenCV `findContours`, filtered by area, perimeter, and
   circularity to isolate coin-like shapes.
2. **Hough Circle Transform** — OpenCV `HoughCircles`, tuned for coin-sized circles.
3. **YOLOv8n** — trained on a small custom dataset of 19 hand-labeled images
   (15 train / 4 validation) of Iranian coins on a plain background.

## Results
On a controlled dataset (plain background, even lighting), both classical methods and
YOLOv8n correctly counted coins in most test images. The classical methods proved more
stable given the very small training set, while YOLOv8n — despite the limited data —
learned to detect coins reasonably well on images similar to its training set, though it
generalized poorly to entirely new images. This highlights a common challenge in deep
learning: model performance is highly dependent on training data size and diversity.

## Tech stack
Python, OpenCV, Tkinter, Ultralytics YOLOv8, Pillow, NumPy

## How it works
Run the script, select an image, and click "Run All Methods" to see a side-by-side
comparison of all three detection methods with their respective coin counts.
