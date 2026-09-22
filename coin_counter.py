"""
Coin Detection and Counting
----------------------------
Compares three approaches for detecting and counting coins in an image:
  1. Classical contour detection (OpenCV findContours)
  2. Classical Hough Circle Transform (OpenCV HoughCircles)
  3. Deep learning object detection (YOLOv8n, trained on a small custom dataset)

Author: Milad Arghiani
"""

import os
import math
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("ultralytics is not installed - run: pip install ultralytics")

# Path to the trained YOLO weights file.
# Uses an environment variable if set, otherwise defaults to a "weights"
# folder next to this script - no hardcoded personal paths.
MODEL_PATH = os.environ.get(
    "COIN_MODEL_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", "best.pt"),
)


class CoinCounterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Coin Counter")
        self.root.geometry("1300x880")

        self.image_path = None
        self.original_image = None

        top_bar = tk.Frame(root)
        top_bar.pack(fill=tk.X, pady=10)

        tk.Button(top_bar, text="Select Image", command=self.select_image,
                  font=("tahoma", 12)).pack(side=tk.LEFT, padx=30)

        tk.Button(top_bar, text="Run All Methods", command=self.run_all_methods,
                  font=("tahoma", 12)).pack(side=tk.LEFT, padx=10)

        self.result_label = tk.Label(
            top_bar, text="Contour: -   |   Hough: -   |   YOLO: -",
            font=("tahoma", 12, "bold"))
        self.result_label.pack(side=tk.LEFT, padx=50)

        panels_frame = tk.Frame(root)
        panels_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.panels = []
        panel_titles = ["Original", "Contour", "Hough", "YOLO"]

        for i, title in enumerate(panel_titles):
            frame = tk.LabelFrame(panels_frame, text=title, font=("tahoma", 10, "bold"),
                                   padx=5, pady=5)
            frame.grid(row=i // 2, column=i % 2, sticky="nsew", padx=6, pady=6)
            label = tk.Label(frame, relief="sunken")
            label.pack(fill=tk.BOTH, expand=True)
            self.panels.append(label)

        panels_frame.columnconfigure(0, weight=1)
        panels_frame.columnconfigure(1, weight=1)
        panels_frame.rowconfigure(0, weight=1)
        panels_frame.rowconfigure(1, weight=1)

        print("App started... select an image to begin")

    def select_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.png *.jpeg *.JPG")])
        if not path:
            return

        image = cv2.imread(path)
        if image is None:
            messagebox.showerror("Error", "Could not open the image")
            return

        self.original_image = image
        self.image_path = path
        self.show_image(image, self.panels[0])
        print("Image loaded:", path)

    def show_image(self, image, panel):
        h, w = image.shape[:2]
        scale = min(480 / max(w, 1), 480 / max(h, 1))
        resized = cv2.resize(image, (int(w * scale), int(h * scale)))

        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tk_image = ImageTk.PhotoImage(Image.fromarray(rgb))
        panel.imgtk = tk_image  # keep a reference to avoid garbage collection
        panel.configure(image=tk_image)

    def preprocess(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2)
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
        return gray, thresh

    def detect_contour(self, thresh, original):
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result = original.copy()
        count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 350 or area > 120000:
                continue
            perimeter = cv2.arcLength(cnt, True)
            if perimeter < 60:
                continue
            circularity = 4 * math.pi * area / (perimeter ** 2)
            if circularity < 0.60:
                continue
            (x, y), r = cv2.minEnclosingCircle(cnt)
            r = int(r)
            if r < 16 or r > 200:
                continue
            cv2.circle(result, (int(x), int(y)), r, (0, 220, 0), 3)
            count += 1
        return result, count

    def detect_hough(self, gray, original):
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1.18,
            minDist=45, param1=55, param2=32,
            minRadius=18, maxRadius=170)
        result = original.copy()
        count = 0
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for (x, y, r) in circles[0, :]:
                if 18 <= r <= 160:
                    cv2.circle(result, (x, y), r, (0, 220, 0), 3)
                    count += 1
        return result, count

    def detect_yolo(self):
        if not YOLO_AVAILABLE:
            return self.original_image.copy(), 0, "ultralytics not installed"

        if not os.path.exists(MODEL_PATH):
            return self.original_image.copy(), 0, "model not found"

        try:
            model = YOLO(MODEL_PATH)
            results = model(self.image_path, conf=0.12, iou=0.45, verbose=True)

            result_image = self.original_image.copy()
            count = 0

            print("\n--- YOLO detections ---")
            for r in results:
                if len(r.boxes) == 0:
                    print("No coins detected")
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0].item()
                    label = f"coin {conf:.2f}"

                    print(f"-> coin found: {label}   [{x1},{y1}] -> [{x2},{y2}]")

                    cv2.rectangle(result_image, (x1, y1), (x2, y2), (255, 0, 0), 3)
                    cv2.putText(result_image, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    count += 1

            print(f"Total YOLO detections: {count}\n")
            return result_image, count, "YOLO"

        except Exception as error:
            print("YOLO error:", error)
            return self.original_image.copy(), 0, "YOLO error"

    def run_all_methods(self):
        if self.original_image is None:
            messagebox.showwarning("Notice", "Please select an image first")
            return

        gray, thresh = self.preprocess(self.original_image)

        contour_img, contour_count = self.detect_contour(thresh, self.original_image)
        self.show_image(contour_img, self.panels[1])

        hough_img, hough_count = self.detect_hough(gray, self.original_image)
        self.show_image(hough_img, self.panels[2])

        yolo_img, yolo_count, yolo_label = self.detect_yolo()
        self.show_image(yolo_img, self.panels[3])

        self.result_label.config(
            text=f"Contour: {contour_count}   |   Hough: {hough_count}   |   {yolo_label}: {yolo_count}")

        print("All methods executed")


if __name__ == "__main__":
    root = tk.Tk()
    app = CoinCounterApp(root)
    root.mainloop()
    print("Program closed")
