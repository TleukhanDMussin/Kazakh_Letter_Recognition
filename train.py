from ultralytics import YOLO
import cv2
import time

model = YOLO('best.pt')  # build from YAML and transfer weights
img = cv2.imread("test4.png")
img = cv2.resize(img, (600, 400))
results = model(source=img, show = True) #glass bottle
print(results)
time.sleep(100)