import cv2
import numpy as np

# Load YOLOv3
net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")  
# cfg first, weights second

# Use OpenCV DNN backend
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Load class labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Get output layers
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
