# 🎥 Webcam Object Detection with YOLOv3

This project implements a **real-time object detection system** using **YOLOv3** with OpenCV’s DNN module. It detects multiple objects from a webcam feed such as people, cars, bottles, and more.

---

## 🚀 Features
- Real-time object detection from webcam  
- Uses **pre-trained YOLOv3 model** on the COCO dataset (80 classes)  
- Bounding boxes with class labels and confidence scores  
- Lightweight and runs on CPU with OpenCV  

---

## 📂 Project Structure
yolo-webcam-detection/
│── object_detection.py # Main Python script
│── yolov3.cfg # YOLOv3 configuration file
│── coco.names # Class labels
│── yolov3.weights # Pre-trained weights (download separately)


---

## 🔽 Download Weights
Due to GitHub file size limits, the **YOLOv3 weights** must be downloaded separately:  

➡️ [Download yolov3.weights from Google Drive](https://drive.google.com/your-link-here)  

After downloading, place the file inside the project folder.

---

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/yolo-webcam-detection.git
   cd yolo-webcam-detection
Install dependencies:


pip install -r requirements.txt
▶️ Usage
Run the object detection script:


python object_detection.py
Press q to quit the webcam window.

🛠️ Tools & Technologies
Python

OpenCV

YOLOv3

COCO Dataset

NumPy

📌 Notes
Make sure your webcam is connected and accessible.

For faster inference, you can use GPU if OpenCV is built with CUDA.

👨‍💻 Author
Developed by Aaryan Kumar at B.P. Poddar Institute of Management and Technology (BPPIMT).


