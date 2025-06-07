# 🚦 Crosswalk Detection in CARLA using YOLOv8

This project demonstrates a real-time crosswalk detection system in the CARLA simulator using a custom-trained YOLOv8 model.  
When a crosswalk is detected, the autonomous vehicle slows down by 50% and resumes its original speed once it has passed the crosswalk — mimicking safe urban driving behavior.

---

## 🧠 How It Works

- The project runs a Tesla Model 3 in the CARLA simulation environment.
- A front-mounted RGB camera continuously captures images of the road.
- These images are processed in real time by a YOLOv8 model trained to detect crosswalks.
- If a crosswalk is detected:
  - A bounding box and label are drawn on the image.
  - A message “Crosswalk Ahead – Slowing Down!” is displayed.
  - The vehicle's velocity is reduced by 50%.
- Once the crosswalk is no longer detected, the vehicle resumes its original speed.

---

## 🛠️ Components

- **Main.py**: Runs the live simulation, processes camera input, performs detection, and controls vehicle behavior.
- **TrainYolo.py**: Trains a YOLOv8 model using a custom dataset defined in `data.yaml`.

---

## 📦 Requirements

- Python 3.7+
- [CARLA Simulator](https://carla.org/)
- [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- OpenCV
- PyTorch

Install dependencies with:
```bash
pip install ultralytics opencv-python torch

▶️ How to Run
Start the CARLA simulator.

Run the main simulation:
python Main.py
(Optional) To retrain the model, use:
python TrainYolo.py

Training Details
Base model: yolov8n.pt

Dataset: defined in data.yaml

Trained for 50 epochs on images resized to 640×640.

👥 Authors
Ron Gershtein
Lihi Kimhazi

Course: Intelligent Software for Autonomous Vehicles
Instructor: Dr. Vadim Talys
Semester: Spring 2025
