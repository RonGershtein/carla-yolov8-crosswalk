import sys
sys.path.append("C:/Users/Ron/Desktop/CARLA_0.9.13 (1)/WindowsNoEditor/PythonAPI/carla/dist/carla-0.9.13-py3.7-win-amd64.egg")

import carla
import numpy as np
import cv2
import torch
from ultralytics import YOLO
import time
import os

# נתיב למודל המאומן שלך
MODEL_PATH = r'C:\Users\Ron\PycharmProjects\PythonProject2\.venv\runs\detect\train2\weights\best.pt'

# טעינת המודל
model = YOLO(MODEL_PATH)

# התחברות ל-Carla
client = carla.Client("localhost", 2000)
client.set_timeout(10.0)
world = client.get_world()
blueprint_library = world.get_blueprint_library()

# יצירת הרכב
vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
spawn_point = world.get_map().get_spawn_points()[0]
vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
vehicle.set_autopilot(True)

# יצירת מצלמה קדמית
camera_bp = blueprint_library.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', '640')
camera_bp.set_attribute('image_size_y', '480')
camera_bp.set_attribute('fov', '90')
camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

# משתנים גלובליים
frame_counter = 0
crosswalk_detected = False
slowed_down = False
original_velocity = None

# פונקציית עיבוד תמונה
def process_image(image):
    global frame_counter, crosswalk_detected, slowed_down, original_velocity

    # אם הרכב הושמד – אין מה לעבד
    if vehicle is None or not vehicle.is_alive:
        return

    frame_counter += 1
    if frame_counter % 2 != 0:
        return

    img_array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))[:, :, :3]
    results = model.predict(source=img_array, imgsz=640, conf=0.5, verbose=False)

    annotated = img_array.copy()
    crosswalk_detected = False

    for result in results:
        for box in result.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = box
            label = result.names[int(cls)]
            if 'crosswalk' in label.lower():
                crosswalk_detected = True
                cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(annotated, f'Crosswalk: {conf:.2f}', (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # הצגת מהירות
    velocity = vehicle.get_velocity()
    speed_kmh = int(3.6 * np.linalg.norm([velocity.x, velocity.y, velocity.z]))
    cv2.putText(annotated, f"Speed: {speed_kmh} km/h", (10, 460),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # האטה או החזרת מהירות
    if crosswalk_detected:
        cv2.putText(annotated, "Crosswalk Ahead - Slowing Down!", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        if not slowed_down:
            original_velocity = vehicle.get_velocity()
            slowed_velocity = carla.Vector3D(original_velocity.x * 0.5,
                                             original_velocity.y * 0.5,
                                             original_velocity.z)
            vehicle.set_target_velocity(slowed_velocity)
            slowed_down = True
    else:
        if slowed_down and original_velocity is not None:
            vehicle.set_target_velocity(original_velocity)
            slowed_down = False

    cv2.imshow("Driver Camera", annotated)
    cv2.waitKey(1)

# הפעלת המצלמה
camera.listen(lambda image: process_image(image))

# לולאת הרצה עד עצירה ידנית
try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    print("Cleaning up...")
    if camera.is_listening:
        camera.stop()
    if camera is not None:
        camera.destroy()
    if vehicle is not None:
        vehicle.destroy()
    cv2.destroyAllWindows()
