import os
import time
import psutil
import threading
import mediapipe as mp
from mediapipe.tasks import python as tasks
from mediapipe.tasks.python import vision
import cv2
import numpy as np

def monitor_cpu(interval=0.1):
    """
    Monitors CPU usage and returns a list of usage values and a stop event.
    """
    cpu_usage_list = []
    stop_event = threading.Event()

    def monitor():
        while not stop_event.is_set():
            cpu_usage_list.append(psutil.cpu_percent(interval=interval))
    
    thread = threading.Thread(target=monitor)
    return cpu_usage_list, stop_event, thread

def extract_landmarks(image, detector):
    """
    Extracts hand landmarks from an image and normalizes them.
    """
    # Convert BGR to RGB (OpenCV uses BGR, Mediapipe wants RGB)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    
    results = detector.detect(mp_image)
    
    data_aux = []
    if results.hand_landmarks:
        # Sort hands by x-position to ensure consistency if multiple hands are present
        sorted_hands = sorted(results.hand_landmarks, key=lambda h: h[0].x)
        
        for hand_landmarks in sorted_hands:
            x_ = []
            y_ = []
            for landmark in hand_landmarks:
                x_.append(landmark.x)
                y_.append(landmark.y)

            for landmark in hand_landmarks:
                data_aux.append(landmark.x - min(x_))
                data_aux.append(landmark.y - min(y_))
        
        # Consistent length (84) - pad with zeros if only one hand is detected
        while len(data_aux) < 84:
            data_aux.append(0.0)
        # In case more than 2 hands were detected (unlikely with num_hands=2), truncate
        data_aux = data_aux[:84]
        
        return data_aux
    return None

def process_data_directory(data_dir):
    """
    Processes all images in the specified directory and extracts landmarks.
    """
    # Initialize the detector
    model_path = os.path.join('models', 'hand_landmarker.task')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please download it first.")

    base_options = tasks.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
    detector = vision.HandLandmarker.create_from_options(options)

    data = []
    labels = []

    cpu_usage_list, stop_monitoring, monitor_thread = monitor_cpu()
    monitor_thread.start()

    start_time_total = time.time()

    for dir_ in os.listdir(data_dir):
        dir_path = os.path.join(data_dir, dir_)
        if not os.path.isdir(dir_path):
            continue
            
        print(f"Processing category: {dir_}")
        for img_path in os.listdir(dir_path):
            full_img_path = os.path.join(dir_path, img_path)
            img = cv2.imread(full_img_path)
            if img is None:
                continue

            landmarks = extract_landmarks(img, detector)
            if landmarks:
                data.append(landmarks)
                labels.append(dir_)

    stop_monitoring.set()
    monitor_thread.join()

    end_time_total = time.time()

    total_time = end_time_total - start_time_total
    avg_cpu = sum(cpu_usage_list) / len(cpu_usage_list) if cpu_usage_list else 0

    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Average CPU usage: {avg_cpu:.2f}%")

    # Close the detector
    detector.close()

    return data, labels
