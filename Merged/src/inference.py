import os
import pickle
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from mediapipe.tasks import python as tasks
from mediapipe.tasks.python import vision
from src.data_processing import extract_landmarks
from PIL import Image, ImageDraw, ImageFont

def render_text_with_font(image, text, position, font_path, font_size=32, color=(0, 255, 0), max_width=None):
    """
    Renders text on an image using a specific TTF/OTF font, with optional wrapping.
    Returns the modified image and the number of lines rendered.
    """
    # Convert OpenCV image (BGR) to PIL image (RGB)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception as e:
        print(f"Warning: Could not load font {font_path}. Using default. Error: {e}")
        font = ImageFont.load_default()
    
    lines_count = 1
    # Handle text wrapping if max_width is provided
    if max_width:
        words = text.split(' ')
        lines = []
        current_line = []
        for word in words:
            test_line = ' '.join(current_line + [word])
            # getbbox returns (left, top, right, bottom)
            bbox = draw.textbbox((0, 0), test_line, font=font)
            if bbox[2] - bbox[0] <= max_width:
                current_line.append(word)
            else:
                lines.append(' '.join(current_line))
                current_line = [word]
        lines.append(' '.join(current_line))
        text = '\n'.join(lines)
        lines_count = len(lines)

    # Draw the text (multiline supported by draw.text)
    draw.text(position, text, font=font, fill=color)
    
    # Convert PIL image back to OpenCV image (BGR)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR), lines_count

def run_inference(model_path, labels_path):
    """
    Runs real-time inference using the webcam.
    """
    # Load model and labels
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found at {labels_path}")

    model = tf.keras.models.load_model(model_path)
    with open(labels_path, 'rb') as f:
        labels_dict = pickle.load(f)

    # Font path
    font_path = "/home/alki/Project/Amharic-sign-Language-Transcription/Noto_Sans_Ethiopic/NotoSansEthiopic-VariableFont_wdth,wght.ttf"
    if not os.path.exists(font_path):
        print(f"Warning: Font not found at {font_path}")

    # Initialize the detector
    task_path = os.path.join('models', 'hand_landmarker.task')
    if not os.path.exists(task_path):
        raise FileNotFoundError(f"Mediapipe task file not found at {task_path}")

    base_options = tasks.BaseOptions(model_asset_path=task_path)
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
    detector = vision.HandLandmarker.create_from_options(options)

    # Transcript state
    transcript = []
    last_added_label = None
    stable_label = None
    stable_counter = 0
    STABILITY_THRESHOLD = 15 # Frames to confirm a sign

    cap = cv2.VideoCapture(0)
    print("Webcam started. Press 'q' to quit.")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape
            
            # Process frame
            landmarks = extract_landmarks(frame, detector)
            
            current_sign_display = ""
            if landmarks:
                prediction = model.predict(np.expand_dims(landmarks, axis=0), verbose=0)
                predicted_class = np.argmax(prediction)
                confidence = np.max(prediction)
                
                if confidence > 0.7:
                    label = labels_dict[predicted_class]
                    current_sign_display = f"{label} ({confidence*100:.1f}%)"
                    
                    # Logic to add to transcript
                    if label == stable_label:
                        stable_counter += 1
                    else:
                        stable_label = label
                        stable_counter = 0
                    
                    if stable_counter >= STABILITY_THRESHOLD:
                        if label != last_added_label:
                            transcript.append(label)
                            last_added_label = label
                        stable_counter = 0 # Reset to allow re-detecting after change
                else:
                    stable_label = None
                    stable_counter = 0
            else:
                stable_label = None
                stable_counter = 0
                last_added_label = None # Allow re-detecting the same sign if hand is moved and returned

            # Create a black bar at the bottom for the transcript
            bar_height = 100
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, h - bar_height), (w, h), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

            # Render current sign (temporary/current)
            if current_sign_display:
                frame, _ = render_text_with_font(frame, current_sign_display, (10, 10), font_path, font_size=24, color=(0, 255, 0))

            # Render scrolling transcript at the bottom
            transcript_text = " ".join(transcript)
            # Dry run to check line count
            _, num_lines = render_text_with_font(frame.copy(), transcript_text, (10, h - bar_height + 10), font_path, font_size=28, color=(255, 255, 255), max_width=w-20)
            
            if num_lines > 2:
                # Clear and keep only the latest word that caused the overflow
                if transcript:
                    transcript = [transcript[-1]]
                transcript_text = " ".join(transcript)
            
            frame, _ = render_text_with_font(frame, transcript_text, (10, h - bar_height + 10), font_path, font_size=28, color=(255, 255, 255), max_width=w-20)

            cv2.imshow('Amharic Sign Language Transcription', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
