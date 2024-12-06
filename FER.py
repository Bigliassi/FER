import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow INFO and WARNING messages

import warnings
warnings.filterwarnings('ignore', message='SymbolDatabase.GetPrototype() is deprecated')
warnings.filterwarnings('ignore', category=UserWarning)  # Ignore all UserWarnings

import logging
logging.basicConfig(filename='emotion_recognition.log', level=logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

import cv2
import numpy as np
from fer import FER
import math
import mediapipe as mp
from statistics import mean

def detect_gesture(frame, pose):
    # Detect head and body position using MediaPipe Pose
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.pose_landmarks:
        # Extract landmarks to detect gestures like looking away or head down
        landmarks = results.pose_landmarks.landmark
        nose_y = landmarks[mp.solutions.pose.PoseLandmark.NOSE].y
        left_eye_y = landmarks[mp.solutions.pose.PoseLandmark.LEFT_EYE].y
        right_eye_y = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_EYE].y

        # Gesture: Head down if nose is significantly lower than eyes
        if nose_y > (left_eye_y + right_eye_y) / 2 + 0.05:
            return "head_down"

        # Gesture: Looking elsewhere (detected via head rotation)
        left_eye_x = landmarks[mp.solutions.pose.PoseLandmark.LEFT_EYE].x
        right_eye_x = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_EYE].x
        eye_distance = abs(left_eye_x - right_eye_x)

        # If eyes are too close together or far apart, might be looking away
        if eye_distance < 0.02 or eye_distance > 0.1:
            return "looking_elsewhere"

    return "neutral"

def analyze_emotion_and_gesture(video_path, emotion_detector, pose):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total number of frames
    frame_interval = math.floor(fps * 5)  # Analyze frame every 5 seconds

    # Store the affective scores for each 30-second window
    frame_count = 0
    emotions_scores = []
    window_scores = []
    gesture_adjustment = 0

    # For suppressing repeated warnings
    face_not_detected_frames = []

    for i in range(0, total_frames, frame_interval):
        # Set the current frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            logging.error(f"Error: Unable to read frame at position {i}.")
            continue  # Skip to the next frame

        # Detect emotions in the current frame
        result = emotion_detector.detect_emotions(frame)
        face_detected = False
        if result:
            face_detected = True
            emotions = result[0]["emotions"]
            positive_emotions = emotions["happy"] + emotions["neutral"]
            negative_emotions = emotions["angry"] + emotions["sad"] + emotions["disgust"] + emotions["fear"]
            # Adjust affective_score to range from 1 to 9
            affective_score = ((positive_emotions - negative_emotions + 1) * 4) + 1  # Normalize to [1, 9]
            affective_score = np.clip(affective_score, 1, 9)
            emotions_scores.append(affective_score)
        else:
            # No face detected, focus on gestures
            face_not_detected_frames.append(i)

        # Detect gestures in the current frame
        gesture = detect_gesture(frame, pose)
        if gesture == "head_down" or gesture == "looking_elsewhere":
            gesture_adjustment = -1  # Reduce affective score by 2 for discomfort gestures
        elif gesture == "neutral":
            gesture_adjustment = 0  # No adjustment
        else:
            # Other gestures can be added here with corresponding adjustments
            pass

        # If no face detected, but negative gesture is detected
        if not face_detected and gesture in ["head_down", "looking_elsewhere"]:
            # Assume negative affective score due to gestures indicating fatigue or disengagement
            affective_score = 3  # Lower score to indicate potential fatigue
            emotions_scores.append(affective_score)

        # Collect scores for 30-second window
        frame_count += 1
        if frame_count % 6 == 0:  # Every 30 seconds (6 frames per 30s)
            # Adjust the average score for the window with gesture sensitivity
            if emotions_scores:
                avg_score = mean(emotions_scores) + gesture_adjustment
            else:
                # If no emotions detected, base score on gesture adjustment
                avg_score = 5 + gesture_adjustment  # Neutral score adjusted by gestures
            avg_score = np.clip(avg_score, 1, 9)  # Ensure score stays in [1, 9]
            window_scores.append(avg_score)
            emotions_scores = []  # Reset for next window
            gesture_adjustment = 0  # Reset gesture adjustment

    cap.release()

    # Log the frames where no face was detected
    if face_not_detected_frames:
        logging.warning(f"No face detected at frames: {face_not_detected_frames}")

    return window_scores

def main():
    # Get the folder path
    folder_path = 'E:/EmotionRecognition'

    # Ask user for the video file name
    video_file = input("Enter the name of the video file (e.g., with .mp4 or .mov extension): ")

    # Construct full video file path
    video_path = os.path.join(folder_path, video_file)

    # Check if the video file exists
    if not os.path.exists(video_path):
        print(f"Error: The video file '{video_file}' does not exist in the folder '{folder_path}'.")
        return

    # Load the pre-trained facial expression recognition model
    emotion_detector = FER(mtcnn=True)

    # Initialize MediaPipe for gesture recognition
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False)

    # Analyze the video for emotions and gestures
    print(f"Analyzing {video_file}...")
    scores = analyze_emotion_and_gesture(video_path, emotion_detector, pose)

    if scores:
        for idx, score in enumerate(scores, 1):
            print(f"30-second window {idx}: Affective Score = {score}/9")
    else:
        print("No affective scores could be calculated.")

if __name__ == "__main__":
    main()
