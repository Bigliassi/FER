import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow INFO messages

import warnings
warnings.filterwarnings('ignore', message='SymbolDatabase.GetPrototype() is deprecated')
warnings.filterwarnings('ignore', category=UserWarning)  # Ignore all UserWarnings

import logging
logging.basicConfig(
    filename='emotion_recognition.log',
    level=logging.DEBUG,  # change to DEBUG for more verbose output
    format='%(asctime)s %(levelname)s:%(message)s'
)
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

import cv2
import numpy as np
import math
from statistics import mean
import datetime
import pathlib

import torch
from transformers import AutoModelForImageClassification, AutoImageProcessor

import mediapipe as mp
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import matplotlib.patches as patches

##############################
#        CONFIGURATION       #
##############################

# Path to your local Hugging Face model folders
# Replace backslashes with forward slashes or ensure proper escaping
# We'll convert them with pathlib below, so "E:/..." or r"E:\..." is okay

LOCAL_MODEL_DIR_1 = r"E:\EmotionRecognition\models\rendy-k-face-emotion-recognizer"
LOCAL_MODEL_DIR_2 = r"E:\EmotionRecognition\models\gerhardien-face-emotion"

# Make sure these folders contain:
#  - model.safetensors (or pytorch_model.bin)
#  - config.json
#  - preprocessor_config.json
# etc.

# Futuristic color palette
PRIMARY_COLOR = "#00f5ff"  # neon-cyan
ACCENT_COLOR  = "#ff00fd"  # neon-magenta
BG_COLOR      = "#121212"  # near-black
TEXT_COLOR    = "#ffffff"  # white

# Plot font settings
plt.rcParams['text.color'] = TEXT_COLOR
plt.rcParams['axes.labelcolor'] = TEXT_COLOR
plt.rcParams['xtick.color'] = TEXT_COLOR
plt.rcParams['ytick.color'] = TEXT_COLOR
plt.rcParams['axes.edgecolor'] = TEXT_COLOR
plt.rcParams['figure.facecolor'] = BG_COLOR
plt.rcParams['axes.facecolor'] = BG_COLOR
plt.rcParams['savefig.facecolor'] = BG_COLOR

##############################
#   LOCAL HF MODEL LOADING   #
##############################

def load_local_face_emotion_model(model_dir):
    """
    Loads a local face-emotion image-classification model from a folder,
    ensuring we treat model_dir as a local path (not a remote repo).
    Returns (model, processor).
    """
    logging.debug(f"Attempting to load model from: {model_dir}")

    # 1) Resolve the path absolutely to ensure huggingface does not parse as remote
    model_path = pathlib.Path(model_dir).resolve()
    logging.info(f"Resolved local model path => {model_path}")

    if not model_path.is_dir():
        logging.error(f"Local folder does not exist: {model_path}")
        raise FileNotFoundError(f"Local model folder not found: {model_path}")

    # Optional: Check for essential files
    essential_files = ["config.json", "preprocessor_config.json"]
    # At least one of "model.safetensors" or "pytorch_model.bin"
    has_safetensors = (model_path / "model.safetensors").is_file()
    has_bin = (model_path / "pytorch_model.bin").is_file()
    missing_essentials = []
    for ef in essential_files:
        if not (model_path / ef).is_file():
            missing_essentials.append(ef)
    if (not has_safetensors) and (not has_bin):
        missing_essentials.append("model.safetensors/pytorch_model.bin")

    if missing_essentials:
        logging.warning(f"Missing crucial files in {model_path}: {missing_essentials}")

    # 2) Load the processor & model with local_files_only=True
    logging.debug("Loading AutoImageProcessor...")
    processor = AutoImageProcessor.from_pretrained(
        str(model_path),
        local_files_only=True
    )
    logging.debug("Loading AutoModelForImageClassification...")
    model = AutoModelForImageClassification.from_pretrained(
        str(model_path),
        local_files_only=True
    )
    logging.info(f"Successfully loaded local model from {model_path}")
    return model, processor

def local_hf_predict_emotion(frame_bgr, model, processor):
    """
    Runs inference on a single frame (OpenCV BGR) using the local Hugging Face
    image-classification model. Returns a dict {emotion_label: probability}.
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    inputs = processor(images=frame_rgb, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1)[0].cpu().numpy()

    id2label = model.config.id2label  # {0:'happy',1:'sad',...}
    result = {}
    for idx, label in id2label.items():
        result[label] = float(probs[idx])
    logging.debug(f"Predicted emotions => {result}")
    return result

def ensemble_two_local_models(frame_bgr, model1, proc1, model2, proc2):
    """
    Gets emotion probabilities from both local models,
    merges them by simple averaging, normalizes to sum=1.
    """
    probs1 = local_hf_predict_emotion(frame_bgr, model1, proc1)
    probs2 = local_hf_predict_emotion(frame_bgr, model2, proc2)

    all_labels = set(probs1.keys()).union(probs2.keys())
    combined = {}
    for lab in all_labels:
        v1 = probs1.get(lab, 0.0)
        v2 = probs2.get(lab, 0.0)
        combined[lab] = (v1 + v2) / 2.0

    s = sum(combined.values())
    if s > 0:
        for k in combined:
            combined[k] /= s
    else:
        combined = {"neutral": 1.0}

    logging.debug(f"Ensembled local models => {combined}")
    return combined

##############################
#   MOUTH OPEN DETECTION     #
##############################

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

UPPER_LIP_INNER = 13
LOWER_LIP_INNER = 14
MOUTH_OPEN_THRESHOLD = 0.02

def detect_mouth_open(frame, face_mesh_obj, threshold=MOUTH_OPEN_THRESHOLD):
    results = face_mesh_obj.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        return False

    face_landmarks = results.multi_face_landmarks[0]
    h, w, _ = frame.shape

    upper_lip = face_landmarks.landmark[UPPER_LIP_INNER]
    lower_lip = face_landmarks.landmark[LOWER_LIP_INNER]

    dist = abs((lower_lip.y * h) - (upper_lip.y * h))
    norm_dist = dist / float(h)
    return norm_dist > threshold

def is_mouth_open_no_smile(combined_probs, mouth_open, smile_threshold=0.3):
    happy_prob = combined_probs.get("happy", 0.0)
    if mouth_open and happy_prob < smile_threshold:
        return True
    return False

##############################
#   POSE / GESTURE DETECTION #
##############################

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)

def detect_gesture(frame, pose_obj):
    results = pose_obj.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        nose_y = landmarks[mp.solutions.pose.PoseLandmark.NOSE].y
        left_eye_y = landmarks[mp.solutions.pose.PoseLandmark.LEFT_EYE].y
        right_eye_y = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_EYE].y

        # If nose is significantly lower than eyes => 'head_down'
        if nose_y > (left_eye_y + right_eye_y) / 2 + 0.03:
            return "head_down"

        # If eyes are too close/far => might be 'looking_elsewhere'
        left_eye_x = landmarks[mp.solutions.pose.PoseLandmark.LEFT_EYE].x
        right_eye_x = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_EYE].x
        eye_dist = abs(left_eye_x - right_eye_x)
        if eye_dist < 0.02 or eye_dist > 0.1:
            return "looking_elsewhere"

    return "neutral"

##############################
#   AFFECTIVE SCORE          #
##############################

def compute_affective_score_ensemble(local_probs):
    positive = local_probs.get("happy", 0.0) + local_probs.get("neutral", 0.0)
    negative = (
        local_probs.get("sad", 0.0) +
        local_probs.get("angry", 0.0) +
        local_probs.get("disgust", 0.0) +
        local_probs.get("fear", 0.0)
    )
    score = ((positive - negative + 1) * 4) + 1
    score = np.clip(score, 1, 9)
    return score

##############################
#   EXERTION SCORE           #
##############################

def compute_exertion_score(
    gesture,
    no_face_detected,
    mouth_open_no_smile=False,
    neg_emotions=0.0,
    consecutive_mouth_open=0,
    history=None,
    happy_prob=0.0
):
    base = 4.0

    if gesture == "head_down":
        base += 2
    elif gesture == "looking_elsewhere":
        base += 1

    if no_face_detected:
        base += 1

    if mouth_open_no_smile:
        base += 1

    if consecutive_mouth_open >= 3:
        base += 1

    if neg_emotions > 0.4:
        base += 1

    if history is not None and len(history) == 6:
        hd_count = sum(1 for h in history if h["gesture"] == "head_down")
        mo_count = sum(1 for h in history if h["mouth_open_no_smile"])
        base += hd_count * 0.1
        base += mo_count * 0.1

    if happy_prob > 0.5:
        base -= 0.5

    return np.clip(base, 1, 9)

##############################
#   MAIN ANALYSIS LOOP       #
##############################

def analyze_emotion_and_gesture(video_path, model1, proc1, model2, proc2):
    logging.info(f"Starting analysis for video => {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        msg = f"Error: Could not open video {video_path}"
        logging.error(msg)
        print(msg)
        return None, None, None

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        logging.warning(f"FPS is non-positive ({fps}). The video might be corrupted.")
    logging.debug(f"Video => FPS={fps}, TotalFrames={total_frames}")

    frame_interval = math.floor(fps * 1) if fps > 0 else 5

    annotated_frames = []
    window_scores = []  # (aff, exh) per 30s

    frames_processed = 0
    window_aff = []
    window_exh = []

    exertion_history = []
    consecutive_mouth_open = 0
    smoothed_exertion = None
    alpha = 0.7

    for i in range(0, total_frames, frame_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            logging.error(f"Error reading frame at position {i}")
            continue

        # 1) Ensemble local models
        local_probs = ensemble_two_local_models(frame, model1, proc1, model2, proc2)

        # 2) Affective Score
        aff_score = compute_affective_score_ensemble(local_probs)

        # 3) Check face detection
        face_detected = True
        sum_others = sum(local_probs.get(e,0) for e in ["angry","sad","fear","disgust","happy","surprise"])
        if local_probs.get("neutral",0) == 1.0 and sum_others == 0:
            face_detected = False

        # 4) Mouth open
        mouth_open = detect_mouth_open(frame, face_mesh)
        mouth_open_nosm = is_mouth_open_no_smile(local_probs, mouth_open)
        if mouth_open_nosm:
            consecutive_mouth_open += 1
        else:
            consecutive_mouth_open = 0

        # 5) Gesture
        gesture = detect_gesture(frame, pose)

        # 6) Negative emotions
        neg_emo = local_probs.get("sad",0) + local_probs.get("fear",0)

        # 7) Rolling history
        exertion_history.append({
            "gesture": gesture,
            "mouth_open_no_smile": mouth_open_nosm,
        })
        if len(exertion_history) > 6:
            exertion_history.pop(0)

        # 8) Raw Exertion
        raw_exh = compute_exertion_score(
            gesture=gesture,
            no_face_detected=(not face_detected),
            mouth_open_no_smile=mouth_open_nosm,
            neg_emotions=neg_emo,
            consecutive_mouth_open=consecutive_mouth_open,
            history=exertion_history,
            happy_prob=local_probs.get("happy",0)
        )

        # 9) Alpha smoothing
        if smoothed_exertion is None:
            smoothed_exertion = raw_exh
        else:
            smoothed_exertion = alpha * smoothed_exertion + (1 - alpha) * raw_exh
        final_exh = np.clip(smoothed_exertion, 1, 9)

        frames_processed += 1
        window_aff.append(aff_score)
        window_exh.append(final_exh)

        # Annotate
        annotated = annotate_frame(frame, local_probs, gesture, i, aff_score, final_exh, mouth_open_nosm)
        annotated_frames.append((i, annotated))

        # Every 6 frames => ~30s
        if frames_processed % 6 == 0:
            avg_aff = np.clip(mean(window_aff), 1, 9)
            avg_exh = np.clip(mean(window_exh), 1, 9)
            window_scores.append((avg_aff, avg_exh))
            window_aff = []
            window_exh = []

    cap.release()

    if not window_scores:
        logging.info("No scores found (possibly empty video).")
        print("No scores found. Possibly an empty or unreadable video.")
        return None, None, None

    logging.info(f"Finished analyzing. Generated {len(window_scores)} window scores.")
    return window_scores, annotated_frames, fps

##############################
#   ANNOTATION / REPORT      #
##############################

def annotate_frame(frame, local_probs, gesture, frame_index, aff_score, exh_score, mouth_open_no_smile):
    annotated = frame.copy()

    cv2.putText(
        annotated,
        f"Gesture: {gesture}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 255),
        2
    )
    text_line2 = f"Affect={aff_score:.2f}/9 Exertion={exh_score:.2f}/9"
    cv2.putText(
        annotated,
        text_line2,
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )
    if mouth_open_no_smile:
        cv2.putText(
            annotated,
            "Mouth Open (No Smile)",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )
    else:
        cv2.putText(
            annotated,
            f"Frame: {frame_index}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )
    return annotated

def generate_report(
    video_file,
    window_scores,
    annotated_frames,
    output_pdf="analysis_report_v2.pdf",
    icon_path="E:/EmotionRecognition/assets/logo_icon.png"  # Example icon
):
    """
    Generates an improved PDF report:
    - Consistent static Y-axis range (1 to 9) for graphs.
    - Polished design with better fonts, alignment, and spacing.
    - Enhanced descriptions and explanations.
    """
    from datetime import datetime
    import matplotlib.patches as patches
    import matplotlib.image as mpimg

    # -----------------------------
    # 1) Improved Matplotlib Styles
    # -----------------------------
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['patch.linewidth'] = 1.8
    plt.rcParams['lines.linewidth'] = 2.5
    plt.rcParams['figure.facecolor'] = "#222222"
    plt.rcParams['axes.facecolor'] = "#121212"
    plt.rcParams['text.color'] = "#ffffff"
    plt.rcParams['axes.labelcolor'] = "#ffffff"
    plt.rcParams['xtick.color'] = "#ffffff"
    plt.rcParams['ytick.color'] = "#ffffff"
    plt.rcParams['axes.edgecolor'] = "#aaaaaa"

    with PdfPages(output_pdf) as pdf:

        # =====================
        # COVER PAGE
        # =====================
        fig, ax = plt.subplots(figsize=(8.5, 11))  # Standard letter size
        ax.set_facecolor("#121212")
        fig.patch.set_facecolor("#121212")
        ax.set_axis_off()

        # Insert Icon or Logo
        if os.path.isfile(icon_path):
            logo = mpimg.imread(icon_path)
            ax.imshow(logo, extent=[0.1, 2.1, 0.1, 2.1])  # Adjust placement

        # Title with Shadow Effect
        ax.text(
            0.5, 0.75, "Emotion & Exertion Analysis Report",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=28, color="#00ffff", weight="bold"
        )
        # Subtitle
        dt_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ax.text(
            0.5, 0.6,
            f"Video analyzed: {video_file}\nGenerated on: {dt_now}",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=14, color="#ffffff"
        )
        pdf.savefig(fig)
        plt.close()

        # =====================
        # EXPLANATIONS PAGE
        # =====================
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.set_facecolor("#121212")
        ax.set_axis_off()

        explanation_text = (
            "This report merges predictions from two local Hugging Face models "
            "to determine emotion probabilities in each frame. Metrics:\n\n"
            "  • Affective Score (1–9): Combines 'happy' & 'neutral', minus 'sad', 'fear', etc.\n"
            "  • Exertion Score (1–9): Detects fatigue via gestures, mouth open, and negative emotions.\n\n"
            "Smoothing (70% previous, 30% new) avoids sudden spikes in Exertion."
        )
        ax.text(
            0.1, 0.9, "Report Details:", fontsize=18, color="#ff00fd", weight="bold"
        )
        ax.text(
            0.1, 0.75, explanation_text, fontsize=12, color="#ffffff", wrap=True
        )
        pdf.savefig(fig)
        plt.close()

        # =====================
        # PLOTS
        # =====================
        aff_values = [ws[0] for ws in window_scores]
        exh_values = [ws[1] for ws in window_scores]
        x_axis = range(1, len(window_scores) + 1)

        fig, ax = plt.subplots(figsize=(8.5, 5))
        ax.set_title("Affective vs. Exertion Scores Over Time", pad=15)
        ax.plot(
            x_axis, aff_values, marker="o", markersize=6,
            color="#00ffff", label="Affective (1–9)"
        )
        ax.plot(
            x_axis, exh_values, marker="s", markersize=6,
            color="#ff00fd", label="Exertion (1–9)"
        )
        ax.set_xlabel("30-second Window #", labelpad=10)
        ax.set_ylabel("Score (1=Low, 9=High)", labelpad=10)
        ax.set_ylim(1, 9)  # Static Y-axis range
        ax.legend(facecolor="#2b2b2b", edgecolor="#999999", loc="upper left")
        pdf.savefig(fig)
        plt.close()

        # =====================
        # ANNOTATED FRAMES
        # =====================
        max_frames = 6
        step = max(1, len(annotated_frames) // max_frames)
        selected_frames = annotated_frames[::step][:max_frames]

        for idx, (frame_idx, frame_img) in enumerate(selected_frames, start=1):
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.imshow(cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB))
            ax.set_title(f"Annotated Frame #{frame_idx}", color="#ffffff", fontsize=14)
            ax.axis("off")
            pdf.savefig(fig)
            plt.close()

    print(f"Enhanced report saved to {output_pdf}")

##############################
#           MAIN             #
##############################

def main():
    # Debug messages about environment
    logging.debug("Starting main function. Checking environment paths.")
    folder_path = 'E:/EmotionRecognition'

    video_file = input("Enter the video file name (e.g., myvideo.mp4): ")
    video_path = os.path.join(folder_path, video_file)
    logging.debug(f"Full video path => {video_path}")

    if not os.path.exists(video_path):
        msg = f"Error: The video file '{video_file}' does not exist in '{folder_path}'."
        logging.error(msg)
        print(msg)
        return

    print("Loading local Hugging Face models from disk...")
    logging.info("Loading local HF models #1 and #2 ...")

    # Load the first local model
    hf_model_1, hf_proc_1 = load_local_face_emotion_model(LOCAL_MODEL_DIR_1)
    # Load the second local model
    hf_model_2, hf_proc_2 = load_local_face_emotion_model(LOCAL_MODEL_DIR_2)

    print(f"Analyzing {video_file} using two local HF face-emotion models. Please wait...")
    logging.info(f"Begin analyze_emotion_and_gesture on {video_path}")

    window_scores, annotated_frames, fps = analyze_emotion_and_gesture(
        video_path,
        hf_model_1,
        hf_proc_1,
        hf_model_2,
        hf_proc_2
    )

    if not window_scores:
        logging.warning("No final scores computed, exiting.")
        return

    # Print final window-level results
    for i, (aff, exh) in enumerate(window_scores, start=1):
        print(f"30s window {i}: Affective={aff:.2f}/9, Exertion={exh:.2f}/9")

    # Generate PDF report
    output_pdf = os.path.splitext(video_file)[0] + "_analysis_report.pdf"
    generate_report(video_file, window_scores, annotated_frames, output_pdf=output_pdf)

if __name__ == "__main__":
    main()
