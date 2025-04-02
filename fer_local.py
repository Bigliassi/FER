# ------------------- fer.py -------------------
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

import pandas as pd
import random

# For the reliability testing
try:
    import pingouin as pg
except ImportError:
    print("Warning: pingouin not installed. ICC and some stats won't be available.")

# SciKeras for bridging Keras & scikit-learn
from scikeras.wrappers import KerasRegressor
import tensorflow as tf
from tensorflow.keras import layers, models

# Additional stats
from scipy.stats import ttest_rel, wilcoxon, shapiro

from sklearn.model_selection import (
    train_test_split,
    KFold,
    GridSearchCV,
    learning_curve
)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression

##############################
#        CONFIGURATION       #
##############################

LOCAL_MODEL_DIR_1 = r"E:\EmotionRecognition\models\rendy-k-face-emotion-recognizer"
LOCAL_MODEL_DIR_2 = r"E:\EmotionRecognition\models\gerhardien-face-emotion"

# Futuristic color palette
PRIMARY_COLOR = "#00f5ff"
ACCENT_COLOR  = "#ff00fd"
BG_COLOR      = "#121212"
TEXT_COLOR    = "#ffffff"

plt.rcParams['text.color'] = TEXT_COLOR
plt.rcParams['axes.labelcolor'] = TEXT_COLOR
plt.rcParams['xtick.color'] = TEXT_COLOR
plt.rcParams['ytick.color'] = TEXT_COLOR
plt.rcParams['axes.edgecolor'] = TEXT_COLOR
plt.rcParams['figure.facecolor'] = BG_COLOR
plt.rcParams['axes.facecolor'] = BG_COLOR
plt.rcParams['savefig.facecolor'] = BG_COLOR

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

##############################
#   LOCAL HF MODEL LOADING   #
##############################

def load_local_face_emotion_model(model_dir):
    logging.debug(f"Attempting to load model from: {model_dir}")
    model_path = pathlib.Path(model_dir).resolve()
    logging.info(f"Resolved local model path => {model_path}")
    if not model_path.is_dir():
        logging.error(f"Local folder does not exist: {model_path}")
        raise FileNotFoundError(f"Local model folder not found: {model_path}")
    processor = AutoImageProcessor.from_pretrained(
        str(model_path),
        local_files_only=True
    )
    model = AutoModelForImageClassification.from_pretrained(
        str(model_path),
        local_files_only=True
    )
    logging.info(f"Successfully loaded local model from {model_path}")
    return model, processor

def local_hf_predict_emotion(frame_bgr, model, processor):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    inputs = processor(images=frame_rgb, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1)[0].cpu().numpy()

    id2label = model.config.id2label
    result = {}
    for idx, label in id2label.items():
        result[label] = float(probs[idx])
    logging.debug(f"Predicted emotions => {result}")
    return result

def ensemble_two_local_models(frame_bgr, model1, proc1, model2, proc2):
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

        if nose_y > (left_eye_y + right_eye_y) / 2 + 0.03:
            return "head_down"
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

    frame_interval = math.floor(fps * 1) if fps > 0 else 5
    annotated_frames = []
    window_scores = []
    frames_processed = 0
    window_aff = []

    for i in range(0, total_frames, frame_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            logging.error(f"Error reading frame at position {i}")
            continue

        local_probs = ensemble_two_local_models(frame, model1, proc1, model2, proc2)
        aff_score = compute_affective_score_ensemble(local_probs)

        sum_others = sum(
            local_probs.get(e, 0)
            for e in ["angry", "sad", "fear", "disgust", "happy", "surprise"]
        )
        face_detected = True
        if local_probs.get("neutral", 0) == 1.0 and sum_others == 0:
            face_detected = False

        mouth_open = detect_mouth_open(frame, face_mesh)
        mouth_open_nosm = is_mouth_open_no_smile(local_probs, mouth_open)
        gesture = detect_gesture(frame, pose)

        frames_processed += 1
        window_aff.append(aff_score)

        annotated = annotate_frame(frame, local_probs, gesture, i, aff_score, mouth_open_nosm)
        annotated_frames.append((i, annotated))

        if frames_processed % 6 == 0:
            avg_aff = np.clip(mean(window_aff), 1, 9)
            window_scores.append(avg_aff)
            window_aff = []

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

def annotate_frame(frame, local_probs, gesture, frame_index, aff_score, mouth_open_no_smile):
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
    text_line2 = f"Affect={aff_score:.2f}/9"
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
    icon_path="E:/EmotionRecognition/assets/logo_icon.png"
):
    from datetime import datetime
    import matplotlib.image as mpimg

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
        # COVER PAGE
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.set_facecolor("#121212")
        fig.patch.set_facecolor("#121212")
        ax.set_axis_off()
        if os.path.isfile(icon_path):
            logo = mpimg.imread(icon_path)
            ax.imshow(logo, extent=[0.1, 2.1, 0.1, 2.1])
        ax.text(
            0.5, 0.75, "Emotion (Affect) Analysis Report",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=28, color="#00ffff", weight="bold"
        )
        dt_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ax.text(
            0.5, 0.6,
            f"Video analyzed: {video_file}\nGenerated on: {dt_now}",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=14, color="#ffffff"
        )
        pdf.savefig(fig)
        plt.close()

        # EXPLANATIONS PAGE
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.set_facecolor("#121212")
        ax.set_axis_off()
        explanation_text = (
            "This report merges predictions from two local Hugging Face models "
            "to determine emotion probabilities in each frame. We compute a simple Affective Score (1–9) "
            "that combines 'happy' & 'neutral', minus 'sad', 'fear', etc.\n\n"
            "Scores are averaged every ~30 seconds. Sample annotated frames are provided, "
            "highlighting gesture and mouth-open detection.\n"
        )
        ax.text(
            0.1, 0.9, "Report Details:", fontsize=18, color="#ff00fd", weight="bold"
        )
        ax.text(
            0.1, 0.75, explanation_text, fontsize=12, color="#ffffff", wrap=True
        )
        pdf.savefig(fig)
        plt.close()

        # PLOTS
        x_axis = range(1, len(window_scores) + 1)
        fig, ax = plt.subplots(figsize=(8.5, 5))
        ax.set_title("Affect Scores Over Time (30s windows)", pad=15)
        ax.plot(
            x_axis, window_scores, marker="o", markersize=6,
            color="#00ffff", label="Affect (1–9)"
        )
        ax.set_xlabel("30-second Window #", labelpad=10)
        ax.set_ylabel("Affect Score (1=Low, 9=High)", labelpad=10)
        ax.set_ylim(1, 9)
        ax.legend(facecolor="#2b2b2b", edgecolor="#999999", loc="upper left")
        pdf.savefig(fig)
        plt.close()

        # ANNOTATED FRAMES
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
#   RELIABILITY CHECK (ICC)  #
##############################

def check_reliability_across_tests(participant_id, list_of_csvs):
    combined = []
    for csvf in list_of_csvs:
        if not os.path.isfile(csvf):
            print(f"File not found: {csvf}")
            continue
        df = pd.read_csv(csvf)
        df['file_source'] = csvf
        combined.append(df)

    if not combined:
        print("No valid CSV files loaded for reliability check.")
        return

    big_df = pd.concat(combined, ignore_index=True)
    if 'pingouin' not in globals():
        print("pingouin not installed. Cannot compute ICC.")
        return
    big_df = big_df.dropna(subset=['participant_report'])
    if len(big_df) < 2:
        print("Not enough data for ICC.")
        return

    df_icc = pd.DataFrame({
        'participant': [participant_id]*len(big_df),
        'rater': big_df['file_source'].values,
        'score': big_df['participant_report'].values
    })
    icc_res = pg.intraclass_corr(data=df_icc, targets='participant', raters='rater', ratings='score')
    print("\nIntraclass Correlation (ICC) results:")
    print(icc_res)

##############################
#   DENSE NN for AFFECT      #
##############################

def create_dense_model(n_hidden=2, n_neurons=32, dropout_rate=0.2, lr=0.001):
    model = models.Sequential()
    model.add(layers.Input(shape=(3,)))
    for _ in range(n_hidden):
        model.add(layers.Dense(n_neurons, activation='relu'))
        if dropout_rate > 0:
            model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(1, activation='linear'))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='mse',
        metrics=['mae']
    )
    return model

keras_reg = KerasRegressor(
    model=create_dense_model,
    n_hidden=2,
    n_neurons=32,
    dropout_rate=0.2,
    lr=0.001,
    epochs=50,
    batch_size=16,
    verbose=0
)

def cohen_d(arr1, arr2):
    diff = arr1 - arr2
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    return mean_diff / std_diff if std_diff != 0 else 0.0

def train_and_evaluate_affect_model(csv_files, test_size=0.2):
    data_frames = []
    for cf in csv_files:
        if not os.path.isfile(cf):
            print(f"File not found: {cf}")
            continue
        df_ = pd.read_csv(cf)
        needed_cols = ['researcher_pred', 'participant_report', 'pretrained_pred']
        if all(col in df_.columns for col in needed_cols):
            data_frames.append(df_[needed_cols].dropna())
        else:
            print(f"Skipping {cf}: missing needed columns.")

    if not data_frames:
        print("No valid data found. Cannot train model.")
        return

    df_all = pd.concat(data_frames, ignore_index=True)
    X = df_all[['researcher_pred', 'pretrained_pred', 'researcher_pred']]
    y = df_all['participant_report']

    if len(X) < 5:
        print("Not enough data samples to train. Need more intervals.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    model_keras = KerasRegressor(model=create_dense_model, verbose=0)
    param_grid = {
        'n_hidden': [1, 2],
        'n_neurons': [16, 32],
        'dropout_rate': [0.0, 0.2],
        'lr': [0.001, 0.0001],
        'batch_size': [8, 16],
        'epochs': [50]
    }
    kfold = KFold(n_splits=3, shuffle=True, random_state=42)
    grid = GridSearchCV(
        estimator=model_keras,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=kfold,
        verbose=1
    )
    grid_result = grid.fit(X_train, y_train)

    best_params = grid_result.best_params_
    print("\nBest Hyperparameters found:")
    print(best_params)

    best_model = grid_result.best_estimator_
    y_pred = best_model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)

    print(f"\nTest MSE: {mse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Test R^2: {r2:.4f}")

    train_sizes, train_scores, val_scores = learning_curve(
        best_model, X_train, y_train, cv=kfold,
        scoring='neg_mean_squared_error',
        train_sizes=np.linspace(0.1, 1.0, 5),
        random_state=42
    )
    train_mse = -train_scores
    val_mse   = -val_scores
    mean_train_mse = np.mean(train_mse, axis=1)
    mean_val_mse   = np.mean(val_mse, axis=1)

    plt.figure()
    plt.plot(train_sizes, mean_train_mse, 'o-', color="r", label="Training MSE")
    plt.plot(train_sizes, mean_val_mse,   'o-', color="g", label="Validation MSE")
    plt.title("Learning Curve (Dense NN)")
    plt.xlabel("Training Examples")
    plt.ylabel("MSE")
    plt.legend(loc="best")
    plt.show()

    y_mean = np.mean(y_train)
    y_base_pred = [y_mean]*len(y_test)
    base_mse = mean_squared_error(y_test, y_base_pred)
    print(f"\nNaive Baseline (Mean) MSE: {base_mse:.4f}")

    model_abs_errors = np.abs(y_test - y_pred)
    base_abs_errors  = np.abs(y_test - y_base_pred)
    diff_errors = model_abs_errors - base_abs_errors
    stat_shapiro, p_shapiro = shapiro(diff_errors) if len(diff_errors) >= 3 else (np.nan, np.nan)
    print(f"\nShapiro test => stat={stat_shapiro:.3f}, p={p_shapiro:.3f}")
    if (not np.isnan(p_shapiro)) and (p_shapiro > 0.05):
        tstat, pval = ttest_rel(model_abs_errors, base_abs_errors)
        print(f"Paired t-test => t={tstat:.3f}, p={pval:.3f}")
        test_used = "Paired t-test"
    else:
        wstat, wpval = wilcoxon(model_abs_errors, base_abs_errors)
        print(f"Wilcoxon => W={wstat:.3f}, p={wpval:.3f}")
        test_used = "Wilcoxon Signed-Rank"

    d_value = cohen_d(model_abs_errors, base_abs_errors)
    print(f"Effect size (Cohen's d): {d_value:.3f}")

    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    y_lin = linreg.predict(X_test)
    mse_lin = mean_squared_error(y_test, y_lin)
    print(f"Linear Regression MSE: {mse_lin:.4f}")

    seeds = [10, 20, 30]
    stabilities = []
    for s in seeds:
        kfold_ = KFold(n_splits=3, shuffle=True, random_state=s)
        scores_ = []
        for train_idx, val_idx in kfold_.split(X_train, y_train):
            X_tr_, X_val_ = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr_, y_val_ = y_train.iloc[train_idx], y_train.iloc[val_idx]
            tmp_model = create_dense_model(
                n_hidden=best_params['n_hidden'],
                n_neurons=best_params['n_neurons'],
                dropout_rate=best_params['dropout_rate'],
                lr=best_params['lr']
            )
            tmp_model.fit(
                X_tr_, y_tr_,
                epochs=best_params['epochs'],
                batch_size=best_params['batch_size'],
                verbose=0
            )
            preds_ = tmp_model.predict(X_val_).flatten()
            scores_.append(mean_squared_error(y_val_, preds_))
        stabilities.append(np.mean(scores_))
    print(f"\nModel stability across seeds {seeds}: MSE => {stabilities}")
    print(f"(Used {test_used} for significance testing vs. baseline)")

##############################
#           MAIN             #
##############################

def main():
    print("This is fer.py main. Usually you call from ui_interface.py")

if __name__ == "__main__":
    main()
