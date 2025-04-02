# ------------------- ui_interface.py -------------------
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import pandas as pd

# Import everything we need from fer.py
from fer_local import (
    load_local_face_emotion_model,
    analyze_emotion_and_gesture,
    generate_report,
    check_reliability_across_tests,
    train_and_evaluate_affect_model
)

# We'll also import these if we want to do extra combining ourselves:
# from fer import create_dense_model, cohen_d  # only if needed

# We'll store the loaded models globally (optional).
global_model_1 = None
global_proc_1 = None
global_model_2 = None
global_proc_2 = None

class EmotionGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Emotion Recognition GUI")
        self.geometry("800x500")

        # Variables
        self.video_path_var = tk.StringVar()
        self.participant_id_var = tk.StringVar()
        self.entries = []  # Will store rows of interval entries

        # Build the interface
        self.create_widgets()

    def create_widgets(self):
        # ------------------
        # 1) VIDEO SELECTION
        # ------------------
        frame_video = ttk.LabelFrame(self, text="Video Selection")
        frame_video.pack(fill="x", padx=5, pady=5)

        btn_browse = ttk.Button(frame_video, text="Select Video", command=self.select_video)
        btn_browse.pack(side="left", padx=5, pady=5)

        self.lbl_video = ttk.Label(frame_video, textvariable=self.video_path_var, width=60)
        self.lbl_video.pack(side="left", padx=5, pady=5)

        # ------------------
        # 2) LOAD MODELS
        # ------------------
        frame_models = ttk.LabelFrame(self, text="Model Loading")
        frame_models.pack(fill="x", padx=5, pady=5)

        btn_load = ttk.Button(frame_models, text="Load HF Models", command=self.load_models)
        btn_load.pack(side="left", padx=5, pady=5)

        self.lbl_load_status = ttk.Label(frame_models, text="Models not loaded")
        self.lbl_load_status.pack(side="left", padx=10)

        # ------------------
        # 3) ANALYZE VIDEO
        # ------------------
        frame_analyze = ttk.LabelFrame(self, text="Analyze")
        frame_analyze.pack(fill="x", padx=5, pady=5)

        btn_analyze = ttk.Button(frame_analyze, text="Analyze Video", command=self.analyze_video)
        btn_analyze.pack(side="left", padx=5, pady=5)

        # ------------------
        # 4) INTERVALS
        # ------------------
        frame_intervals = ttk.LabelFrame(self, text="Define Intervals")
        frame_intervals.pack(fill="both", expand=True, padx=5, pady=5)

        btn_add_interval = ttk.Button(frame_intervals, text="Add Interval", command=self.add_interval)
        btn_add_interval.pack(anchor="w", padx=5, pady=3)

        self.frame_interval_rows = ttk.Frame(frame_intervals)
        self.frame_interval_rows.pack(fill="x", padx=5, pady=5)

        btn_save_csv = ttk.Button(frame_intervals, text="Save Intervals to CSV", command=self.save_intervals_csv)
        btn_save_csv.pack(anchor="e", padx=5, pady=5)

        # ------------------
        # 5) RELIABILITY
        # ------------------
        frame_reliability = ttk.LabelFrame(self, text="Reliability Check")
        frame_reliability.pack(fill="x", padx=5, pady=5)

        lbl_pid = ttk.Label(frame_reliability, text="Participant ID:")
        lbl_pid.pack(side="left", padx=2)
        self.entry_pid = ttk.Entry(frame_reliability, textvariable=self.participant_id_var, width=10)
        self.entry_pid.pack(side="left", padx=2)

        btn_reliab = ttk.Button(frame_reliability, text="Check Reliability", command=self.check_reliability_gui)
        btn_reliab.pack(side="left", padx=5, pady=5)

        # ------------------
        # 6) TRAIN MODEL
        # ------------------
        frame_train = ttk.LabelFrame(self, text="Train Custom DNN")
        frame_train.pack(fill="x", padx=5, pady=5)

        btn_train = ttk.Button(frame_train, text="Train Model (select CSVs)", command=self.train_model_gui)
        btn_train.pack(side="left", padx=5, pady=5)

    # --------------------------------------------------------------
    #   BUTTON COMMANDS
    # --------------------------------------------------------------
    def select_video(self):
        filetypes = [("MP4 files", "*.mp4"), ("All files", "*.*")]
        path = filedialog.askopenfilename(title="Select a Video", filetypes=filetypes)
        if path:
            self.video_path_var.set(path)

    def load_models(self):
        global global_model_1, global_proc_1, global_model_2, global_proc_2
        video_path = self.video_path_var.get()
        if not video_path:
            # Not strictly necessary, but we check if user at least selected something
            messagebox.showwarning("Warning", "No video selected yet, but we'll load models anyway.")
        try:
            from fer import LOCAL_MODEL_DIR_1, LOCAL_MODEL_DIR_2
            global_model_1, global_proc_1 = load_local_face_emotion_model(LOCAL_MODEL_DIR_1)
            global_model_2, global_proc_2 = load_local_face_emotion_model(LOCAL_MODEL_DIR_2)
            self.lbl_load_status.config(text="Models Loaded!")
            messagebox.showinfo("Success", "Models loaded successfully.")
        except Exception as e:
            messagebox.showerror("Error Loading Models", str(e))

    def analyze_video(self):
        global global_model_1, global_proc_1, global_model_2, global_proc_2
        if not global_model_1 or not global_proc_1:
            messagebox.showerror("Error", "Models not loaded yet.")
            return
        video_path = self.video_path_var.get()
        if not video_path or not os.path.exists(video_path):
            messagebox.showerror("Error", "Invalid video path.")
            return

        window_scores, annotated_frames, fps = analyze_emotion_and_gesture(
            video_path,
            global_model_1,
            global_proc_1,
            global_model_2,
            global_proc_2
        )
        if not window_scores:
            messagebox.showerror("Error", "No scores found, possibly unreadable video.")
            return

        result_str = ""
        for i, aff in enumerate(window_scores, start=1):
            result_str += f"30s window {i}: Affect={aff:.2f}/9\n"
        messagebox.showinfo("Analysis Complete", result_str)

        output_pdf = os.path.splitext(video_path)[0] + "_analysis_report.pdf"
        generate_report(video_path, window_scores, annotated_frames, output_pdf=output_pdf)
        messagebox.showinfo("Report Generated", f"PDF saved to {output_pdf}")

    def add_interval(self):
        """
        Dynamically add a new row of entries for: start_time, end_time, researcher_pred, participant_report, pretrained_pred
        """
        row_frame = ttk.Frame(self.frame_interval_rows)
        row_frame.pack(fill="x", pady=2)

        lbl_start = ttk.Label(row_frame, text="Start:")
        lbl_start.grid(row=0, column=0, padx=2)
        ent_start = ttk.Entry(row_frame, width=8)
        ent_start.grid(row=0, column=1, padx=2)

        lbl_end = ttk.Label(row_frame, text="End:")
        lbl_end.grid(row=0, column=2, padx=2)
        ent_end = ttk.Entry(row_frame, width=8)
        ent_end.grid(row=0, column=3, padx=2)

        lbl_rp = ttk.Label(row_frame, text="Researcher:")
        lbl_rp.grid(row=0, column=4, padx=2)
        ent_rp = ttk.Entry(row_frame, width=5)
        ent_rp.grid(row=0, column=5, padx=2)

        lbl_pp = ttk.Label(row_frame, text="Participant:")
        lbl_pp.grid(row=0, column=6, padx=2)
        ent_pp = ttk.Entry(row_frame, width=5)
        ent_pp.grid(row=0, column=7, padx=2)

        lbl_pre = ttk.Label(row_frame, text="Pretrained:")
        lbl_pre.grid(row=0, column=8, padx=2)
        ent_pre = ttk.Entry(row_frame, width=5)
        ent_pre.grid(row=0, column=9, padx=2)

        self.entries.append((ent_start, ent_end, ent_rp, ent_pp, ent_pre))

    def save_intervals_csv(self):
        """
        Reads all interval fields from self.entries and appends them to a CSV.
        If the CSV doesn't exist, create it.
        If it does exist, read existing data, append new rows, and overwrite.
        """
        if not self.entries:
            messagebox.showerror("Error", "No intervals defined.")
            return

        video_path = self.video_path_var.get()
        if not video_path:
            messagebox.showwarning("Missing Video", "No video selected; using 'unknown_video' as ID.")
            video_id = "unknown_video"
        else:
            video_id = os.path.splitext(os.path.basename(video_path))[0]

        # Collect new rows
        new_rows = []
        for i, (s, e, rp, pp, pre) in enumerate(self.entries, start=1):
            start_time = s.get()
            end_time   = e.get()
            try:
                researcher_pred = float(rp.get())
            except ValueError:
                researcher_pred = float('nan')
            try:
                participant_report = float(pp.get())
            except ValueError:
                participant_report = float('nan')
            try:
                pretrained_pred = float(pre.get())
            except ValueError:
                pretrained_pred = float('nan')

            new_rows.append({
                "video_id": video_id,
                "interval_index": i,
                "start_time": start_time,
                "end_time": end_time,
                "researcher_pred": researcher_pred,
                "participant_report": participant_report,
                "pretrained_pred": pretrained_pred
            })

        if not new_rows:
            messagebox.showerror("Error", "No valid rows to save.")
            return

        csv_name = f"{video_id}_intervals.csv"

        # If CSV exists, append; else create
        if os.path.isfile(csv_name):
            # Read old data, append new, overwrite
            old_df = pd.read_csv(csv_name)
            new_df = pd.DataFrame(new_rows)
            combined_df = pd.concat([old_df, new_df], ignore_index=True)
            combined_df.to_csv(csv_name, index=False)
            msg = f"Appended {len(new_rows)} intervals to existing file '{csv_name}'."
        else:
            new_df = pd.DataFrame(new_rows)
            new_df.to_csv(csv_name, index=False)
            msg = f"Created new CSV '{csv_name}' with {len(new_rows)} intervals."

        messagebox.showinfo("CSV Saved", msg)

    def check_reliability_gui(self):
        participant_id = self.participant_id_var.get()
        if not participant_id:
            messagebox.showerror("Error", "Please enter a participant ID first.")
            return
        csv_files = filedialog.askopenfilenames(
            title="Select CSV files for reliability check",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if not csv_files:
            return

        import io
        import sys
        backup_stdout = sys.stdout
        sys.stdout = io.StringIO()
        check_reliability_across_tests(participant_id, csv_files)
        output = sys.stdout.getvalue()
        sys.stdout = backup_stdout

        messagebox.showinfo("Reliability Results", output)

    def train_model_gui(self):
        """
        Asks the user for CSV files to train the model on.
        Then for each CSV, we ask "Is this the first or second test?" 
        We store that in test_session column if not already present.
        """
        csv_files = filedialog.askopenfilenames(
            title="Select CSV files for training the DNN",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if not csv_files:
            return

        # We'll update each CSV to add 'test_session'
        updated_csvs = []
        for cf in csv_files:
            if not os.path.isfile(cf):
                continue

            # Ask user about which test session
            session = self.ask_session_number(cf)
            if session is None:
                # user canceled
                continue

            df_ = pd.read_csv(cf)
            if "test_session" not in df_.columns:
                df_["test_session"] = session
            else:
                # Overwrite or only fill missing?
                df_["test_session"] = session

            df_.to_csv(cf, index=False)
            updated_csvs.append(cf)

        # Now call the train_and_evaluate_affect_model on updated_csvs
        if not updated_csvs:
            messagebox.showwarning("No CSV", "No CSV to train on after session assignment.")
            return

        import io
        import sys
        backup_stdout = sys.stdout
        sys.stdout = io.StringIO()

        train_and_evaluate_affect_model(updated_csvs)

        output = sys.stdout.getvalue()
        sys.stdout = backup_stdout

        messagebox.showinfo("Training Complete", output)

    def ask_session_number(self, csv_file):
        """
        Simple popup to ask "Is this first or second test?"
        Returns 1 or 2 or None if canceled.
        """
        # We'll build a small modal dialog
        dialog = tk.Toplevel(self)
        dialog.title("Session Number")
        lbl = ttk.Label(dialog, text=f"CSV: {os.path.basename(csv_file)}\nIs this from session #1 or #2?")
        lbl.pack(padx=10, pady=10)

        choice_var = tk.StringVar(value="1")  # default "1"

        def on_ok():
            dialog.destroy()

        def on_cancel():
            choice_var.set("")  # empty means None
            dialog.destroy()

        rb1 = ttk.Radiobutton(dialog, text="Session #1", variable=choice_var, value="1")
        rb1.pack(anchor="w", padx=10, pady=2)
        rb2 = ttk.Radiobutton(dialog, text="Session #2", variable=choice_var, value="2")
        rb2.pack(anchor="w", padx=10, pady=2)

        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=5)
        btn_ok = ttk.Button(btn_frame, text="OK", command=on_ok)
        btn_ok.pack(side="left", padx=5)
        btn_cancel = ttk.Button(btn_frame, text="Cancel", command=on_cancel)
        btn_cancel.pack(side="left", padx=5)

        # Make it modal
        dialog.transient(self)
        dialog.grab_set()
        self.wait_window(dialog)

        val = choice_var.get()
        if val in ["1", "2"]:
            return int(val)
        return None


if __name__ == "__main__":
    app = EmotionGUI()
    app.mainloop()
