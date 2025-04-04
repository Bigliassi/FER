# ------------------- ui_interface.py -------------------
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, Canvas, Scale, font
import os
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
from PIL import Image, ImageTk

# Import everything we need from fer_local.py
from fer_local import (
    load_local_face_emotion_model,
    analyze_emotion_and_gesture,
    generate_report,
    check_reliability_across_tests,
    train_and_evaluate_affect_model
)

# We'll store the loaded models globally
global_model_1 = None
global_proc_1 = None
global_model_2 = None
global_proc_2 = None

class ModernWidget(ttk.Frame):
    """Base class for modern-looking widgets with consistent styling"""
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.configure(padding="10")
        
class EmotionGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Emotion Recognition System")
        self.geometry("1200x800")
        self.configure(bg="#121212")
        
        # Define colors and fonts
        self.bg_color = "#121212"
        self.accent_color = "#00f5ff"  
        self.secondary_color = "#ff00fd"
        self.text_color = "#ffffff"
        
        # Create and configure custom fonts
        self.default_font = font.nametofont("TkDefaultFont")
        self.default_font.configure(family="Segoe UI", size=10)
        
        self.heading_font = font.Font(family="Segoe UI", size=14, weight="bold")
        self.subheading_font = font.Font(family="Segoe UI", size=12, weight="bold")
        self.label_font = font.Font(family="Segoe UI", size=10)
        self.button_font = font.Font(family="Segoe UI", size=10, weight="bold")
        
        # Apply a modern theme
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Configure common styles
        self.style.configure('TFrame', background=self.bg_color)
        self.style.configure('TLabel', background=self.bg_color, foreground=self.text_color, font=self.label_font)
        self.style.configure('TLabelframe', background=self.bg_color, foreground=self.text_color, font=self.subheading_font)
        self.style.configure('TLabelframe.Label', background=self.bg_color, foreground=self.accent_color, font=self.subheading_font)
        
        # Button styling
        self.style.configure('TButton', 
                           background=self.accent_color, 
                           foreground=self.bg_color, 
                           font=self.button_font,
                           borderwidth=0, 
                           focusthickness=3, 
                           focuscolor=self.secondary_color)
        self.style.map('TButton', 
                     background=[('active', self.secondary_color), 
                                 ('pressed', self.bg_color)],
                     foreground=[('pressed', self.accent_color)])
        
        # Notebook styling
        self.style.configure('TNotebook', background=self.bg_color, tabmargins=[2, 5, 2, 0])
        self.style.configure('TNotebook.Tab', 
                           background=self.bg_color, 
                           foreground=self.text_color,
                           font=self.label_font,
                           padding=[10, 5],
                           borderwidth=0)
        self.style.map('TNotebook.Tab', 
                     background=[('selected', self.accent_color)],
                     foreground=[('selected', self.bg_color)])
        
        # Entry styling
        self.style.configure('TEntry', 
                           fieldbackground='#333333', 
                           foreground=self.text_color,
                           borderwidth=1,
                           relief='flat',
                           insertcolor=self.accent_color)
        
        # Variables
        self.video_path_var = tk.StringVar()
        self.participant_id_var = tk.StringVar()
        self.video_duration = tk.StringVar(value="Duration: N/A")
        self.crop_start = tk.DoubleVar(value=0)
        self.crop_end = tk.DoubleVar(value=100)
        self.portions = tk.IntVar(value=1)
        self.portions_data = []  # Will store data for each portion
        
        # Main container with tabs
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Create tabs
        self.tab_video = ttk.Frame(self.notebook, style='TFrame')
        self.tab_analysis = ttk.Frame(self.notebook, style='TFrame')
        self.tab_reliability = ttk.Frame(self.notebook, style='TFrame')
        self.tab_model = ttk.Frame(self.notebook, style='TFrame')
        self.tab_visualization = ttk.Frame(self.notebook, style='TFrame')
        
        self.notebook.add(self.tab_video, text="Video Selection")
        self.notebook.add(self.tab_analysis, text="Analysis")
        self.notebook.add(self.tab_reliability, text="Reliability")
        self.notebook.add(self.tab_model, text="Model Validation")
        self.notebook.add(self.tab_visualization, text="Decision Visualization")
        
        # Build each tab's interface
        self.create_video_tab()
        self.create_analysis_tab()
        self.create_reliability_tab()
        self.create_model_tab()
        self.create_visualization_tab()
        
        # Status bar
        self.status_frame = ttk.Frame(self, style='TFrame')
        self.status_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=15, pady=5)
        self.status_label = ttk.Label(self.status_frame, text="Ready")
        self.status_label.pack(side=tk.LEFT)
        self.model_status = ttk.Label(self.status_frame, text="Models: Not Loaded")
        self.model_status.pack(side=tk.RIGHT)
        
        # Store original versions of video frame (to prevent stretching)
        self.original_preview_img = None

    def create_video_tab(self):
        """Create the video selection and processing tab"""
        frame = ttk.Frame(self.tab_video, style='TFrame')
        frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Video selection section
        video_select_frame = ttk.LabelFrame(frame, text="Select Video", style='TFrame')
        video_select_frame.pack(fill=tk.X, padx=5, pady=10)
        
        btn_browse = ttk.Button(video_select_frame, text="Browse Video", command=self.select_video)
        btn_browse.pack(side=tk.LEFT, padx=10, pady=15)
        
        path_label = ttk.Label(video_select_frame, textvariable=self.video_path_var)
        path_label.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)
        
        duration_label = ttk.Label(video_select_frame, textvariable=self.video_duration)
        duration_label.pack(side=tk.RIGHT, padx=10)
        
        # Video preview frame
        preview_frame = ttk.LabelFrame(frame, text="Video Preview", style='TFrame')
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=10)
        
        preview_container = ttk.Frame(preview_frame)
        preview_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        preview_container.pack_propagate(False)  # Prevent propagation to maintain size
        
        self.preview_canvas = Canvas(preview_container, bg=self.bg_color, highlightthickness=0)
        self.preview_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Video cropping section
        crop_frame = ttk.LabelFrame(frame, text="Video Cropping", style='TFrame')
        crop_frame.pack(fill=tk.X, padx=5, pady=10)
        
        ttk.Label(crop_frame, text="Start %:").grid(row=0, column=0, padx=10, pady=10)
        start_scale = Scale(crop_frame, from_=0, to=99, orient=tk.HORIZONTAL, 
                          variable=self.crop_start, bg=self.bg_color, fg=self.text_color,
                          highlightthickness=0, troughcolor=self.accent_color)
        start_scale.grid(row=0, column=1, padx=10, pady=10, sticky=tk.EW)
        
        ttk.Label(crop_frame, text="End %:").grid(row=0, column=2, padx=10, pady=10)
        end_scale = Scale(crop_frame, from_=1, to=100, orient=tk.HORIZONTAL, 
                        variable=self.crop_end, bg=self.bg_color, fg=self.text_color,
                        highlightthickness=0, troughcolor=self.accent_color)
        end_scale.grid(row=0, column=3, padx=10, pady=10, sticky=tk.EW)
        end_scale.set(100)
        
        # Configure grid weights for scaling
        crop_frame.columnconfigure(1, weight=1)
        crop_frame.columnconfigure(3, weight=1)
        
        # Video portions section
        portions_frame = ttk.LabelFrame(frame, text="Video Portions", style='TFrame')
        portions_frame.pack(fill=tk.X, padx=5, pady=10)
        
        ttk.Label(portions_frame, text="Number of equal portions:").pack(side=tk.LEFT, padx=10, pady=10)
        portions_spin = ttk.Spinbox(portions_frame, from_=1, to=10, textvariable=self.portions, width=5)
        portions_spin.pack(side=tk.LEFT, padx=10, pady=10)
        
        # Buttons to proceed
        btn_frame = ttk.Frame(frame, style='TFrame')
        btn_frame.pack(fill=tk.X, padx=5, pady=15)
        
        btn_load_models = ttk.Button(btn_frame, text="1. Load Models", command=self.load_models)
        btn_load_models.pack(side=tk.LEFT, padx=10, pady=10)
        
        btn_process = ttk.Button(btn_frame, text="2. Process Video & Continue to Analysis", 
                               command=self.process_video_and_switch)
        btn_process.pack(side=tk.RIGHT, padx=10, pady=10)

    def create_analysis_tab(self):
        """Create the analysis tab for entering and viewing affective states"""
        frame = ttk.Frame(self.tab_analysis, style='TFrame')
        frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Top frame for results visualization
        viz_frame = ttk.LabelFrame(frame, text="Affective States Visualization", style='TFrame')
        viz_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=10)
        
        # We'll add a matplotlib figure for visualization (now using line graph)
        self.fig = Figure(figsize=(5, 4), dpi=100, facecolor=self.bg_color)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor(self.bg_color)
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Bottom frame for entering values
        data_frame = ttk.LabelFrame(frame, text="Enter Affective States Data", style='TFrame')
        data_frame.pack(fill=tk.X, padx=5, pady=10)
        
        # We'll create a scrollable frame to hold the portion entries
        scroll_container = ttk.Frame(data_frame)
        scroll_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        scrollbar = ttk.Scrollbar(scroll_container, orient="vertical")
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        canvas = tk.Canvas(scroll_container, bg=self.bg_color, highlightthickness=0,
                         yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar.config(command=canvas.yview)
        
        self.portions_container = ttk.Frame(canvas, style='TFrame')
        canvas.create_window((0, 0), window=self.portions_container, anchor="nw", tags="portions_frame")
        
        def on_configure(event):
            """Update scrollregion when the frame changes size"""
            canvas.configure(scrollregion=canvas.bbox("all"))
            
        self.portions_container.bind("<Configure>", on_configure)
        
        # Validation button
        validate_frame = ttk.Frame(data_frame, style='TFrame')
        validate_frame.pack(fill=tk.X, padx=5, pady=10)
        
        btn_validate = ttk.Button(validate_frame, text="Validate Responses", 
                                command=self.validate_responses)
        btn_validate.pack(side=tk.RIGHT, padx=10, pady=5)
        
        # Buttons for analysis actions
        btn_frame = ttk.Frame(frame, style='TFrame')
        btn_frame.pack(fill=tk.X, padx=5, pady=10)
        
        btn_save = ttk.Button(btn_frame, text="Save Data to CSV", command=self.save_analysis_data)
        btn_save.pack(side=tk.LEFT, padx=10, pady=5)
        
        btn_generate = ttk.Button(btn_frame, text="Generate Full Report", command=self.generate_full_report)
        btn_generate.pack(side=tk.RIGHT, padx=10, pady=5)
        
        btn_train_new = ttk.Button(btn_frame, text="Train Model with New Data", 
                                command=self.train_with_new_data)
        btn_train_new.pack(side=tk.RIGHT, padx=10, pady=5)

    def create_reliability_tab(self):
        """Create the reliability check tab"""
        frame = ttk.Frame(self.tab_reliability, style='TFrame')
        frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Participant ID section
        id_frame = ttk.Frame(frame, style='TFrame')
        id_frame.pack(fill=tk.X, padx=5, pady=10)
        
        ttk.Label(id_frame, text="Participant ID:").pack(side=tk.LEFT, padx=10, pady=10)
        pid_entry = ttk.Entry(id_frame, textvariable=self.participant_id_var, width=10)
        pid_entry.pack(side=tk.LEFT, padx=10, pady=10)
        
        # CSV selection section
        csv_frame = ttk.LabelFrame(frame, text="Select Test Session CSVs", style='TFrame')
        csv_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=10)
        
        self.csv_listbox = tk.Listbox(csv_frame, bg="#333333", fg=self.text_color, 
                                    highlightthickness=0, selectbackground=self.secondary_color,
                                    font=self.label_font)
        self.csv_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        csv_btn_frame = ttk.Frame(csv_frame, style='TFrame')
        csv_btn_frame.pack(side=tk.RIGHT, padx=10, pady=10)
        
        btn_add_csv = ttk.Button(csv_btn_frame, text="Add CSV", command=self.add_csv_to_list)
        btn_add_csv.pack(fill=tk.X, padx=5, pady=5)
        
        btn_remove_csv = ttk.Button(csv_btn_frame, text="Remove Selected", 
                                  command=lambda: self.csv_listbox.delete(tk.ACTIVE))
        btn_remove_csv.pack(fill=tk.X, padx=5, pady=5)
        
        # Button to create a new CSV for the first test
        btn_create_csv = ttk.Button(csv_btn_frame, text="Create New CSV", 
                                  command=self.create_new_reliability_csv)
        btn_create_csv.pack(fill=tk.X, padx=5, pady=5)
        
        # Results section
        results_frame = ttk.LabelFrame(frame, text="Reliability Results", style='TFrame')
        results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=10)
        
        self.reliability_text = tk.Text(results_frame, bg="#333333", fg=self.text_color, 
                                      highlightthickness=0, height=10, font=self.label_font,
                                      padx=10, pady=10)
        self.reliability_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Button to run reliability check
        btn_check = ttk.Button(frame, text="Run Reliability Check", command=self.check_reliability)
        btn_check.pack(side=tk.BOTTOM, padx=10, pady=15)

    def create_model_tab(self):
        """Create the model validation tab"""
        frame = ttk.Frame(self.tab_model, style='TFrame')
        frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # CSV selection for model training
        csv_frame = ttk.LabelFrame(frame, text="Select CSVs for Model Training", style='TFrame')
        csv_frame.pack(fill=tk.X, padx=5, pady=10)
        
        self.train_listbox = tk.Listbox(csv_frame, bg="#333333", fg=self.text_color, 
                                      highlightthickness=0, selectbackground=self.secondary_color,
                                      font=self.label_font)
        self.train_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        train_btn_frame = ttk.Frame(csv_frame, style='TFrame')
        train_btn_frame.pack(side=tk.RIGHT, padx=10, pady=10)
        
        btn_add_train = ttk.Button(train_btn_frame, text="Add CSV", command=self.add_train_csv)
        btn_add_train.pack(fill=tk.X, padx=5, pady=5)
        
        btn_remove_train = ttk.Button(train_btn_frame, text="Remove Selected", 
                                    command=lambda: self.train_listbox.delete(tk.ACTIVE))
        btn_remove_train.pack(fill=tk.X, padx=5, pady=5)
        
        # Button to create a new CSV for model validation
        btn_create_model_csv = ttk.Button(train_btn_frame, text="Create New CSV", 
                                       command=self.create_new_model_csv)
        btn_create_model_csv.pack(fill=tk.X, padx=5, pady=5)
        
        # Model validation results
        results_frame = ttk.LabelFrame(frame, text="Model Validation Results", style='TFrame')
        results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=10)
        
        self.model_results_text = tk.Text(results_frame, bg="#333333", fg=self.text_color, 
                                        highlightthickness=0, height=15, font=self.label_font,
                                        padx=10, pady=10)
        self.model_results_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Button to train and validate model
        btn_validate = ttk.Button(frame, text="Train and Validate Model", command=self.train_and_validate)
        btn_validate.pack(side=tk.BOTTOM, padx=10, pady=15)
        
    def create_visualization_tab(self):
        """Create the visualization tab to show model decision making"""
        frame = ttk.Frame(self.tab_visualization, style='TFrame')
        frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Information section
        info_frame = ttk.LabelFrame(frame, text="Model Decision Visualization", style='TFrame')
        info_frame.pack(fill=tk.X, padx=5, pady=10)
        
        info_text = ("This tab visualizes how the machine learning model makes decisions.\n"
                    "It shows the difference between the pretrained model and the updated model\n"
                    "that incorporates feedback from researchers and participants.")
        
        info_label = ttk.Label(info_frame, text=info_text, justify=tk.CENTER)
        info_label.pack(padx=10, pady=10)
        
        # Visualization container with two plots
        viz_container = ttk.Frame(frame, style='TFrame')
        viz_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=10)
        
        # Original model visualization
        orig_frame = ttk.LabelFrame(viz_container, text="Original Model Decisions", style='TFrame')
        orig_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.orig_fig = Figure(figsize=(4, 4), dpi=100, facecolor=self.bg_color)
        self.orig_ax = self.orig_fig.add_subplot(111)
        self.orig_ax.set_facecolor(self.bg_color)
        self.orig_canvas = FigureCanvasTkAgg(self.orig_fig, master=orig_frame)
        self.orig_canvas_widget = self.orig_canvas.get_tk_widget()
        self.orig_canvas_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Updated model visualization
        updated_frame = ttk.LabelFrame(viz_container, text="Updated Model Decisions", style='TFrame')
        updated_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.updated_fig = Figure(figsize=(4, 4), dpi=100, facecolor=self.bg_color)
        self.updated_ax = self.updated_fig.add_subplot(111)
        self.updated_ax.set_facecolor(self.bg_color)
        self.updated_canvas = FigureCanvasTkAgg(self.updated_fig, master=updated_frame)
        self.updated_canvas_widget = self.updated_canvas.get_tk_widget()
        self.updated_canvas_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Button to update visualizations
        btn_update_viz = ttk.Button(frame, text="Update Visualizations", command=self.update_decision_visualizations)
        btn_update_viz.pack(side=tk.BOTTOM, padx=10, pady=15)

    # --------------------------------------------------------------
    #   BUTTON COMMANDS
    # --------------------------------------------------------------
    def select_video(self):
        """Browse and select a video file, then show duration and preview"""
        filetypes = [("Video files", "*.mp4 *.avi *.mov"), ("All files", "*.*")]
        path = filedialog.askopenfilename(title="Select a Video", filetypes=filetypes)
        if not path:
            return
            
        self.video_path_var.set(path)
        
        # Get video duration and display a preview frame
        try:
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                messagebox.showerror("Error", "Could not open video file.")
                return
                
            # Get duration
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_sec = frame_count / fps if fps > 0 else 0
            duration_min = int(duration_sec // 60)
            duration_sec_remainder = int(duration_sec % 60)
            self.video_duration.set(f"Duration: {duration_min}m {duration_sec_remainder}s")
            
            # Get a preview frame (about 10% into the video)
            preview_frame_pos = int(frame_count * 0.1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, preview_frame_pos)
            ret, frame = cap.read()
            if ret:
                # Get canvas dimensions
                self.preview_canvas.update()
                canvas_width = self.preview_canvas.winfo_width() or 400
                canvas_height = self.preview_canvas.winfo_height() or 300
                
                # Convert frame to RGB for PIL
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Create PIL Image
                pil_img = Image.fromarray(frame_rgb)
                
                # Calculate resize dimensions while maintaining aspect ratio
                img_width, img_height = pil_img.size
                ratio = min(canvas_width/img_width, canvas_height/img_height)
                new_width = int(img_width * ratio)
                new_height = int(img_height * ratio)
                
                # Resize image
                pil_img = pil_img.resize((new_width, new_height), Image.LANCZOS)
                
                # Convert to PhotoImage and store original reference
                self.original_preview_img = ImageTk.PhotoImage(pil_img)
                
                # Calculate center position
                x_pos = (canvas_width - new_width) // 2
                y_pos = (canvas_height - new_height) // 2
                
                # Clear previous and display new image
                self.preview_canvas.delete("all")
                self.preview_canvas.create_image(x_pos, y_pos, anchor=tk.NW, image=self.original_preview_img)
                
            cap.release()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read video: {str(e)}")

    def load_models(self):
        """Load the emotion recognition models"""
        global global_model_1, global_proc_1, global_model_2, global_proc_2
        
        self.status_label.config(text="Loading models...")
        self.update_idletasks()
        
        try:
            from fer_local import LOCAL_MODEL_DIR_1, LOCAL_MODEL_DIR_2
            global_model_1, global_proc_1 = load_local_face_emotion_model(LOCAL_MODEL_DIR_1)
            global_model_2, global_proc_2 = load_local_face_emotion_model(LOCAL_MODEL_DIR_2)
            
            self.model_status.config(text="Models: Loaded")
            self.status_label.config(text="Models loaded successfully")
            messagebox.showinfo("Success", "Models loaded successfully.")
        except Exception as e:
            self.status_label.config(text="Error loading models")
            messagebox.showerror("Error Loading Models", str(e))

    def process_video_and_switch(self):
        """Process the video according to crop settings and switch to analysis tab"""
        global global_model_1, global_proc_1, global_model_2, global_proc_2
        
        if not global_model_1 or not global_proc_1:
            messagebox.showerror("Error", "Models not loaded yet. Please load models first.")
            return
            
        video_path = self.video_path_var.get()
        if not video_path or not os.path.exists(video_path):
            messagebox.showerror("Error", "Invalid video path. Please select a video first.")
            return
            
        # Get crop percentages
        start_pct = self.crop_start.get() / 100.0
        end_pct = self.crop_end.get() / 100.0
        
        # Validate crop values
        if start_pct >= end_pct:
            messagebox.showerror("Error", "Start percentage must be less than end percentage.")
            return
            
        # Get video info
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not open video file.")
            return
            
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate frame positions
        start_frame = int(frame_count * start_pct)
        end_frame = int(frame_count * end_pct)
        
        # Clear previous portion data
        self.portions_data = []
        
        # Get number of portions
        num_portions = self.portions.get()
        if num_portions < 1:
            num_portions = 1
            
        # Calculate frames per portion
        frames_per_portion = (end_frame - start_frame) // num_portions
        
        self.status_label.config(text="Analyzing video...")
        self.update_idletasks()
        
        # Process each portion
        for i in range(num_portions):
            portion_start = start_frame + i * frames_per_portion
            portion_end = portion_start + frames_per_portion if i < num_portions - 1 else end_frame
            
            # Convert to time for display
            start_time = portion_start / fps if fps > 0 else 0
            end_time = portion_end / fps if fps > 0 else 0
            
            # Analyze portion (extract frames in this range)
            cap.set(cv2.CAP_PROP_POS_FRAMES, portion_start)
            portion_frames = []
            
            # Choose every nth frame to keep processing manageable
            sample_interval = max(1, (portion_end - portion_start) // 20)
            
            for frame_idx in range(portion_start, portion_end, sample_interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    portion_frames.append(frame)
                    
            # Process this portion with emotion models
            if portion_frames:
                # Extract emotion from sampled frames
                portion_scores = []
                for frame in portion_frames:
                    local_probs = self.ensemble_models(frame)
                    score = self.compute_score(local_probs)
                    portion_scores.append(score)
                
                pretrained_score = sum(portion_scores) / len(portion_scores) if portion_scores else 0
                
                # Store portion data
                self.portions_data.append({
                    'portion': i + 1,
                    'start_time': start_time,
                    'end_time': end_time,
                    'start_frame': portion_start,
                    'end_frame': portion_end,
                    'researcher_score': 0.0,  # Will be filled by user
                    'participant_score': 0.0,  # Will be filled by user
                    'pretrained_score': pretrained_score
                })
        
        cap.release()
        
        # Create UI elements for each portion
        self.create_portion_entries()
        
        # Switch to analysis tab
        self.notebook.select(self.tab_analysis)
        self.status_label.config(text="Video processed. Enter researcher and participant scores.")

    def create_portion_entries(self):
        """Create entry fields for each video portion"""
        # Clear previous entries
        for widget in self.portions_container.winfo_children():
            widget.destroy()
            
        # Create header
        header_frame = ttk.Frame(self.portions_container, style='TFrame')
        header_frame.pack(fill=tk.X, pady=(0, 5))
        
        headers = ["Portion", "Time Range", "Researcher Score (1-9)", 
                  "Participant Score (1-9)", "Pretrained Score"]
        widths = [10, 20, 20, 20, 20]
        
        for i, (header, width) in enumerate(zip(headers, widths)):
            lbl = ttk.Label(header_frame, text=header, width=width)
            lbl.grid(row=0, column=i, padx=2)
            
        # Add entries for each portion
        for i, portion_data in enumerate(self.portions_data):
            row_frame = ttk.Frame(self.portions_container, style='TFrame')
            row_frame.pack(fill=tk.X, pady=2)
            
            # Portion number
            portion_lbl = ttk.Label(row_frame, text=f"{portion_data['portion']}", width=10)
            portion_lbl.grid(row=0, column=0, padx=2)
            
            # Time range
            start_min = int(portion_data['start_time'] // 60)
            start_sec = int(portion_data['start_time'] % 60)
            end_min = int(portion_data['end_time'] // 60)
            end_sec = int(portion_data['end_time'] % 60)
            time_range = f"{start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d}"
            time_lbl = ttk.Label(row_frame, text=time_range, width=20)
            time_lbl.grid(row=0, column=1, padx=2)
            
            # Researcher score entry
            researcher_var = tk.DoubleVar(value=5.0)  # Default mid-value
            researcher_entry = ttk.Spinbox(row_frame, from_=1.0, to=9.0, increment=0.5,
                                         textvariable=researcher_var, width=10)
            researcher_entry.grid(row=0, column=2, padx=2)
            
            # Participant score entry
            participant_var = tk.DoubleVar(value=5.0)  # Default mid-value
            participant_entry = ttk.Spinbox(row_frame, from_=1.0, to=9.0, increment=0.5,
                                          textvariable=participant_var, width=10)
            participant_entry.grid(row=0, column=3, padx=2)
            
            # Pretrained score display
            pretrained_score = f"{portion_data['pretrained_score']:.2f}"
            pretrained_lbl = ttk.Label(row_frame, text=pretrained_score, width=10)
            pretrained_lbl.grid(row=0, column=4, padx=2)
            
            # Store variables in portion data for later retrieval
            self.portions_data[i]['researcher_var'] = researcher_var
            self.portions_data[i]['participant_var'] = participant_var
            
        # Update visualization
        self.update_visualization()

    def update_visualization(self):
        """Update the visualization with current data as a line graph"""
        self.ax.clear()
        
        # Check if we have data
        if not self.portions_data:
            return
            
        # Prepare data for plotting
        portion_nums = [p['portion'] for p in self.portions_data]
        researcher_scores = [p.get('researcher_var', tk.DoubleVar(value=0)).get() for p in self.portions_data]
        participant_scores = [p.get('participant_var', tk.DoubleVar(value=0)).get() for p in self.portions_data]
        pretrained_scores = [p['pretrained_score'] for p in self.portions_data]
        
        # Plot lines
        self.ax.plot(portion_nums, researcher_scores, 'o-', color=self.accent_color, 
                   linewidth=2, markersize=8, label='Researcher')
        self.ax.plot(portion_nums, participant_scores, 'o-', color=self.secondary_color, 
                   linewidth=2, markersize=8, label='Participant')
        self.ax.plot(portion_nums, pretrained_scores, 'o-', color='#ffffff', 
                   linewidth=2, markersize=8, label='Pretrained')
        
        # Styling
        self.ax.set_xlabel('Portion')
        self.ax.set_ylabel('Affective Score (1-9)')
        self.ax.set_title('Affective Scores by Source')
        self.ax.set_xticks(portion_nums)
        self.ax.set_ylim(0, 10)
        self.ax.grid(True, linestyle='--', alpha=0.3)
        self.ax.legend(facecolor="#333333", edgecolor="#555555")
        
        # Update theme colors for labels
        self.ax.xaxis.label.set_color(self.text_color)
        self.ax.yaxis.label.set_color(self.text_color)
        self.ax.title.set_color(self.text_color)
        self.ax.tick_params(colors=self.text_color)
        
        # Set tick colors
        for spine in self.ax.spines.values():
            spine.set_edgecolor(self.text_color)
        
        # Add a subtle background grid
        self.ax.set_facecolor("#1a1a1a")
        
        # Redraw canvas
        self.canvas.draw()

    def save_analysis_data(self):
        """Save analysis data to CSV"""
        if not self.portions_data:
            messagebox.showerror("Error", "No data to save.")
            return
            
        # Update portion data with current UI values
        for i, portion_data in enumerate(self.portions_data):
            researcher_var = portion_data.get('researcher_var')
            participant_var = portion_data.get('participant_var')
            
            if researcher_var and participant_var:
                self.portions_data[i]['researcher_score'] = researcher_var.get()
                self.portions_data[i]['participant_score'] = participant_var.get()
        
        # Get video ID from path
        video_path = self.video_path_var.get()
        if not video_path:
            video_id = "unknown_video"
        else:
            video_id = os.path.splitext(os.path.basename(video_path))[0]
            
        # Prepare data for CSV
        csv_data = []
        for p in self.portions_data:
            csv_data.append({
                "video_id": video_id,
                "portion": p['portion'],
                "start_time": p['start_time'],
                "end_time": p['end_time'],
                "researcher_pred": p['researcher_score'],
                "participant_report": p['participant_score'],
                "pretrained_pred": p['pretrained_score']
            })
            
        # Create DataFrame and save
        df = pd.DataFrame(csv_data)
        csv_name = f"{video_id}_analysis.csv"
        
        df.to_csv(csv_name, index=False)
        messagebox.showinfo("Success", f"Data saved to {csv_name}")
        self.status_label.config(text=f"Data saved to {csv_name}")

    def generate_full_report(self):
        """Generate a full PDF report with analysis results"""
        if not self.portions_data:
            messagebox.showerror("Error", "No data to generate report from.")
            return
            
        video_path = self.video_path_var.get()
        if not video_path or not os.path.exists(video_path):
            messagebox.showerror("Error", "Invalid video path.")
            return
            
        # Create window scores from portion data (format needed by generate_report)
        window_scores = [p.get('pretrained_score', 5.0) for p in self.portions_data]
        
        # We need annotated frames, but we don't have them
        # So we'll use dummy frames for now
        annotated_frames = []
        
        output_pdf = os.path.splitext(video_path)[0] + "_analysis_report.pdf"
        
        try:
            generate_report(video_path, window_scores, annotated_frames, output_pdf=output_pdf)
            messagebox.showinfo("Report Generated", f"PDF saved to {output_pdf}")
            self.status_label.config(text=f"Report generated: {output_pdf}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate report: {str(e)}")

    def add_csv_to_list(self):
        """Add a CSV file to the reliability check list"""
        csv_files = filedialog.askopenfilenames(
            title="Select CSV files for reliability check",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if not csv_files:
            return
            
        for csv_file in csv_files:
            if csv_file and csv_file not in self.csv_listbox.get(0, tk.END):
                self.csv_listbox.insert(tk.END, csv_file)

    def add_train_csv(self):
        """Add a CSV file to the model training list"""
        csv_files = filedialog.askopenfilenames(
            title="Select CSV files for model training",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if not csv_files:
            return
            
        for csv_file in csv_files:
            if csv_file and csv_file not in self.train_listbox.get(0, tk.END):
                self.train_listbox.insert(tk.END, csv_file)

    def check_reliability(self):
        """Check reliability across tests"""
        participant_id = self.participant_id_var.get()
        if not participant_id:
            messagebox.showerror("Error", "Please enter a participant ID first.")
            return
            
        csv_files = list(self.csv_listbox.get(0, tk.END))
        if not csv_files:
            messagebox.showerror("Error", "No CSV files selected.")
            return
            
        import io
        import sys
        backup_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        try:
            check_reliability_across_tests(participant_id, csv_files)
            output = sys.stdout.getvalue()
        except Exception as e:
            output = f"Error: {str(e)}"
        finally:
            sys.stdout = backup_stdout
            
        # Display results
        self.reliability_text.delete(1.0, tk.END)
        self.reliability_text.insert(tk.END, output)
        self.status_label.config(text="Reliability check complete")

    def train_and_validate(self):
        """Train and validate the model with selected CSV files"""
        csv_files = list(self.train_listbox.get(0, tk.END))
        if not csv_files:
            messagebox.showerror("Error", "No CSV files selected for training.")
            return
            
        # For each CSV, ask about test session if not already present
        updated_csvs = []
        for csv_file in csv_files:
            if not os.path.isfile(csv_file):
                continue
                
            # Check if it already has test_session column
            df = pd.read_csv(csv_file)
            if "test_session" not in df.columns:
                # Ask user about which test session
                session = self.ask_session_number(csv_file)
                if session is None:
                    continue
                    
                df["test_session"] = session
                df.to_csv(csv_file, index=False)
                
            updated_csvs.append(csv_file)
            
        if not updated_csvs:
            messagebox.showerror("Error", "No valid CSV files after session assignment.")
            return
            
        self.status_label.config(text="Training model...")
        self.update_idletasks()
        
        import io
        import sys
        backup_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        try:
            train_and_evaluate_affect_model(updated_csvs)
            output = sys.stdout.getvalue()
        except Exception as e:
            output = f"Error: {str(e)}"
        finally:
            sys.stdout = backup_stdout
            
        # Display results
        self.model_results_text.delete(1.0, tk.END)
        self.model_results_text.insert(tk.END, output)
        self.status_label.config(text="Model training and validation complete")

    def ask_session_number(self, csv_file):
        """Ask which test session a CSV represents (1 or 2)"""
        dialog = tk.Toplevel(self)
        dialog.title("Session Number")
        dialog.configure(bg=self.bg_color)
        
        lbl = ttk.Label(dialog, text=f"CSV: {os.path.basename(csv_file)}\nIs this from test session #1 or #2?")
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
        
        btn_frame = ttk.Frame(dialog, style='TFrame')
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

    # --------------------------------------------------------------
    #   NEW METHODS
    # --------------------------------------------------------------
    def validate_responses(self):
        """Validate and update the responses entered by the user"""
        if not self.portions_data:
            messagebox.showerror("Error", "No data to validate.")
            return
            
        # Update portion data with current UI values and validate ranges
        valid = True
        for i, portion_data in enumerate(self.portions_data):
            researcher_var = portion_data.get('researcher_var')
            participant_var = portion_data.get('participant_var')
            
            if researcher_var and participant_var:
                researcher_val = researcher_var.get()
                participant_val = participant_var.get()
                
                # Validate ranges (1-9)
                if not (1 <= researcher_val <= 9):
                    messagebox.showerror("Error", f"Researcher score in portion {i+1} must be between 1 and 9.")
                    valid = False
                    break
                
                if not (1 <= participant_val <= 9):
                    messagebox.showerror("Error", f"Participant score in portion {i+1} must be between 1 and 9.")
                    valid = False
                    break
                
                # Update values in our data structure
                self.portions_data[i]['researcher_score'] = researcher_val
                self.portions_data[i]['participant_score'] = participant_val
        
        if valid:
            # Update visualization with new data
            self.update_visualization()
            messagebox.showinfo("Success", "Responses validated and updated.")
            self.status_label.config(text="Responses validated")
    
    def train_with_new_data(self):
        """Train the model with the new responses from this analysis"""
        if not self.portions_data:
            messagebox.showerror("Error", "No data to train with.")
            return
            
        # First, save the current data to CSV
        video_path = self.video_path_var.get()
        if not video_path:
            video_id = "unknown_video"
        else:
            video_id = os.path.splitext(os.path.basename(video_path))[0]
            
        # Ensure all values are updated
        self.validate_responses()
        
        # Create CSV path
        csv_path = f"{video_id}_analysis.csv"
        self.save_analysis_data()  # Save to CSV
        
        # Now train with this CSV plus any existing training data
        all_csvs = list(self.train_listbox.get(0, tk.END))
        if csv_path not in all_csvs:
            all_csvs.append(csv_path)
        
        if not all_csvs:
            messagebox.showerror("Error", "No CSV files available for training.")
            return
        
        # For each CSV, ensure it has test_session column
        updated_csvs = []
        for cf in all_csvs:
            if not os.path.isfile(cf):
                continue
                
            df = pd.read_csv(cf)
            if "test_session" not in df.columns:
                # Ask user which test session this is
                session = self.ask_session_number(cf)
                if session is None:
                    continue
                    
                df["test_session"] = session
                df.to_csv(cf, index=False)
                
            updated_csvs.append(cf)
        
        if not updated_csvs:
            messagebox.showerror("Error", "No valid CSV files after session assignment.")
            return
            
        self.status_label.config(text="Training model with new data...")
        self.update_idletasks()
        
        import io
        import sys
        backup_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        try:
            train_and_evaluate_affect_model(updated_csvs)
            output = sys.stdout.getvalue()
        except Exception as e:
            output = f"Error: {str(e)}"
        finally:
            sys.stdout = backup_stdout
            
        # Display results and switch to Model tab
        self.model_results_text.delete(1.0, tk.END)
        self.model_results_text.insert(tk.END, output)
        
        # Update the decision visualization
        self.update_decision_visualizations()
        
        # Switch to model validation tab to show results
        self.notebook.select(self.tab_model)
        self.status_label.config(text="Model trained with new data")
        messagebox.showinfo("Training Complete", "Model trained with new data. See Model Validation tab for results.")
    
    def create_new_reliability_csv(self):
        """Create a new CSV file for the first reliability test"""
        participant_id = self.participant_id_var.get()
        if not participant_id:
            messagebox.showerror("Error", "Please enter a participant ID first.")
            return
            
        # Get output filename
        filename = filedialog.asksaveasfilename(
            title="Save New Reliability CSV",
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
            initialfile=f"{participant_id}_test1.csv"
        )
        
        if not filename:
            return
        
        # Create empty DataFrame with required columns
        df = pd.DataFrame(columns=[
            "video_id", "portion", "start_time", "end_time", 
            "researcher_pred", "participant_report", "pretrained_pred", "test_session"
        ])
        
        # Save to CSV
        df.to_csv(filename, index=False)
        
        # Add to listbox
        if filename not in self.csv_listbox.get(0, tk.END):
            self.csv_listbox.insert(tk.END, filename)
            
        messagebox.showinfo("Success", f"Created new CSV file: {filename}")
        self.status_label.config(text=f"Created new reliability CSV: {filename}")
    
    def create_new_model_csv(self):
        """Create a new CSV file for model training"""
        # Get output filename
        filename = filedialog.asksaveasfilename(
            title="Save New Model Training CSV",
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
            initialfile="model_training_data.csv"
        )
        
        if not filename:
            return
        
        # Create empty DataFrame with required columns
        df = pd.DataFrame(columns=[
            "video_id", "portion", "start_time", "end_time", 
            "researcher_pred", "participant_report", "pretrained_pred", "test_session"
        ])
        
        # Save to CSV
        df.to_csv(filename, index=False)
        
        # Add to listbox
        if filename not in self.train_listbox.get(0, tk.END):
            self.train_listbox.insert(tk.END, filename)
            
        messagebox.showinfo("Success", f"Created new CSV file: {filename}")
        self.status_label.config(text=f"Created new model training CSV: {filename}")
    
    def update_decision_visualizations(self):
        """Update the visualizations showing how the model makes decisions"""
        # Clear previous visualizations
        self.orig_ax.clear()
        self.updated_ax.clear()
        
        # Check if we have any CSV files for visualization
        train_csvs = list(self.train_listbox.get(0, tk.END))
        if not train_csvs:
            self.orig_ax.text(0.5, 0.5, "No training data available", 
                           horizontalalignment='center', verticalalignment='center',
                           transform=self.orig_ax.transAxes, color=self.text_color)
            self.updated_ax.text(0.5, 0.5, "No training data available", 
                              horizontalalignment='center', verticalalignment='center',
                              transform=self.updated_ax.transAxes, color=self.text_color)
            self.orig_canvas.draw()
            self.updated_canvas.draw()
            return
        
        try:
            # Load data from CSVs
            all_data = []
            for csv_file in train_csvs:
                if os.path.isfile(csv_file):
                    df = pd.read_csv(csv_file)
                    all_data.append(df)
            
            if not all_data:
                messagebox.showerror("Error", "No valid CSV files found.")
                return
                
            # Combine all data
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Original model visualization (pretrained vs participant)
            pretrained = combined_df['pretrained_pred'].values
            participant = combined_df['participant_report'].values
            
            self.orig_ax.scatter(pretrained, participant, alpha=0.7, color=self.accent_color, s=80)
            
            # Add diagonal line (perfect prediction)
            min_val = min(min(pretrained), min(participant))
            max_val = max(max(pretrained), max(participant))
            self.orig_ax.plot([min_val, max_val], [min_val, max_val], 'w--', alpha=0.5)
            
            # Fit a simple regression line to show trend
            from sklearn.linear_model import LinearRegression
            X = pretrained.reshape(-1, 1)
            y = participant.reshape(-1, 1)
            model = LinearRegression().fit(X, y)
            x_pred = np.array([min_val, max_val]).reshape(-1, 1)
            y_pred = model.predict(x_pred)
            self.orig_ax.plot(x_pred, y_pred, color=self.secondary_color, linewidth=2)
            
            # Calculate correlation and R²
            correlation = np.corrcoef(pretrained, participant)[0, 1]
            r2 = model.score(X, y)
            
            # Add correlation info
            self.orig_ax.text(0.05, 0.95, f"Correlation: {correlation:.2f}\nR²: {r2:.2f}", 
                           transform=self.orig_ax.transAxes, color=self.text_color,
                           verticalalignment='top')
            
            # Style the plot
            self.orig_ax.set_xlabel("Pretrained Model Prediction")
            self.orig_ax.set_ylabel("Participant Report")
            self.orig_ax.set_title("Original Model vs Participant")
            self.orig_ax.set_facecolor("#1a1a1a")
            self.orig_ax.grid(True, linestyle='--', alpha=0.3)
            
            # Set labels and ticks
            self.orig_ax.xaxis.label.set_color(self.text_color)
            self.orig_ax.yaxis.label.set_color(self.text_color)
            self.orig_ax.title.set_color(self.text_color)
            self.orig_ax.tick_params(colors=self.text_color)
            
            # Set equal aspect ratio
            self.orig_ax.set_aspect('equal', adjustable='box')
            
            # Updated model visualization (researcher vs participant)
            researcher = combined_df['researcher_pred'].values
            
            self.updated_ax.scatter(researcher, participant, alpha=0.7, color=self.accent_color, s=80)
            
            # Add diagonal line (perfect prediction)
            min_val = min(min(researcher), min(participant))
            max_val = max(max(researcher), max(participant))
            self.updated_ax.plot([min_val, max_val], [min_val, max_val], 'w--', alpha=0.5)
            
            # Fit a simple regression line to show trend
            X = researcher.reshape(-1, 1)
            y = participant.reshape(-1, 1)
            model = LinearRegression().fit(X, y)
            x_pred = np.array([min_val, max_val]).reshape(-1, 1)
            y_pred = model.predict(x_pred)
            self.updated_ax.plot(x_pred, y_pred, color=self.secondary_color, linewidth=2)
            
            # Calculate correlation and R²
            correlation = np.corrcoef(researcher, participant)[0, 1]
            r2 = model.score(X, y)
            
            # Add correlation info
            self.updated_ax.text(0.05, 0.95, f"Correlation: {correlation:.2f}\nR²: {r2:.2f}", 
                              transform=self.updated_ax.transAxes, color=self.text_color,
                              verticalalignment='top')
            
            # Style the plot
            self.updated_ax.set_xlabel("Researcher Assessment")
            self.updated_ax.set_ylabel("Participant Report")
            self.updated_ax.set_title("Researcher vs Participant")
            self.updated_ax.set_facecolor("#1a1a1a")
            self.updated_ax.grid(True, linestyle='--', alpha=0.3)
            
            # Set labels and ticks
            self.updated_ax.xaxis.label.set_color(self.text_color)
            self.updated_ax.yaxis.label.set_color(self.text_color)
            self.updated_ax.title.set_color(self.text_color)
            self.updated_ax.tick_params(colors=self.text_color)
            
            # Set equal aspect ratio
            self.updated_ax.set_aspect('equal', adjustable='box')
            
            # Update canvases
            self.orig_canvas.draw()
            self.updated_canvas.draw()
            
            self.status_label.config(text="Decision visualizations updated")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update visualizations: {str(e)}")
            print(f"Visualization error: {str(e)}")

    # --------------------------------------------------------------
    #   HELPER METHODS
    # --------------------------------------------------------------
    def ensemble_models(self, frame):
        """Call ensemble_two_local_models but handle potential errors"""
        global global_model_1, global_proc_1, global_model_2, global_proc_2
        
        try:
            from fer_local import ensemble_two_local_models
            return ensemble_two_local_models(frame, global_model_1, global_proc_1, global_model_2, global_proc_2)
        except Exception as e:
            print(f"Error in ensemble models: {e}")
            return {"neutral": 1.0}  # Default if error

    def compute_score(self, probs):
        """Compute affective score from probabilities"""
        try:
            from fer_local import compute_affective_score_ensemble
            return compute_affective_score_ensemble(probs)
        except Exception as e:
            print(f"Error computing score: {e}")
            return 5.0  # Default mid-value if error


if __name__ == "__main__":
    app = EmotionGUI()
    app.mainloop()
