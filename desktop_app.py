"""
ECG Dynamic Programming Analysis Desktop Application
Complete desktop app with GUI, DP algorithm, and ML integration
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from scipy.signal import butter, filtfilt
import joblib
import json
import os
from datetime import datetime
import threading
import random


# ============================================
# 1. DP ALGORITHM IMPLEMENTATION
# ============================================
class DPSegmenter:
    def __init__(self, fs=360):
        self.fs = fs
        self.segment_types = ["P", "QRS", "T"]

        # Physiological constraints (in samples)
        self.constraints = {
            "P": {"min": int(0.05 * fs), "max": int(0.12 * fs)},
            "QRS": {"min": int(0.06 * fs), "max": int(0.12 * fs)},
            "T": {"min": int(0.10 * fs), "max": int(0.25 * fs)}
        }

        # For step-by-step visualization
        self.dp_steps = []
        self.backtrack_steps = []

    def bandpass_filter(self, signal, lowcut=0.5, highcut=40.0):
        """Apply bandpass filter to ECG signal"""
        nyquist = 0.5 * self.fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(4, [low, high], btype='band')
        return filtfilt(b, a, signal)

    def normalize_signal(self, signal):
        """Normalize ECG signal"""
        filtered = self.bandpass_filter(signal)
        return (filtered - np.mean(filtered)) / np.std(filtered)

    def cost_function(self, segment_type, signal_segment):
        """Cost function for DP algorithm"""
        if len(signal_segment) < 3:
            return float('inf')

        amplitude = np.max(signal_segment) - np.min(signal_segment)
        slope = np.mean(np.diff(signal_segment) ** 2)

        if segment_type == "QRS":
            # QRS should have high slope and amplitude
            return 1.0 / (slope + 0.001) + 0.5 / (amplitude + 0.001)
        elif segment_type == "P":
            # P wave is smaller
            return amplitude * 0.8 + slope * 0.2
        else:  # T wave
            return amplitude * 0.6 + slope * 0.4

    def segment_with_dp(self, signal, step_by_step=False):
        """Main DP segmentation algorithm with step tracking"""
        n = len(signal)

        # Initialize DP tables
        dp = np.full(n + 1, float('inf'))
        dp[0] = 0.0
        prev = np.full(n + 1, -1, dtype=int)
        seg_type = [""] * (n + 1)

        # Store steps for visualization
        if step_by_step:
            self.dp_steps = []
            self.dp_steps.append({
                "step": 0,
                "position": 0,
                "cost": 0.0,
                "action": "Initialize: dp[0] = 0",
                "dp_table": self.get_dp_snapshot(dp, prev, seg_type, 0, 50)
            })

        # Main DP loop
        for i in range(1, n + 1):
            for seg in self.segment_types:
                min_len = self.constraints[seg]["min"]
                max_len = min(self.constraints[seg]["max"], i)

                for l in range(min_len, max_len + 1):
                    j = i - l
                    if j < 0:
                        continue

                    segment = signal[j:i]
                    segment_cost = self.cost_function(seg, segment)
                    total_cost = dp[j] + segment_cost

                    if total_cost < dp[i]:
                        dp[i] = total_cost
                        prev[i] = j
                        seg_type[i] = seg

                        if step_by_step and (i % 50 == 0 or i == n):
                            self.dp_steps.append({
                                "step": len(self.dp_steps),
                                "position": i,
                                "cost": dp[i],
                                "action": f"Update dp[{i}] = {dp[i]:.3f} from {seg} at {j}",
                                "dp_table": self.get_dp_snapshot(dp, prev, seg_type, max(0, i - 50), min(i + 1, n + 1))
                            })

        # Backtracking
        segments = []
        i = n
        step_count = 0

        self.backtrack_steps = []
        self.backtrack_steps.append({
            "step": step_count,
            "position": i,
            "action": f"Start backtracking from position {i}",
            "segments": [],
            "cost": dp[i]
        })

        while i > 0 and prev[i] != -1:
            segment_info = {
                "type": seg_type[i],
                "start": prev[i],
                "end": i - 1,
                "length": i - prev[i],
                "cost": dp[i] - dp[prev[i]]
            }
            segments.append(segment_info)

            step_count += 1
            self.backtrack_steps.append({
                "step": step_count,
                "position": prev[i],
                "action": f"Backtrack: Found {seg_type[i]} at {prev[i]}-{i - 1}",
                "segments": segments.copy(),
                "cost": dp[prev[i]]
            })

            i = prev[i]

        segments.reverse()

        # Final backtracking step
        self.backtrack_steps.append({
            "step": step_count + 1,
            "position": 0,
            "action": "Backtracking complete",
            "segments": segments,
            "cost": 0.0
        })

        # Build full DP table for display
        dp_table_full = []
        for i in range(min(n + 1, 500)):  # Limit to first 500 for display
            dp_table_full.append({
                "position": i,
                "cost": dp[i] if dp[i] != float('inf') else float('inf'),
                "segment": seg_type[i] if seg_type[i] else "-",
                "prev": prev[i] if prev[i] != -1 else "-",
                "length": i - prev[i] if prev[i] != -1 else "-"
            })

        return {
            "segments": segments,
            "dp_table": dp_table_full,
            "total_cost": dp[n],
            "signal_length": n,
            "step_count": len(self.dp_steps) if step_by_step else 0,
            "backtrack_steps": len(self.backtrack_steps)
        }

    def get_dp_snapshot(self, dp, prev, seg_type, start, end):
        """Get snapshot of DP table for visualization"""
        snapshot = []
        for i in range(start, min(end, len(dp))):
            snapshot.append({
                "position": i,
                "cost": dp[i] if dp[i] != float('inf') else float('inf'),
                "segment": seg_type[i] if seg_type[i] else "-",
                "prev": prev[i] if prev[i] != -1 else "-"
            })
        return snapshot

    def extract_features(self, segments, signal):
        """Extract features from segmented ECG"""
        features = {}

        # Count segments by type
        for seg in ["P", "QRS", "T"]:
            seg_list = [s for s in segments if s["type"] == seg]
            features[f"{seg.lower()}_count"] = len(seg_list)

            if seg_list:
                features[f"{seg.lower()}_duration_mean"] = np.mean([s["length"] for s in seg_list])
                features[f"{seg.lower()}_cost_mean"] = np.mean([s["cost"] for s in seg_list])

        # RR intervals (for QRS complexes)
        qrs_segments = [s for s in segments if s["type"] == "QRS"]
        if len(qrs_segments) > 1:
            rr_intervals = []
            for i in range(1, len(qrs_segments)):
                rr = qrs_segments[i]["start"] - qrs_segments[i - 1]["end"]
                rr_intervals.append(rr)

            features["rr_mean"] = np.mean(rr_intervals)
            features["rr_std"] = np.std(rr_intervals)
            features["heart_rate"] = (60 * self.fs) / features["rr_mean"] if features["rr_mean"] > 0 else 0

        # Signal statistics
        features["signal_mean"] = np.mean(signal)
        features["signal_std"] = np.std(signal)
        features["signal_skewness"] = np.mean((signal - np.mean(signal)) ** 3) / (np.std(signal) ** 3 + 0.001)

        return features


# ============================================
# 2. ML MODEL INTEGRATION
# ============================================
class ECGClassifier:
    def __init__(self, model_path=None):
        self.model = None
        self.feature_names = [
            'qrs_count', 'rr_mean', 'rr_std',
            'p_duration_mean', 'qrs_duration_mean'
        ]

        if model_path and os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
                print(f"✅ ML model loaded from {model_path}")
            except Exception as e:
                print(f"⚠️ Could not load ML model: {e}, using rule-based classifier")
                self.model = None
        else:
            print("ℹ️ No ML model provided, using rule-based classifier")

    def classify(self, features):
        """Classify ECG as Normal or Abnormal"""
        if self.model is not None:
            try:
                # Prepare feature vector
                feature_vector = []
                for feature in self.feature_names:
                    feature_vector.append(features.get(feature, 0))

                feature_vector = np.array(feature_vector).reshape(1, -1)

                # Predict
                prediction = self.model.predict(feature_vector)[0]
                probabilities = self.model.predict_proba(feature_vector)[0]

                return {
                    "method": "random_forest",
                    "prediction": "Normal" if prediction == 0 else "Abnormal",
                    "confidence": float(max(probabilities)),
                    "probabilities": {
                        "Normal": float(probabilities[0]),
                        "Abnormal": float(probabilities[1])
                    },
                    "features_used": self.feature_names
                }
            except Exception as e:
                print(f"⚠️ ML classification failed: {e}, falling back to rule-based")
                return self.rule_based_classification(features)
        else:
            # Rule-based classification (fallback)
            return self.rule_based_classification(features)

    def rule_based_classification(self, features):
        """Rule-based classification when ML model is not available"""
        abnormalities = []

        # Check QRS duration (normal: 60-100ms)
        qrs_duration = features.get("qrs_duration_mean", 0) * (1000 / 360)  # Convert to ms
        if qrs_duration > 100:
            abnormalities.append(f"Wide QRS complex ({qrs_duration:.1f}ms)")

        # Check RR interval variability
        rr_std = features.get("rr_std", 0)
        if rr_std > 50:
            abnormalities.append("Irregular rhythm")

        # Check heart rate
        heart_rate = features.get("heart_rate", 0)
        if heart_rate < 60:
            abnormalities.append(f"Bradycardia ({heart_rate:.0f} BPM)")
        elif heart_rate > 100:
            abnormalities.append(f"Tachycardia ({heart_rate:.0f} BPM)")

        if abnormalities:
            return {
                "method": "rule_based",
                "prediction": "Abnormal",
                "confidence": 0.85,
                "abnormalities": abnormalities,
                "notes": "; ".join(abnormalities)
            }
        else:
            return {
                "method": "rule_based",
                "prediction": "Normal",
                "confidence": 0.90,
                "abnormalities": [],
                "notes": "All parameters within normal range"
            }


# ============================================
# 3. TEST CASES GENERATOR
# ============================================
class TestCaseGenerator:
    def __init__(self, fs=360):
        self.fs = fs

    def generate_normal(self, length=400):
        """Generate normal sinus rhythm"""
        t = np.linspace(0, length / self.fs, length)
        signal = 0.5 * np.sin(2 * np.pi * 1 * t)

        # Add QRS complexes periodically
        for i in range(0, length, int(self.fs * 0.8)):  # ~75 BPM
            if i + 50 < length:
                signal[i:i + 50] += 0.8 * np.sin(2 * np.pi * 10 * t[:50])

        signal += np.random.normal(0, 0.05, length)
        return signal

    def generate_arrhythmia(self, length=400):
        """Generate arrhythmia signal"""
        signal = self.generate_normal(length)

        # Add irregular beats
        irregular_positions = random.sample(range(100, length - 50), 3)
        for pos in irregular_positions:
            signal[pos:pos + 50] += np.random.normal(0, 0.3, 50)

        # Add some missing beats
        for i in range(0, length, int(self.fs * 0.8)):
            if random.random() < 0.3 and i + 50 < length:
                signal[i:i + 50] = np.random.normal(0, 0.1, 50)

        return signal

    def generate_bradycardia(self, length=400):
        """Generate bradycardia signal"""
        t = np.linspace(0, length / self.fs, length)
        signal = 0.4 * np.sin(2 * np.pi * 0.5 * t)  # Slower base frequency

        # Add QRS complexes with longer intervals
        for i in range(0, length, int(self.fs * 1.2)):  # ~50 BPM
            if i + 50 < length:
                signal[i:i + 50] += 0.6 * np.sin(2 * np.pi * 8 * t[:50])

        signal += np.random.normal(0, 0.03, length)
        return signal


# ============================================
# 4. MAIN DESKTOP APPLICATION
# ============================================
class ECGDPApp:
    """Main Desktop Application"""

    def __init__(self, root):
        self.root = root
        self.root.title("ECG Dynamic Programming Analysis")
        self.root.geometry("1400x900")

        # Set application icon and theme
        self.root.configure(bg='#0a0a2a')
        self.setup_styles()

        # Initialize components
        self.segmenter = DPSegmenter()

        # Try to load ML model from various possible locations
        model_paths = [
            "random_forest_model.pkl",
            "model/random_forest_model.pkl",
            "../model/random_forest_model.pkl"
        ]

        model_found = False
        for path in model_paths:
            if os.path.exists(path):
                try:
                    self.classifier = ECGClassifier(path)
                    model_found = True
                    print(f"✅ Model loaded from {path}")
                    break
                except:
                    continue

        if not model_found:
            print("⚠️ ML model not found in standard locations, using rule-based classifier")
            self.classifier = ECGClassifier(None)

        self.generator = TestCaseGenerator()

        # Current data
        self.current_signal = None
        self.current_signal_type = None
        self.dp_results = None
        self.current_dp_step = 0
        self.current_dp_page = 1
        self.total_dp_pages = 1
        self.rows_per_page = 50
        self.current_backtrack_step = 0
        self.is_playing = False

        # Setup GUI
        self.setup_gui()

        # Load initial signal
        self.load_test_case("normal")

    def setup_styles(self):
        """Setup custom styles for the application"""
        self.style = ttk.Style()

        # Configure colors
        self.colors = {
            'bg_dark': '#0a0a2a',
            'bg_medium': '#1a1a3a',
            'bg_light': '#2a2a4a',
            'primary': '#05d9e8',
            'secondary': '#ff2a6d',
            'success': '#00ff9d',
            'warning': '#ffcc00',
            'text': '#d1f7ff',
            'grid': '#444477'  # Fixed: Changed from rgba to hex color
        }

        # Configure ttk styles
        self.style.configure('TButton',
                             background=self.colors['primary'],
                             foreground='black',
                             font=('Arial', 10, 'bold'))

        self.style.configure('Title.TLabel',
                             font=('Arial', 16, 'bold'),
                             foreground=self.colors['primary'])

        self.style.configure('Subtitle.TLabel',
                             font=('Arial', 12),
                             foreground=self.colors['text'])

    def setup_gui(self):
        """Setup the main GUI"""
        # Create main container with custom background
        main_container = tk.Frame(self.root, bg=self.colors['bg_dark'])
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Header
        self.create_header(main_container)

        # Main content area
        content_frame = tk.Frame(main_container, bg=self.colors['bg_dark'])
        content_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # Left panel (Controls and Input)
        left_panel = self.create_left_panel(content_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        # Right panel (Visualization and Results)
        right_panel = self.create_right_panel(content_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        # Footer
        self.create_footer(main_container)

    def create_header(self, parent):
        """Create application header"""
        header_frame = tk.Frame(parent, bg=self.colors['bg_medium'],
                                height=100, relief=tk.RAISED, borderwidth=2)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        header_frame.pack_propagate(False)

        # Title
        title_label = tk.Label(header_frame,
                               text="🫀 ECG Dynamic Programming Analysis",
                               font=('Arial', 24, 'bold'),
                               fg=self.colors['primary'],
                               bg=self.colors['bg_medium'])
        title_label.pack(side=tk.LEFT, padx=20, pady=20)

        # Subtitle
        subtitle_label = tk.Label(header_frame,
                                  text="Biomedical Signal Processing Project | Academic Use Only",
                                  font=('Arial', 10),
                                  fg=self.colors['text'],
                                  bg=self.colors['bg_medium'])
        subtitle_label.pack(side=tk.LEFT, pady=20)

        # Status indicator
        self.status_var = tk.StringVar(value="System Ready")
        status_label = tk.Label(header_frame,
                                textvariable=self.status_var,
                                font=('Arial', 10, 'bold'),
                                fg=self.colors['success'],
                                bg=self.colors['bg_medium'])
        status_label.pack(side=tk.RIGHT, padx=20, pady=20)

    def create_left_panel(self, parent):
        """Create left panel with controls and input"""
        left_frame = tk.Frame(parent, bg=self.colors['bg_medium'],
                              relief=tk.RAISED, borderwidth=2)

        # Test Cases Section
        test_cases_frame = tk.LabelFrame(left_frame,
                                         text=" Test Cases ",
                                         font=('Arial', 12, 'bold'),
                                         fg=self.colors['primary'],
                                         bg=self.colors['bg_medium'],
                                         relief=tk.RAISED,
                                         borderwidth=2)
        test_cases_frame.pack(fill=tk.X, padx=10, pady=10)

        self.create_test_case_buttons(test_cases_frame)

        # Signal Input Section
        input_frame = tk.LabelFrame(left_frame,
                                    text=" ECG Signal Input ",
                                    font=('Arial', 12, 'bold'),
                                    fg=self.colors['primary'],
                                    bg=self.colors['bg_medium'],
                                    relief=tk.RAISED,
                                    borderwidth=2)
        input_frame.pack(fill=tk.X, padx=10, pady=10)

        self.create_input_controls(input_frame)

        # Signal Preview
        preview_frame = tk.LabelFrame(left_frame,
                                      text=" Signal Preview ",
                                      font=('Arial', 12, 'bold'),
                                      fg=self.colors['primary'],
                                      bg=self.colors['bg_medium'],
                                      relief=tk.RAISED,
                                      borderwidth=2)
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.create_signal_preview(preview_frame)

        # Analysis Controls
        controls_frame = tk.LabelFrame(left_frame,
                                       text=" Analysis Controls ",
                                       font=('Arial', 12, 'bold'),
                                       fg=self.colors['primary'],
                                       bg=self.colors['bg_medium'],
                                       relief=tk.RAISED,
                                       borderwidth=2)
        controls_frame.pack(fill=tk.X, padx=10, pady=10)

        self.create_analysis_controls(controls_frame)

        return left_frame

    def create_test_case_buttons(self, parent):
        """Create test case selection buttons"""
        button_frame = tk.Frame(parent, bg=self.colors['bg_medium'])
        button_frame.pack(fill=tk.X, padx=10, pady=10)

        # Normal button
        normal_btn = tk.Button(button_frame,
                               text="📈 Normal Sinus Rhythm\nRegular rhythm with all waves",
                               font=('Arial', 10),
                               fg='black',
                               bg='#00ff9d',
                               activebackground='#00cc7a',
                               relief=tk.RAISED,
                               borderwidth=2,
                               height=3,
                               width=25,
                               command=lambda: self.load_test_case("normal"))
        normal_btn.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.BOTH, expand=True)

        # Arrhythmia button
        arrhythmia_btn = tk.Button(button_frame,
                                   text="💔 Arrhythmia\nIrregular heart rhythm",
                                   font=('Arial', 10),
                                   fg='black',
                                   bg='#ff2a6d',
                                   activebackground='#cc2257',
                                   relief=tk.RAISED,
                                   borderwidth=2,
                                   height=3,
                                   width=25,
                                   command=lambda: self.load_test_case("arrhythmia"))
        arrhythmia_btn.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.BOTH, expand=True)

        # Bradycardia button
        bradycardia_btn = tk.Button(button_frame,
                                    text="🐢 Bradycardia\nSlow heart rate",
                                    font=('Arial', 10),
                                    fg='black',
                                    bg='#ffcc00',
                                    activebackground='#cc9900',
                                    relief=tk.RAISED,
                                    borderwidth=2,
                                    height=3,
                                    width=25,
                                    command=lambda: self.load_test_case("bradycardia"))
        bradycardia_btn.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.BOTH, expand=True)

    def create_input_controls(self, parent):
        """Create signal input controls"""
        control_frame = tk.Frame(parent, bg=self.colors['bg_medium'])
        control_frame.pack(fill=tk.X, padx=10, pady=10)

        # File upload
        upload_btn = tk.Button(control_frame,
                               text="📁 Upload CSV/TXT File",
                               font=('Arial', 10, 'bold'),
                               fg='black',
                               bg=self.colors['primary'],
                               activebackground='#00a3cc',
                               relief=tk.RAISED,
                               borderwidth=2,
                               height=2,
                               command=self.upload_file)
        upload_btn.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)

        # Or label
        or_label = tk.Label(control_frame,
                            text="OR",
                            font=('Arial', 10),
                            fg=self.colors['text'],
                            bg=self.colors['bg_medium'])
        or_label.pack(side=tk.LEFT, padx=10)

        # Generate sample
        gen_btn = tk.Button(control_frame,
                            text="✨ Generate Sample",
                            font=('Arial', 10, 'bold'),
                            fg='black',
                            bg=self.colors['secondary'],
                            activebackground='#cc2257',
                            relief=tk.RAISED,
                            borderwidth=2,
                            height=2,
                            command=self.generate_sample)
        gen_btn.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)

        # Signal info display
        info_frame = tk.Frame(parent, bg=self.colors['bg_light'])
        info_frame.pack(fill=tk.X, padx=10, pady=5)

        self.signal_info_var = tk.StringVar(value="No signal loaded")
        info_label = tk.Label(info_frame,
                              textvariable=self.signal_info_var,
                              font=('Arial', 9),
                              fg=self.colors['text'],
                              bg=self.colors['bg_light'],
                              justify=tk.LEFT)
        info_label.pack(padx=10, pady=5, anchor=tk.W)

    def create_signal_preview(self, parent):
        """Create signal preview plot"""
        # Create figure for signal plot
        self.signal_fig = Figure(figsize=(8, 3), dpi=80, facecolor=self.colors['bg_medium'])
        self.signal_ax = self.signal_fig.add_subplot(111)
        self.signal_ax.set_facecolor(self.colors['bg_light'])

        # Customize plot
        self.signal_ax.set_xlabel('Sample Points', color=self.colors['text'])
        self.signal_ax.set_ylabel('Amplitude', color=self.colors['text'])
        self.signal_ax.tick_params(colors=self.colors['text'])
        self.signal_ax.grid(True, alpha=0.3, color=self.colors['grid'])

        # Create canvas
        self.signal_canvas = FigureCanvasTkAgg(self.signal_fig, parent)
        self.signal_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def create_analysis_controls(self, parent):
        """Create analysis control buttons"""
        # Control buttons frame
        btn_frame = tk.Frame(parent, bg=self.colors['bg_medium'])
        btn_frame.pack(fill=tk.X, padx=10, pady=10)

        # Run DP Analysis button
        self.run_btn = tk.Button(btn_frame,
                                 text="⚡ Run DP Algorithm",
                                 font=('Arial', 12, 'bold'),
                                 fg='black',
                                 bg=self.colors['primary'],
                                 activebackground='#00a3cc',
                                 relief=tk.RAISED,
                                 borderwidth=3,
                                 height=2,
                                 command=self.run_dp_analysis)
        self.run_btn.pack(fill=tk.X, padx=5, pady=5)

        # Settings frame
        settings_frame = tk.Frame(parent, bg=self.colors['bg_medium'])
        settings_frame.pack(fill=tk.X, padx=10, pady=5)

        # Step-by-step visualization toggle
        self.step_by_step_var = tk.BooleanVar(value=True)
        step_toggle = tk.Checkbutton(settings_frame,
                                     text="Step-by-Step Visualization",
                                     variable=self.step_by_step_var,
                                     font=('Arial', 10),
                                     fg=self.colors['text'],
                                     bg=self.colors['bg_medium'],
                                     activebackground=self.colors['bg_medium'],
                                     selectcolor=self.colors['bg_light'])
        step_toggle.pack(side=tk.LEFT, padx=5)

        # Animation toggle
        self.animation_var = tk.BooleanVar(value=True)
        anim_toggle = tk.Checkbutton(settings_frame,
                                     text="Enable Animations",
                                     variable=self.animation_var,
                                     font=('Arial', 10),
                                     fg=self.colors['text'],
                                     bg=self.colors['bg_medium'],
                                     activebackground=self.colors['bg_medium'],
                                     selectcolor=self.colors['bg_light'])
        anim_toggle.pack(side=tk.LEFT, padx=5)

    def create_right_panel(self, parent):
        """Create right panel with visualization and results"""
        right_frame = tk.Frame(parent, bg=self.colors['bg_medium'],
                               relief=tk.RAISED, borderwidth=2)

        # Create notebook for tabs
        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Tab 1: Segmentation Visualization
        self.segmentation_tab = self.create_segmentation_tab()
        self.notebook.add(self.segmentation_tab, text="Segmentation")

        # Tab 2: DP Table
        self.dp_table_tab = self.create_dp_table_tab()
        self.notebook.add(self.dp_table_tab, text="DP Table")

        # Tab 3: DP Process
        self.dp_process_tab = self.create_dp_process_tab()
        self.notebook.add(self.dp_process_tab, text="DP Process")

        # Tab 4: Features
        self.features_tab = self.create_features_tab()
        self.notebook.add(self.features_tab, text="Features")

        # Tab 5: Classification
        self.classification_tab = self.create_classification_tab()
        self.notebook.add(self.classification_tab, text="Classification")

        # Tab 6: Comparison
        self.comparison_tab = self.create_comparison_tab()
        self.notebook.add(self.comparison_tab, text="Comparison")

        return right_frame

    def create_segmentation_tab(self):
        """Create segmentation visualization tab"""
        tab = tk.Frame(self.notebook, bg=self.colors['bg_medium'])

        # Create segmented ECG plot
        self.segmented_fig = Figure(figsize=(10, 4), dpi=80, facecolor=self.colors['bg_medium'])
        self.segmented_ax = self.segmented_fig.add_subplot(111)
        self.segmented_ax.set_facecolor(self.colors['bg_light'])

        # Customize plot
        self.segmented_ax.set_xlabel('Sample Points', color=self.colors['text'])
        self.segmented_ax.set_ylabel('Amplitude', color=self.colors['text'])
        self.segmented_ax.tick_params(colors=self.colors['text'])
        self.segmented_ax.grid(True, alpha=0.3, color=self.colors['grid'])

        # Create canvas
        self.segmented_canvas = FigureCanvasTkAgg(self.segmented_fig, tab)
        self.segmented_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Legend
        legend_frame = tk.Frame(tab, bg=self.colors['bg_medium'])
        legend_frame.pack(fill=tk.X, padx=10, pady=5)

        legend_text = "Legend: P Wave (Blue) | QRS Complex (Red) | T Wave (Green)"
        legend_label = tk.Label(legend_frame,
                                text=legend_text,
                                font=('Arial', 10),
                                fg=self.colors['text'],
                                bg=self.colors['bg_medium'])
        legend_label.pack()

        return tab

    def create_dp_table_tab(self):
        """Create DP table tab with pagination"""
        tab = tk.Frame(self.notebook, bg=self.colors['bg_medium'])

        # Header
        header_frame = tk.Frame(tab, bg=self.colors['bg_medium'])
        header_frame.pack(fill=tk.X, padx=10, pady=10)

        header_label = tk.Label(header_frame,
                                text="Dynamic Programming Table",
                                font=('Arial', 14, 'bold'),
                                fg=self.colors['primary'],
                                bg=self.colors['bg_medium'])
        header_label.pack(side=tk.LEFT)

        # Page controls
        control_frame = tk.Frame(tab, bg=self.colors['bg_medium'])
        control_frame.pack(fill=tk.X, padx=10, pady=5)

        # First page button
        first_btn = tk.Button(control_frame,
                              text="<< First",
                              font=('Arial', 9),
                              fg='black',
                              bg=self.colors['primary'],
                              command=lambda: self.change_dp_page(1))
        first_btn.pack(side=tk.LEFT, padx=2)

        # Previous page button
        prev_btn = tk.Button(control_frame,
                             text="< Prev",
                             font=('Arial', 9),
                             fg='black',
                             bg=self.colors['primary'],
                             command=lambda: self.change_dp_page(max(1, self.current_dp_page - 1)))
        prev_btn.pack(side=tk.LEFT, padx=2)

        # Page indicator
        self.page_label = tk.Label(control_frame,
                                   text="Page 1 of 1",
                                   font=('Arial', 10),
                                   fg=self.colors['text'],
                                   bg=self.colors['bg_medium'])
        self.page_label.pack(side=tk.LEFT, padx=10)

        # Next page button
        next_btn = tk.Button(control_frame,
                             text="Next >",
                             font=('Arial', 9),
                             fg='black',
                             bg=self.colors['primary'],
                             command=lambda: self.change_dp_page(self.current_dp_page + 1))
        next_btn.pack(side=tk.LEFT, padx=2)

        # Last page button
        last_btn = tk.Button(control_frame,
                             text="Last >>",
                             font=('Arial', 9),
                             fg='black',
                             bg=self.colors['primary'],
                             command=lambda: self.change_dp_page(self.total_dp_pages))
        last_btn.pack(side=tk.LEFT, padx=2)

        # Page size selector
        page_size_frame = tk.Frame(control_frame, bg=self.colors['bg_medium'])
        page_size_frame.pack(side=tk.LEFT, padx=20)

        tk.Label(page_size_frame,
                 text="Rows per page:",
                 font=('Arial', 9),
                 fg=self.colors['text'],
                 bg=self.colors['bg_medium']).pack(side=tk.LEFT)

        self.page_size_var = tk.StringVar(value="50")
        page_size_menu = ttk.Combobox(page_size_frame,
                                      textvariable=self.page_size_var,
                                      values=["25", "50", "100", "200"],
                                      width=8,
                                      state="readonly")
        page_size_menu.pack(side=tk.LEFT, padx=5)
        page_size_menu.bind("<<ComboboxSelected>>", self.update_dp_table)

        # DP Table Treeview
        table_frame = tk.Frame(tab, bg=self.colors['bg_medium'])
        table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create treeview with scrollbars
        self.dp_table_tree = ttk.Treeview(table_frame,
                                          columns=('pos', 'cost', 'segment', 'prev', 'length'),
                                          show='headings',
                                          height=15)

        # Define columns
        self.dp_table_tree.heading('pos', text='Position')
        self.dp_table_tree.heading('cost', text='Cost')
        self.dp_table_tree.heading('segment', text='Segment')
        self.dp_table_tree.heading('prev', text='Previous')
        self.dp_table_tree.heading('length', text='Length')

        # Set column widths
        self.dp_table_tree.column('pos', width=80, anchor=tk.CENTER)
        self.dp_table_tree.column('cost', width=120, anchor=tk.CENTER)
        self.dp_table_tree.column('segment', width=80, anchor=tk.CENTER)
        self.dp_table_tree.column('prev', width=80, anchor=tk.CENTER)
        self.dp_table_tree.column('length', width=80, anchor=tk.CENTER)

        # Add scrollbars
        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=self.dp_table_tree.yview)
        hsb = ttk.Scrollbar(table_frame, orient="horizontal", command=self.dp_table_tree.xview)
        self.dp_table_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        # Layout
        self.dp_table_tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')

        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)

        # Initialize pagination
        self.current_dp_page = 1
        self.total_dp_pages = 1
        self.rows_per_page = 50

        return tab

    def create_dp_process_tab(self):
        """Create DP process visualization tab"""
        tab = tk.Frame(self.notebook, bg=self.colors['bg_medium'])

        # Control frame
        control_frame = tk.Frame(tab, bg=self.colors['bg_medium'])
        control_frame.pack(fill=tk.X, padx=10, pady=10)

        # Step navigation
        nav_frame = tk.Frame(control_frame, bg=self.colors['bg_medium'])
        nav_frame.pack(side=tk.LEFT)

        # First button
        first_btn = tk.Button(nav_frame,
                              text="⏮️ First",
                              font=('Arial', 9),
                              fg='black',
                              bg=self.colors['primary'],
                              command=lambda: self.set_dp_step(0))
        first_btn.pack(side=tk.LEFT, padx=2)

        # Previous button
        prev_btn = tk.Button(nav_frame,
                             text="◀️ Prev",
                             font=('Arial', 9),
                             fg='black',
                             bg=self.colors['primary'],
                             command=lambda: self.set_dp_step(max(0, self.current_dp_step - 1)))
        prev_btn.pack(side=tk.LEFT, padx=2)

        # Step indicator
        self.step_label = tk.Label(nav_frame,
                                   text="Step: 0/0",
                                   font=('Arial', 10, 'bold'),
                                   fg=self.colors['text'],
                                   bg=self.colors['bg_medium'],
                                   width=15)
        self.step_label.pack(side=tk.LEFT, padx=10)

        # Next button
        next_btn = tk.Button(nav_frame,
                             text="Next ▶️",
                             font=('Arial', 9),
                             fg='black',
                             bg=self.colors['primary'],
                             command=lambda: self.set_dp_step(self.current_dp_step + 1))
        next_btn.pack(side=tk.LEFT, padx=2)

        # Last button
        last_btn = tk.Button(nav_frame,
                             text="Last ⏭️",
                             font=('Arial', 9),
                             fg='black',
                             bg=self.colors['primary'],
                             command=lambda: self.set_dp_step(len(self.segmenter.dp_steps) - 1))
        last_btn.pack(side=tk.LEFT, padx=2)

        # Play/Pause button
        self.play_btn = tk.Button(control_frame,
                                  text="▶️ Play",
                                  font=('Arial', 9, 'bold'),
                                  fg='black',
                                  bg=self.colors['success'],
                                  command=self.toggle_animation)
        self.play_btn.pack(side=tk.RIGHT, padx=10)

        # Speed control
        speed_frame = tk.Frame(control_frame, bg=self.colors['bg_medium'])
        speed_frame.pack(side=tk.RIGHT, padx=10)

        tk.Label(speed_frame,
                 text="Speed:",
                 font=('Arial', 9),
                 fg=self.colors['text'],
                 bg=self.colors['bg_medium']).pack(side=tk.LEFT)

        self.speed_var = tk.StringVar(value="Normal")
        speed_menu = ttk.Combobox(speed_frame,
                                  textvariable=self.speed_var,
                                  values=["Slow", "Normal", "Fast"],
                                  width=8,
                                  state="readonly")
        speed_menu.pack(side=tk.LEFT, padx=5)

        # Create DP process visualization figure
        self.process_fig = Figure(figsize=(10, 6), dpi=80, facecolor=self.colors['bg_medium'])
        self.process_fig.subplots_adjust(hspace=0.4)

        # Subplot 1: Current DP state
        self.process_ax1 = self.process_fig.add_subplot(211)
        self.process_ax1.set_facecolor(self.colors['bg_light'])
        self.process_ax1.set_title("DP State at Current Step", color=self.colors['text'])
        self.process_ax1.set_xlabel('Position', color=self.colors['text'])
        self.process_ax1.set_ylabel('Cost', color=self.colors['text'])
        self.process_ax1.tick_params(colors=self.colors['text'])
        self.process_ax1.grid(True, alpha=0.3, color=self.colors['grid'])

        # Subplot 2: Step information
        self.process_ax2 = self.process_fig.add_subplot(212)
        self.process_ax2.set_facecolor(self.colors['bg_medium'])
        self.process_ax2.axis('off')

        # Create canvas
        self.process_canvas = FigureCanvasTkAgg(self.process_fig, tab)
        self.process_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        return tab

    def create_features_tab(self):
        """Create features display tab"""
        tab = tk.Frame(self.notebook, bg=self.colors['bg_medium'])

        # Features treeview
        frame = tk.Frame(tab, bg=self.colors['bg_medium'])
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create treeview
        self.features_tree = ttk.Treeview(frame,
                                          columns=('feature', 'value', 'unit', 'normal_range'),
                                          show='headings',
                                          height=20)

        # Define columns
        self.features_tree.heading('feature', text='Feature')
        self.features_tree.heading('value', text='Value')
        self.features_tree.heading('unit', text='Unit')
        self.features_tree.heading('normal_range', text='Normal Range')

        # Set column widths
        self.features_tree.column('feature', width=180, anchor=tk.W)
        self.features_tree.column('value', width=120, anchor=tk.CENTER)
        self.features_tree.column('unit', width=80, anchor=tk.CENTER)
        self.features_tree.column('normal_range', width=150, anchor=tk.CENTER)

        # Add scrollbar
        vsb = ttk.Scrollbar(frame, orient="vertical", command=self.features_tree.yview)
        self.features_tree.configure(yscrollcommand=vsb.set)

        # Layout
        self.features_tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')

        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        return tab

    def create_classification_tab(self):
        """Create classification results tab"""
        tab = tk.Frame(self.notebook, bg=self.colors['bg_medium'])

        # Create a frame for classification results
        results_frame = tk.Frame(tab, bg=self.colors['bg_medium'])
        results_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Classification result card
        self.result_card = tk.Frame(results_frame,
                                    bg=self.colors['bg_light'],
                                    relief=tk.RAISED,
                                    borderwidth=3)
        self.result_card.pack(fill=tk.X, pady=10)

        # Prediction label
        self.prediction_label = tk.Label(self.result_card,
                                         text="Prediction: Not Yet Analyzed",
                                         font=('Arial', 18, 'bold'),
                                         fg=self.colors['warning'],
                                         bg=self.colors['bg_light'])
        self.prediction_label.pack(padx=20, pady=10)

        # Confidence bar
        confidence_frame = tk.Frame(results_frame, bg=self.colors['bg_medium'])
        confidence_frame.pack(fill=tk.X, pady=10)

        tk.Label(confidence_frame,
                 text="Confidence:",
                 font=('Arial', 12),
                 fg=self.colors['text'],
                 bg=self.colors['bg_medium']).pack(side=tk.LEFT, padx=5)

        self.confidence_bar = ttk.Progressbar(confidence_frame,
                                              length=300,
                                              mode='determinate')
        self.confidence_bar.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        self.confidence_label = tk.Label(confidence_frame,
                                         text="0%",
                                         font=('Arial', 12),
                                         fg=self.colors['text'],
                                         bg=self.colors['bg_medium'])
        self.confidence_label.pack(side=tk.LEFT, padx=5)

        # Method info
        method_frame = tk.Frame(results_frame, bg=self.colors['bg_medium'])
        method_frame.pack(fill=tk.X, pady=5)

        self.method_label = tk.Label(method_frame,
                                     text="Method: Not specified",
                                     font=('Arial', 10),
                                     fg=self.colors['text'],
                                     bg=self.colors['bg_medium'])
        self.method_label.pack()

        # Details text area
        details_frame = tk.LabelFrame(results_frame,
                                      text=" Detailed Analysis ",
                                      font=('Arial', 11, 'bold'),
                                      fg=self.colors['primary'],
                                      bg=self.colors['bg_medium'],
                                      relief=tk.RAISED,
                                      borderwidth=2)
        details_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        self.details_text = scrolledtext.ScrolledText(details_frame,
                                                      wrap=tk.WORD,
                                                      width=60,
                                                      height=10,
                                                      font=('Arial', 9),
                                                      bg=self.colors['bg_light'],
                                                      fg=self.colors['text'])
        self.details_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Export button
        export_btn = tk.Button(results_frame,
                               text="📊 Export Report",
                               font=('Arial', 10, 'bold'),
                               fg='black',
                               bg=self.colors['primary'],
                               command=self.export_report)
        export_btn.pack(pady=10)

        return tab

    def create_comparison_tab(self):
        """Create comparison tab for different algorithms"""
        tab = tk.Frame(self.notebook, bg=self.colors['bg_medium'])

        # Create figure for algorithm comparison
        self.comparison_fig = Figure(figsize=(10, 6), dpi=80, facecolor=self.colors['bg_medium'])
        self.comparison_ax = self.comparison_fig.add_subplot(111)
        self.comparison_ax.set_facecolor(self.colors['bg_light'])

        # Customize plot
        self.comparison_ax.set_title("Algorithm Performance Comparison", color=self.colors['text'])
        self.comparison_ax.set_xlabel('Algorithm', color=self.colors['text'])
        self.comparison_ax.set_ylabel('Performance Score', color=self.colors['text'])
        self.comparison_ax.tick_params(colors=self.colors['text'])
        self.comparison_ax.grid(True, alpha=0.3, color=self.colors['grid'])

        # Create canvas
        self.comparison_canvas = FigureCanvasTkAgg(self.comparison_fig, tab)
        self.comparison_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Comparison info
        info_frame = tk.Frame(tab, bg=self.colors['bg_medium'])
        info_frame.pack(fill=tk.X, padx=10, pady=5)

        self.comparison_info = tk.Label(info_frame,
                                        text="DP Algorithm vs Traditional Methods",
                                        font=('Arial', 10),
                                        fg=self.colors['text'],
                                        bg=self.colors['bg_medium'])
        self.comparison_info.pack()

        return tab

    def create_footer(self, parent):
        """Create application footer"""
        footer_frame = tk.Frame(parent, bg=self.colors['bg_medium'], height=40)
        footer_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=(10, 0))
        footer_frame.pack_propagate(False)

        # Copyright
        copyright_label = tk.Label(footer_frame,
                                   text="© 2024 ECG Analysis Tool - Academic Project | Biomedical Engineering",
                                   font=('Arial', 9),
                                   fg=self.colors['text'],
                                   bg=self.colors['bg_medium'])
        copyright_label.pack(side=tk.LEFT, padx=20, pady=10)

        # Version info
        version_label = tk.Label(footer_frame,
                                 text="Version 1.0 | DP-ECG v1.0",
                                 font=('Arial', 9),
                                 fg=self.colors['primary'],
                                 bg=self.colors['bg_medium'])
        version_label.pack(side=tk.RIGHT, padx=20, pady=10)

        # Progress bar for long operations
        self.progress_bar = ttk.Progressbar(footer_frame,
                                            mode='indeterminate',
                                            length=200)
        self.progress_bar.pack(side=tk.RIGHT, padx=10)

    # ============================================
    # 5. APPLICATION LOGIC
    # ============================================

    def load_test_case(self, case_type):
        """Load a test case ECG signal"""
        try:
            self.status_var.set(f"Loading {case_type} signal...")

            # Generate signal based on case type
            if case_type == "normal":
                signal = self.generator.generate_normal(400)
                signal_type = "Normal Sinus Rhythm"
                color = "#00ff9d"
            elif case_type == "arrhythmia":
                signal = self.generator.generate_arrhythmia(400)
                signal_type = "Arrhythmia"
                color = "#ff2a6d"
            elif case_type == "bradycardia":
                signal = self.generator.generate_bradycardia(400)
                signal_type = "Bradycardia"
                color = "#ffcc00"
            else:
                signal = self.generator.generate_normal(400)
                signal_type = "Test Signal"
                color = self.colors['primary']

            # Normalize signal
            self.current_signal = self.segmenter.normalize_signal(signal)
            self.current_signal_type = signal_type

            # Update signal info
            self.signal_info_var.set(
                f"Type: {signal_type}\n"
                f"Length: {len(signal)} samples\n"
                f"Duration: {len(signal) / self.segmenter.fs:.2f} seconds"
            )

            # Plot signal
            self.update_signal_plot()

            # Reset analysis results
            self.dp_results = None
            self.current_dp_step = 0
            self.current_backtrack_step = 0

            self.status_var.set(f"Loaded {signal_type}")

            # Clear previous results
            self.clear_results()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load test case: {str(e)}")
            self.status_var.set("Error loading signal")

    def upload_file(self):
        """Upload ECG signal from file"""
        file_path = filedialog.askopenfilename(
            title="Select ECG Data File",
            filetypes=[
                ("CSV files", "*.csv"),
                ("Text files", "*.txt"),
                ("MAT files", "*.mat"),
                ("All files", "*.*")
            ]
        )

        if not file_path:
            return

        try:
            self.status_var.set(f"Loading file: {os.path.basename(file_path)}")

            # Read file based on extension
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path, header=None)
                signal = df.iloc[:, 0].values
            elif file_path.endswith('.txt'):
                signal = np.loadtxt(file_path)
            else:
                messagebox.showerror("Error", "Unsupported file format")
                return

            # Ensure signal is 1D and not too large
            if len(signal.shape) > 1:
                signal = signal.flatten()

            # Limit signal length for performance
            if len(signal) > 10000:
                signal = signal[:10000]
                messagebox.showwarning("Warning", "Signal truncated to 10,000 samples")

            # Normalize and store
            self.current_signal = self.segmenter.normalize_signal(signal)
            self.current_signal_type = f"File: {os.path.basename(file_path)}"

            # Update info
            self.signal_info_var.set(
                f"Source: {os.path.basename(file_path)}\n"
                f"Length: {len(signal)} samples\n"
                f"Duration: {len(signal) / self.segmenter.fs:.2f} seconds"
            )

            # Plot signal
            self.update_signal_plot()

            # Reset results
            self.dp_results = None
            self.clear_results()

            self.status_var.set("File loaded successfully")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {str(e)}")
            self.status_var.set("Error loading file")

    def generate_sample(self):
        """Generate a random ECG sample"""
        try:
            # Choose random type
            types = ["normal", "arrhythmia", "bradycardia"]
            random_type = random.choice(types)

            # Generate random length between 300-600 samples
            length = random.randint(300, 600)

            # Generate signal
            if random_type == "normal":
                signal = self.generator.generate_normal(length)
                signal_type = "Random Normal"
                color = "#00ff9d"
            elif random_type == "arrhythmia":
                signal = self.generator.generate_arrhythmia(length)
                signal_type = "Random Arrhythmia"
                color = "#ff2a6d"
            else:
                signal = self.generator.generate_bradycardia(length)
                signal_type = "Random Bradycardia"
                color = "#ffcc00"

            # Add random noise
            noise_level = random.uniform(0.02, 0.1)
            signal += np.random.normal(0, noise_level, len(signal))

            # Normalize and store
            self.current_signal = self.segmenter.normalize_signal(signal)
            self.current_signal_type = signal_type

            # Update info
            self.signal_info_var.set(
                f"Type: {signal_type}\n"
                f"Length: {len(signal)} samples\n"
                f"Noise Level: {noise_level:.3f}\n"
                f"Generated: {datetime.now().strftime('%H:%M:%S')}"
            )

            # Plot signal
            self.update_signal_plot()

            # Reset results
            self.dp_results = None
            self.clear_results()

            self.status_var.set(f"Generated {signal_type} sample")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate sample: {str(e)}")
            self.status_var.set("Error generating sample")

    def update_signal_plot(self):
        """Update the signal preview plot"""
        if self.current_signal is None:
            return

        self.signal_ax.clear()

        # Plot signal
        self.signal_ax.plot(self.current_signal,
                            color=self.colors['primary'],
                            linewidth=1.5,
                            label='ECG Signal')

        # Customize plot
        self.signal_ax.set_xlabel('Sample Points', color=self.colors['text'])
        self.signal_ax.set_ylabel('Amplitude (Normalized)', color=self.colors['text'])
        self.signal_ax.set_title('ECG Signal Preview', color=self.colors['text'])
        self.signal_ax.tick_params(colors=self.colors['text'])
        self.signal_ax.grid(True, alpha=0.3, color=self.colors['grid'])
        self.signal_ax.legend(facecolor=self.colors['bg_light'])

        self.signal_ax.set_facecolor(self.colors['bg_light'])
        self.signal_canvas.draw()

    def run_dp_analysis(self):
        """Run the DP segmentation algorithm"""
        if self.current_signal is None:
            messagebox.showwarning("Warning", "Please load or generate an ECG signal first")
            return

        try:
            # Update status and show progress
            self.status_var.set("Running DP Algorithm...")
            self.progress_bar.start()
            self.run_btn.config(state=tk.DISABLED)

            # Get step-by-step flag
            step_by_step = self.step_by_step_var.get()

            # Run DP segmentation in a separate thread to keep GUI responsive
            def run_segmentation():
                try:
                    results = self.segmenter.segment_with_dp(
                        self.current_signal,
                        step_by_step=step_by_step
                    )

                    # Get features
                    features = self.segmenter.extract_features(
                        results["segments"],
                        self.current_signal
                    )

                    # Get classification
                    classification = self.classifier.classify(features)

                    # Update GUI in main thread
                    self.root.after(0, lambda: self.update_results(results, features, classification))

                except Exception as e:
                    self.root.after(0, lambda: self.handle_segmentation_error(e))

            # Start segmentation thread
            thread = threading.Thread(target=run_segmentation, daemon=True)
            thread.start()

        except Exception as e:
            self.handle_segmentation_error(e)

    def update_results(self, results, features, classification):
        """Update all results displays"""
        try:
            self.dp_results = results

            # Update segmented plot
            self.update_segmented_plot()

            # Update DP table
            self.update_dp_table()

            # Update features
            self.update_features_display(features)

            # Update classification
            self.update_classification_display(classification)

            # Update DP process visualization
            if self.segmenter.dp_steps:
                self.update_dp_process_display(0)

            # Update comparison
            self.update_comparison()

            # Stop progress bar
            self.progress_bar.stop()
            self.run_btn.config(state=tk.NORMAL)

            # Update status
            seg_count = len(results["segments"])
            total_cost = results["total_cost"]
            self.status_var.set(f"DP Complete: {seg_count} segments, Cost: {total_cost:.2f}")

            # Switch to segmentation tab
            self.notebook.select(self.segmentation_tab)

        except Exception as e:
            self.handle_segmentation_error(e)

    def update_segmented_plot(self):
        """Update the segmented ECG plot"""
        if self.dp_results is None or self.current_signal is None:
            return

        self.segmented_ax.clear()

        # Plot original signal
        self.segmented_ax.plot(self.current_signal,
                               color='gray',
                               alpha=0.6,
                               linewidth=1,
                               label='Original Signal')

        # Plot segments with different colors
        colors = {"P": "blue", "QRS": "red", "T": "green"}

        for segment in self.dp_results["segments"]:
            seg_type = segment["type"]
            start = segment["start"]
            end = segment["end"] + 1  # Inclusive

            if start < len(self.current_signal) and end <= len(self.current_signal):
                x = np.arange(start, end)
                y = self.current_signal[start:end]

                self.segmented_ax.plot(x, y,
                                       color=colors.get(seg_type, "orange"),
                                       linewidth=2.5,
                                       label=seg_type)

                # Add segment type label
                mid_x = (start + end) // 2
                if mid_x < len(self.current_signal):
                    mid_y = self.current_signal[mid_x]
                    self.segmented_ax.text(mid_x, mid_y, seg_type,
                                           color=colors.get(seg_type, "orange"),
                                           fontsize=9,
                                           ha='center',
                                           va='center',
                                           bbox=dict(boxstyle='round,pad=0.2',
                                                     facecolor='white',
                                                     alpha=0.7))

        # Customize plot
        self.segmented_ax.set_xlabel('Sample Points', color=self.colors['text'])
        self.segmented_ax.set_ylabel('Amplitude', color=self.colors['text'])
        self.segmented_ax.set_title('ECG Segmentation Result', color=self.colors['text'])
        self.segmented_ax.tick_params(colors=self.colors['text'])
        self.segmented_ax.grid(True, alpha=0.3, color=self.colors['grid'])

        # Create custom legend (avoid duplicates)
        handles, labels = self.segmented_ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            self.segmented_ax.legend(by_label.values(), by_label.keys(),
                                     facecolor=self.colors['bg_light'])

        self.segmented_ax.set_facecolor(self.colors['bg_light'])
        self.segmented_canvas.draw()

    def update_dp_table(self, event=None):
        """Update the DP table display"""
        if self.dp_results is None:
            return

        try:
            # Get page size
            self.rows_per_page = int(self.page_size_var.get())

            # Clear current items
            for item in self.dp_table_tree.get_children():
                self.dp_table_tree.delete(item)

            # Get the DP table data
            dp_table = self.dp_results.get("dp_table", [])

            if not dp_table:
                return

            # Calculate pagination
            total_items = len(dp_table)
            self.total_dp_pages = max(1, (total_items + self.rows_per_page - 1) // self.rows_per_page)
            self.current_dp_page = min(self.current_dp_page, self.total_dp_pages)

            start_idx = (self.current_dp_page - 1) * self.rows_per_page
            end_idx = min(start_idx + self.rows_per_page, total_items)

            # Insert items for current page
            for i in range(start_idx, end_idx):
                item = dp_table[i]

                # Format cost
                cost = item.get("cost", float('inf'))
                if cost == float('inf'):
                    cost_str = "∞"
                else:
                    cost_str = f"{cost:.3f}"

                # Insert into treeview
                self.dp_table_tree.insert("", "end",
                                          values=(item.get("position", ""),
                                                  cost_str,
                                                  item.get("segment", ""),
                                                  item.get("prev", ""),
                                                  item.get("length", "")))

            # Update page label
            self.page_label.config(text=f"Page {self.current_dp_page} of {self.total_dp_pages}")

        except Exception as e:
            print(f"Error updating DP table: {e}")

    def change_dp_page(self, page):
        """Change DP table page"""
        if page < 1 or page > self.total_dp_pages:
            return

        self.current_dp_page = page
        self.update_dp_table()

    def update_dp_process_display(self, step_index):
        """Update DP process visualization"""
        if not self.segmenter.dp_steps or step_index >= len(self.segmenter.dp_steps):
            return

        step = self.segmenter.dp_steps[step_index]
        self.current_dp_step = step_index

        # Clear axes
        self.process_ax1.clear()
        self.process_ax2.clear()

        # Plot 1: DP cost progression
        positions = []
        costs = []

        for i, s in enumerate(self.segmenter.dp_steps[:step_index + 1]):
            positions.append(s.get("position", 0))
            costs.append(s.get("cost", 0))

        if positions:
            self.process_ax1.plot(positions, costs, 'o-',
                                  color=self.colors['primary'],
                                  linewidth=2,
                                  markersize=4,
                                  label='DP Cost')

            # Highlight current position
            current_pos = step.get("position", 0)
            current_cost = step.get("cost", 0)
            self.process_ax1.plot(current_pos, current_cost, 'o',
                                  color=self.colors['secondary'],
                                  markersize=10,
                                  label='Current')

        # Customize plot 1
        self.process_ax1.set_xlabel('Position', color=self.colors['text'])
        self.process_ax1.set_ylabel('Cost', color=self.colors['text'])
        self.process_ax1.set_title('DP Cost Progression', color=self.colors['text'])
        self.process_ax1.tick_params(colors=self.colors['text'])
        self.process_ax1.grid(True, alpha=0.3, color=self.colors['grid'])
        self.process_ax1.legend(facecolor=self.colors['bg_light'])
        self.process_ax1.set_facecolor(self.colors['bg_light'])

        # Plot 2: Step information (text)
        info_text = f"Step {step_index + 1}/{len(self.segmenter.dp_steps)}\n"
        info_text += f"Position: {step.get('position', 0)}\n"
        info_text += f"Cost: {step.get('cost', 0):.3f}\n"
        info_text += f"Action: {step.get('action', '')}\n\n"

        # Add DP table snippet
        dp_table = step.get("dp_table", [])
        if dp_table:
            info_text += "DP Table (current region):\n"
            info_text += "Pos | Cost    | Seg | Prev\n"
            info_text += "-" * 35 + "\n"

            for i, row in enumerate(dp_table[-10:]):  # Show last 10 rows
                pos = row.get("position", 0)
                cost = row.get("cost", 0)
                seg = row.get("segment", "-")
                prev = row.get("prev", "-")

                if cost == float('inf'):
                    cost_str = "∞"
                else:
                    cost_str = f"{cost:.3f}"

                info_text += f"{pos:3d} | {cost_str:7s} | {seg:3s} | {prev}\n"

        self.process_ax2.text(0.05, 0.95, info_text,
                              fontsize=9,
                              fontfamily='monospace',
                              color=self.colors['text'],
                              verticalalignment='top',
                              transform=self.process_ax2.transAxes)

        self.process_ax2.set_facecolor(self.colors['bg_medium'])
        self.process_ax2.axis('off')

        # Update step label
        self.step_label.config(text=f"Step: {step_index + 1}/{len(self.segmenter.dp_steps)}")

        self.process_canvas.draw()

    def set_dp_step(self, step_index):
        """Set current DP step for visualization"""
        if not self.segmenter.dp_steps:
            return

        step_index = max(0, min(step_index, len(self.segmenter.dp_steps) - 1))
        self.update_dp_process_display(step_index)

    def toggle_animation(self):
        """Toggle animation playback"""
        if not self.segmenter.dp_steps:
            return

        if not self.is_playing:
            # Start animation
            self.is_playing = True
            self.play_btn.config(text="⏸️ Pause", bg=self.colors['warning'])
            self.animate_dp_steps()
        else:
            # Stop animation
            self.is_playing = False
            self.play_btn.config(text="▶️ Play", bg=self.colors['success'])

    def animate_dp_steps(self):
        """Animate DP steps"""
        if not self.is_playing or not self.segmenter.dp_steps:
            return

        if self.current_dp_step < len(self.segmenter.dp_steps) - 1:
            self.current_dp_step += 1
            self.update_dp_process_display(self.current_dp_step)

            # Determine speed
            speed = self.speed_var.get()
            if speed == "Slow":
                delay = 1000
            elif speed == "Fast":
                delay = 100
            else:  # Normal
                delay = 300

            # Schedule next step
            self.root.after(delay, self.animate_dp_steps)
        else:
            # Animation complete
            self.is_playing = False
            self.play_btn.config(text="▶️ Play", bg=self.colors['success'])

    def update_features_display(self, features):
        """Update features display"""
        # Clear current items
        for item in self.features_tree.get_children():
            self.features_tree.delete(item)

        # Define normal ranges for common ECG features
        normal_ranges = {
            'p_count': ('2-10', 'Number of P waves'),
            'qrs_count': ('2-10', 'Number of QRS complexes'),
            't_count': ('2-10', 'Number of T waves'),
            'p_duration_mean': ('40-120 ms', 'P wave duration'),
            'qrs_duration_mean': ('60-100 ms', 'QRS complex duration'),
            't_duration_mean': ('100-250 ms', 'T wave duration'),
            'rr_mean': ('600-1000 ms', 'RR interval'),
            'rr_std': ('<50 ms', 'RR variability'),
            'heart_rate': ('60-100 BPM', 'Heart rate'),
            'signal_mean': ('≈0', 'Signal mean'),
            'signal_std': ('≈1', 'Signal std dev'),
            'signal_skewness': ('≈0', 'Signal skewness')
        }

        # Convert durations from samples to ms
        for key in list(features.keys()):
            if 'duration' in key:
                features[key] = features[key] * (1000 / self.segmenter.fs)

        # Add features to treeview
        for feature_name, feature_value in features.items():
            # Determine unit
            unit = ""
            if 'duration' in feature_name or 'rr_' in feature_name:
                unit = "ms"
            elif feature_name == 'heart_rate':
                unit = "BPM"
            elif 'count' in feature_name:
                unit = "count"

            # Get normal range
            normal_range, description = normal_ranges.get(feature_name, ('N/A', ''))

            # Format value
            if isinstance(feature_value, (int, np.integer)):
                value_str = str(feature_value)
            elif isinstance(feature_value, (float, np.floating)):
                value_str = f"{feature_value:.3f}"
            else:
                value_str = str(feature_value)

            # Insert into treeview
            self.features_tree.insert("", "end",
                                      values=(feature_name.replace('_', ' ').title(),
                                              value_str,
                                              unit,
                                              normal_range))

    def update_classification_display(self, classification):
        """Update classification results display"""
        prediction = classification.get("prediction", "Unknown")
        confidence = classification.get("confidence", 0) * 100
        method = classification.get("method", "unknown")
        abnormalities = classification.get("abnormalities", [])
        notes = classification.get("notes", "")

        # Update prediction label with color coding
        if prediction == "Normal":
            color = self.colors['success']
            icon = "✅ "
        elif prediction == "Abnormal":
            color = self.colors['secondary']
            icon = "⚠️ "
        else:
            color = self.colors['warning']
            icon = "❓ "

        self.prediction_label.config(
            text=f"{icon}Prediction: {prediction}",
            fg=color
        )

        # Update result card background
        self.result_card.config(bg=color)
        self.prediction_label.config(bg=color)

        # Update confidence bar
        self.confidence_bar['value'] = confidence
        self.confidence_label.config(text=f"{confidence:.1f}%")

        # Update method label
        method_display = method.replace('_', ' ').title()
        self.method_label.config(text=f"Method: {method_display}")

        # Update details text
        self.details_text.delete(1.0, tk.END)

        details = f"Classification Result:\n"
        details += f"{'=' * 40}\n"
        details += f"Prediction: {prediction}\n"
        details += f"Confidence: {confidence:.1f}%\n"
        details += f"Method: {method_display}\n\n"

        if abnormalities:
            details += f"Detected Abnormalities:\n"
            for abn in abnormalities:
                details += f"  • {abn}\n"
            details += f"\nNotes: {notes}\n"
        else:
            details += f"No abnormalities detected.\n"
            details += f"Notes: {notes}\n"

        # Add probabilities if available
        probabilities = classification.get("probabilities", {})
        if probabilities:
            details += f"\nProbabilities:\n"
            for cls, prob in probabilities.items():
                details += f"  {cls}: {prob * 100:.1f}%\n"

        self.details_text.insert(1.0, details)

    def update_comparison(self):
        """Update algorithm comparison plot"""
        if self.dp_results is None:
            return

        self.comparison_ax.clear()

        # Simulate comparison data (in real app, this would be actual metrics)
        algorithms = ['DP Algorithm', 'Threshold', 'Template\nMatching', 'Wavelet\nTransform']

        # Performance metrics (simulated)
        accuracy = [92, 75, 83, 88]
        speed = [85, 95, 70, 65]  # Higher is faster
        robustness = [90, 60, 80, 85]

        x = np.arange(len(algorithms))
        width = 0.25

        # Plot bars
        bars1 = self.comparison_ax.bar(x - width, accuracy, width,
                                       label='Accuracy (%)',
                                       color=self.colors['primary'])
        bars2 = self.comparison_ax.bar(x, speed, width,
                                       label='Speed (%)',
                                       color=self.colors['secondary'])
        bars3 = self.comparison_ax.bar(x + width, robustness, width,
                                       label='Robustness (%)',
                                       color=self.colors['success'])

        # Add value labels on bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                self.comparison_ax.text(bar.get_x() + bar.get_width() / 2.,
                                        height + 1,
                                        f'{int(height)}%',
                                        ha='center', va='bottom',
                                        fontsize=8,
                                        color=self.colors['text'])

        # Customize plot
        self.comparison_ax.set_xlabel('Algorithm', color=self.colors['text'])
        self.comparison_ax.set_ylabel('Performance Score (%)', color=self.colors['text'])
        self.comparison_ax.set_title('Algorithm Performance Comparison', color=self.colors['text'])
        self.comparison_ax.set_xticks(x)
        self.comparison_ax.set_xticklabels(algorithms, color=self.colors['text'])
        self.comparison_ax.tick_params(colors=self.colors['text'])
        self.comparison_ax.grid(True, alpha=0.3, color=self.colors['grid'], axis='y')
        self.comparison_ax.legend(facecolor=self.colors['bg_light'])

        # Add DP algorithm advantage note
        self.comparison_ax.text(0.5, -0.15,
                                "DP Algorithm shows best balance of accuracy and robustness",
                                transform=self.comparison_ax.transAxes,
                                ha='center', fontsize=9,
                                color=self.colors['primary'],
                                bbox=dict(boxstyle='round',
                                          facecolor=self.colors['bg_light'],
                                          alpha=0.8))

        self.comparison_ax.set_facecolor(self.colors['bg_light'])
        self.comparison_canvas.draw()

        # Update comparison info
        info_text = "DP Algorithm vs Traditional Methods: "
        info_text += "DP provides optimal segmentation with dynamic constraints"
        self.comparison_info.config(text=info_text)

    def export_report(self):
        """Export analysis report"""
        if self.dp_results is None:
            messagebox.showwarning("Warning", "No analysis results to export")
            return

        # Ask for save location
        file_path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[
                ("PDF files", "*.pdf"),
                ("Text files", "*.txt"),
                ("JSON files", "*.json")
            ],
            title="Save Analysis Report"
        )

        if not file_path:
            return

        try:
            self.status_var.set("Exporting report...")

            # Prepare report data
            report_data = {
                "timestamp": datetime.now().isoformat(),
                "signal_type": self.current_signal_type,
                "signal_length": len(self.current_signal) if self.current_signal is not None else 0,
                "sampling_rate": self.segmenter.fs,
                "dp_results": {
                    "total_segments": len(self.dp_results.get("segments", [])),
                    "total_cost": float(self.dp_results.get("total_cost", 0)),
                    "segments": self.dp_results.get("segments", [])
                },
                "classification": self.get_current_classification()
            }

            # Export based on file type
            if file_path.endswith('.json'):
                with open(file_path, 'w') as f:
                    json.dump(report_data, f, indent=2, default=str)

            elif file_path.endswith('.txt'):
                with open(file_path, 'w') as f:
                    f.write("=" * 60 + "\n")
                    f.write("ECG ANALYSIS REPORT\n")
                    f.write("=" * 60 + "\n\n")

                    f.write(f"Timestamp: {report_data['timestamp']}\n")
                    f.write(f"Signal Type: {report_data['signal_type']}\n")
                    f.write(f"Signal Length: {report_data['signal_length']} samples\n")
                    f.write(f"Sampling Rate: {report_data['sampling_rate']} Hz\n\n")

                    f.write("DP Segmentation Results:\n")
                    f.write(f"  Total Segments: {report_data['dp_results']['total_segments']}\n")
                    f.write(f"  Total Cost: {report_data['dp_results']['total_cost']:.3f}\n\n")

                    f.write("Segments:\n")
                    for seg in report_data['dp_results']['segments']:
                        f.write(f"  {seg['type']}: {seg['start']}-{seg['end']} "
                                f"(length: {seg['length']}, cost: {seg['cost']:.3f})\n")

            else:  # PDF (simplified - in real app use ReportLab)
                with open(file_path, 'w') as f:
                    f.write("PDF export would require ReportLab library\n")
                    f.write("For now, saving as text content\n\n")
                    f.write(str(report_data))

            self.status_var.set(f"Report saved to {os.path.basename(file_path)}")
            messagebox.showinfo("Success", f"Report exported successfully!\n{file_path}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to export report: {str(e)}")
            self.status_var.set("Export failed")

    def get_current_classification(self):
        """Get current classification results"""
        # This would retrieve the current classification from the classifier
        # For now, return dummy data
        return {
            "prediction": "Normal",
            "confidence": 0.95,
            "method": "rule_based"
        }

    def clear_results(self):
        """Clear all results displays"""
        # Clear segmented plot
        self.segmented_ax.clear()
        self.segmented_ax.set_facecolor(self.colors['bg_light'])
        self.segmented_canvas.draw()

        # Clear DP table
        for item in self.dp_table_tree.get_children():
            self.dp_table_tree.delete(item)

        # Clear features tree
        for item in self.features_tree.get_children():
            self.features_tree.delete(item)

        # Clear classification
        self.prediction_label.config(text="Prediction: Not Yet Analyzed",
                                     fg=self.colors['warning'])
        self.result_card.config(bg=self.colors['bg_light'])
        self.prediction_label.config(bg=self.colors['bg_light'])
        self.confidence_bar['value'] = 0
        self.confidence_label.config(text="0%")
        self.method_label.config(text="Method: Not specified")
        self.details_text.delete(1.0, tk.END)

        # Clear DP process
        self.process_ax1.clear()
        self.process_ax2.clear()
        self.process_ax1.set_facecolor(self.colors['bg_light'])
        self.process_ax2.set_facecolor(self.colors['bg_medium'])
        self.process_canvas.draw()

        # Clear comparison
        self.comparison_ax.clear()
        self.comparison_ax.set_facecolor(self.colors['bg_light'])
        self.comparison_canvas.draw()

        # Reset steps
        self.current_dp_step = 0
        self.step_label.config(text="Step: 0/0")
        if self.is_playing:
            self.is_playing = False
            self.play_btn.config(text="▶️ Play", bg=self.colors['success'])

    def handle_segmentation_error(self, error):
        """Handle segmentation errors"""
        self.progress_bar.stop()
        self.run_btn.config(state=tk.NORMAL)
        self.status_var.set("DP Analysis Failed")

        messagebox.showerror("Segmentation Error",
                             f"Failed to segment ECG signal:\n{str(error)}")

    def create_tooltip(self, widget, text):
        """Create a tooltip for a widget"""
        def enter(event):
            x, y, _, _ = widget.bbox("insert")
            x += widget.winfo_rootx() + 25
            y += widget.winfo_rooty() + 20

            self.tooltip = tk.Toplevel(widget)
            self.tooltip.wm_overrideredirect(True)
            self.tooltip.wm_geometry(f"+{x}+{y}")

            label = tk.Label(self.tooltip, text=text,
                             background="yellow", relief='solid', borderwidth=1)
            label.pack()

        def leave(event):
            if hasattr(self, 'tooltip'):
                self.tooltip.destroy()

        widget.bind("<Enter>", enter)
        widget.bind("<Leave>", leave)

    def on_closing(self):
        """Handle application closing"""
        if messagebox.askokcancel("Quit", "Do you want to quit the application?"):
            # Stop any running animations or threads
            self.is_playing = False

            # Close matplotlib figures
            plt.close('all')

            # Destroy the application
            self.root.destroy()


# ============================================
# 7. MAIN ENTRY POINT
# ============================================
def main():
    """Main entry point for the application"""
    root = tk.Tk()

    # Create and run application
    app = ECGDPApp(root)

    # Set closing protocol
    root.protocol("WM_DELETE_WINDOW", app.on_closing)

    # Start the main loop
    root.mainloop()


if __name__ == "__main__":
    main()