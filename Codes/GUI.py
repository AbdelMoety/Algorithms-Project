import matplotlib
matplotlib.use("TkAgg")

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter import font as tkfont
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime


# ==========================================
# BIOMEDICAL ALGORITHMS
# ==========================================

STATE_NAMES = ["Baseline", "P", "PR", "QRS", "ST", "T"]
NUM_STATES = len(STATE_NAMES)
INF = float('inf')

PREV_STATES = {
    0: [0, 5], 1: [1, 0], 2: [2, 1], 
    3: [3, 2], 4: [4, 3], 5: [5, 4]
}

def get_emission_cost(voltage, state_index):
    if state_index in [0, 2, 4]: 
        return abs(voltage - 0.0) * 1.5  
    elif state_index in [1, 5]:
        dist_to_bump = abs(voltage - 0.2)
        dist_to_base = abs(voltage - 0.0)
        return min(dist_to_bump, dist_to_base * 0.8) 
    elif state_index == 3:
        if voltage < 0.15: return 10.0 
        return abs(voltage - 1.0) * 0.2
    return 100.0

def run_viterbi(signal_array):
    N = len(signal_array)
    dp = [[INF] * NUM_STATES for _ in range(N)]
    path = [[0] * NUM_STATES for _ in range(N)]
    dp[0][0] = get_emission_cost(signal_array[0], 0)
    
    for t in range(1, N):
        val = signal_array[t]
        for curr in range(NUM_STATES):
            best_prev_cost = INF
            best_prev = -1
            for prev in PREV_STATES[curr]:
                trans_cost = 0.5 if prev != curr else 0.0
                total_cost = dp[t-1][prev] + get_emission_cost(val, curr) + trans_cost
                if total_cost < best_prev_cost:
                    best_prev_cost = total_cost
                    best_prev = prev
            dp[t][curr] = best_prev_cost
            path[t][curr] = best_prev

    final_state = 0
    min_final = INF
    for s in range(NUM_STATES):
        if dp[N-1][s] < min_final:
            min_final = dp[N-1][s]
            final_state = s
            
    best_path = [0] * N
    best_path[N-1] = final_state
    for t in range(N-1, 0, -1):
        prev = path[t][best_path[t]]
        best_path[t-1] = prev
        
    # Return both the path and the DP table
    return [STATE_NAMES[i] for i in best_path], dp, path

def extract_features(segment_labels, signal_values):
    qrs_width = segment_labels.count("QRS")
    
    st_indices = [i for i, x in enumerate(segment_labels) if x == "ST"]
    if st_indices:
        st_level = sum([signal_values[i] for i in st_indices]) / len(st_indices)
    else:
        st_level = 0.0 
        
    pr_interval = segment_labels.count("P") + segment_labels.count("PR")

    return pd.DataFrame([[qrs_width, st_level, pr_interval]], 
                        columns=["QRS_Width", "ST_Level", "PR_Interval"])

def preprocess_signal(raw_signal):
    s = pd.Series(raw_signal)
    cleaned = s.rolling(window=5, center=True).mean().fillna(s)
    
    s_min = cleaned.min()
    s_max = cleaned.max()
    if s_max - s_min == 0: return cleaned.values
    normalized = (cleaned - s_min) / (s_max - s_min)
    return normalized.values


# ==========================================
# ENHANCED GUI APPLICATION
# ==========================================

class ModernECG_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced ECG Arrhythmia Analysis System")
        self.root.geometry("1400x900")
        self.root.configure(bg='#1e1e1e')
        
        # State variables
        self.signal_data = None
        self.model = None
        self.scaler = None
        self.analysis_history = []
        self.current_segments = None
        self.processed_signal = None
        self.dp_table = None
        self.path_table = None
        
        # Color scheme
        self.colors = {
            'bg_dark': '#1e1e1e',
            'bg_medium': '#2d2d2d',
            'bg_light': '#3d3d3d',
            'accent': '#00d4ff',
            'success': '#00e676',
            'error': '#ff1744',
            'warning': '#ffd600',
            'text': '#ffffff',
            'text_secondary': '#b0b0b0'
        }
        
        self.setup_ui()
        
    def setup_ui(self):
        # Create main container with sidebar and content
        main_container = tk.Frame(self.root, bg=self.colors['bg_dark'])
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Left Sidebar
        self.create_sidebar(main_container)
        
        # Right Content Area
        content_frame = tk.Frame(main_container, bg=self.colors['bg_dark'])
        content_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Header
        self.create_header(content_frame)
        
        # Visualization Area
        self.create_visualization_area(content_frame)
        
        # Results Dashboard
        self.create_results_dashboard(content_frame)
        
        # Status Bar
        self.create_status_bar(content_frame)
        
    def create_sidebar(self, parent):
        sidebar = tk.Frame(parent, bg=self.colors['bg_medium'], width=280, relief=tk.RAISED, bd=1)
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=0, pady=0)
        sidebar.pack_propagate(False)
        
        # Title
        title_frame = tk.Frame(sidebar, bg=self.colors['accent'], height=80)
        title_frame.pack(fill=tk.X, pady=0)
        
        tk.Label(title_frame, text="⚡ ECG ANALYZER", 
                font=('Segoe UI', 18, 'bold'), 
                bg=self.colors['accent'], 
                fg=self.colors['bg_dark']).pack(pady=25)
        
        # Control Panel Section
        controls_frame = tk.Frame(sidebar, bg=self.colors['bg_medium'])
        controls_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=20)
        
        tk.Label(controls_frame, text="DATA LOADING", 
                font=('Segoe UI', 10, 'bold'),
                bg=self.colors['bg_medium'], 
                fg=self.colors['text_secondary']).pack(anchor='w', pady=(0, 10))
        
        # Load buttons with icons
        self.create_sidebar_button(controls_frame, "📊 Load ECG Signal", self.load_csv, self.colors['accent'])
        self.create_sidebar_button(controls_frame, "🧠 Load ML Model", self.load_model, '#9c27b0')
        self.create_sidebar_button(controls_frame, "⚖️ Load Scaler", self.load_scaler, '#ff9800')
        
        # Status indicators
        tk.Label(controls_frame, text="STATUS", 
                font=('Segoe UI', 10, 'bold'),
                bg=self.colors['bg_medium'], 
                fg=self.colors['text_secondary']).pack(anchor='w', pady=(25, 10))
        
        status_container = tk.Frame(controls_frame, bg=self.colors['bg_light'], relief=tk.FLAT, bd=1)
        status_container.pack(fill=tk.X, pady=5)
        
        self.signal_indicator = self.create_status_indicator(status_container, "Signal")
        self.model_indicator = self.create_status_indicator(status_container, "Model")
        self.scaler_indicator = self.create_status_indicator(status_container, "Scaler")
        
        # Action Buttons
        tk.Label(controls_frame, text="ACTIONS", 
                font=('Segoe UI', 10, 'bold'),
                bg=self.colors['bg_medium'], 
                fg=self.colors['text_secondary']).pack(anchor='w', pady=(25, 10))
        
        # Run Analysis Button
        run_btn = tk.Button(controls_frame, text="▶ RUN ANALYSIS", 
                           command=self.run_pipeline,
                           font=('Segoe UI', 12, 'bold'),
                           bg=self.colors['success'],
                           fg='white',
                           activebackground='#00c853',
                           relief=tk.FLAT,
                           cursor='hand2',
                           pady=15)
        run_btn.pack(fill=tk.X, pady=(0, 10))
        
        # Additional buttons
        self.create_sidebar_button(controls_frame, "💾 Export Results", self.export_results, '#607d8b')
        self.create_sidebar_button(controls_frame, "🔄 Clear All", self.clear_all, '#f44336')
        
        # Info Section
        info_frame = tk.Frame(sidebar, bg=self.colors['bg_light'], relief=tk.FLAT)
        info_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10, padx=10)
        
        tk.Label(info_frame, text="ℹ️ System Info", 
                font=('Segoe UI', 9, 'bold'),
                bg=self.colors['bg_light'], 
                fg=self.colors['text']).pack(pady=5)
        
        self.info_label = tk.Label(info_frame, 
                                   text="Ready to analyze\nECG signals", 
                                   font=('Segoe UI', 8),
                                   bg=self.colors['bg_light'], 
                                   fg=self.colors['text_secondary'],
                                   justify=tk.LEFT)
        self.info_label.pack(pady=5, padx=10)
        
    def create_sidebar_button(self, parent, text, command, color):
        btn = tk.Button(parent, text=text, 
                       command=command,
                       font=('Segoe UI', 10),
                       bg=color,
                       fg='white',
                       activebackground=color,
                       relief=tk.FLAT,
                       cursor='hand2',
                       anchor='w',
                       padx=15,
                       pady=12)
        btn.pack(fill=tk.X, pady=5)
        
        # Hover effect
        btn.bind('<Enter>', lambda e: btn.configure(bg=self.adjust_color(color, 1.2)))
        btn.bind('<Leave>', lambda e: btn.configure(bg=color))
        
        return btn
    
    def create_status_indicator(self, parent, label):
        frame = tk.Frame(parent, bg=self.colors['bg_light'])
        frame.pack(fill=tk.X, padx=10, pady=5)
        
        indicator = tk.Label(frame, text="●", font=('Segoe UI', 16), 
                           bg=self.colors['bg_light'], fg='#757575')
        indicator.pack(side=tk.LEFT, padx=(0, 10))
        
        tk.Label(frame, text=label, font=('Segoe UI', 9),
                bg=self.colors['bg_light'], fg=self.colors['text']).pack(side=tk.LEFT)
        
        return indicator
        
    def create_header(self, parent):
        header = tk.Frame(parent, bg=self.colors['bg_medium'], height=60)
        header.pack(fill=tk.X, padx=20, pady=(20, 10))
        
        tk.Label(header, text="ECG Signal Analysis Dashboard", 
                font=('Segoe UI', 20, 'bold'),
                bg=self.colors['bg_medium'], 
                fg=self.colors['text']).pack(side=tk.LEFT, padx=10, pady=10)
        
        # Timestamp
        self.time_label = tk.Label(header, 
                                   text=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                   font=('Segoe UI', 10),
                                   bg=self.colors['bg_medium'], 
                                   fg=self.colors['text_secondary'])
        self.time_label.pack(side=tk.RIGHT, padx=10)
        self.update_time()
        
    def create_visualization_area(self, parent):
        viz_container = tk.Frame(parent, bg=self.colors['bg_medium'], relief=tk.FLAT, bd=1)
        viz_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Tabs for different views
        self.notebook = ttk.Notebook(viz_container)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Configure notebook style
        style = ttk.Style()
        style.theme_use('default')
        style.configure('TNotebook', background=self.colors['bg_medium'], borderwidth=0)
        style.configure('TNotebook.Tab', background=self.colors['bg_light'], 
                       foreground=self.colors['text'], padding=[20, 10])
        style.map('TNotebook.Tab', background=[('selected', self.colors['accent'])],
                 foreground=[('selected', self.colors['bg_dark'])])
        
        # Create tabs
        self.plot_frame_raw = tk.Frame(self.notebook, bg='white')
        self.plot_frame_processed = tk.Frame(self.notebook, bg='white')
        self.plot_frame_segmented = tk.Frame(self.notebook, bg='white')
        self.plot_frame_dp = tk.Frame(self.notebook, bg='white')
        
        self.notebook.add(self.plot_frame_raw, text='📈 Raw Signal')
        self.notebook.add(self.plot_frame_processed, text='🔧 Processed Signal')
        self.notebook.add(self.plot_frame_segmented, text='🎯 Segmentation')
        self.notebook.add(self.plot_frame_dp, text='🔢 DP Table')
        
    def create_results_dashboard(self, parent):
        results_frame = tk.Frame(parent, bg=self.colors['bg_medium'], height=200, relief=tk.FLAT, bd=1)
        results_frame.pack(fill=tk.X, padx=20, pady=(0, 10))
        results_frame.pack_propagate(False)
        
        # Title
        tk.Label(results_frame, text="ANALYSIS RESULTS", 
                font=('Segoe UI', 12, 'bold'),
                bg=self.colors['bg_medium'], 
                fg=self.colors['text']).pack(pady=(15, 10))
        
        # Metrics container
        metrics_container = tk.Frame(results_frame, bg=self.colors['bg_medium'])
        metrics_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 15))
        
        # Create metric cards
        self.metric_qrs = self.create_metric_card(metrics_container, "QRS Width", "samples", 0)
        self.metric_st = self.create_metric_card(metrics_container, "ST Level", "mV", 1)
        self.metric_pr = self.create_metric_card(metrics_container, "PR Interval", "samples", 2)
        
        # Diagnosis card
        diag_frame = tk.Frame(metrics_container, bg=self.colors['bg_light'], relief=tk.FLAT, bd=2)
        diag_frame.grid(row=0, column=3, padx=15, sticky='nsew')
        
        tk.Label(diag_frame, text="DIAGNOSIS", 
                font=('Segoe UI', 10, 'bold'),
                bg=self.colors['bg_light'], 
                fg=self.colors['text_secondary']).pack(pady=(15, 5))
        
        self.lbl_diagnosis = tk.Label(diag_frame, text="PENDING", 
                                     font=('Segoe UI', 24, 'bold'),
                                     bg=self.colors['bg_light'], 
                                     fg=self.colors['warning'])
        self.lbl_diagnosis.pack(pady=20)
        
        # Configure grid weights
        for i in range(4):
            metrics_container.grid_columnconfigure(i, weight=1)
            
    def create_metric_card(self, parent, title, unit, col):
        frame = tk.Frame(parent, bg=self.colors['bg_light'], relief=tk.FLAT, bd=2)
        frame.grid(row=0, column=col, padx=10, sticky='nsew')
        
        tk.Label(frame, text=title.upper(), 
                font=('Segoe UI', 9, 'bold'),
                bg=self.colors['bg_light'], 
                fg=self.colors['text_secondary']).pack(pady=(15, 5))
        
        value_label = tk.Label(frame, text="--", 
                              font=('Segoe UI', 28, 'bold'),
                              bg=self.colors['bg_light'], 
                              fg=self.colors['accent'])
        value_label.pack()
        
        tk.Label(frame, text=unit, 
                font=('Segoe UI', 8),
                bg=self.colors['bg_light'], 
                fg=self.colors['text_secondary']).pack(pady=(0, 15))
        
        return value_label
        
    def create_status_bar(self, parent):
        status_frame = tk.Frame(parent, bg=self.colors['bg_light'], height=40, relief=tk.FLAT)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.status_var = tk.StringVar()
        self.status_var.set("⚡ System Ready - Load signal and model to begin analysis")
        
        self.status_label = tk.Label(status_frame, textvariable=self.status_var,
                                     font=('Segoe UI', 9),
                                     bg=self.colors['bg_light'], 
                                     fg=self.colors['text'],
                                     anchor='w')
        self.status_label.pack(side=tk.LEFT, padx=20, pady=10)
        
        # Progress bar (hidden by default)
        self.progress = ttk.Progressbar(status_frame, mode='indeterminate', length=200)
        
    def adjust_color(self, color, factor):
        # Lighten or darken color
        color = color.lstrip('#')
        r, g, b = tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
        r = min(255, int(r * factor))
        g = min(255, int(g * factor))
        b = min(255, int(b * factor))
        return f'#{r:02x}{g:02x}{b:02x}'
        
    def update_time(self):
        self.time_label.config(text=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self.root.after(1000, self.update_time)
        
    # ==========================================
    # CORE FUNCTIONALITY
    # ==========================================
    
    def load_csv(self):
        path = filedialog.askopenfilename(
            title="Select ECG Signal File",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if path:
            try:
                df = pd.read_csv(path)
                if df.shape[1] >= 300:
                    self.signal_data = df.iloc[0, :300].values.astype(float)
                else:
                    self.signal_data = df.iloc[:, 0].values.astype(float)[:300]
                
                self.signal_indicator.config(fg=self.colors['success'])
                self.status_var.set(f"✓ Signal loaded: {os.path.basename(path)} ({len(self.signal_data)} samples)")
                self.info_label.config(text=f"Signal: {os.path.basename(path)}\nSamples: {len(self.signal_data)}")
                
                self.plot_signal(self.signal_data, self.plot_frame_raw, "Raw ECG Signal")
                
            except Exception as e:
                self.signal_indicator.config(fg=self.colors['error'])
                messagebox.showerror("Load Error", f"Could not load CSV file:\n{str(e)}")
                
    def load_model(self):
        path = filedialog.askopenfilename(
            title="Select ML Model",
            filetypes=[("Pickle Files", "*.pkl"), ("All Files", "*.*")]
        )
        if path:
            try:
                self.model = pickle.load(open(path, 'rb'))
                self.model_indicator.config(fg=self.colors['success'])
                self.status_var.set(f"✓ Model loaded: {os.path.basename(path)}")
            except Exception as e:
                self.model_indicator.config(fg=self.colors['error'])
                messagebox.showerror("Load Error", f"Invalid model file:\n{str(e)}")
                
    def load_scaler(self):
        path = filedialog.askopenfilename(
            title="Select Feature Scaler",
            filetypes=[("Pickle Files", "*.pkl"), ("All Files", "*.*")]
        )
        if path:
            try:
                self.scaler = pickle.load(open(path, 'rb'))
                self.scaler_indicator.config(fg=self.colors['success'])
                self.status_var.set(f"✓ Scaler loaded: {os.path.basename(path)}")
            except Exception as e:
                self.scaler_indicator.config(fg=self.colors['error'])
                messagebox.showerror("Load Error", f"Invalid scaler file:\n{str(e)}")
                
    def plot_signal(self, signal, frame, title, segments=None):
        for widget in frame.winfo_children():
            widget.destroy()
            
        fig = Figure(figsize=(10, 4), dpi=100, facecolor='#f5f5f5')
        ax = fig.add_subplot(111)
        
        ax.plot(signal, color='#2196F3', linewidth=1.5, alpha=0.8, label='ECG Signal')
        
        if segments:
            t_indices = np.arange(len(signal))
            seg_array = np.array(segments)
            
            colors = {'P': '#4CAF50', 'QRS': '#F44336', 'T': '#2196F3', 
                     'PR': '#FF9800', 'ST': '#9C27B0', 'Baseline': '#757575'}
            
            for seg_name, color in colors.items():
                mask = seg_array == seg_name
                if mask.any():
                    ax.scatter(t_indices[mask], signal[mask], 
                             c=color, s=15, label=seg_name, alpha=0.7, zorder=5)
            
            ax.legend(loc='upper right', framealpha=0.9)
        
        ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
        ax.set_xlabel("Sample Index", fontsize=10)
        ax.set_ylabel("Normalized Amplitude", fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.3, color='gray')
        ax.set_facecolor('#fafafa')
        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def plot_dp_table(self, dp_table, path_table, frame):
        """Display the DP table as an actual table with cell values"""
        for widget in frame.winfo_children():
            widget.destroy()
        
        # Create a container with scrollbars
        container = tk.Frame(frame, bg='white')
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_frame = tk.Frame(container, bg='white')
        title_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(title_frame, text="Viterbi Dynamic Programming Table", 
                font=('Segoe UI', 14, 'bold'),
                bg='white', fg='#1e1e1e').pack(side=tk.LEFT)
        
        info_text = f"({len(dp_table)} time steps × {NUM_STATES} states)"
        tk.Label(title_frame, text=info_text, 
                font=('Segoe UI', 10),
                bg='white', fg='#666').pack(side=tk.LEFT, padx=10)
        
        # Create frame for table with scrollbars
        table_container = tk.Frame(container, bg='white')
        table_container.pack(fill=tk.BOTH, expand=True)
        
        # Create canvas and scrollbars
        canvas = tk.Canvas(table_container, bg='white', highlightthickness=0)
        v_scrollbar = tk.Scrollbar(table_container, orient='vertical', command=canvas.yview)
        h_scrollbar = tk.Scrollbar(table_container, orient='horizontal', command=canvas.xview)
        
        scrollable_frame = tk.Frame(canvas, bg='white')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Pack scrollbars and canvas
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create the table using grid
        table_frame = tk.Frame(scrollable_frame, bg='white')
        table_frame.pack(padx=5, pady=5)
        
        # Cell styling
        header_font = tkfont.Font(family='Segoe UI', size=9, weight='bold')
        cell_font = tkfont.Font(family='Consolas', size=8)
        
        # Header row - Time steps
        tk.Label(table_frame, text="State \\ Time", font=header_font,
                bg='#37474f', fg='white', relief=tk.RAISED, bd=1,
                padx=8, pady=5).grid(row=0, column=0, sticky='nsew')
        
        # Limit display to reasonable number of columns
        N = len(dp_table)
        display_step = max(1, N // 100)  # Show at most 100 columns
        display_indices = list(range(0, N, display_step))
        
        # Column headers (time steps)
        for col_idx, t in enumerate(display_indices, start=1):
            tk.Label(table_frame, text=f"t={t}", font=header_font,
                    bg='#546e7a', fg='white', relief=tk.RAISED, bd=1,
                    width=10, pady=5).grid(row=0, column=col_idx, sticky='nsew')
        
        # State rows
        for state_idx, state_name in enumerate(STATE_NAMES):
            # Row header (state name)
            tk.Label(table_frame, text=state_name, font=header_font,
                    bg='#546e7a', fg='white', relief=tk.RAISED, bd=1,
                    padx=8, pady=5).grid(row=state_idx+1, column=0, sticky='nsew')
            
            # Cell values
            for col_idx, t in enumerate(display_indices, start=1):
                value = dp_table[t][state_idx]
                
                # Format the value
                if value == INF:
                    display_value = "∞"
                    bg_color = '#ffebee'
                    fg_color = '#c62828'
                else:
                    display_value = f"{value:.2f}"
                    # Color based on value (lighter = lower cost = better)
                    normalized = min(1.0, value / 10.0)  # Normalize for coloring
                    
                    # Gradient from green (low cost) to red (high cost)
                    if normalized < 0.5:
                        bg_color = '#e8f5e9'  # Light green
                    elif normalized < 1.0:
                        bg_color = '#fff9c4'  # Light yellow
                    else:
                        bg_color = '#ffebee'  # Light red
                    fg_color = '#1e1e1e'
                
                # Highlight if this is part of the optimal path
                if hasattr(self, 'current_segments') and t < len(self.current_segments):
                    if self.current_segments[t] == state_name:
                        bg_color = '#4fc3f7'  # Bright blue for optimal path
                        fg_color = 'white'
                        display_value = f"★{display_value}"
                
                cell_label = tk.Label(table_frame, text=display_value, font=cell_font,
                                     bg=bg_color, fg=fg_color, relief=tk.SOLID, bd=1,
                                     width=10, pady=3)
                cell_label.grid(row=state_idx+1, column=col_idx, sticky='nsew')
        
        # Add statistics at the bottom
        stats_frame = tk.Frame(container, bg='#f5f5f5', relief=tk.SOLID, bd=1)
        stats_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Calculate statistics
        valid_costs = [dp_table[t][s] for t in range(N) for s in range(NUM_STATES) if dp_table[t][s] != INF]
        
        stats_text = f"Total Cells: {N * NUM_STATES}  |  "
        if valid_costs:
            stats_text += f"Min Cost: {min(valid_costs):.2f}  |  "
            stats_text += f"Max Cost: {max(valid_costs):.2f}  |  "
            stats_text += f"Avg Cost: {sum(valid_costs)/len(valid_costs):.2f}"
        
        if display_step > 1:
            stats_text += f"  |  ⚠️ Showing every {display_step}th column"
        
        tk.Label(stats_frame, text=stats_text, font=('Segoe UI', 9),
                bg='#f5f5f5', fg='#424242', pady=8).pack()
        
        # Legend
        legend_frame = tk.Frame(container, bg='white')
        legend_frame.pack(fill=tk.X, pady=(5, 0))
        
        tk.Label(legend_frame, text="Legend:", font=('Segoe UI', 9, 'bold'),
                bg='white', fg='#1e1e1e').pack(side=tk.LEFT, padx=5)
        
        tk.Label(legend_frame, text="★", font=('Segoe UI', 10, 'bold'),
                bg='#4fc3f7', fg='white', relief=tk.SOLID, bd=1,
                padx=5).pack(side=tk.LEFT, padx=2)
        tk.Label(legend_frame, text="Optimal Path", font=('Segoe UI', 9),
                bg='white', fg='#1e1e1e').pack(side=tk.LEFT, padx=5)
        
        tk.Label(legend_frame, text="  ", bg='#e8f5e9', relief=tk.SOLID, bd=1,
                padx=8).pack(side=tk.LEFT, padx=2)
        tk.Label(legend_frame, text="Low Cost", font=('Segoe UI', 9),
                bg='white', fg='#1e1e1e').pack(side=tk.LEFT, padx=5)
        
        tk.Label(legend_frame, text="  ", bg='#ffebee', relief=tk.SOLID, bd=1,
                padx=8).pack(side=tk.LEFT, padx=2)
        tk.Label(legend_frame, text="High Cost", font=('Segoe UI', 9),
                bg='white', fg='#1e1e1e').pack(side=tk.LEFT, padx=5)
        
        # Mouse wheel scrolling
        def on_mouse_wheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        canvas.bind_all("<MouseWheel>", on_mouse_wheel)
        
    def run_pipeline(self):
        if self.signal_data is None:
            messagebox.showwarning("Missing Data", "Please load an ECG signal first.")
            return
        if self.model is None or self.scaler is None:
            messagebox.showwarning("Missing Components", "Please load both ML model and scaler.")
            return
        
        try:
            # Show progress
            self.progress.pack(side=tk.RIGHT, padx=20)
            self.progress.start(10)
            self.status_var.set("⚙️ Processing signal...")
            self.root.update()
            
            # Preprocessing
            self.processed_signal = preprocess_signal(self.signal_data)
            self.plot_signal(self.processed_signal, self.plot_frame_processed, 
                           "Preprocessed Signal (Denoised & Normalized)")
            
            self.status_var.set("🔍 Running Viterbi segmentation...")
            self.root.update()
            
            # Segmentation
            self.current_segments, self.dp_table, self.path_table = run_viterbi(self.processed_signal)
            self.plot_signal(self.processed_signal, self.plot_frame_segmented, 
                           "Segmented ECG Signal", self.current_segments)
            
            # Visualize DP table
            self.plot_dp_table(self.dp_table, self.path_table, self.plot_frame_dp)
            
            # Switch to segmented view
            self.notebook.select(2)
            
            self.status_var.set("📊 Extracting features...")
            self.root.update()
            
            # Feature extraction
            features_df = extract_features(self.current_segments, self.processed_signal)
            
            # Update metrics
            self.metric_qrs.config(text=f"{int(features_df['QRS_Width'][0])}")
            self.metric_st.config(text=f"{features_df['ST_Level'][0]:.4f}")
            self.metric_pr.config(text=f"{int(features_df['PR_Interval'][0])}")
            
            self.status_var.set("🧠 Running ML prediction...")
            self.root.update()
            
            # ML Prediction
            X_scaled = self.scaler.transform(features_df)
            prediction = self.model.predict(X_scaled)[0]
            
            # Update diagnosis
            if prediction == 0:
                self.lbl_diagnosis.config(text="NORMAL", fg=self.colors['success'])
                diagnosis_text = "Normal Rhythm"
            else:
                self.lbl_diagnosis.config(text="ABNORMAL", fg=self.colors['error'])
                diagnosis_text = "Arrhythmia Detected"
            
            # Save to history
            self.analysis_history.append({
                'timestamp': datetime.now(),
                'diagnosis': diagnosis_text,
                'qrs': features_df['QRS_Width'][0],
                'st': features_df['ST_Level'][0],
                'pr': features_df['PR_Interval'][0]
            })
            
            self.progress.stop()
            self.progress.pack_forget()
            self.status_var.set(f"✓ Analysis complete - Diagnosis: {diagnosis_text}")
            
        except Exception as e:
            self.progress.stop()
            self.progress.pack_forget()
            self.status_var.set("❌ Analysis failed")
            messagebox.showerror("Analysis Error", f"An error occurred:\n{str(e)}")
            
    def export_results(self):
        if not self.analysis_history:
            messagebox.showinfo("No Data", "No analysis results to export.")
            return
            
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        
        if path:
            try:
                df = pd.DataFrame(self.analysis_history)
                df.to_csv(path, index=False)
                self.status_var.set(f"✓ Results exported to {os.path.basename(path)}")
                messagebox.showinfo("Success", "Results exported successfully!")
            except Exception as e:
                messagebox.showerror("Export Error", f"Could not export results:\n{str(e)}")
                
    def clear_all(self):
        if messagebox.askyesno("Clear All", "This will clear all loaded data and results. Continue?"):
            self.signal_data = None
            self.current_segments = None
            self.processed_signal = None
            self.dp_table = None
            self.path_table = None
            
            self.signal_indicator.config(fg='#757575')
            
            self.metric_qrs.config(text="--")
            self.metric_st.config(text="--")
            self.metric_pr.config(text="--")
            self.lbl_diagnosis.config(text="PENDING", fg=self.colors['warning'])
            
            for frame in [self.plot_frame_raw, self.plot_frame_processed, 
                         self.plot_frame_segmented, self.plot_frame_dp]:
                for widget in frame.winfo_children():
                    widget.destroy()
            
            self.status_var.set("⚡ System cleared - Ready for new analysis")
            self.info_label.config(text="Ready to analyze\nECG signals")


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    root = tk.Tk()
    app = ModernECG_GUI(root)
    root.mainloop()