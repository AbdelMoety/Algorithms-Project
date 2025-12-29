import sys
import numpy as np
import joblib
from scipy.signal import butter, filtfilt

from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel,
    QFileDialog, QVBoxLayout, QHBoxLayout
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure


# =========================
# Signal Processing
# =========================
def bandpass(signal, fs=360, low=0.5, high=40):
    b, a = butter(2, [low/(fs/2), high/(fs/2)], btype='band')
    return filtfilt(b, a, signal)

def normalize(signal):
    signal = bandpass(signal)
    return (signal - np.mean(signal)) / np.std(signal)


# =========================
# Dynamic Programming
# =========================
SEGMENTS = ["P", "QRS", "T"]

def cost(seg, x):
    if seg == "QRS":
        return np.var(x) * 0.5
    return np.var(x)

def dp_segmentation(signal, min_len=10, max_len=80):
    n = len(signal)
    dp = [float("inf")] * (n+1)
    choice = [None] * (n+1)
    steps = []
    dp[0] = 0

    for i in range(1, n):
        for l in range(min_len, max_len):
            j = i - l
            if j < 0:
                continue
            for s in SEGMENTS:
                c = cost(s, signal[j:i])
                if dp[j] + c < dp[i]:
                    dp[i] = dp[j] + c
                    choice[i] = (j, s)
                    steps.append(f"dp[{i}] <- {s} from {j}")

    return dp, choice, steps


def backtrack(choice, n):
    segs = []
    i = n
    while choice[i]:
        j, s = choice[i]
        segs.append({"type": s, "start": j, "end": i})
        i = j
    return segs[::-1]


# =========================
# Feature Extraction
# =========================
def extract_features(segs):
    p = [s for s in segs if s["type"] == "P"]
    qrs = [s for s in segs if s["type"] == "QRS"]
    t = [s for s in segs if s["type"] == "T"]

    return np.array([
        np.mean([s["end"] - s["start"] for s in p]) if p else 0,
        np.mean([s["end"] - s["start"] for s in qrs]) if qrs else 0,
        np.mean([s["end"] - s["start"] for s in t]) if t else 0,
        len(qrs)
    ])


# =========================
# GUI Application
# =========================
class ECGApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🫀 ECG DP Analyzer")
        self.resize(1000, 650)

        self.model = joblib.load("random_forest_model.pkl")

        self.label = QLabel("ECG Analysis using Dynamic Programming")
        self.label.setStyleSheet("font-size:22px;font-weight:bold;")

        self.loadBtn = QPushButton("Load ECG Signal")
        self.runBtn = QPushButton("Run DP Analysis")
        self.stepBtn = QPushButton("Next DP Step")

        self.loadBtn.clicked.connect(self.load_signal)
        self.runBtn.clicked.connect(self.run_analysis)
        self.stepBtn.clicked.connect(self.next_step)

        self.fig = Figure(facecolor="#121212")
        self.canvas = FigureCanvasQTAgg(self.fig)

        btns = QHBoxLayout()
        btns.addWidget(self.loadBtn)
        btns.addWidget(self.runBtn)
        btns.addWidget(self.stepBtn)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addLayout(btns)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        self.signal = None
        self.steps = []
        self.step_i = 0

        self.setStyleSheet("""
        QWidget { background-color:#121212; color:white; }
        QPushButton {
            background-color:#7b2cbf;
            color:white; padding:10px;
            border-radius:10px;
        }
        QPushButton:hover {
            background-color:#9d4edd;
        }
        """)

    def load_signal(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open ECG CSV", "", "*.csv")
        if path:
            self.signal = np.loadtxt(path, delimiter=",")[:400]
            self.plot(self.signal, "Raw ECG Signal")

    def run_analysis(self):
        clean = normalize(self.signal)
        dp, choice, self.steps = dp_segmentation(clean)
        segs = backtrack(choice, len(clean))

        features = extract_features(segs).reshape(1, -1)
        pred = self.model.predict(features)[0]

        ax = self.fig.clear() or self.fig.add_subplot(111)
        ax.plot(clean, color="cyan")

        colors = {"P":"blue", "QRS":"red", "T":"green"}
        for s in segs:
            ax.axvspan(s["start"], s["end"],
                       color=colors[s["type"]], alpha=0.3)

        title = "NORMAL ECG ✅" if pred == 0 else "ABNORMAL ECG ❌"
        ax.set_title(title, color="white")
        ax.tick_params(colors="white")
        self.canvas.draw()

    def next_step(self):
        if self.step_i < len(self.steps):
            self.label.setText(self.steps[self.step_i])
            self.step_i += 1

    def plot(self, x, title):
        ax = self.fig.clear() or self.fig.add_subplot(111)
        ax.plot(x, color="cyan")
        ax.set_title(title, color="white")
        ax.tick_params(colors="white")
        self.canvas.draw()


# =========================
# Run
# =========================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = ECGApp()
    win.show()
    sys.exit(app.exec_())
