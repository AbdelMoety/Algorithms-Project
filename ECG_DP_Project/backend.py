
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import json
import traceback
from datetime import datetime

app = Flask(__name__)
CORS(app)

print("🚀 ECG DP Backend Server Started!")

# Generate sample ECG signal
def generate_ecg_signal(length=400, signal_type="normal"):
    fs = 360
    signal = []

    for i in range(length):
        t = i / fs
        value = np.sin(2 * np.pi * 0.8 * t) * 0.5

        # Add QRS complex periodically
        if 0.3 < (t % 0.8) < 0.4:
            value += 1.0 * np.sin(2 * np.pi * 15 * (t % 0.1))

        # Add noise
        value += np.random.normal(0, 0.05)

        # Modify based on type
        if signal_type == "arrhythmia" and np.random.random() < 0.2:
            value += np.random.uniform(-0.5, 0.5)
        elif signal_type == "bradycardia":
            value *= 0.6

        signal.append(float(value))

    return signal

# DP Algorithm Implementation
class DPSegmenter:
    def __init__(self):
        self.segment_types = ["P", "QRS", "T"]

    def segment_with_dp(self, signal):
        """Complete DP algorithm with step-by-step tracking"""
        n = len(signal)

        # DP table initialization
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        backtrack = [None] * (n + 1)

        # For visualization: track all steps
        steps = []

        # Base case step
        steps.append({
            "step": 0,
            "action": "Initialize DP table",
            "dp_table": [{"i": 0, "cost": 0, "prev": "-", "type": "-"}],
            "description": "dp[0] = 0 (base case)"
        })

        # DP iterations (simplified for demo)
        for i in range(1, min(n + 1, 100)):  # Limit for demo
            for seg_type in self.segment_types:
                for l in [20, 40, 30]:  # Sample lengths
                    j = i - l
                    if j >= 0:
                        # Calculate cost
                        segment = signal[j:i]
                        cost = dp[j] + self.calculate_cost(seg_type, segment)

                        if cost < dp[i]:
                            dp[i] = cost
                            backtrack[i] = (j, seg_type, l)

                            # Record step
                            steps.append({
                                "step": len(steps),
                                "action": f"Update position {i}",
                                "dp_table": self.get_dp_snapshot(dp[:i+1], backtrack[:i+1]),
                                "description": f"dp[{i}] = {cost:.2f} from {seg_type} at {j}"
                            })

        # Backtracking steps
        backtrack_steps = []
        i = min(n, 99)
        current_step = 0

        while i > 0 and backtrack[i]:
            prev_pos, seg_type, length = backtrack[i]

            backtrack_steps.append({
                "step": current_step,
                "from": i,
                "to": prev_pos,
                "type": seg_type,
                "action": f"Backtrack: Found {seg_type} at {prev_pos}-{i}",
                "segments_found": backtrack_steps[-1]["segments_found"] + [seg_type] if backtrack_steps else [seg_type]
            })

            i = prev_pos
            current_step += 1

        return {
            "dp_table": steps,
            "backtracking": backtrack_steps,
            "segments": self.reconstruct_segments(backtrack, n),
            "total_cost": dp[min(n, 99)]
        }

    def calculate_cost(self, seg_type, segment):
        """Simple cost function"""
        if len(segment) == 0:
            return float('inf')

        amplitude = max(segment) - min(segment)
        if seg_type == "QRS":
            return 1.0 / (amplitude + 0.001)
        elif seg_type == "P":
            return amplitude * 0.5
        else:  # T
            return amplitude * 0.3

    def get_dp_snapshot(self, dp, backtrack):
        """Get current DP table state"""
        snapshot = []
        for i in range(len(dp)):
            if backtrack[i]:
                prev, typ, length = backtrack[i]
                snapshot.append({
                    "i": i,
                    "cost": round(dp[i], 2),
                    "prev": prev,
                    "type": typ,
                    "length": length
                })
            else:
                snapshot.append({
                    "i": i,
                    "cost": round(dp[i], 2),
                    "prev": "-",
                    "type": "-",
                    "length": "-"
                })
        return snapshot[:20]  # First 20 rows

    def reconstruct_segments(self, backtrack, n):
        """Reconstruct segments from backtracking table"""
        segments = []
        i = min(n, 99)

        while i > 0 and backtrack[i]:
            prev, typ, length = backtrack[i]
            segments.append({
                "type": typ,
                "start": prev,
                "end": i - 1,
                "length": length,
                "cost": 0.5  # Sample cost
            })
            i = prev

        return list(reversed(segments))

# Initialize segmenter
segmenter = DPSegmenter()

@app.route('/')
def home():
    return jsonify({
        "service": "ECG DP Analysis API",
        "status": "running",
        "endpoints": ["/health", "/analyze", "/test-cases", "/dp-visualization"]
    })

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

@app.route('/test-cases')
def get_test_cases():
    return jsonify({
        "normal": {"name": "Normal Sinus Rhythm", "description": "Regular ECG pattern"},
        "arrhythmia": {"name": "Arrhythmia", "description": "Irregular heart rhythm"},
        "bradycardia": {"name": "Bradycardia", "description": "Slow heart rate"}
    })

@app.route('/analyze', methods=['POST'])
def analyze_ecg():
    try:
        data = request.get_json()
        signal = data.get('signal', [])

        if not signal:
            # Generate sample signal
            signal_type = data.get('type', 'normal')
            signal = generate_ecg_signal(400, signal_type)

        # Run DP algorithm
        results = segmenter.segment_with_dp(signal)

        return jsonify({
            "success": True,
            "dp_steps": results["dp_table"],
            "backtracking_steps": results["backtracking"],
            "segments": results["segments"],
            "total_cost": results["total_cost"],
            "signal_info": {
                "length": len(signal),
                "type": data.get('type', 'normal')
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/dp-visualization')
def dp_visualization():
    """Special endpoint for DP table visualization"""
    return jsonify({
        "dp_formulation": {
            "subproblems": "dp[i] = min cost to segment first i samples",
            "recurrence": "dp[i] = min(dp[j] + cost(segment_type, signal[j:i]))",
            "base_case": "dp[0] = 0",
            "complexity": "O(n × m × k) where n=length, m=types, k=max_length"
        },
        "sample_dp_table": [
            {"i": 0, "cost": 0, "prev": "-", "type": "-", "action": "Base case"},
            {"i": 20, "cost": 1.2, "prev": 0, "type": "P", "action": "Update from P"},
            {"i": 60, "cost": 3.5, "prev": 20, "type": "QRS", "action": "Update from QRS"},
            {"i": 90, "cost": 5.1, "prev": 60, "type": "T", "action": "Update from T"}
        ]
    })

if __name__ == '__main__':
    print("🌐 Backend running on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
