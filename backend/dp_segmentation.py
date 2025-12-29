"""
Dynamic Programming for ECG Signal Segmentation
Identifies P, QRS, and T waves using optimal path finding
"""

import numpy as np

class DPSegmenter:
    """
    Dynamic Programming segmentation of ECG signals
    """

    def __init__(self, fs=360):
        """
        Initialize DP segmenter

        Args:
            fs (int): Sampling frequency (Hz)
        """
        self.fs = fs
        self.segment_types = ["P", "QRS", "T"]

        # Physiological constraints (in samples)
        self.min_duration = {
            "P": int(0.06 * fs),    # 60ms minimum
            "QRS": int(0.06 * fs),  # 60ms minimum
            "T": int(0.10 * fs)     # 100ms minimum
        }

        self.max_duration = {
            "P": int(0.12 * fs),    # 120ms maximum
            "QRS": int(0.12 * fs),  # 120ms maximum
            "T": int(0.25 * fs)     # 250ms maximum
        }

    def calculate_cost(self, segment_type, signal_segment):
        """
        Calculate cost for assigning a segment type to signal portion

        Different wave types have different characteristics:
        - QRS: High slope, high amplitude
        - P/T: Lower slope, rounded shape

        Args:
            segment_type (str): "P", "QRS", or "T"
            signal_segment (np.array): Portion of ECG signal

        Returns:
            float: Cost value (lower = better match)
        """
        if len(signal_segment) < 3:
            return float('inf')

        # Extract features
        amplitude = np.max(signal_segment) - np.min(signal_segment)
        slope = np.mean(np.diff(signal_segment) ** 2)
        variance = np.var(signal_segment)

        if segment_type == "QRS":
            # QRS should have high slope and amplitude
            cost = 1.0 / (slope + 0.01) + 1.0 / (amplitude + 0.01)
        elif segment_type == "P":
            # P wave is smaller and before QRS
            cost = variance + 0.5 * slope
        else:  # T wave
            # T wave is after QRS, moderate amplitude
            cost = variance + 0.3 * slope

        return cost

    def segment(self, signal):
        """
        Main DP segmentation algorithm

        Args:
            signal (np.array): Normalized ECG signal

        Returns:
            dict: DP tables and segmentation results
        """
        n = len(signal)

        # DP table: cost of optimal segmentation ending at position i
        dp = np.full(n + 1, float('inf'))
        dp[0] = 0  # Base case: cost at start is 0

        # Backtracking table: stores (prev_position, segment_type)
        backtrack = [None] * (n + 1)

        # Segment start positions
        segment_starts = [[] for _ in range(n + 1)]

        print(f"DP Segmentation: Processing {n} points...")

        # Fill DP table (bottom-up)
        for i in range(1, n + 1):
            for seg_type in self.segment_types:
                min_len = self.min_duration[seg_type]
                max_len = self.max_duration[seg_type]

                for l in range(min_len, min(max_len, i) + 1):
                    j = i - l  # Start position of segment

                    if j < 0:
                        continue

                    # Extract segment
                    segment = signal[j:i]

                    # Calculate cost
                    segment_cost = self.calculate_cost(seg_type, segment)
                    total_cost = dp[j] + segment_cost

                    # Update DP table if better
                    if total_cost < dp[i]:
                        dp[i] = total_cost
                        backtrack[i] = (j, seg_type, segment_cost)

        # Reconstruct path
        segments = []
        i = n

        while i > 0 and backtrack[i] is not None:
            j, seg_type, cost = backtrack[i]
            segments.append({
                'type': seg_type,
                'start': j,
                'end': i - 1,  # Convert to 0-based indexing
                'length': i - j, # <--- ده السطر اللي كان ناقص
                'duration': (i - j) / self.fs * 1000,  # Convert to ms
                'cost': float(cost)
            })
            i = j

        # Reverse to get chronological order
        segments.reverse()

        return {
            'dp_table': dp.tolist(),
            'segments': segments,
            'total_cost': float(dp[n]),
            'backtrack_table': [(i, str(bt)) for i, bt in enumerate(backtrack) if bt is not None]
        }