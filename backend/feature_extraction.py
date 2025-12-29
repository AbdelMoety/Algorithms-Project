# """
# Feature Extraction from Segmented ECG
# Extracts clinically relevant features for ML classification
# """
#
# import numpy as np
#
#
# def extract_wave_features(segment_type, signal_portion, fs=360):
#     """
#     Extract specific features for a wave type
#
#     Args:
#         segment_type (str): Type of wave (P, QRS, T)
#         signal_portion (np.array): ECG signal for this segment
#         fs (int): Sampling frequency
#
#     Returns:
#         dict: Extracted features
#     """
#     if len(signal_portion) == 0:
#         return {}
#
#     features = {
#         'duration_ms': (len(signal_portion) / fs) * 1000,
#         'amplitude': float(np.max(signal_portion) - np.min(signal_portion)),
#         'mean_amplitude': float(np.mean(signal_portion)),
#         'std_amplitude': float(np.std(signal_portion)),
#         'area': float(np.sum(np.abs(signal_portion))),
#         'slope_mean': float(np.mean(np.diff(signal_portion))),
#         'slope_max': float(np.max(np.abs(np.diff(signal_portion))))
#     }
#
#     # Type-specific features
#     if segment_type == "QRS":
#         features['qrs_complexity'] = float(np.sum(np.diff(signal_portion) ** 2))
#         features['r_peak'] = float(np.max(signal_portion))
#     elif segment_type == "P":
#         features['p_symmetry'] = float(
#             np.mean(signal_portion[:len(signal_portion) // 2]) /
#             (np.mean(signal_portion[len(signal_portion) // 2:]) + 0.001)
#         )
#
#     return features
#
#
# def calculate_intervals(segments):
#     """
#     Calculate intervals between waves (RR, PR, QT intervals)
#
#     Args:
#         segments (list): List of identified segments
#
#     Returns:
#         dict: Interval features
#     """
#     # Find QRS complexes
#     qrs_segments = [s for s in segments if s['type'] == 'QRS']
#
#     intervals = {}
#
#     if len(qrs_segments) > 1:
#         # RR intervals (between consecutive QRS)
#         rr_intervals = []
#         for i in range(1, len(qrs_segments)):
#             rr = qrs_segments[i]['start'] - qrs_segments[i - 1]['end']
#             rr_intervals.append(rr)
#
#         intervals['rr_mean'] = float(np.mean(rr_intervals))
#         intervals['rr_std'] = float(np.std(rr_intervals))
#         intervals['rr_cv'] = intervals['rr_std'] / intervals['rr_mean'] if intervals['rr_mean'] > 0 else 0
#
#     # Find P waves before QRS and T waves after QRS
#     for qrs in qrs_segments:
#         # Find preceding P wave
#         p_waves = [s for s in segments if s['type'] == 'P' and s['end'] < qrs['start']]
#         if p_waves:
#             closest_p = max(p_waves, key=lambda x: x['end'])
#             intervals[f'pr_interval_{qrs["start"]}'] = qrs['start'] - closest_p['end']
#
#         # Find following T wave
#         t_waves = [s for s in segments if s['type'] == 'T' and s['start'] > qrs['end']]
#         if t_waves:
#             closest_t = min(t_waves, key=lambda x: x['start'])
#             intervals[f'qt_interval_{qrs["end"]}'] = closest_t['end'] - qrs['start']
#
#     return intervals
#
#
# def extract_all_features(segments, signal, fs=360):
#     """
#     Extract comprehensive features from segmented ECG
#
#     Args:
#         segments (list): List of identified segments
#         signal (np.array): Complete ECG signal
#         fs (int): Sampling frequency
#
#     Returns:
#         dict: All extracted features
#     """
#     if not segments:
#         return {}
#
#     features = {}
#
#     # 1. Count of each wave type
#     wave_counts = {}
#     for seg_type in ["P", "QRS", "T"]:
#         count = len([s for s in segments if s['type'] == seg_type])
#         wave_counts[f'{seg_type.lower()}_count'] = count
#         features[f'{seg_type.lower()}_count'] = count
#
#     # 2. Wave-specific features
#     for seg in segments:
#         seg_signal = signal[seg['start']:seg['end'] + 1]
#         wave_features = extract_wave_features(seg['type'], seg_signal, fs)
#
#         for key, value in wave_features.items():
#             features[f'{seg["type"]}_{key}_{seg["start"]}'] = value
#
#     # 3. Interval features
#     interval_features = calculate_intervals(segments)
#     features.update(interval_features)
#
#     # 4. Global signal features
#     features['total_duration'] = len(signal) / fs
#     features['signal_mean'] = float(np.mean(signal))
#     features['signal_std'] = float(np.std(signal))
#     features['signal_skewness'] = float(np.mean((signal - np.mean(signal)) ** 3) / (np.std(signal) ** 3 + 0.001))
#
#     # 5. Create feature vector for ML (selecting most important features)
#     ml_features = [
#         wave_counts.get('qrs_count', 0),
#         features.get('rr_mean', 0) if 'rr_mean' in features else 0,
#         features.get('rr_std', 0) if 'rr_std' in features else 0,
#         np.mean([features.get(f'QRS_duration_ms_{s["start"]}', 0)
#                  for s in segments if s['type'] == 'QRS']) if any(s['type'] == 'QRS' for s in segments) else 0,
#         np.mean([features.get(f'QRS_amplitude_{s["start"]}', 0)
#                  for s in segments if s['type'] == 'QRS']) if any(s['type'] == 'QRS' for s in segments) else 0,
#     ]
#
#     return {
#         'detailed_features': features,
#         'ml_feature_vector': ml_features,
#         'feature_names': ['QRS_count', 'RR_mean', 'RR_std', 'QRS_duration_mean', 'QRS_amplitude_mean']
#     }
"""
Feature Extraction from Segmented ECG
Extracts clinically relevant features for ML classification
"""

import numpy as np

def extract_wave_features(segment_type, signal_portion, fs=360):
    """
    Extract specific features for a wave type

    Args:
        segment_type (str): Type of wave (P, QRS, T)
        signal_portion (np.array): ECG signal for this segment
        fs (int): Sampling frequency

    Returns:
        dict: Extracted features
    """
    if len(signal_portion) == 0:
        return {}

    features = {
        'duration_ms': (len(signal_portion) / fs) * 1000,
        'amplitude': float(np.max(signal_portion) - np.min(signal_portion)),
        'mean_amplitude': float(np.mean(signal_portion)),
        'std_amplitude': float(np.std(signal_portion)),
        'area': float(np.sum(np.abs(signal_portion))),
        'slope_mean': float(np.mean(np.diff(signal_portion))),
        'slope_max': float(np.max(np.abs(np.diff(signal_portion))))
    }

    # Type-specific features
    if segment_type == "QRS":
        features['qrs_complexity'] = float(np.sum(np.diff(signal_portion) ** 2))
        features['r_peak'] = float(np.max(signal_portion))
    elif segment_type == "P":
        features['p_symmetry'] = float(
            np.mean(signal_portion[:len(signal_portion) // 2]) /
            (np.mean(signal_portion[len(signal_portion) // 2:]) + 0.001)
        )

    return features


def calculate_intervals(segments):
    """
    Calculate intervals between waves (RR, PR, QT intervals)

    Args:
        segments (list): List of identified segments

    Returns:
        dict: Interval features
    """
    # Find QRS complexes
    qrs_segments = [s for s in segments if s['type'] == 'QRS']

    intervals = {}

    if len(qrs_segments) > 1:
        # RR intervals (between consecutive QRS)
        rr_intervals = []
        for i in range(1, len(qrs_segments)):
            rr = qrs_segments[i]['start'] - qrs_segments[i - 1]['end']
            rr_intervals.append(rr)

        intervals['rr_mean'] = float(np.mean(rr_intervals))
        intervals['rr_std'] = float(np.std(rr_intervals))
        intervals['rr_cv'] = intervals['rr_std'] / intervals['rr_mean'] if intervals['rr_mean'] > 0 else 0

    # Find P waves before QRS and T waves after QRS
    for qrs in qrs_segments:
        # Find preceding P wave
        p_waves = [s for s in segments if s['type'] == 'P' and s['end'] < qrs['start']]
        if p_waves:
            closest_p = max(p_waves, key=lambda x: x['end'])
            intervals[f'pr_interval_{qrs["start"]}'] = qrs['start'] - closest_p['end']

        # Find following T wave
        t_waves = [s for s in segments if s['type'] == 'T' and s['start'] > qrs['end']]
        if t_waves:
            closest_t = min(t_waves, key=lambda x: x['start'])
            intervals[f'qt_interval_{qrs["end"]}'] = closest_t['end'] - qrs['start']

    return intervals


def extract_all_features(segments, signal, fs=360):
    """
    Extract comprehensive features from segmented ECG

    Args:
        segments (list): List of identified segments
        signal (np.array): Complete ECG signal
        fs (int): Sampling frequency

    Returns:
        dict: All extracted features
    """
    if not segments:
        return {}

    features = {}

    # 1. Count of each wave type
    wave_counts = {}
    for seg_type in ["P", "QRS", "T"]:
        count = len([s for s in segments if s['type'] == seg_type])
        wave_counts[f'{seg_type.lower()}_count'] = count
        features[f'{seg_type.lower()}_count'] = count

    # 2. Wave-specific features
    for seg in segments:
        seg_signal = signal[seg['start']:seg['end'] + 1]
        wave_features = extract_wave_features(seg['type'], seg_signal, fs)

        for key, value in wave_features.items():
            features[f'{seg["type"]}_{key}_{seg["start"]}'] = value

    # 3. Interval features
    interval_features = calculate_intervals(segments)
    features.update(interval_features)

    # 4. Global signal features
    features['total_duration'] = len(signal) / fs
    features['signal_mean'] = float(np.mean(signal))
    features['signal_std'] = float(np.std(signal))
    features['signal_skewness'] = float(np.mean((signal - np.mean(signal)) ** 3) / (np.std(signal) ** 3 + 0.001))

    # 5. Create feature vector for ML (Modified to match the 3 features model)
    ml_features = [
        wave_counts.get('qrs_count', 0),
        features.get('rr_mean', 0) if 'rr_mean' in features else 0,
        features.get('rr_std', 0) if 'rr_std' in features else 0
    ]

    return {
        'detailed_features': features,
        'ml_feature_vector': ml_features,
        'feature_names': ['QRS_count', 'RR_mean', 'RR_std']
    }