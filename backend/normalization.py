"""
ECG Signal Normalization and Cleaning Module
Removes noise and normalizes ECG signals for DP segmentation
"""

import numpy as np
from scipy.signal import butter, filtfilt


def bandpass_filter(signal, fs=360, lowcut=0.5, highcut=40.0):
    """
    Apply bandpass filter to remove noise from ECG signal

    Args:
        signal (np.array): Raw ECG signal
        fs (int): Sampling frequency (Hz)
        lowcut (float): Lower cutoff frequency (Hz)
        highcut (float): Upper cutoff frequency (Hz)

    Returns:
        np.array: Filtered signal
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(4, [low, high], btype='band')
    y = filtfilt(b, a, signal)
    return y


def moving_average(signal, window_size=5):
    """
    Apply moving average smoothing

    Args:
        signal (np.array): Input signal
        window_size (int): Size of moving window

    Returns:
        np.array: Smoothed signal
    """
    window = np.ones(window_size) / window_size
    return np.convolve(signal, window, mode='same')


def normalize_signal(signal):
    """
    Complete normalization pipeline for ECG signal

    Steps:
    1. Remove baseline wander using high-pass filter
    2. Remove high-frequency noise using low-pass filter
    3. Standardize signal to zero mean and unit variance

    Args:
        signal (np.array): Raw ECG signal (400 points)

    Returns:
        dict: Contains cleaned signal and metadata
    """
    if len(signal) > 400:
        signal = signal[:400]  # Use first 400 points

    # Step 1: Remove high-frequency noise
    filtered = bandpass_filter(signal)

    # Step 2: Smooth the signal
    smoothed = moving_average(filtered)

    # Step 3: Standardize (z-score normalization)
    normalized = (smoothed - np.mean(smoothed)) / np.std(smoothed)

    return {
        'raw': signal.tolist(),
        'filtered': filtered.tolist(),
        'normalized': normalized.tolist(),
        'mean': float(np.mean(smoothed)),
        'std': float(np.std(smoothed))
    }