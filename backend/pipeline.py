"""
Main Pipeline Orchestrator
Coordinates all processing steps from signal to classification
"""

import numpy as np
import joblib
from normalization import normalize_signal
from dp_segmentation import DPSegmenter
from backtracking import reconstruct_segments, visualize_backtracking_steps
from feature_extraction import extract_all_features


class ECGPipeline:
    """Complete ECG processing pipeline"""

    def __init__(self, model_path="D:/2nd year SBME/First semester/Algorithm/ECG_DP_Project/model/random_forest_model.pkl"):
        """
        Initialize pipeline with ML model

        Args:
            model_path (str): Path to trained Random Forest model
        """
        self.segmenter = DPSegmenter()
        try:
            self.model = joblib.load(model_path)
            self.model_loaded = True
            print(f"ML model loaded from {model_path}")
        except:
            self.model = None
            self.model_loaded = False
            print("Warning: ML model not found, classification will use rule-based method")

    def process(self, signal, step_by_step=False):
        """
        Process ECG signal through complete pipeline

        Args:
            signal (list or np.array): Raw ECG signal
            step_by_step (bool): Whether to return intermediate steps

        Returns:
            dict: Complete analysis results
        """
        print(f"Processing ECG signal of length {len(signal)}...")

        # Step 1: Normalization
        print("Step 1: Normalizing signal...")
        norm_result = normalize_signal(np.array(signal))
        clean_signal = np.array(norm_result['normalized'])

        # Step 2: DP Segmentation
        print("Step 2: Running DP segmentation...")
        dp_result = self.segmenter.segment(clean_signal)

        # Step 3: Backtracking
        print("Step 3: Backtracking optimal path...")
        segments = dp_result['segments']

        # Step 4: Feature Extraction
        print("Step 4: Extracting features...")
        features = extract_all_features(segments, clean_signal)

        # Step 5: Classification
        print("Step 5: Classifying signal...")
        classification = self.classify(features)

        # Step 6: Generate visualization data if requested
        visualization = {}
        if step_by_step:
            print("Generating step-by-step visualization...")
            visualization = self.generate_visualization_data(
                signal, clean_signal, segments, dp_result
            )

        # Compile results
        results = {
            'success': True,
            'signal_info': {
                'original_length': len(signal),
                'processed_length': len(clean_signal),
                'normalization': norm_result
            },
            'segmentation': {
                'segments': segments,
                'total_segments': len(segments),
                'dp_total_cost': dp_result['total_cost'],
                'by_type': {
                    'P': len([s for s in segments if s['type'] == 'P']),
                    'QRS': len([s for s in segments if s['type'] == 'QRS']),
                    'T': len([s for s in segments if s['type'] == 'T'])
                }
            },
            'features': features,
            'classification': classification,
            'visualization': visualization if step_by_step else None
        }

        print("Pipeline processing complete!")
        return results

    def classify(self, features):
        """
        Classify ECG as Normal or Abnormal

        Args:
            features (dict): Extracted features

        Returns:
            dict: Classification results
        """
        if self.model_loaded and 'ml_feature_vector' in features:
            # Use ML model
            feature_vector = np.array(features['ml_feature_vector']).reshape(1, -1)
            prediction = self.model.predict(feature_vector)[0]
            probabilities = self.model.predict_proba(feature_vector)[0]

            return {
                'method': 'ml_model',
                'prediction': 'Normal' if prediction == 0 else 'Abnormal',
                'confidence': float(max(probabilities)),
                'probabilities': {
                    'Normal': float(probabilities[0]),
                    'Abnormal': float(probabilities[1])
                }
            }
        else:
            # Rule-based classification (fallback)
            return self.rule_based_classification(features)

    def rule_based_classification(self, features):
        """
        Rule-based ECG classification (when ML model is not available)

        Args:
            features (dict): Extracted features

        Returns:
            dict: Classification results
        """
        detailed = features.get('detailed_features', {})

        # Basic rules for ECG classification
        abnormalities = []

        # Check QRS duration (normal: 60-100ms)
        qrs_durations = [v for k, v in detailed.items() if 'QRS_duration_ms' in k]
        if qrs_durations:
            avg_qrs = np.mean(qrs_durations)
            if avg_qrs > 100:  # > 100ms is abnormal
                abnormalities.append(f"Wide QRS complex ({avg_qrs:.1f}ms)")

        # Check RR interval variability
        if 'rr_std' in detailed and detailed['rr_std'] > 50:
            abnormalities.append("Irregular rhythm")

        # Check for missing P waves
        if detailed.get('p_count', 0) < detailed.get('qrs_count', 0) / 2:
            abnormalities.append("Missing P waves")

        if abnormalities:
            return {
                'method': 'rule_based',
                'prediction': 'Abnormal',
                'confidence': 0.85,
                'abnormalities': abnormalities,
                'notes': '; '.join(abnormalities)
            }
        else:
            return {
                'method': 'rule_based',
                'prediction': 'Normal',
                'confidence': 0.90,
                'abnormalities': [],
                'notes': 'All parameters within normal range'
            }

    def generate_visualization_data(self, raw_signal, clean_signal, segments, dp_result):
        """
        Generate data for step-by-step visualization

        Args:
            raw_signal (np.array): Original signal
            clean_signal (np.array): Cleaned signal
            segments (list): Identified segments
            dp_result (dict): DP results

        Returns:
            dict: Visualization data
        """
        # Backtracking steps
        backtrack_steps = visualize_backtracking_steps(segments, len(clean_signal))

        # DP table sample (first 50 positions)
        dp_table_sample = []
        for i, cost in enumerate(dp_result.get('dp_table', [])[:51]):
            dp_table_sample.append({
                'position': i,
                'cost': round(cost, 2) if cost != float('inf') else '∞'
            })

        # Signal samples for visualization
        signal_samples = {
            'raw': raw_signal[:100].tolist(),  # First 100 points
            'clean': clean_signal[:100].tolist(),
            'full_clean': clean_signal.tolist()
        }

        return {
            'backtrack_steps': backtrack_steps,
            'dp_table_sample': dp_table_sample,
            'signal_samples': signal_samples,
            'total_steps': len(backtrack_steps)
        }