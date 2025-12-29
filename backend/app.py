"""
Flask API Server for ECG DP Analysis
Provides REST endpoints for frontend communication
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import json
from pipeline import ECGPipeline
import os

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Initialize pipeline
pipeline = ECGPipeline()


@app.route('/')
def home():
    """API Home endpoint"""
    return jsonify({
        'service': 'ECG DP Analysis API',
        'version': '1.0',
        'endpoints': {
            '/analyze': 'POST - Analyze ECG signal',
            '/health': 'GET - API health check',
            '/test-cases': 'GET - Get predefined test cases'
        }
    })


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': pipeline.model_loaded,
        'timestamp': np.datetime64('now').astype(str)
    })


@app.route('/test-cases', methods=['GET'])
def get_test_cases():
    """Get predefined test cases for demonstration"""

    test_cases = {
        'normal_sinus_rhythm': {
            'name': 'Normal Sinus Rhythm',
            'description': 'Regular rhythm with all waves present',
            'expected_segments': ['P', 'QRS', 'T', 'P', 'QRS', 'T'],
            'expected_classification': 'Normal'
        },
        'arrhythmia': {
            'name': 'Arrhythmia',
            'description': 'Irregular heart rhythm',
            'expected_segments': ['P', 'QRS', 'T', 'QRS', 'T'],  # Missing P wave
            'expected_classification': 'Abnormal'
        },
        'bradycardia': {
            'name': 'Bradycardia',
            'description': 'Slow heart rate',
            'expected_segments': ['P', 'QRS', 'T'],
            'expected_classification': 'Abnormal'
        }
    }

    return jsonify(test_cases)


@app.route('/analyze', methods=['POST'])
def analyze_ecg():
    """
    Main analysis endpoint

    Expected JSON payload:
    {
        "signal": [array of numbers],
        "step_by_step": true/false,
        "metadata": {
            "sample_rate": 360,
            "patient_id": "optional"
        }
    }
    """
    try:
        # Get request data
        data = request.get_json()

        if not data or 'signal' not in data:
            return jsonify({
                'success': False,
                'error': 'No signal data provided'
            }), 400

        # Extract signal
        signal = np.array(data['signal'], dtype=np.float64)

        # Check signal length
        if len(signal) < 100:
            return jsonify({
                'success': False,
                'error': f'Signal too short ({len(signal)} points). Minimum 100 points required.'
            }), 400

        if len(signal) > 1000:
            signal = signal[:1000]  # Limit to first 1000 points

        # Get analysis options
        step_by_step = data.get('step_by_step', False)
        metadata = data.get('metadata', {})

        # Process through pipeline
        results = pipeline.process(signal, step_by_step=step_by_step)

        # Add metadata to results
        results['metadata'] = metadata
        results['request_timestamp'] = np.datetime64('now').astype(str)

        return jsonify(results)

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': 'Check server logs for details'
        }), 500


@app.route('/batch-analyze', methods=['POST'])
def batch_analyze():
    """
    Batch analysis endpoint for multiple signals

    Expected JSON payload:
    {
        "signals": [
            {"id": "1", "signal": [array], "metadata": {}},
            {"id": "2", "signal": [array], "metadata": {}}
        ]
    }
    """
    try:
        data = request.get_json()

        if not data or 'signals' not in data:
            return jsonify({'error': 'No signals provided'}), 400

        results = []
        for item in data['signals']:
            signal_id = item.get('id', f'signal_{len(results) + 1}')
            signal = np.array(item['signal'], dtype=np.float64)

            # Process signal
            analysis = pipeline.process(signal, step_by_step=False)
            analysis['id'] = signal_id
            analysis['metadata'] = item.get('metadata', {})

            results.append(analysis)

        return jsonify({
            'success': True,
            'total_processed': len(results),
            'results': results
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

#
# @app.route('/api/dp-table/full', methods=['POST'])
# def get_full_dp_table():
#     """Get complete DP table with pagination"""
#     try:
#         data = request.get_json()
#         signal = np.array(data['signal'], dtype=np.float64)
#
#         # Run DP
#         results = segmenter.segment_with_visualization(signal)
#
#         # Return full table
#         return jsonify({
#             'success': True,
#             'dp_table': results['dp_table']['full_table'],
#             'total_rows': len(results['dp_table']['full_table']),
#             'page_size': 50,  # Rows per page
#             'total_pages': (len(results['dp_table']['full_table']) + 49) // 50
#         })
#
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500


@app.route('/api/dp-table/page/<int:page>', methods=['POST'])
def get_dp_table_page(page):
    """Get specific page of DP table"""
    try:
        data = request.get_json()
        signal = np.array(data['signal'], dtype=np.float64)

        results = segmenter.segment_with_visualization(signal)
        full_table = results['dp_table']['full_table']

        page_size = 50
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, len(full_table))

        return jsonify({
            'success': True,
            'page': page,
            'data': full_table[start_idx:end_idx],
            'start_position': start_idx,
            'end_position': end_idx,
            'has_previous': page > 1,
            'has_next': end_idx < len(full_table)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Get port from environment variable or use default
    port = int(os.environ.get('PORT', 5000))

    print(f"""
    ===========================================
    ECG DP Analysis API Server
    ===========================================
    Local: http://localhost:{port}
    Health: http://localhost:{port}/health

    Endpoints:
    - POST /analyze : Analyze ECG signal
    - GET /test-cases : Get test cases
    - POST /batch-analyze : Analyze multiple signals

    Press Ctrl+C to stop the server
    ===========================================
    """)

    app.run(host='0.0.0.0', port=port, debug=True)