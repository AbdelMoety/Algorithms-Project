/**
 * API Communication Module
 * Handles all communication with the Flask backend
 */

const API_BASE_URL = 'http://localhost:5000'; // Flask backend URL

class ECG_API {
    constructor() {
        this.baseUrl = API_BASE_URL;
    }

    /**
     * Test API connection
     */
    async testConnection() {
        try {
            const response = await fetch(`${this.baseUrl}/health`);
            const data = await response.json();
            return {
                connected: true,
                data: data
            };
        } catch (error) {
            console.error('API Connection Error:', error);
            return {
                connected: false,
                error: error.message
            };
        }
    }

    /**
     * Get predefined test cases
     */
    async getTestCases() {
        try {
            const response = await fetch(`${this.baseUrl}/test-cases`);
            if (!response.ok) throw new Error('Failed to fetch test cases');
            return await response.json();
        } catch (error) {
            console.error('Error fetching test cases:', error);
            throw error;
        }
    }

    /**
     * Analyze ECG signal
     * @param {Array} signal - ECG signal array
     * @param {Object} options - Analysis options
     */
    async analyzeSignal(signal, options = {}) {
        const payload = {
            signal: signal,
            step_by_step: options.stepByStep || true,
            metadata: options.metadata || {}
        };

        try {
            const response = await fetch(`${this.baseUrl}/analyze`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Analysis failed');
            }

            return await response.json();
        } catch (error) {
            console.error('Error analyzing signal:', error);
            throw error;
        }
    }

    /**
     * Analyze multiple signals in batch
     * @param {Array} signals - Array of signal objects
     */
    async batchAnalyze(signals) {
        const payload = { signals };

        try {
            const response = await fetch(`${this.baseUrl}/batch-analyze`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                throw new Error('Batch analysis failed');
            }

            return await response.json();
        } catch (error) {
            console.error('Error in batch analysis:', error);
            throw error;
        }
    }

    /**
     * Generate sample ECG signal
     * @param {string} type - Signal type (normal, arrhythmia, bradycardia)
     * @param {number} length - Signal length in points
     */
    generateSampleSignal(type = 'normal', length = 400) {
        const fs = 360; // Sampling frequency
        const duration = length / fs;

        let signal = [];

        switch (type) {
            case 'normal':
                // Normal sinus rhythm
                signal = this._generateNormalRhythm(length, fs);
                break;

            case 'arrhythmia':
                // Arrhythmia - irregular rhythm
                signal = this._generateArrhythmia(length, fs);
                break;

            case 'bradycardia':
                // Bradycardia - slow rhythm
                signal = this._generateBradycardia(length, fs);
                break;

            default:
                signal = this._generateNormalRhythm(length, fs);
        }

        return {
            signal: signal,
            metadata: {
                type: type,
                sample_rate: fs,
                length: length,
                duration: duration.toFixed(2) + 's'
            }
        };
    }

    /**
     * Generate normal sinus rhythm
     */
    _generateNormalRhythm(length, fs) {
        const signal = [];
        const heartRate = 75; // BPM
        const rrInterval = 60 / heartRate * fs; // Samples per beat

        for (let i = 0; i < length; i++) {
            const t = i / fs;

            // Generate P wave
            const pWave = this._generatePWave(i, rrInterval);

            // Generate QRS complex
            const qrsComplex = this._generateQRS(i, rrInterval);

            // Generate T wave
            const tWave = this._generateTWave(i, rrInterval);

            // Add baseline
            const baseline = Math.random() * 0.01; // Small noise

            signal.push(pWave + qrsComplex + tWave + baseline);
        }

        return signal;
    }

    /**
     * Generate arrhythmia signal
     */
    _generateArrhythmia(length, fs) {
        const signal = [];
        const heartRate = 85; // BPM with variability

        for (let i = 0; i < length; i++) {
            const t = i / fs;

            // Irregular RR intervals
            const rrVariation = 0.8 + Math.random() * 0.4; // 0.8 to 1.2
            const rrInterval = (60 / heartRate * fs) * rrVariation;

            // Sometimes skip P wave
            const hasPWave = Math.random() > 0.3;
            const pWave = hasPWave ? this._generatePWave(i, rrInterval) : 0;

            // QRS with possible abnormalities
            const qrsComplex = this._generateQRS(i, rrInterval);

            // T wave
            const tWave = this._generateTWave(i, rrInterval);

            // Extra noise for arrhythmia
            const noise = (Math.random() - 0.5) * 0.05;

            signal.push(pWave + qrsComplex + tWave + noise);
        }

        return signal;
    }

    /**
     * Generate bradycardia signal
     */
    _generateBradycardia(length, fs) {
        const signal = [];
        const heartRate = 45; // Slow BPM
        const rrInterval = 60 / heartRate * fs;

        for (let i = 0; i < length; i++) {
            const t = i / fs;

            // Generate waves with longer intervals
            const pWave = this._generatePWave(i, rrInterval);
            const qrsComplex = this._generateQRS(i, rrInterval);
            const tWave = this._generateTWave(i, rrInterval);

            signal.push(pWave + qrsComplex + tWave);
        }

        return signal;
    }

    /**
     * Generate P wave component
     */
    _generatePWave(i, rrInterval) {
        const beatPosition = i % rrInterval;
        const pStart = rrInterval * 0.1;
        const pEnd = rrInterval * 0.2;

        if (beatPosition >= pStart && beatPosition < pEnd) {
            const positionInP = (beatPosition - pStart) / (pEnd - pStart);
            // Gaussian shape for P wave
            return 0.1 * Math.exp(-Math.pow((positionInP - 0.5) * 5, 2));
        }
        return 0;
    }

    /**
     * Generate QRS complex component
     */
    _generateQRS(i, rrInterval) {
        const beatPosition = i % rrInterval;
        const qrsStart = rrInterval * 0.3;
        const qrsEnd = rrInterval * 0.4;

        if (beatPosition >= qrsStart && beatPosition < qrsEnd) {
            const positionInQRS = (beatPosition - qrsStart) / (qrsEnd - qrsStart);

            // QRS shape: negative Q, positive R, negative S
            if (positionInQRS < 0.3) {
                // Q wave (negative)
                return -0.15 * (positionInQRS / 0.3);
            } else if (positionInQRS < 0.6) {
                // R wave (positive peak)
                return 0.5 * Math.sin(Math.PI * (positionInQRS - 0.3) / 0.3);
            } else {
                // S wave (negative)
                return -0.2 * ((positionInQRS - 0.6) / 0.4);
            }
        }
        return 0;
    }

    /**
     * Generate T wave component
     */
    _generateTWave(i, rrInterval) {
        const beatPosition = i % rrInterval;
        const tStart = rrInterval * 0.5;
        const tEnd = rrInterval * 0.8;

        if (beatPosition >= tStart && beatPosition < tEnd) {
            const positionInT = (beatPosition - tStart) / (tEnd - tStart);
            // Asymmetric shape for T wave
            return 0.2 * Math.sin(Math.PI * positionInT) * Math.exp(-positionInT * 2);
        }
        return 0;
    }

    /**
     * Parse uploaded file
     * @param {File} file - Uploaded file
     */
    async parseFile(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();

            reader.onload = (event) => {
                try {
                    const content = event.target.result;
                    let signal;

                    // Determine file type and parse accordingly
                    if (file.name.endsWith('.csv')) {
                        signal = this._parseCSV(content);
                    } else if (file.name.endsWith('.txt')) {
                        signal = this._parseTXT(content);
                    } else if (file.name.endsWith('.json')) {
                        signal = this._parseJSON(content);
                    } else {
                        reject(new Error('Unsupported file format'));
                        return;
                    }

                    resolve(signal);
                } catch (error) {
                    reject(error);
                }
            };

            reader.onerror = () => {
                reject(new Error('Failed to read file'));
            };

            reader.readAsText(file);
        });
    }

    /**
     * Parse CSV/TXT/JSON content (Universal Parser)
     * Handles horizontal (Excel rows) and vertical (Text lines) data
     */
    _parseCSV(content) {
        // 1. تنظيف الملف: تحويل كل الفواصل والمسافات والأسطر الجديدة لمسافة واحدة
        // Regex looks for: comma, semicolon, space, tab, or newline
        const allText = content.replace(/[\r\n,;\t]+/g, ' ');

        // 2. تقسيم النص لمصفوفة بناءً على المسافات
        const parts = allText.trim().split(/\s+/);

        const signal = [];

        // 3. تحويل النصوص لأرقام
        parts.forEach(part => {
            // شيل أي حروف غريبة وسيب الأرقام والنقطة والسالب
            const cleanPart = part.replace(/[^0-9.-]/g, '');

            if (cleanPart) {
                const value = parseFloat(cleanPart);
                if (!isNaN(value)) {
                    signal.push(value);
                }
            }
        });

        return signal;
    }
    /**
     * Parse TXT file content
     */
    _parseTXT(content) {
        const lines = content.split('\n');
        const signal = [];

        lines.forEach(line => {
            const trimmed = line.trim();
            if (trimmed) {
                const value = parseFloat(trimmed);
                if (!isNaN(value)) {
                    signal.push(value);
                }
            }
        });

        return signal;
    }

    /**
     * Parse JSON file content
     */
    _parseJSON(content) {
        const data = JSON.parse(content);

        // Handle different JSON formats
        if (Array.isArray(data)) {
            return data;
        } else if (data.signal) {
            return data.signal;
        } else if (data.ecg) {
            return data.ecg;
        } else {
            throw new Error('Invalid JSON format');
        }
    }

    /**
     * Validate signal
     * @param {Array} signal - ECG signal
     */
    validateSignal(signal) {
        if (!Array.isArray(signal)) {
            return {
                valid: false,
                error: 'Signal must be an array'
            };
        }

        if (signal.length < 100) {
            return {
                valid: false,
                error: `Signal too short (${signal.length} points). Minimum 100 points required.`
            };
        }

        if (signal.length > 10000) {
            return {
                valid: false,
                error: `Signal too long (${signal.length} points). Maximum 10000 points allowed.`
            };
        }

        // Check for non-numeric values
        const invalidValues = signal.filter(val =>
            typeof val !== 'number' || isNaN(val) || !isFinite(val)
        );

        if (invalidValues.length > 0) {
            return {
                valid: false,
                error: `Signal contains ${invalidValues.length} invalid values`
            };
        }

        return {
            valid: true,
            length: signal.length,
            min: Math.min(...signal),
            max: Math.max(...signal),
            mean: signal.reduce((a, b) => a + b, 0) / signal.length
        };
    }
}

// Create global API instance
window.ECG_API = new ECG_API();