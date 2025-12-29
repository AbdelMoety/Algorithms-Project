/**
 * Main Application Controller
 * Orchestrates the entire ECG DP Analysis application
 */

class ECG_Application {
    constructor() {
        this.api = window.ECG_API;
        this.visualizer = window.ECG_Visualizer;
        this.currentSignal = null;
        this.analysisResults = null;
        this.currentTestCase = null;

        // Initialize when DOM is loaded
        document.addEventListener('DOMContentLoaded', () => this.init());
    }

    /**
     * Initialize application
     */
    async init() {
        console.log('ECG DP Analysis Application Starting...');

        // Test API connection
        await this.testAPIConnection();

        // Initialize visualizer
        this.visualizer.init();

        // Bind UI events
        this.bindEvents();

        // Load test cases
        await this.loadTestCases();

        // Initial UI state
        this.updateUIState('ready');

        console.log('Application initialized successfully');
    }

    /**
     * Test API connection
     */
    async testAPIConnection() {
        try {
            const connection = await this.api.testConnection();

            if (connection.connected) {
                console.log('API Connection Successful:', connection.data);
                this.showNotification('Connected to analysis server', 'success');
            } else {
                console.warn('API Connection Failed:', connection.error);
                this.showNotification(
                    'Backend server not running. Some features may be limited.',
                    'warning'
                );
            }
        } catch (error) {
            console.error('API Test Error:', error);
        }
    }

    /**
     * Load test cases from API
     */
    async loadTestCases() {
        try {
            const testCases = await this.api.getTestCases();
            console.log('Test cases loaded:', testCases);

            // Store test cases for later use
            this.testCases = testCases;

        } catch (error) {
            console.warn('Could not load test cases from API:', error);

            // Use default test cases
            this.testCases = {
                normal: {
                    name: 'Normal Sinus Rhythm',
                    description: 'Regular heart rhythm with properly formed P, QRS, and T waves.',
                    characteristics: [
                        'Heart rate: 60-100 BPM',
                        'Regular RR intervals',
                        'P wave before every QRS',
                        'Normal QRS duration (< 120ms)',
                        'Upright T waves'
                    ],
                    clinical_notes: 'This represents a healthy, normal ECG pattern.'
                },
                arrhythmia: {
                    name: 'Arrhythmia',
                    description: 'Irregular heart rhythm with potential missing or abnormal waves.',
                    characteristics: [
                        'Irregular RR intervals',
                        'Missing or abnormal P waves',
                        'Possible premature ventricular contractions',
                        'Variable QRS morphology',
                        'Inconsistent wave patterns'
                    ],
                    clinical_notes: 'May indicate cardiac electrical system abnormalities.'
                },
                bradycardia: {
                    name: 'Bradycardia',
                    description: 'Slow heart rate with prolonged intervals between beats.',
                    characteristics: [
                        'Heart rate < 60 BPM',
                        'Prolonged RR intervals',
                        'Normal P-QRS-T sequence',
                        'Possible sinus arrhythmia',
                        'Normal wave morphology'
                    ],
                    clinical_notes: 'Can be normal in athletes or indicate conduction system issues.'
                }
            };
        }
    }

    /**
     * Bind all UI event listeners
     */
    bindEvents() {
        // File upload
        this.bindFileUpload();

        // Generate sample button
        this.bindGenerateSample();

        // Analyze button
        this.bindAnalyzeButton();

        // Test case buttons
        this.bindTestCaseButtons();

        // Close case details
        this.bindCloseDetails();

        // Process step indicators
        this.bindProcessSteps();

        // Settings toggles
        this.bindSettingsToggles();

        // Footer links
        this.bindFooterLinks();
    }

    /**
     * Bind file upload events
     */
    bindFileUpload() {
        const uploadArea = document.getElementById('upload-area');
        const fileInput = document.getElementById('file-input');

        if (!uploadArea || !fileInput) return;

        // Click on upload area triggers file input
        uploadArea.addEventListener('click', () => fileInput.click());

        // File input change
        fileInput.addEventListener('change', (e) => this.handleFileUpload(e.target.files[0]));

        // Drag and drop
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('drag-over');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('drag-over');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('drag-over');

            if (e.dataTransfer.files.length > 0) {
                this.handleFileUpload(e.dataTransfer.files[0]);
            }
        });
    }

    /**
     * Handle file upload
     */
    async handleFileUpload(file) {
        if (!file) return;

        try {
            this.updateUIState('loading');
            this.showNotification(`Loading ${file.name}...`, 'info');

            // Parse file
            const signal = await this.api.parseFile(file);

            // Validate signal
            const validation = this.api.validateSignal(signal);
            if (!validation.valid) {
                throw new Error(validation.error);
            }

            // Store signal
            this.currentSignal = {
                data: signal,
                source: 'file',
                filename: file.name,
                metadata: validation
            };

            // Update UI
            this.updateSignalPreview(signal);
            this.updateUIState('signal_loaded');
            this.showNotification('Signal loaded successfully', 'success');

            // Clear test case selection
            this.clearTestCaseSelection();

        } catch (error) {
            console.error('File upload error:', error);
            this.showNotification(`Error: ${error.message}`, 'error');
            this.updateUIState('ready');
        }
    }

    /**
     * Bind generate sample button
     */
    bindGenerateSample() {
        const generateBtn = document.getElementById('generate-sample');
        if (!generateBtn) return;

        generateBtn.addEventListener('click', () => {
            const sampleType = document.querySelector('input[name="sample-type"]:checked').value;
            this.generateSampleSignal(sampleType);
        });
    }

    /**
     * Generate sample signal
     */
    generateSampleSignal(type) {
        try {
            this.updateUIState('loading');

            // Generate signal
            const result = this.api.generateSampleSignal(type, 400);
            const signal = result.signal;

            // Store signal
            this.currentSignal = {
                data: signal,
                source: 'generated',
                type: type,
                metadata: result.metadata
            };

            // Update UI
            this.updateSignalPreview(signal);
            this.updateUIState('signal_loaded');
            this.showNotification(`Generated ${type} rhythm signal`, 'success');

            // Clear test case selection
            this.clearTestCaseSelection();

        } catch (error) {
            console.error('Generate sample error:', error);
            this.showNotification(`Error: ${error.message}`, 'error');
            this.updateUIState('ready');
        }
    }

    /**
     * Bind analyze button
     */
    bindAnalyzeButton() {
        const analyzeBtn = document.getElementById('btn-analyze');
        if (!analyzeBtn) return;

        analyzeBtn.addEventListener('click', () => this.runAnalysis());
    }

    /**
     * Run ECG analysis
     */
    async runAnalysis() {
        if (!this.currentSignal) {
            this.showNotification('Please load or generate a signal first', 'warning');
            return;
        }

        try {
            // Update UI state
            this.updateUIState('analyzing');
            this.visualizer.showLoading('Running DP Analysis...');

            // Get analysis options
            const stepByStep = document.getElementById('step-by-step-toggle').checked;
            const enableAnimations = document.getElementById('animation-toggle').checked;

            // Run analysis
            const results = await this.api.analyzeSignal(this.currentSignal.data, {
                stepByStep: stepByStep,
                metadata: {
                    source: this.currentSignal.source,
                    type: this.currentSignal.type,
                    ...this.currentSignal.metadata
                }
            });

            if (!results.success) {
                throw new Error(results.error || 'Analysis failed');
            }

            // Store results
            this.analysisResults = results;

            // Update visualizations
            this.updateVisualizations(results);

            // Update UI state
            this.updateUIState('analysis_complete');
            this.visualizer.showSuccess('Analysis complete');
            this.showNotification('ECG analysis completed successfully', 'success');

            // Update process steps
            this.updateProcessSteps('complete');

        } catch (error) {
            console.error('Analysis error:', error);
            this.visualizer.showError(error.message);
            this.showNotification(`Analysis failed: ${error.message}`, 'error');
            this.updateUIState('signal_loaded');
            this.updateProcessSteps('error');
        }
    }

    /**
     * Update all visualizations with results
     */
    updateVisualizations(results) {
        // Clear previous visualizations
        this.visualizer.clear();

        // Plot segmented ECG
        this.visualizer.plotSegmentedECG(results);

        // Update DP table (if tab is active)
        if (document.querySelector('[data-tab="dp-process"]').classList.contains('active')) {
            this.visualizer.updateDPTable();
        }

        // Update features (if tab is active)
        if (document.querySelector('[data-tab="features"]').classList.contains('active')) {
            this.visualizer.updateFeaturesDisplay();
        }

        // Update classification (if tab is active)
        if (document.querySelector('[data-tab="classification"]').classList.contains('active')) {
            this.visualizer.updateClassificationDisplay();
        }

        // Update signal stats
        this.updateSignalStats(results);

    }

    /**
     * Update signal statistics
     */
    updateSignalStats(results) {
        const signal = results.signal_info?.normalization?.normalized;
        if (!signal) return;

        const stats = {
            points: signal.length,
            duration: (signal.length / 360).toFixed(2) + 's',
            mean: results.signal_info?.normalization?.mean?.toFixed(3) ||
                  (signal.reduce((a, b) => a + b, 0) / signal.length).toFixed(3),
            std: results.signal_info?.normalization?.std?.toFixed(3) ||
                 this.calculateStd(signal).toFixed(3)
        };

        document.getElementById('stat-points').textContent = stats.points;
        document.getElementById('stat-duration').textContent = stats.duration;
        document.getElementById('stat-mean').textContent = stats.mean;
    }

    /**
     * Calculate standard deviation
     */
    calculateStd(array) {
        const mean = array.reduce((a, b) => a + b, 0) / array.length;
        const squaredDiffs = array.map(value => Math.pow(value - mean, 2));
        const avgSquaredDiff = squaredDiffs.reduce((a, b) => a + b, 0) / array.length;
        return Math.sqrt(avgSquaredDiff);
    }

    /**
     * Bind test case buttons
     */
    bindTestCaseButtons() {
        const caseButtons = document.querySelectorAll('.case-btn');

        caseButtons.forEach(button => {
            button.addEventListener('click', () => {
                const caseType = button.dataset.case;
                this.loadTestCase(caseType, button);
            });
        });
    }

    /**
     * Load test case
     */
    loadTestCase(caseType, button) {
        // Update button states
        document.querySelectorAll('.case-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        button.classList.add('active');

        // Store current test case
        this.currentTestCase = caseType;

        // Generate corresponding signal
        this.generateSampleSignal(caseType);

        // Show case details
        this.showCaseDetails(caseType);
    }

    /**
     * Show case details
     */
    showCaseDetails(caseType) {
        const caseData = this.testCases?.[caseType];
        if (!caseData) return;

        const detailsContent = document.getElementById('details-content');
        const caseDetails = document.getElementById('case-details');

        if (!detailsContent || !caseDetails) return;

        detailsContent.innerHTML = `
            <div class="case-detail-header">
                <h4>${caseData.name}</h4>
                <div class="case-type-badge ${caseType}-badge">
                    ${caseType.toUpperCase()}
                </div>
            </div>
            <p class="case-description">${caseData.description}</p>
            
            <div class="case-section">
                <h5><i class="fas fa-list-check"></i> Characteristics</h5>
                <ul class="case-characteristics">
                    ${caseData.characteristics?.map(item => 
                        `<li><i class="fas fa-heart-circle-check"></i> ${item}</li>`
                    ).join('') || '<li>No characteristics defined</li>'}
                </ul>
            </div>
            
            <div class="case-section">
                <h5><i class="fas fa-stethoscope"></i> Clinical Notes</h5>
                <p class="clinical-notes">${caseData.clinical_notes}</p>
            </div>
            
            <div class="case-section">
                <h5><i class="fas fa-brain"></i> DP Algorithm Focus</h5>
                <p class="algorithm-focus">
                    The Dynamic Programming algorithm will specifically look for:
                    ${caseType === 'normal' ? 
                        'regular wave intervals and consistent morphology' :
                      caseType === 'arrhythmia' ? 
                        'irregular intervals and missing/abnormal waves' :
                        'prolonged intervals and slow heart rate patterns'}
                </p>
            </div>
        `;

        caseDetails.classList.add('active');
        caseDetails.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }

    /**
     * Clear test case selection
     */
    clearTestCaseSelection() {
        document.querySelectorAll('.case-btn').forEach(btn => {
            btn.classList.remove('active');
        });

        const caseDetails = document.getElementById('case-details');
        if (caseDetails) {
            caseDetails.classList.remove('active');
        }

        this.currentTestCase = null;
    }

    /**
     * Bind close details button
     */
    bindCloseDetails() {
        const closeBtn = document.getElementById('close-details');
        if (!closeBtn) return;

        closeBtn.addEventListener('click', () => {
            const caseDetails = document.getElementById('case-details');
            if (caseDetails) {
                caseDetails.classList.remove('active');
            }
        });
    }

    /**
     * Bind process step indicators
     */
    bindProcessSteps() {
        // Steps will be updated during analysis
    }

    /**
     * Update process steps visualization
     */
    updateProcessSteps(state) {
        const steps = ['normalization', 'dp', 'backtrack', 'classify'];

        steps.forEach((stepId, index) => {
            const stepEl = document.getElementById(`step-${stepId}`);
            if (!stepEl) return;

            // Reset all steps
            stepEl.classList.remove('active', 'completed');

            // Update based on state
            if (state === 'analyzing') {
                if (index < 1) {
                    stepEl.classList.add('active');
                } else if (index > 0) {
                    stepEl.classList.add('completed');
                }
            } else if (state === 'complete') {
                stepEl.classList.add('completed');
            } else if (state === 'error') {
                stepEl.classList.add('error');
            }
        });
    }

    /**
     * Bind settings toggles
     */
    bindSettingsToggles() {
        // Toggles are already bound via HTML, but we can add additional logic here
        const animationToggle = document.getElementById('animation-toggle');
        if (animationToggle) {
            animationToggle.addEventListener('change', (e) => {
                if (e.target.checked) {
                    this.showNotification('Animations enabled', 'info');
                } else {
                    this.showNotification('Animations disabled', 'info');
                }
            });
        }
    }

    /**
     * Bind footer links
     */
    bindFooterLinks() {
        document.getElementById('btn-about')?.addEventListener('click', (e) => {
            e.preventDefault();
            this.showAboutModal();
        });

        document.getElementById('btn-documentation')?.addEventListener('click', (e) => {
            e.preventDefault();
            this.showDocumentation();
        });

        document.getElementById('btn-github')?.addEventListener('click', (e) => {
            e.preventDefault();
            window.open('https://github.com/your-repo/ecg-dp-analysis', '_blank');
        });
    }

    /**
     * Show about modal
     */
    showAboutModal() {
        const modalContent = `
            <div class="about-modal">
                <h3><i class="fas fa-heartbeat"></i> ECG DP Analysis System</h3>
                <p>This application demonstrates the use of Dynamic Programming for ECG signal analysis in biomedical applications.</p>
                
                <div class="about-section">
                    <h4>Key Features:</h4>
                    <ul>
                        <li><strong>Dynamic Programming Segmentation:</strong> Optimal identification of P, QRS, and T waves</li>
                        <li><strong>Step-by-Step Visualization:</strong> Interactive demonstration of DP algorithm</li>
                        <li><strong>Feature Extraction:</strong> Clinical feature calculation from segmented waves</li>
                        <li><strong>ML Classification:</strong> Random Forest model for Normal/Abnormal classification</li>
                    </ul>
                </div>
                
                <div class="about-section">
                    <h4>Educational Purpose:</h4>
                    <p>This system is designed for educational demonstration of DP algorithms in biomedical signal processing. It is not intended for clinical diagnosis.</p>
                </div>
                
                <div class="about-footer">
                    <p><strong>Team Project - Biomedical Engineering</strong><br>
                    Dynamic Programming in Biomedical Applications</p>
                </div>
            </div>
        `;

        this.showModal('About the Project', modalContent);
    }

    /**
     * Show documentation
     */
    showDocumentation() {
        const docContent = `
            <div class="documentation-modal">
                <h3><i class="fas fa-book"></i> Documentation</h3>
                
                <div class="doc-section">
                    <h4>How to Use:</h4>
                    <ol>
                        <li><strong>Load ECG Signal:</strong> Upload a CSV/TXT file or generate a sample signal</li>
                        <li><strong>Select Test Case:</strong> Choose from Normal, Arrhythmia, or Bradycardia examples</li>
                        <li><strong>Run Analysis:</strong> Click "Run DP Analysis" to process the signal</li>
                        <li><strong>Explore Results:</strong> Use tabs to view segmentation, DP process, features, and classification</li>
                        <li><strong>Step-by-Step:</strong> Use controls to visualize the DP algorithm step by step</li>
                    </ol>
                </div>
                
                <div class="doc-section">
                    <h4>DP Algorithm Details:</h4>
                    <p><strong>Time Complexity:</strong> O(n × m × k) where n is signal length, m is segment types, k is possible segment lengths</p>
                    <p><strong>Space Complexity:</strong> O(n × m) for DP table storage</p>
                    <p><strong>Optimization:</strong> Minimizes cost function based on wave characteristics</p>
                </div>
                
                <div class="doc-section">
                    <h4>File Format:</h4>
                    <p>Accepted formats: CSV, TXT, JSON</p>
                    <p>Each line should contain a single numeric value (ECG amplitude)</p>
                    <p>Recommended: 400-1000 samples at 360Hz sampling rate</p>
                </div>
            </div>
        `;

        this.showModal('Documentation', docContent);
    }

    /**
     * Show modal dialog
     */
    showModal(title, content) {
        // Create modal if it doesn't exist
        let modal = document.getElementById('app-modal');
        if (!modal) {
            modal = document.createElement('div');
            modal.id = 'app-modal';
            modal.className = 'app-modal';
            modal.innerHTML = `
                <div class="modal-overlay"></div>
                <div class="modal-content">
                    <div class="modal-header">
                        <h3 class="modal-title"></h3>
                        <button class="modal-close"><i class="fas fa-times"></i></button>
                    </div>
                    <div class="modal-body"></div>
                </div>
            `;
            document.body.appendChild(modal);

            // Add close functionality
            modal.querySelector('.modal-overlay').addEventListener('click', () => this.hideModal());
            modal.querySelector('.modal-close').addEventListener('click', () => this.hideModal());
        }

        // Update modal content
        modal.querySelector('.modal-title').textContent = title;
        modal.querySelector('.modal-body').innerHTML = content;

        // Show modal
        modal.classList.add('active');
        document.body.style.overflow = 'hidden';
    }

    /**
     * Hide modal
     */
    hideModal() {
        const modal = document.getElementById('app-modal');
        if (modal) {
            modal.classList.remove('active');
            document.body.style.overflow = '';
        }
    }

    /**
     * Update signal preview plot
     */
    updateSignalPreview(signal) {
        this.visualizer.plotRawSignal(signal, {
            title: 'ECG Signal Preview'
        });
    }

    /**
     * Update UI state
     */
    updateUIState(state) {
        const analyzeBtn = document.getElementById('btn-analyze');

        switch (state) {
            case 'ready':
                if (analyzeBtn) analyzeBtn.disabled = true;
                break;

            case 'signal_loaded':
                if (analyzeBtn) analyzeBtn.disabled = false;
                break;

            case 'loading':
            case 'analyzing':
                if (analyzeBtn) analyzeBtn.disabled = true;
                break;

            case 'analysis_complete':
                if (analyzeBtn) analyzeBtn.disabled = false;
                break;
        }

        // Update status indicator
        const statusEl = document.getElementById('results-status');
        if (statusEl) {
            const statusMessages = {
                'ready': '<i class="fas fa-hourglass-half"></i><span>Ready for analysis</span>',
                'signal_loaded': '<i class="fas fa-check-circle"></i><span>Signal loaded</span>',
                'loading': '<div class="spinner" style="width: 16px; height: 16px;"></div><span>Loading...</span>',
                'analyzing': '<div class="spinner" style="width: 16px; height: 16px;"></div><span>Analyzing...</span>',
                'analysis_complete': '<i class="fas fa-check-circle"></i><span>Analysis complete</span>'
            };

            statusEl.innerHTML = statusMessages[state] || statusMessages['ready'];
        }
    }

    /**
     * Show notification
     */
    showNotification(message, type = 'info') {
        // Create notification container if it doesn't exist
        let container = document.getElementById('notifications');
        if (!container) {
            container = document.createElement('div');
            container.id = 'notifications';
            container.className = 'notifications-container';
            document.body.appendChild(container);
        }

        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <div class="notification-icon">
                ${type === 'success' ? '<i class="fas fa-check-circle"></i>' :
                  type === 'error' ? '<i class="fas fa-exclamation-circle"></i>' :
                  type === 'warning' ? '<i class="fas fa-exclamation-triangle"></i>' :
                  '<i class="fas fa-info-circle"></i>'}
            </div>
            <div class="notification-content">${message}</div>
            <button class="notification-close"><i class="fas fa-times"></i></button>
        `;

        // Add to container
        container.appendChild(notification);

        // Add close functionality
        notification.querySelector('.notification-close').addEventListener('click', () => {
            notification.classList.add('hiding');
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        });

        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.classList.add('hiding');
                setTimeout(() => {
                    if (notification.parentNode) {
                        notification.parentNode.removeChild(notification);
                    }
                }, 300);
            }
        }, 5000);

        // Animate in
        setTimeout(() => {
            notification.classList.add('show');
        }, 10);
    }

    /**
     * Export results
     */
    exportResults(format = 'json') {
        if (!this.analysisResults) {
            this.showNotification('No results to export', 'warning');
            return;
        }

        let content, filename, mimeType;

        if (format === 'json') {
            content = JSON.stringify(this.analysisResults, null, 2);
            filename = `ecg-analysis-${Date.now()}.json`;
            mimeType = 'application/json';
        } else if (format === 'csv') {
            // Create CSV from results
            const segments = this.analysisResults.segmentation?.segments || [];
            const csvRows = [
                ['Type', 'Start', 'End', 'Duration (ms)', 'Cost'],
                ...segments.map(s => [s.type, s.start, s.end, s.duration?.toFixed(1) || '', s.cost?.toFixed(2) || ''])
            ];

            content = csvRows.map(row => row.join(',')).join('\n');
            filename = `ecg-segments-${Date.now()}.csv`;
            mimeType = 'text/csv';
        }

        // Create download link
        const blob = new Blob([content], { type: mimeType });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        this.showNotification(`Results exported as ${format.toUpperCase()}`, 'success');
    }
}

// Initialize application
window.ECG_App = new ECG_Application();

// Add CSS for notifications and modal
const extraStyles = `
/* Notifications */
.notifications-container {
    position: fixed;
    top: 20px;
    right: 20px;
    z-index: 9999;
    display: flex;
    flex-direction: column;
    gap: 10px;
    max-width: 400px;
}

.notification {
    background: linear-gradient(135deg, rgba(26, 26, 58, 0.95), rgba(74, 26, 106, 0.95));
    border: 1px solid rgba(5, 217, 232, 0.3);
    border-radius: 10px;
    padding: 15px 20px;
    display: flex;
    align-items: center;
    gap: 15px;
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
    transform: translateX(100%);
    opacity: 0;
    transition: all 0.3s ease;
    backdrop-filter: blur(10px);
}

.notification.show {
    transform: translateX(0);
    opacity: 1;
}

.notification.hiding {
    transform: translateX(100%);
    opacity: 0;
}

.notification-success {
    border-left: 4px solid #00ff9d;
}

.notification-error {
    border-left: 4px solid #ff2a6d;
}

.notification-warning {
    border-left: 4px solid #ffcc00;
}

.notification-info {
    border-left: 4px solid #05d9e8;
}

.notification-icon {
    font-size: 1.5rem;
}

.notification-success .notification-icon {
    color: #00ff9d;
}

.notification-error .notification-icon {
    color: #ff2a6d;
}

.notification-warning .notification-icon {
    color: #ffcc00;
}

.notification-info .notification-icon {
    color: #05d9e8;
}

.notification-content {
    flex: 1;
    font-size: 0.95rem;
}

.notification-close {
    background: none;
    border: none;
    color: rgba(255, 255, 255, 0.5);
    cursor: pointer;
    padding: 5px;
    font-size: 1rem;
    transition: color 0.3s;
}

.notification-close:hover {
    color: #fff;
}

/* Modal */
.app-modal {
    display: none;
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    z-index: 10000;
}

.app-modal.active {
    display: block;
}

.modal-overlay {
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0, 0, 0, 0.7);
    backdrop-filter: blur(5px);
}

.modal-content {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    background: linear-gradient(135deg, rgba(26, 26, 58, 0.95), rgba(74, 26, 106, 0.95));
    border: 1px solid rgba(5, 217, 232, 0.3);
    border-radius: 15px;
    width: 90%;
    max-width: 600px;
    max-height: 80vh;
    overflow-y: auto;
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.5);
    animation: modalSlideUp 0.3s ease;
}

@keyframes modalSlideUp {
    from {
        opacity: 0;
        transform: translate(-50%, -40%);
    }
    to {
        opacity: 1;
        transform: translate(-50%, -50%);
    }
}

.modal-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 20px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}

.modal-title {
    color: #05d9e8;
    font-family: 'Orbitron', sans-serif;
    font-size: 1.5rem;
    margin: 0;
}

.modal-close {
    background: none;
    border: none;
    color: rgba(255, 255, 255, 0.5);
    font-size: 1.5rem;
    cursor: pointer;
    padding: 5px;
    transition: color 0.3s;
}

.modal-close:hover {
    color: #fff;
}

.modal-body {
    padding: 20px;
    color: #d1f7ff;
}

.modal-body h4 {
    color: #05d9e8;
    margin-top: 20px;
    margin-bottom: 10px;
}

.modal-body ul, .modal-body ol {
    padding-left: 20px;
    margin-bottom: 15px;
}

.modal-body li {
    margin-bottom: 8px;
}

/* Additional UI Styles */
.drag-over {
    border-color: #05d9e8 !important;
    background: rgba(5, 217, 232, 0.1) !important;
}

.current-segment {
    background: rgba(5, 217, 232, 0.1) !important;
    border-left: 3px solid #05d9e8;
}

.segment-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 15px;
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
}

.p-badge {
    background: rgba(5, 217, 232, 0.2);
    color: #05d9e8;
    border: 1px solid #05d9e8;
}

.qrs-badge {
    background: rgba(255, 42, 109, 0.2);
    color: #ff2a6d;
    border: 1px solid #ff2a6d;
}

.t-badge {
    background: rgba(0, 255, 157, 0.2);
    color: #00ff9d;
    border: 1px solid #00ff9d;
}

.case-type-badge {
    display: inline-block;
    padding: 5px 15px;
    border-radius: 20px;
    font-size: 0.9rem;
    font-weight: 600;
    text-transform: uppercase;
    background: rgba(5, 217, 232, 0.2);
    color: #05d9e8;
    border: 1px solid #05d9e8;
}

.normal-badge {
    background: rgba(0, 255, 157, 0.2);
    color: #00ff9d;
    border: 1px solid #00ff9d;
}

.arrhythmia-badge {
    background: rgba(255, 42, 109, 0.2);
    color: #ff2a6d;
    border: 1px solid #ff2a6d;
}

.bradycardia-badge {
    background: rgba(255, 204, 0, 0.2);
    color: #ffcc00;
    border: 1px solid #ffcc00;
}

.case-detail-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 15px;
}

.case-section {
    margin-top: 20px;
    padding-top: 20px;
    border-top: 1px solid rgba(255, 255, 255, 0.1);
}

.case-section h5 {
    color: #05d9e8;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 10px;
}

.case-characteristics {
    list-style: none;
    padding-left: 0;
}

.case-characteristics li {
    padding: 8px 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    display: flex;
    align-items: center;
    gap: 10px;
}

.case-characteristics li:last-child {
    border-bottom: none;
}

.case-characteristics i {
    color: #05d9e8;
}

.clinical-notes, .algorithm-focus {
    background: rgba(0, 0, 0, 0.2);
    padding: 15px;
    border-radius: 8px;
    border-left: 3px solid #05d9e8;
}
`;

// Add styles to document
const styleSheet = document.createElement('style');
styleSheet.textContent = extraStyles;
document.head.appendChild(styleSheet);
