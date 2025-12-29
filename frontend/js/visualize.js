/**
 * Visualization Module (Enhanced for Project Requirements)
 * Handles interactive charts, DP animations, and Backtracking visualization
 */

class ECG_Visualizer {
    constructor() {
        this.currentPlot = null;
        this.currentStep = 0;
        this.totalSteps = 0;
        this.isAnimating = false;
        this.animationInterval = null;
        this.segments = [];
        this.signalData = null;

        // Color scheme
        this.colors = {
            pWave: '#05d9e8',
            qrsWave: '#ff2a6d',
            tWave: '#00ff9d',
            costCurve: '#ffcc00', // لون منحنى الـ DP
            text: '#d1f7ff',
            grid: 'rgba(255, 255, 255, 0.1)'
        };
    }

    init() {
        console.log('ECG Visualizer initialized');
        this.bindEvents();
    }

    bindEvents() {
        document.getElementById('prev-step')?.addEventListener('click', () => this.prevStep());
        document.getElementById('next-step')?.addEventListener('click', () => this.nextStep());
        document.getElementById('play-steps')?.addEventListener('click', () => this.togglePlay());

        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', (e) => this.switchTab(e.target.dataset.tab));
        });
    }

    // 1. رسم الإشارة الأصلية (Preview)
    plotRawSignal(signal, options = {}) {
        const container = options.container || 'signal-preview-plot';
        const trace = {
            x: Array.from({length: signal.length}, (_, i) => i),
            y: signal,
            type: 'scatter',
            mode: 'lines',
            line: { color: this.colors.pWave, width: 2 },
            name: 'ECG'
        };

        const layout = {
            title: options.title || 'ECG Signal',
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0.2)',
            font: { color: this.colors.text },
            xaxis: { gridcolor: this.colors.grid, title: 'Time (Samples)' },
            yaxis: { gridcolor: this.colors.grid, title: 'Amplitude' },
            margin: { t: 30, r: 20, b: 40, l: 50 },
            height: 200
        };

        Plotly.newPlot(container, [trace], layout, {displayModeBar: false});
    }

    // 2. تجهيز الرسمة الرئيسية (Segmentation + DP Cost)
    plotSegmentedECG(data) {
        this.signalData = data;
        this.segments = data.segmentation?.segments || [];
        this.totalSteps = data.visualization?.backtrack_steps?.length || 1;

        const signal = data.signal_info?.normalization?.normalized || [];
        const dpTable = data.visualization?.dp_table_sample?.map(d => d.cost === '∞' ? null : d.cost) || [];

        // Trace 1: The ECG Signal
        const signalTrace = {
            x: Array.from({length: signal.length}, (_, i) => i),
            y: signal,
            xaxis: 'x',
            yaxis: 'y',
            type: 'scatter',
            mode: 'lines',
            line: { color: 'rgba(255, 255, 255, 0.3)', width: 1 },
            name: 'Raw Signal'
        };

        // Trace 2: DP Cost Curve (Visualization of the DP Table construction)
        // This satisfies "DP table construction" visualization requirement
        const dpTrace = {
            x: Array.from({length: dpTable.length}, (_, i) => i),
            y: dpTable,
            xaxis: 'x',
            yaxis: 'y2', // Second Y-axis
            type: 'scatter',
            mode: 'lines',
            fill: 'tozeroy',
            line: { color: this.colors.costCurve, width: 2 },
            name: 'DP Min Cost (Accumulated)'
        };

        // Prepare Segment Traces (Hidden initially)
        const segmentTraces = this._createSegmentTraces(signal);

        const layout = {
            title: 'Backtracking Visualization & DP Cost',
            grid: { rows: 2, columns: 1, pattern: 'independent' },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0.2)',
            font: { color: this.colors.text },
            xaxis: { title: 'Time (Samples)', gridcolor: this.colors.grid },
            yaxis: { title: 'ECG Amplitude', gridcolor: this.colors.grid, domain: [0.3, 1] }, // Top 70%
            yaxis2: { title: 'DP Cost', gridcolor: this.colors.grid, domain: [0, 0.2] },     // Bottom 20%
            showlegend: true,
            legend: { x: 1, y: 1 },
            height: 400
        };

        // Combine traces: Signal, DP Cost, and Segments
        Plotly.newPlot('ecg-plot', [signalTrace, dpTrace, ...segmentTraces], layout, {displayModeBar: false});

        // Initialize at Step 0
        this.currentStep = 0;
        this.updateStepInfo(0);
        this.enableStepControls(true);
        this.updateFeaturesDisplay();
        this.updateClassificationDisplay();
        this.updateDPTableDisplay();
    }

    _createSegmentTraces(signal) {
        const traces = [];
        ['P', 'QRS', 'T'].forEach(type => {
            traces.push({
                x: [null], y: [null], // Empty initially
                type: 'scatter',
                mode: 'lines',
                line: { color: this._getSegmentColor(type), width: 3 },
                name: `${type} Wave`,
                fill: 'toself',
                fillcolor: this._getSegmentColor(type, 0.3)
            });
        });
        return traces;
    }

    _getSegmentColor(type, opacity = 1) {
        const map = { 'P': this.colors.pWave, 'QRS': this.colors.qrsWave, 'T': this.colors.tWave };
        let color = map[type] || '#fff';
        if (opacity < 1) color = color.replace(')', `, ${opacity})`).replace('rgb', 'rgba');
        return color;
    }

    // 3. تحديث الأنيميشن خطوة بخطوة (Backtracking Visualization)
    updateStep(step) {
        if (!this.signalData) return;

        this.currentStep = step;
        const stepData = this.signalData.visualization.backtrack_steps[step];
        const signal = this.signalData.signal_info.normalization.normalized;

        // Build arrays for currently visible segments
        const pX = [], pY = [], qrsX = [], qrsY = [], tX = [], tY = [];

        stepData.segments.forEach(seg => {
            const range = Array.from({length: seg.end - seg.start + 1}, (_, i) => seg.start + i);
            const vals = range.map(i => signal[i]);

            // Insert NaNs to break lines between disjoint segments
            const xArr = (seg.type === 'P') ? pX : (seg.type === 'QRS') ? qrsX : tX;
            const yArr = (seg.type === 'P') ? pY : (seg.type === 'QRS') ? qrsY : tY;

            xArr.push(...range, null);
            yArr.push(...vals, null);
        });

        // Update Plotly Traces efficiently
        // Indices 2, 3, 4 correspond to P, QRS, T traces we created earlier
        Plotly.restyle('ecg-plot', { x: [pX, qrsX, tX], y: [pY, qrsY, tY] }, [2, 3, 4]);

        // Update Text Info
        this.updateStepInfo(step);
        this.updateProgressBar(step, this.totalSteps);
        this.highlightCurrentSegment(stepData.current_segment);
    }

    updateStepInfo(step) {
        if (!this.signalData?.visualization) return;
        const stepData = this.signalData.visualization.backtrack_steps[step];

        document.getElementById('current-step').innerText = `Step ${step}/${this.totalSteps - 1}`;
        document.getElementById('step-description').innerText = stepData.description;

        // Auto-scroll logic for table
        this.updateSegmentsTable(stepData.segments);
    }

    // Controls Logic
    prevStep() { if (this.currentStep > 0) this.updateStep(this.currentStep - 1); }
    nextStep() { if (this.currentStep < this.totalSteps - 1) this.updateStep(this.currentStep + 1); }

    togglePlay() {
        if (this.isAnimating) {
            clearInterval(this.animationInterval);
            this.isAnimating = false;
            document.getElementById('play-steps').innerHTML = '<i class="fas fa-play"></i> Play';
        } else {
            this.isAnimating = true;
            document.getElementById('play-steps').innerHTML = '<i class="fas fa-pause"></i> Pause';
            this.animationInterval = setInterval(() => {
                if (this.currentStep < this.totalSteps - 1) this.nextStep();
                else this.togglePlay();
            }, 800); // Speed of animation
        }
    }

    updateProgressBar(current, total) {
        const percent = ((current) / (total - 1)) * 100;
        document.getElementById('step-progress-fill').style.width = `${percent}%`;
    }

    // Display Helpers
    updateSegmentsTable(segments) {
        const tbody = document.querySelector('#segments-table tbody');
        tbody.innerHTML = segments.map((s, i) => `
            <tr class="${i === segments.length - 1 ? 'current-segment' : ''}">
                <td>${i + 1}</td>
                <td><span class="segment-badge ${s.type}-badge">${s.type}</span></td>
                <td>${s.start}</td>
                <td>${s.end}</td>
                <td>${s.duration.toFixed(1)}</td>
                <td>${s.cost.toFixed(2)}</td>
            </tr>
        `).join('');
    }

    highlightCurrentSegment(seg) {
        // Implementation handled in updateSegmentsTable via class
    }

    updateDPTableDisplay() {
        const tbody = document.querySelector('#dp-table tbody');
        const dpData = this.signalData.visualization.dp_table_sample;
        if(dpData && tbody) {
            tbody.innerHTML = dpData.map(row => `
                <tr>
                    <td>${row.position}</td>
                    <td>${row.cost}</td>
                    <td>-</td>
                    <td>-</td>
                </tr>
            `).join('');
        }
    }

    updateFeaturesDisplay() {
        const grid = document.getElementById('features-grid');
        const feats = this.signalData.features.detailed_features;
        const keys = ['qrs_count', 'rr_mean', 'signal_mean', 'total_duration'];

        grid.innerHTML = keys.map(k => `
            <div class="feature-card">
                <h5>${k.replace('_', ' ').toUpperCase()}</h5>
                <div class="feature-value">${typeof feats[k] === 'number' ? feats[k].toFixed(2) : feats[k]}</div>
            </div>
        `).join('');
    }

    updateClassificationDisplay() {
        const res = this.signalData.classification;
        const div = document.getElementById('classification-result');
        div.className = `classification-result classification-${res.prediction.toLowerCase()}`;
        div.innerHTML = `
            <div class="result-content">
                <div class="result-icon"><i class="fas fa-${res.prediction === 'Normal' ? 'check-circle' : 'exclamation-triangle'}"></i></div>
                <h3 class="result-title">${res.prediction} ECG</h3>
                <p>Confidence: ${(res.confidence * 100).toFixed(1)}%</p>
                <p class="result-method">Method: ${res.method}</p>
                ${res.abnormalities?.length ? `<ul>${res.abnormalities.map(a => `<li>${a}</li>`).join('')}</ul>` : ''}
            </div>
        `;
    }

    switchTab(tab) {
        document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.getElementById(`${tab}-tab`).classList.add('active');
        document.querySelector(`[data-tab="${tab}"]`).classList.add('active');
    }

    showLoading(msg) { document.getElementById('results-status').innerHTML = `<i class="fas fa-spinner fa-spin"></i> ${msg}`; }
    showSuccess(msg) { document.getElementById('results-status').innerHTML = `<i class="fas fa-check"></i> ${msg}`; }
    showError(msg) { document.getElementById('results-status').innerHTML = `<i class="fas fa-times"></i> ${msg}`; }
    clear() { /* ... */ }
    enableStepControls(enable) {
        document.querySelectorAll('.btn-step').forEach(b => b.disabled = !enable);
    }
}

window.ECG_Visualizer = new ECG_Visualizer();