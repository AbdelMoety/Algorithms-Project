/**
 * DP Table Visualizer
 * Shows step-by-step DP table construction
 */

class DPVisualizer {
    constructor() {
        this.currentStep = 0;
        this.totalSteps = 0;
        this.dpSteps = [];
        this.isPlaying = false;
        this.playInterval = null;
        this.speed = 1000; // ms per step

        // DP table state
        this.dpTable = [];
        this.currentOperation = null;

        // Initialize event listeners
        this.init();
    }

    init() {
        console.log("DP Visualizer initialized");
        this.bindEvents();

        // Load sample data for demo
        this.loadSampleData();
    }

    bindEvents() {
        // Control buttons
        document.getElementById('prev-btn')?.addEventListener('click', () => this.prevStep());
        document.getElementById('next-btn')?.addEventListener('click', () => this.nextStep());
        document.getElementById('play-btn')?.addEventListener('click', () => this.togglePlay());

        // Speed control
        const speedSlider = document.createElement('input');
        speedSlider.type = 'range';
        speedSlider.min = '100';
        speedSlider.max = '2000';
        speedSlider.value = '1000';
        speedSlider.style.marginLeft = '20px';
        speedSlider.oninput = (e) => {
            this.speed = 2100 - e.target.value; // Reverse so right is faster
        };

        document.querySelector('.step-controls').appendChild(speedSlider);
    }

    loadSampleData() {
        // Create sample DP steps for demonstration
        this.dpSteps = [
            {
                step: 0,
                description: "Initialization: dp[0] = 0 (base case)",
                table: [
                    { i: 0, cost: 0, prev: '-', type: '-', len: '-', action: 'Base case' }
                ],
                operation: "Setting base case: dp[0] = 0"
            },
            {
                step: 1,
                description: "Processing position 20: Found P wave candidate",
                table: [
                    { i: 0, cost: 0, prev: '-', type: '-', len: '-', action: 'Base case' },
                    { i: 20, cost: 1.2, prev: 0, type: 'P', len: 20, action: 'Update from P wave' }
                ],
                operation: "dp[20] = min(dp[0] + cost(P, signal[0:20])) = 1.2"
            },
            {
                step: 2,
                description: "Processing position 60: QRS complex found",
                table: [
                    { i: 0, cost: 0, prev: '-', type: '-', len: '-', action: 'Base case' },
                    { i: 20, cost: 1.2, prev: 0, type: 'P', len: 20, action: 'P wave' },
                    { i: 60, cost: 3.5, prev: 20, type: 'QRS', len: 40, action: 'Update from QRS' }
                ],
                operation: "dp[60] = min(dp[20] + cost(QRS, signal[20:60])) = 3.5"
            },
            {
                step: 3,
                description: "Processing position 90: T wave identified",
                table: [
                    { i: 0, cost: 0, prev: '-', type: '-', len: '-', action: 'Base case' },
                    { i: 20, cost: 1.2, prev: 0, type: 'P', len: 20, action: 'P wave' },
                    { i: 60, cost: 3.5, prev: 20, type: 'QRS', len: 40, action: 'QRS complex' },
                    { i: 90, cost: 5.1, prev: 60, type: 'T', len: 30, action: 'Update from T wave' }
                ],
                operation: "dp[90] = min(dp[60] + cost(T, signal[60:90])) = 5.1"
            },
            {
                step: 4,
                description: "Final position 140: Complete segmentation",
                table: [
                    { i: 0, cost: 0, prev: '-', type: '-', len: '-', action: 'Base case' },
                    { i: 20, cost: 1.2, prev: 0, type: 'P', len: 20, action: 'P wave' },
                    { i: 60, cost: 3.5, prev: 20, type: 'QRS', len: 40, action: 'QRS complex' },
                    { i: 90, cost: 5.1, prev: 60, type: 'T', len: 30, action: 'T wave' },
                    { i: 140, cost: 6.8, prev: 90, type: 'P', len: 50, action: 'Next P wave' }
                ],
                operation: "Optimal total cost = dp[140] = 6.8"
            }
        ];

        this.totalSteps = this.dpSteps.length;
        this.updateStepInfo();
    }

    updateStepInfo() {
        const stepInfo = this.dpSteps[this.currentStep];
        if (!stepInfo) return;

        // Update description
        document.getElementById('step-description').innerHTML = `
            <h3>Step ${stepInfo.step + 1}/${this.totalSteps}</h3>
            <p>${stepInfo.description}</p>
            <div class="formula-box">
                ${stepInfo.operation}
            </div>
        `;

        // Update step counter
        document.getElementById('step-counter').textContent =
            `Step ${this.currentStep + 1}/${this.totalSteps}`;

        // Update progress bar
        const progress = (this.currentStep / (this.totalSteps - 1)) * 100;
        document.getElementById('dp-progress').style.width = `${progress}%`;

        // Update DP table
        this.updateDPTable(stepInfo.table);

        // Update current operation
        document.getElementById('current-operation').innerHTML = `
            <p><strong>Current Operation:</strong> ${stepInfo.operation}</p>
            <p><strong>Recurrence Applied:</strong> dp[i] = min(dp[j] + cost(type, signal[j:i]))</p>
        `;
    }

    updateDPTable(tableData) {
        const tableBody = document.getElementById('dp-table-body');
        tableBody.innerHTML = '';

        tableData.forEach((row, index) => {
            const tr = document.createElement('tr');

            // Highlight current row if it was just updated
            const isUpdated = index === tableData.length - 1 && this.currentStep > 0;
            const rowClass = isUpdated ? 'updated' : '';

            tr.innerHTML = `
                <td class="${rowClass}">${row.i}</td>
                <td class="${rowClass}">${row.cost}</td>
                <td class="${rowClass}">${row.prev}</td>
                <td class="${rowClass}">${row.type}</td>
                <td class="${rowClass}">${row.len}</td>
                <td class="${rowClass}">${row.action}</td>
            `;

            tableBody.appendChild(tr);
        });
    }

    prevStep() {
        if (this.currentStep > 0) {
            this.currentStep--;
            this.updateStepInfo();
        }
        this.updateButtons();
    }

    nextStep() {
        if (this.currentStep < this.totalSteps - 1) {
            this.currentStep++;
            this.updateStepInfo();
        }
        this.updateButtons();
    }

    togglePlay() {
        if (this.isPlaying) {
            this.pause();
        } else {
            this.play();
        }
    }

    play() {
        if (this.currentStep >= this.totalSteps - 1) {
            this.currentStep = 0;
        }

        this.isPlaying = true;
        const playBtn = document.getElementById('play-btn');
        playBtn.innerHTML = '<i class="fas fa-pause"></i> Pause';
        playBtn.classList.add('pause');
        playBtn.classList.remove('play');

        this.playInterval = setInterval(() => {
            if (this.currentStep < this.totalSteps - 1) {
                this.nextStep();
            } else {
                this.pause();
            }
        }, this.speed);
    }

    pause() {
        this.isPlaying = false;
        clearInterval(this.playInterval);

        const playBtn = document.getElementById('play-btn');
        playBtn.innerHTML = '<i class="fas fa-play"></i> Play';
        playBtn.classList.add('play');
        playBtn.classList.remove('pause');
    }

    resetSteps() {
        this.pause();
        this.currentStep = 0;
        this.updateStepInfo();
        this.updateButtons();
    }

    updateButtons() {
        const prevBtn = document.getElementById('prev-btn');
        const nextBtn = document.getElementById('next-btn');

        prevBtn.disabled = this.currentStep === 0;
        nextBtn.disabled = this.currentStep === this.totalSteps - 1;
    }

    // Load real DP data from API
    async loadDPDataFromAPI(signal) {
        try {
            const response = await fetch('http://localhost:5000/api/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    signal: signal,
                    step_by_step: true
                })
            });

            const data = await response.json();

            if (data.success) {
                this.processDPResults(data);
            }

        } catch (error) {
            console.error('Error loading DP data:', error);
        }
    }

    processDPResults(results) {
        // Process DP steps from backend
        if (results.dp_table && results.dp_table.steps) {
            this.dpSteps = results.dp_table.steps.map((step, index) => ({
                step: index,
                description: step.description || `Step ${index}`,
                table: this.createTableSnapshot(results, step),
                operation: step.action || 'DP update'
            }));

            this.totalSteps = this.dpSteps.length;
            this.currentStep = 0;
            this.updateStepInfo();
        }
    }

    createTableSnapshot(results, step) {
        // Create table snapshot from step data
        // This is a simplified version - you'd need to adapt based on your backend response
        const snapshot = [];

        // Add base case
        snapshot.push({
            i: 0,
            cost: 0,
            prev: '-',
            type: '-',
            len: '-',
            action: 'Base case'
        });

        // Add other rows based on step data
        if (step.position) {
            snapshot.push({
                i: step.position,
                cost: step.cost || 0,
                prev: step.prev_position || 0,
                type: step.segment_type || '-',
                len: step.segment_length || 0,
                action: step.action || 'Update'
            });
        }

        return snapshot;
    }
}

// Initialize DP Visualizer
const dpVisualizer = new DPVisualizer();

// Global functions for HTML buttons
function prevStep() { dpVisualizer.prevStep(); }
function nextStep() { dpVisualizer.nextStep(); }
function togglePlay() { dpVisualizer.togglePlay(); }
function resetSteps() { dpVisualizer.resetSteps(); }

// Comparison functions
async function runComparison() {
    try {
        const response = await fetch('http://localhost:5000/api/compare-parameters', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                signal: Array.from({length: 200}, (_, i) => Math.sin(i * 0.1)) // Sample signal
            })
        });

        const data = await response.json();

        if (data.success) {
            displayComparisonResults(data.comparisons);
        }

    } catch (error) {
        console.error('Comparison error:', error);
    }
}

function displayComparisonResults(comparisons) {
    let html = '<h3>Parameter Comparison Results</h3>';
    html += '<table class="comparison-table">';
    html += '<tr><th>Parameter Set</th><th>Segments</th><th>Total Cost</th><th>P Waves</th><th>QRS Waves</th><th>T Waves</th><th>Operations</th></tr>';

    for (const [paramSet, results] of Object.entries(comparisons)) {
        html += `
            <tr>
                <td><strong>${paramSet}</strong></td>
                <td>${results.segments_found}</td>
                <td>${results.total_cost.toFixed(3)}</td>
                <td>${results.p_waves}</td>
                <td>${results.qrs_waves}</td>
                <td>${results.t_waves}</td>
                <td>${results.operations}</td>
            </tr>
        `;
    }

    html += '</table>';

    document.getElementById('comparison-results').innerHTML = html;
}

async function saveCurrentResults() {
    // Get current results from visualizer
    const currentResults = {
        timestamp: new Date().toISOString(),
        steps: dpVisualizer.dpSteps.length,
        currentStep: dpVisualizer.currentStep
    };

    try {
        const response = await fetch('http://localhost:5000/api/save-results', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(currentResults)
        });

        const data = await response.json();
        if (data.success) {
            alert('Results saved successfully!');
            loadSavedResults(); // Refresh list
        }

    } catch (error) {
        console.error('Save error:', error);
    }
}

async function loadSavedResults() {
    try {
        const response = await fetch('http://localhost:5000/api/load-results');
        const data = await response.json();

        if (data.results && data.results.length > 0) {
            let html = '<h3>Saved Results</h3>';

            data.results.forEach(result => {
                html += `
                    <div class="result-item" onclick="loadResult('${result.filename}')">
                        <div><strong>${result.timestamp}</strong></div>
                        <div>Segments: ${result.segments} | Cost: ${result.total_cost}</div>
                        <div>Signal Length: ${result.signal_length}</div>
                    </div>
                `;
            });

            document.getElementById('saved-results').innerHTML = html;
        }

    } catch (error) {
        console.error('Load error:', error);
    }
}

async function loadResult(filename) {
    // Load and display a saved result
    alert(`Loading: ${filename}`);
    // You would implement this based on your backend
}