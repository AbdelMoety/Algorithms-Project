/**
 * Backtracking Visualizer
 * Shows step-by-step reconstruction of optimal path
 */

class BacktrackingVisualizer {
    constructor() {
        this.currentStep = 0;
        this.backtrackSteps = [];
        this.isPlaying = false;
        this.playInterval = null;

        // Sample backtracking data
        this.loadSampleData();
    }

    loadSampleData() {
        this.backtrackSteps = [
            {
                step: 0,
                position: 140,
                action: "Start backtracking from final position",
                description: "Start at position 140 with total cost 6.8",
                segment: null,
                cost: 6.8,
                segments: []
            },
            {
                step: 1,
                position: 140,
                action: "Backtrack to position 90",
                description: "Found P wave: positions 90-139 (length: 50)",
                segment: {
                    type: 'P',
                    start: 90,
                    end: 139,
                    length: 50,
                    cost: 1.7
                },
                cost: 5.1,
                segments: [{type: 'P', start: 90, end: 139}]
            },
            {
                step: 2,
                position: 90,
                action: "Backtrack to position 60",
                description: "Found T wave: positions 60-89 (length: 30)",
                segment: {
                    type: 'T',
                    start: 60,
                    end: 89,
                    length: 30,
                    cost: 1.6
                },
                cost: 3.5,
                segments: [
                    {type: 'P', start: 90, end: 139},
                    {type: 'T', start: 60, end: 89}
                ]
            },
            {
                step: 3,
                position: 60,
                action: "Backtrack to position 20",
                description: "Found QRS complex: positions 20-59 (length: 40)",
                segment: {
                    type: 'QRS',
                    start: 20,
                    end: 59,
                    length: 40,
                    cost: 2.3
                },
                cost: 1.2,
                segments: [
                    {type: 'P', start: 90, end: 139},
                    {type: 'T', start: 60, end: 89},
                    {type: 'QRS', start: 20, end: 59}
                ]
            },
            {
                step: 4,
                position: 20,
                action: "Backtrack to position 0",
                description: "Found P wave: positions 0-19 (length: 20)",
                segment: {
                    type: 'P',
                    start: 0,
                    end: 19,
                    length: 20,
                    cost: 1.2
                },
                cost: 0,
                segments: [
                    {type: 'P', start: 90, end: 139},
                    {type: 'T', start: 60, end: 89},
                    {type: 'QRS', start: 20, end: 59},
                    {type: 'P', start: 0, end: 19}
                ]
            },
            {
                step: 5,
                position: 0,
                action: "Backtracking complete",
                description: "Reached start position. Found 4 segments in optimal path.",
                segment: null,
                cost: 0,
                segments: [
                    {type: 'P', start: 0, end: 19},
                    {type: 'QRS', start: 20, end: 59},
                    {type: 'T', start: 60, end: 89},
                    {type: 'P', start: 90, end: 139}
                ]
            }
        ];
    }

    displayStep(stepIndex) {
        const step = this.backtrackSteps[stepIndex];
        if (!step) return;

        const pathContainer = document.getElementById('backtracking-path');
        const segmentsContainer = document.getElementById('reconstructed-segments');

        // Display current step in path
        let pathHTML = '<h3>Backtracking Steps</h3>';

        this.backtrackSteps.forEach((s, idx) => {
            const activeClass = idx === stepIndex ? 'active' : '';
            const waveClass = s.segment ? `path-step ${s.segment.type.toLowerCase()}-wave` : 'path-step';

            pathHTML += `
                <div class="${waveClass} ${activeClass}">
                    <strong>Step ${s.step}:</strong> ${s.action}
                    <div style="font-size: 0.9em; opacity: 0.8; margin-top: 5px;">
                        ${s.description}
                    </div>
                    ${s.segment ? `
                        <div style="margin-top: 5px;">
                            <span class="badge">${s.segment.type}</span>
                            Positions: ${s.segment.start}-${s.segment.end}
                            | Cost: ${s.segment.cost}
                        </div>
                    ` : ''}
                </div>
            `;
        });

        pathContainer.innerHTML = pathHTML;

        // Display reconstructed segments
        let segmentsHTML = '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">';

        step.segments.forEach((seg, idx) => {
            let color = '#05d9e8';
            if (seg.type === 'QRS') color = '#ff2a6d';
            if (seg.type === 'T') color = '#00ff9d';

            segmentsHTML += `
                <div style="border-left: 4px solid ${color}; padding: 15px; background: rgba(255,255,255,0.05); border-radius: 8px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <strong style="color: ${color};">${seg.type} Wave</strong>
                        <span class="badge">Step ${step.segments.length - idx}</span>
                    </div>
                    <div>Position: ${seg.start} - ${seg.end}</div>
                    <div>Length: ${seg.end - seg.start + 1} samples</div>
                    <div style="margin-top: 10px; height: 10px; background: rgba(255,255,255,0.1); border-radius: 5px;">
                        <div style="width: ${(seg.end / 140) * 100}%; height: 100%; background: ${color}; border-radius: 5px;"></div>
                    </div>
                </div>
            `;
        });

        segmentsHTML += '</div>';

        if (step.segments.length === 0) {
            segmentsHTML = '<p>No segments reconstructed yet. Continue backtracking...</p>';
        }

        segmentsContainer.innerHTML = segmentsHTML;

        // Scroll to current step
        const activeStep = pathContainer.querySelector('.path-step.active');
        if (activeStep) {
            activeStep.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
    }

    nextStep() {
        if (this.currentStep < this.backtrackSteps.length - 1) {
            this.currentStep++;
            this.displayStep(this.currentStep);
        }
    }

    prevStep() {
        if (this.currentStep > 0) {
            this.currentStep--;
            this.displayStep(this.currentStep);
        }
    }

    play() {
        if (this.isPlaying) return;

        this.isPlaying = true;
        this.playInterval = setInterval(() => {
            if (this.currentStep < this.backtrackSteps.length - 1) {
                this.nextStep();
            } else {
                this.pause();
            }
        }, 1500);
    }

    pause() {
        this.isPlaying = false;
        clearInterval(this.playInterval);
    }
}

// Initialize backtracking visualizer
const backtrackingVis = new BacktrackingVisualizer();

// Global functions
function prevBacktrackStep() { backtrackingVis.prevStep(); }
function nextBacktrackStep() { backtrackingVis.nextStep(); }
function playBacktracking() {
    if (backtrackingVis.isPlaying) {
        backtrackingVis.pause();
    } else {
        backtrackingVis.play();
    }
}

// Initialize with first step
window.addEventListener('load', () => {
    backtrackingVis.displayStep(0);
});