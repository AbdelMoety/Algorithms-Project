/**
 * Complete DP Table Viewer with Pagination
 * Shows the entire DP table with navigation controls
 */

class DPTableViewer {
    constructor() {
        this.currentPage = 1;
        this.totalPages = 1;
        this.pageSize = 50;
        this.dpTable = [];
        this.isLoading = false;

        this.init();
    }

    init() {
        console.log("DP Table Viewer Initialized");
        this.createUI();
        this.bindEvents();
    }

    createUI() {
        // Create container for DP table
        const container = document.createElement('div');
        container.id = 'dp-table-container';
        container.className = 'dp-table-full-container';
        container.innerHTML = `
            <div class="dp-table-header">
                <h3><i class="fas fa-table"></i> Complete DP Table</h3>
                <div class="table-info">
                    <span id="table-info">Loading...</span>
                </div>
            </div>
            
            <div class="table-controls">
                <button class="control-btn" id="first-page">
                    <i class="fas fa-fast-backward"></i> First
                </button>
                <button class="control-btn" id="prev-page">
                    <i class="fas fa-chevron-left"></i> Previous
                </button>
                
                <div class="page-info">
                    Page <input type="number" id="page-input" min="1" value="1" style="width: 60px;">
                    of <span id="total-pages">1</span>
                </div>
                
                <button class="control-btn" id="next-page">
                    Next <i class="fas fa-chevron-right"></i>
                </button>
                <button class="control-btn" id="last-page">
                    Last <i class="fas fa-fast-forward"></i>
                </button>
                
                <div class="page-size">
                    Rows per page:
                    <select id="page-size-select">
                        <option value="25">25</option>
                        <option value="50" selected>50</option>
                        <option value="100">100</option>
                        <option value="200">200</option>
                    </select>
                </div>
                
                <button class="control-btn" id="download-table">
                    <i class="fas fa-download"></i> Download CSV
                </button>
            </div>
            
            <div class="table-wrapper">
                <table class="dp-table-full" id="dp-table-full">
                    <thead>
                        <tr>
                            <th>Position (i)</th>
                            <th>Cost dp[i]</th>
                            <th>Segment Type</th>
                            <th>Previous (j)</th>
                            <th>Length (i-j)</th>
                            <th>State</th>
                        </tr>
                    </thead>
                    <tbody id="dp-table-body">
                        <tr><td colspan="6" style="text-align: center;">Loading DP table...</td></tr>
                    </tbody>
                </table>
            </div>
            
            <div class="table-footer">
                <div class="legend">
                    <div class="legend-item">
                        <span class="legend-dot infinite"></span>
                        <span>Infinite Cost (∞)</span>
                    </div>
                    <div class="legend-item">
                        <span class="legend-dot optimal"></span>
                        <span>Optimal Path</span>
                    </div>
                    <div class="legend-item">
                        <span class="legend-dot updated"></span>
                        <span>Recently Updated</span>
                    </div>
                </div>
                <div class="row-count" id="row-count">Showing 0 rows</div>
            </div>
        `;

        // Add to page
        const mainContent = document.querySelector('.main-content') || document.body;
        mainContent.appendChild(container);

        // Add CSS
        this.addStyles();
    }

    addStyles() {
        const styles = `
            .dp-table-full-container {
                background: rgba(0, 0, 0, 0.3);
                border-radius: 15px;
                padding: 20px;
                margin: 20px 0;
                border: 1px solid rgba(5, 217, 232, 0.3);
            }
            
            .dp-table-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 20px;
                padding-bottom: 15px;
                border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            }
            
            .table-controls {
                display: flex;
                gap: 15px;
                align-items: center;
                margin-bottom: 20px;
                flex-wrap: wrap;
                padding: 15px;
                background: rgba(0, 0, 0, 0.2);
                border-radius: 10px;
            }
            
            .control-btn {
                padding: 8px 16px;
                background: rgba(5, 217, 232, 0.2);
                border: 1px solid #05d9e8;
                color: #05d9e8;
                border-radius: 6px;
                cursor: pointer;
                display: flex;
                align-items: center;
                gap: 8px;
                transition: all 0.3s;
            }
            
            .control-btn:hover {
                background: #05d9e8;
                color: #0a0a2a;
            }
            
            .control-btn:disabled {
                opacity: 0.5;
                cursor: not-allowed;
            }
            
            .page-info {
                display: flex;
                align-items: center;
                gap: 10px;
                font-weight: bold;
            }
            
            .page-size {
                margin-left: auto;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            
            .table-wrapper {
                overflow-x: auto;
                max-height: 600px;
                overflow-y: auto;
                border-radius: 10px;
                border: 1px solid rgba(255, 255, 255, 0.1);
            }
            
            .dp-table-full {
                width: 100%;
                border-collapse: collapse;
                font-family: 'Courier New', monospace;
                font-size: 0.9em;
            }
            
            .dp-table-full th {
                position: sticky;
                top: 0;
                background: rgba(5, 217, 232, 0.3);
                padding: 12px;
                text-align: center;
                border: 1px solid rgba(255, 255, 255, 0.2);
                color: #05d9e8;
                z-index: 10;
            }
            
            .dp-table-full td {
                padding: 10px;
                text-align: center;
                border: 1px solid rgba(255, 255, 255, 0.1);
                transition: all 0.3s;
            }
            
            .dp-table-full tr:hover td {
                background: rgba(5, 217, 232, 0.1);
            }
            
            .dp-table-full .infinite {
                color: #ff2a6d;
                font-weight: bold;
            }
            
            .dp-table-full .optimal {
                background: rgba(0, 255, 157, 0.1);
                border-left: 3px solid #00ff9d;
            }
            
            .dp-table-full .updated {
                background: rgba(255, 204, 0, 0.1);
                animation: highlight 2s;
            }
            
            @keyframes highlight {
                0% { background: rgba(255, 204, 0, 0.5); }
                100% { background: rgba(255, 204, 0, 0.1); }
            }
            
            .table-footer {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-top: 20px;
                padding-top: 15px;
                border-top: 1px solid rgba(255, 255, 255, 0.1);
            }
            
            .legend {
                display: flex;
                gap: 20px;
            }
            
            .legend-item {
                display: flex;
                align-items: center;
                gap: 8px;
                font-size: 0.9em;
            }
            
            .legend-dot {
                width: 12px;
                height: 12px;
                border-radius: 50%;
            }
            
            .legend-dot.infinite {
                background: #ff2a6d;
            }
            
            .legend-dot.optimal {
                background: #00ff9d;
            }
            
            .legend-dot.updated {
                background: #ffcc00;
            }
            
            .row-count {
                font-weight: bold;
                color: #05d9e8;
            }
            
            /* Scrollbar styling */
            .table-wrapper::-webkit-scrollbar {
                width: 10px;
                height: 10px;
            }
            
            .table-wrapper::-webkit-scrollbar-track {
                background: rgba(255, 255, 255, 0.05);
                border-radius: 5px;
            }
            
            .table-wrapper::-webkit-scrollbar-thumb {
                background: #05d9e8;
                border-radius: 5px;
            }
            
            .table-wrapper::-webkit-scrollbar-thumb:hover {
                background: #00ff9d;
            }
        `;

        const styleSheet = document.createElement('style');
        styleSheet.textContent = styles;
        document.head.appendChild(styleSheet);
    }

    bindEvents() {
        // Page navigation
        document.getElementById('first-page')?.addEventListener('click', () => this.goToPage(1));
        document.getElementById('prev-page')?.addEventListener('click', () => this.prevPage());
        document.getElementById('next-page')?.addEventListener('click', () => this.nextPage());
        document.getElementById('last-page')?.addEventListener('click', () => this.goToPage(this.totalPages));

        // Page input
        document.getElementById('page-input')?.addEventListener('change', (e) => {
            const page = parseInt(e.target.value);
            if (page >= 1 && page <= this.totalPages) {
                this.goToPage(page);
            }
        });

        // Page size selector
        document.getElementById('page-size-select')?.addEventListener('change', (e) => {
            this.pageSize = parseInt(e.target.value);
            this.currentPage = 1;
            this.loadTable();
        });

        // Download button
        document.getElementById('download-table')?.addEventListener('click', () => this.downloadCSV());
    }

    async loadTable(signal) {
        if (!signal || signal.length === 0) {
            this.showMessage('Please load a signal first', 'error');
            return;
        }

        this.isLoading = true;
        this.showMessage('Loading complete DP table...', 'loading');

        try {
            // Fetch complete DP table
            const response = await fetch('http://localhost:5000/api/dp-table/full', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ signal: signal })
            });

            const data = await response.json();

            if (data.success) {
                this.dpTable = data.dp_table;
                this.totalPages = data.total_pages;

                this.updateUI();
                this.renderCurrentPage();

                this.showMessage(`DP table loaded: ${this.dpTable.length} rows`, 'success');
            } else {
                this.showMessage('Failed to load DP table: ' + data.error, 'error');
            }

        } catch (error) {
            console.error('Error loading DP table:', error);
            this.showMessage('Connection error. Using demo data...', 'error');
            this.loadDemoData();
        } finally {
            this.isLoading = false;
        }
    }

    loadDemoData() {
        // Create demo DP table with 400 positions
        this.dpTable = [];

        for (let i = 0; i <= 400; i++) {
            let cost = '∞';
            let segment = '-';
            let prev = '-';
            let length = '-';

            if (i === 0) {
                cost = '0.000';
                segment = 'Base';
            } else if (i % 50 === 0) {
                // Simulate segment updates
                const types = ['P', 'QRS', 'T'];
                segment = types[Math.floor(Math.random() * types.length)];
                prev = Math.max(0, i - Math.floor(Math.random() * 50) + 10);
                cost = (Math.random() * 5).toFixed(3);
                length = i - prev;
            }

            this.dpTable.push({
                position: i,
                cost: cost,
                segment: segment,
                prev: prev,
                length: length
            });
        }

        this.totalPages = Math.ceil(this.dpTable.length / this.pageSize);
        this.updateUI();
        this.renderCurrentPage();

        this.showMessage('Demo DP table loaded', 'warning');
    }

    renderCurrentPage() {
        const tableBody = document.getElementById('dp-table-body');
        if (!tableBody) return;

        const startIdx = (this.currentPage - 1) * this.pageSize;
        const endIdx = Math.min(startIdx + this.pageSize, this.dpTable.length);
        const pageData = this.dpTable.slice(startIdx, endIdx);

        let html = '';

        pageData.forEach((row, index) => {
            const globalIndex = startIdx + index;
            const isOptimal = row.segment !== '-' && row.segment !== 'Base';
            const isInfinite = row.cost === '∞';

            const rowClass = isOptimal ? 'optimal' : isInfinite ? 'infinite' : '';
            const costClass = isInfinite ? 'infinite' : '';

            html += `
                <tr class="${rowClass}">
                    <td>${row.position}</td>
                    <td class="${costClass}">${row.cost}</td>
                    <td>${row.segment}</td>
                    <td>${row.prev}</td>
                    <td>${row.length}</td>
                    <td>
                        ${row.segment === 'Base' ? 'Base Case' : 
                          row.segment === '-' ? 'Not Calculated' : 
                          `Optimal ${row.segment} Wave`}
                    </td>
                </tr>
            `;
        });

        tableBody.innerHTML = html;

        // Update row count
        document.getElementById('row-count').textContent =
            `Showing rows ${startIdx + 1}-${endIdx} of ${this.dpTable.length}`;

        // Highlight current page in table
        this.highlightCurrentPage();
    }

    highlightCurrentPage() {
        // Remove previous highlights
        document.querySelectorAll('.dp-table-full tr').forEach(row => {
            row.classList.remove('current-page');
        });

        // Highlight first row of current page
        const rows = document.querySelectorAll('#dp-table-body tr');
        if (rows.length > 0) {
            rows[0].classList.add('updated');
            rows[0].scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
    }

    updateUI() {
        // Update page input
        const pageInput = document.getElementById('page-input');
        if (pageInput) {
            pageInput.value = this.currentPage;
            pageInput.max = this.totalPages;
        }

        // Update total pages display
        document.getElementById('total-pages').textContent = this.totalPages;

        // Update buttons state
        document.getElementById('first-page').disabled = this.currentPage === 1;
        document.getElementById('prev-page').disabled = this.currentPage === 1;
        document.getElementById('next-page').disabled = this.currentPage === this.totalPages;
        document.getElementById('last-page').disabled = this.currentPage === this.totalPages;

        // Update table info
        document.getElementById('table-info').innerHTML = `
            Total Positions: ${this.dpTable.length} | 
            Pages: ${this.totalPages} | 
            Current Page: ${this.currentPage}
        `;
    }

    goToPage(page) {
        if (page >= 1 && page <= this.totalPages && page !== this.currentPage) {
            this.currentPage = page;
            this.renderCurrentPage();
            this.updateUI();
        }
    }

    prevPage() {
        if (this.currentPage > 1) {
            this.goToPage(this.currentPage - 1);
        }
    }

    nextPage() {
        if (this.currentPage < this.totalPages) {
            this.goToPage(this.currentPage + 1);
        }
    }

    showMessage(message, type = 'info') {
        const tableInfo = document.getElementById('table-info');
        if (tableInfo) {
            let icon = 'ℹ️';
            if (type === 'success') icon = '✅';
            if (type === 'error') icon = '❌';
            if (type === 'warning') icon = '⚠️';
            if (type === 'loading') icon = '⏳';

            tableInfo.innerHTML = `${icon} ${message}`;

            // Clear message after 3 seconds if not loading
            if (type !== 'loading') {
                setTimeout(() => {
                    this.updateUI();
                }, 3000);
            }
        }
    }

    downloadCSV() {
        if (this.dpTable.length === 0) {
            this.showMessage('No data to download', 'error');
            return;
        }

        // Create CSV content
        let csv = 'Position,Cost,Segment Type,Previous Position,Length,State\n';

        this.dpTable.forEach(row => {
            const state = row.segment === 'Base' ? 'Base Case' :
                         row.segment === '-' ? 'Not Calculated' :
                         `Optimal ${row.segment} Wave`;

            csv += `${row.position},${row.cost},${row.segment},${row.prev},${row.length},${state}\n`;
        });

        // Create download link
        const blob = new Blob([csv], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `dp_table_${new Date().toISOString().split('T')[0]}.csv`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        this.showMessage('CSV downloaded successfully', 'success');
    }

    // Method to be called from main application
    loadFromAnalysisResults(results) {
        if (results && results.dp_table && results.dp_table.full_table) {
            this.dpTable = results.dp_table.full_table;
            this.totalPages = Math.ceil(this.dpTable.length / this.pageSize);
            this.currentPage = 1;

            this.updateUI();
            this.renderCurrentPage();

            this.showMessage(`DP table loaded from analysis results`, 'success');
            return true;
        }
        return false;
    }
}

// Create global instance
window.DPTableViewer = new DPTableViewer();

// Export functions for use in main.js
window.showDPTableFull = function(signal) {
    window.DPTableViewer.loadTable(signal);
};

window.showDPTableFromResults = function(results) {
    window.DPTableViewer.loadFromAnalysisResults(results);
};