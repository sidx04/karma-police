class ChatPanel {
    constructor() {
        this.selectedGraphs = new Map();
        this.isOpen = false;
        this.apiEndpoint = '/api/chat';
        this.conversationHistory = [];
        this.init();
    }

    init() {
        this.attachEventListeners();
        this.observeWidgets();
    }

    attachEventListeners() {
        const toggle = document.getElementById('chat-toggle');
        const close = document.getElementById('chat-close');
        const sendBtn = document.getElementById('chat-send');
        const input = document.getElementById('chat-input');

        toggle.addEventListener('click', () => this.togglePanel());
        close.addEventListener('click', () => this.closePanel());
        sendBtn.addEventListener('click', () => this.sendMessage());
        
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        input.addEventListener('input', (e) => {
            e.target.style.height = 'auto';
            e.target.style.height = Math.min(e.target.scrollHeight, 150) + 'px';
        });
    }

    observeWidgets() {
        const widgets = document.querySelectorAll('.widget, .card');
        widgets.forEach((widget, index) => {
            const header = widget.querySelector('.widget-header, .card-header');
            if (header && !widget.querySelector('.graph-checkbox')) {
                const checkbox = document.createElement('input');
                checkbox.type = 'checkbox';
                checkbox.className = 'graph-checkbox';
                checkbox.dataset.graphId = `graph-${index}`;
                checkbox.dataset.graphName = header.textContent.trim();
                
                const checkboxWrapper = document.createElement('div');
                checkboxWrapper.className = 'checkbox-wrapper';
                checkboxWrapper.appendChild(checkbox);
                
                header.style.position = 'relative';
                header.insertBefore(checkboxWrapper, header.firstChild);
                
                checkbox.addEventListener('change', (e) => {
                    this.handleGraphSelection(e.target, widget);
                });
            }
        });
    }

    handleGraphSelection(checkbox, widget) {
        const graphId = checkbox.dataset.graphId;
        const graphName = checkbox.dataset.graphName;

        if (checkbox.checked) {
            const graphData = this.collectGraphData(widget, graphName);
            this.selectedGraphs.set(graphId, graphData);
        } else {
            this.selectedGraphs.delete(graphId);
        }

        this.updateSelectedGraphsDisplay();
    }

    collectGraphData(widget, graphName) {
        const data = {
            name: graphName,
            timestamp: new Date().toISOString(),
            metrics: {}
        };

        const value = widget.querySelector('.widget-value');
        if (value) {
            data.metrics.current = value.textContent.trim();
        }

        const subtitle = widget.querySelector('.widget-subtitle');
        if (subtitle) {
            data.metrics.subtitle = subtitle.textContent.trim();
        }

        const detailItems = widget.querySelectorAll('.detail-item');
        detailItems.forEach(item => {
            const label = item.querySelector('.detail-label');
            const value = item.querySelector('.detail-value');
            if (label && value) {
                data.metrics[label.textContent.trim()] = value.textContent.trim();
            }
        });

        const metricItems = widget.querySelectorAll('.metric-item');
        metricItems.forEach(item => {
            const label = item.querySelector('.metric-label');
            const value = item.querySelector('.metric-value');
            if (label && value) {
                data.metrics[label.textContent.trim()] = value.textContent.trim();
            }
        });

        const table = widget.querySelector('table');
        if (table) {
            const rows = Array.from(table.querySelectorAll('tbody tr'));
            data.metrics.tableData = rows.slice(0, 5).map(row => {
                const cells = Array.from(row.querySelectorAll('td'));
                return cells.map(cell => cell.textContent.trim());
            });
        }

        const detailPanel = widget.closest('.detail-panel');
        if (detailPanel && detailPanel.classList.contains('active')) {
            const canvases = detailPanel.querySelectorAll('canvas');
            canvases.forEach(canvas => {
                const chartId = canvas.id;
                if (window[chartId]) {
                    const chart = window[chartId];
                    if (chart.data && chart.data.datasets) {
                        data.metrics.charts = data.metrics.charts || {};
                        data.metrics.charts[chartId] = {
                            labels: chart.data.labels?.slice(-10) || [],
                            datasets: chart.data.datasets.map(ds => ({
                                label: ds.label,
                                data: ds.data?.slice(-10) || []
                            }))
                        };
                    }
                }
            });
        }

        return data;
    }

    updateSelectedGraphsDisplay() {
        const container = document.getElementById('selected-graphs');
        container.innerHTML = '';

        if (this.selectedGraphs.size === 0) {
            return;
        }

        this.selectedGraphs.forEach((data, id) => {
            const chip = document.createElement('div');
            chip.className = 'graph-chip';
            chip.innerHTML = `
                <i class="ph ph-chart-line"></i>
                <span>${data.name}</span>
                <button class="remove-chip" data-graph-id="${id}">
                    <i class="ph ph-x"></i>
                </button>
            `;
            
            chip.querySelector('.remove-chip').addEventListener('click', (e) => {
                e.stopPropagation();
                this.removeGraph(id);
            });
            
            container.appendChild(chip);
        });
    }

    removeGraph(graphId) {
        this.selectedGraphs.delete(graphId);
        const checkbox = document.querySelector(`[data-graph-id="${graphId}"]`);
        if (checkbox) {
            checkbox.checked = false;
        }
        this.updateSelectedGraphsDisplay();
    }

    clearAllGraphs() {
        // Uncheck all checkboxes
        this.selectedGraphs.forEach((data, graphId) => {
            const checkbox = document.querySelector(`[data-graph-id="${graphId}"]`);
            if (checkbox) {
                checkbox.checked = false;
            }
        });
        
        // Clear the map and update display
        this.selectedGraphs.clear();
        this.updateSelectedGraphsDisplay();
    }

    async sendMessage() {
        const input = document.getElementById('chat-input');
        const userMessage = input.value.trim();

        if (!userMessage) return;

        this.addMessage(userMessage, 'user');
        input.value = '';
        input.style.height = 'auto';

        let graphContext = '';
        if (this.selectedGraphs.size > 0) {
            graphContext = '\n\nSelected Graph Data:\n';
            Array.from(this.selectedGraphs.values()).forEach(graph => {
                graphContext += `\n${graph.name}:\n`;
                Object.entries(graph.metrics).forEach(([key, value]) => {
                    if (key !== 'tableData') {
                        graphContext += `  - ${key}: ${value}\n`;
                    } else if (Array.isArray(value) && value.length > 0) {
                        graphContext += `  - Table Data:\n`;
                        value.forEach(row => {
                            graphContext += `    ${row.join(' | ')}\n`;
                        });
                    }
                });
            });
        }

        this.conversationHistory.push({
            role: 'user',
            content: userMessage + graphContext
        });

        const loadingId = this.addMessage('Analyzing...', 'assistant', true);

        try {
            const response = await this.callAI(userMessage, graphContext);
            
            this.conversationHistory.push({
                role: 'assistant',
                content: response
            });
            
            this.removeMessage(loadingId);
            this.addMessage(response, 'assistant');
            
            // Clear selected graphs after sending
            this.clearAllGraphs();
        } catch (error) {
            this.removeMessage(loadingId);
            this.addMessage('Sorry, I encountered an error analyzing the data. Please try again.', 'assistant', false, true);
            console.error('Chat error:', error);
        }
    }

    async callAI(userMessage, graphContext) {
        const response = await fetch(this.apiEndpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: userMessage,
                graphContext: graphContext,
                conversationHistory: this.conversationHistory.slice(-10)
            })
        });
        
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.error || 'API request failed');
        }
        
        const data = await response.json();
        return data.response;
    }

    addMessage(content, role = 'user', isLoading = false, isError = false) {
        const messagesContainer = document.getElementById('chat-messages');
        const messageId = `msg-${Date.now()}`;
        
        const messageDiv = document.createElement('div');
        messageDiv.className = `chat-message ${role}${isLoading ? ' loading' : ''}${isError ? ' error' : ''}`;
        messageDiv.id = messageId;
        
        if (isLoading) {
            messageDiv.innerHTML = `
                <div class="message-content">
                    <div class="loading-dots">
                        <span></span><span></span><span></span>
                    </div>
                </div>
            `;
        } else {
            messageDiv.innerHTML = `
                <div class="message-content">${this.formatMessage(content)}</div>
            `;
        }
        
        messagesContainer.appendChild(messageDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
        
        return messageId;
    }

    removeMessage(messageId) {
        const message = document.getElementById(messageId);
        if (message) {
            message.remove();
        }
    }

    formatMessage(content) {
        return content
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\n/g, '<br>');
    }

    togglePanel() {
        this.isOpen = !this.isOpen;
        const sidebar = document.getElementById('chat-sidebar');
        
        if (this.isOpen) {
            sidebar.classList.add('open');
        } else {
            sidebar.classList.remove('open');
        }
    }

    closePanel() {
        this.isOpen = false;
        const sidebar = document.getElementById('chat-sidebar');
        sidebar.classList.remove('open');
    }
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        window.chatPanel = new ChatPanel();
    });
} else {
    window.chatPanel = new ChatPanel();
}

// Export for module usage
export default ChatPanel;
