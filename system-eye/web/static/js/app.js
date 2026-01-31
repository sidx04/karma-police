// Utility functions
function formatBytes(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function formatNumber(num) {
    return num.toLocaleString();
}

// Chart configurations
const chartColors = {
    primary: 'rgb(102, 126, 234)',
    secondary: 'rgb(118, 75, 162)',
    success: 'rgb(16, 185, 129)',
    warning: 'rgb(245, 158, 11)',
    danger: 'rgb(239, 68, 68)',
};

// History data
const historySize = 60;
const cpuHistory = [];
const memoryHistory = [];
const diskReadHistory = [];
const diskWriteHistory = [];
const networkRxHistory = [];
const networkTxHistory = [];
const timeLabels = [];

// Initialize charts
window.cpuChart = new Chart(document.getElementById('cpuChart'), {
    type: 'line',
    data: {
        labels: timeLabels,
        datasets: [{
            label: 'CPU Usage %',
            data: cpuHistory,
            borderColor: chartColors.primary,
            backgroundColor: 'rgba(102, 126, 234, 0.1)',
            tension: 0.4,
            fill: true
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            y: {
                beginAtZero: true,
                max: 100
            }
        }
    }
});

window.cpuCoreChart = null;

window.memoryChart = new Chart(document.getElementById('memoryChart'), {
    type: 'line',
    data: {
        labels: timeLabels,
        datasets: [{
            label: 'Memory Usage %',
            data: memoryHistory,
            borderColor: chartColors.secondary,
            backgroundColor: 'rgba(118, 75, 162, 0.1)',
            tension: 0.4,
            fill: true
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            y: {
                beginAtZero: true,
                max: 100
            }
        }
    }
});

window.memoryPieChart = null;

window.diskChart = new Chart(document.getElementById('diskChart'), {
    type: 'line',
    data: {
        labels: timeLabels,
        datasets: [
            {
                label: 'Read (MB)',
                data: diskReadHistory,
                borderColor: chartColors.success,
                tension: 0.4
            },
            {
                label: 'Write (MB)',
                data: diskWriteHistory,
                borderColor: chartColors.danger,
                tension: 0.4
            }
        ]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            y: {
                beginAtZero: true
            }
        }
    }
});

window.networkChart = new Chart(document.getElementById('networkChart'), {
    type: 'line',
    data: {
        labels: timeLabels,
        datasets: [
            {
                label: 'RX (MB)',
                data: networkRxHistory,
                borderColor: chartColors.success,
                tension: 0.4
            },
            {
                label: 'TX (MB)',
                data: networkTxHistory,
                borderColor: chartColors.warning,
                tension: 0.4
            }
        ]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            y: {
                beginAtZero: true
            }
        }
    }
});

// Update functions
function updateMetrics(data) {
    const now = new Date().toLocaleTimeString();
    document.getElementById('last-update').textContent = now;

    // Add to history
    timeLabels.push(now);
    cpuHistory.push(data.cpu.overall);
    memoryHistory.push(data.memory.usage_percent);
    diskReadHistory.push(data.disk.total.read_bytes / 1024 / 1024);
    diskWriteHistory.push(data.disk.total.write_bytes / 1024 / 1024);
    networkRxHistory.push(data.network.total.rx_bytes / 1024 / 1024);
    networkTxHistory.push(data.network.total.tx_bytes / 1024 / 1024);

    // Keep only last N samples
    if (timeLabels.length > historySize) {
        timeLabels.shift();
        cpuHistory.shift();
        memoryHistory.shift();
        diskReadHistory.shift();
        diskWriteHistory.shift();
        networkRxHistory.shift();
        networkTxHistory.shift();
    }

    // Update CPU widget - use 3 decimals for high core count systems
    document.getElementById('cpu-overall').textContent = data.cpu.overall.toFixed(3) + '%';
    document.getElementById('cpu-progress').style.width = data.cpu.overall + '%';
    document.getElementById('cpu-subtitle').textContent =
        'Load: ' + data.cpu.load_avg_1.toFixed(2) + ' ' +
        data.cpu.load_avg_5.toFixed(2) + ' ' +
        data.cpu.load_avg_15.toFixed(2);

    // Update CPU details (cores/threads in widget)
    const cpuDetails = document.getElementById('cpu-details');
    if (cpuDetails && data.cpu.per_core) {
        const detailValues = cpuDetails.querySelectorAll('.detail-value');
        if (detailValues.length >= 2) {
            detailValues[0].textContent = data.cpu.per_core.length; // Cores
            detailValues[1].textContent = data.cpu.per_core.length; // Threads (same as cores for now)
        }
    }

    // Update Memory widget
    const memUsedGB = (data.memory.used / 1024 / 1024 / 1024).toFixed(1);
    const memTotalGB = (data.memory.total / 1024 / 1024 / 1024).toFixed(1);
    const memAvailGB = (data.memory.free / 1024 / 1024 / 1024).toFixed(1);
    document.getElementById('memory-overall').textContent = data.memory.usage_percent.toFixed(1) + '%';
    document.getElementById('memory-progress').style.width = data.memory.usage_percent + '%';
    document.getElementById('memory-subtitle').textContent = memUsedGB + ' / ' + memTotalGB + ' GB';

    // Update Memory details (used/available in widget)
    const memoryDetails = document.getElementById('memory-details');
    if (memoryDetails) {
        const detailValues = memoryDetails.querySelectorAll('.detail-value');
        if (detailValues.length >= 2) {
            detailValues[0].textContent = memUsedGB + ' GB'; // Used
            detailValues[1].textContent = memAvailGB + ' GB'; // Available
        }
    }

    // Update Disk widget
    const totalIO = data.disk.total.read_bytes + data.disk.total.write_bytes;
    document.getElementById('disk-overall').textContent = formatBytes(totalIO);
    document.getElementById('disk-read').textContent = formatBytes(data.disk.total.read_bytes);
    document.getElementById('disk-write').textContent = formatBytes(data.disk.total.write_bytes);

    // Update Network widget
    const totalNet = data.network.total.rx_bytes + data.network.total.tx_bytes;
    document.getElementById('network-overall').textContent = formatBytes(totalNet);
    document.getElementById('network-rx').textContent = formatBytes(data.network.total.rx_bytes);
    document.getElementById('network-tx').textContent = formatBytes(data.network.total.tx_bytes);

    // Update GPU widget
    if (data.gpu && data.gpu.devices && data.gpu.devices.length > 0) {
        const gpu = data.gpu.devices[0];
        document.getElementById('gpu-overall').textContent = gpu.utilization + '%';
        document.getElementById('gpu-progress').style.width = gpu.utilization + '%';
        document.getElementById('gpu-subtitle').textContent =
            gpu.temperature + '°C · ' +
            gpu.memory.usage_percent.toFixed(1) + '% mem · ' +
            gpu.power_usage + 'W';

        // Update GPU details (in widget)
        const gpuDetails = document.getElementById('gpu-details');
        if (gpuDetails) {
            const detailValues = gpuDetails.querySelectorAll('.detail-value');
            if (detailValues.length >= 2) {
                detailValues[0].textContent = formatBytes(gpu.memory.used) + ' / ' + formatBytes(gpu.memory.total);
                detailValues[1].textContent = gpu.temperature + '°C';
            }
        }

        // Update GPU detail panel
        document.getElementById('gpu-name').textContent = gpu.name;
        document.getElementById('gpu-util-detail').textContent = gpu.utilization + '%';
        document.getElementById('gpu-mem-used').textContent = formatBytes(gpu.memory.used);
        document.getElementById('gpu-mem-total').textContent = formatBytes(gpu.memory.total);
        document.getElementById('gpu-mem-percent').textContent = gpu.memory.usage_percent.toFixed(1) + '%';
        document.getElementById('gpu-temp-detail').textContent = gpu.temperature + '°C';
        document.getElementById('gpu-power-detail').textContent = gpu.power_usage + 'W';
        document.getElementById('gpu-fan-speed').textContent = gpu.fan_speed + '%';
    }

    // Update CPU detail panel
    if (data.cpu.per_core && data.cpu.per_core.length > 0) {
        if (!window.cpuCoreChart) {
            window.cpuCoreChart = new Chart(document.getElementById('cpuCoreChart'), {
                type: 'bar',
                data: {
                    labels: data.cpu.per_core.map((_, i) => 'Core ' + i),
                    datasets: [{
                        label: 'CPU Usage %',
                        data: data.cpu.per_core,
                        backgroundColor: chartColors.primary
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100
                        }
                    }
                }
            });
        } else {
            window.cpuCoreChart.data.labels = data.cpu.per_core.map((_, i) => 'Core ' + i);
            window.cpuCoreChart.data.datasets[0].data = data.cpu.per_core;
            window.cpuCoreChart.update();
        }

        // Update CPU detail panel metrics
        const cpuPanel = document.getElementById('cpu-detail-panel');
        if (cpuPanel) {
            const metricValues = cpuPanel.querySelectorAll('.metric-value');
            if (metricValues.length >= 5) {
                metricValues[0].textContent = data.cpu.per_core.length; // Physical Cores
                metricValues[1].textContent = data.cpu.per_core.length; // Logical Processors
                metricValues[2].textContent = data.cpu.load_avg_1.toFixed(2); // Load 1m
                metricValues[3].textContent = data.cpu.load_avg_5.toFixed(2); // Load 5m
                metricValues[4].textContent = data.cpu.load_avg_15.toFixed(2); // Load 15m
            }
        }
    }

    // Update memory pie chart
    if (!window.memoryPieChart) {
        window.memoryPieChart = new Chart(document.getElementById('memoryPieChart'), {
            type: 'doughnut',
            data: {
                labels: ['Used', 'Cached', 'Buffers', 'Free'],
                datasets: [{
                    data: [data.memory.used, data.memory.cached, data.memory.buffers, data.memory.free],
                    backgroundColor: [chartColors.primary, chartColors.secondary, chartColors.success, '#e5e7eb']
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false
            }
        });
    } else {
        window.memoryPieChart.data.datasets[0].data = [data.memory.used, data.memory.cached, data.memory.buffers, data.memory.free];
        window.memoryPieChart.update();
    }

    // Update Memory detail panel metrics
    const memoryPanel = document.getElementById('memory-detail-panel');
    if (memoryPanel) {
        const metricValues = memoryPanel.querySelectorAll('.metric-value');
        if (metricValues.length >= 5) {
            metricValues[0].textContent = formatBytes(data.memory.total); // Total Memory
            metricValues[1].textContent = formatBytes(data.memory.used); // Used Memory
            metricValues[2].textContent = formatBytes(data.memory.free); // Available Memory
            metricValues[3].textContent = formatBytes(data.memory.cached); // Cached
            metricValues[4].textContent = formatBytes(data.memory.buffers); // Buffers
        }
    }

    // Update disk table
    if (data.disk.devices) {
        const diskTable = document.getElementById('disk-table').getElementsByTagName('tbody')[0];
        diskTable.innerHTML = '';
        data.disk.devices.slice(0, 10).forEach(device => {
            const row = diskTable.insertRow();
            row.innerHTML =
                '<td>' + device.name + '</td>' +
                '<td>' + formatBytes(device.io.read_bytes) + '</td>' +
                '<td>' + formatBytes(device.io.write_bytes) + '</td>' +
                '<td>' + (device.io.read_latency || 0).toFixed(2) + ' / ' + (device.io.write_latency || 0).toFixed(2) + '</td>';
        });
    }

    // Update network table
    if (data.network.interfaces) {
        const netTable = document.getElementById('network-table').getElementsByTagName('tbody')[0];
        netTable.innerHTML = '';
        data.network.interfaces.forEach(iface => {
            const row = netTable.insertRow();
            const errors = (iface.traffic.rx_errors || 0) + (iface.traffic.tx_errors || 0);
            row.innerHTML =
                '<td>' + iface.name + '</td>' +
                '<td>' + formatBytes(iface.traffic.rx_bytes) + '</td>' +
                '<td>' + formatBytes(iface.traffic.tx_bytes) + '</td>' +
                '<td>' + errors + '</td>';
        });
    }

    // Update process metrics
    if (data.processes) {
        document.getElementById('process-total').textContent = data.processes.total_count || 0;
        document.getElementById('process-running').textContent = data.processes.running_count || 0;
        document.getElementById('process-sleeping').textContent = data.processes.sleeping_count || 0;

        // Top CPU processes
        if (data.processes.top_cpu) {
            const cpuTable = document.getElementById('process-cpu-table').getElementsByTagName('tbody')[0];
            cpuTable.innerHTML = '';
            data.processes.top_cpu.slice(0, 5).forEach(proc => {
                const row = cpuTable.insertRow();
                row.innerHTML =
                    '<td>' + proc.pid + '</td>' +
                    '<td>' + (proc.name || '--') + '</td>' +
                    '<td>' + (proc.cpu_percent || 0).toFixed(2) + '%</td>' +
                    '<td>' + (proc.memory_percent || 0).toFixed(2) + '%</td>' +
                    '<td>' + proc.state + '</td>';
            });
        }

        // Top memory processes
        if (data.processes.top_memory) {
            const memTable = document.getElementById('process-memory-table').getElementsByTagName('tbody')[0];
            memTable.innerHTML = '';
            data.processes.top_memory.slice(0, 5).forEach(proc => {
                const row = memTable.insertRow();
                row.innerHTML =
                    '<td>' + proc.pid + '</td>' +
                    '<td>' + (proc.name || '--') + '</td>' +
                    '<td>' + (proc.memory_percent || 0).toFixed(2) + '%</td>' +
                    '<td>' + formatBytes(proc.memory_rss || 0) + '</td>' +
                    '<td>' + proc.state + '</td>';
            });
        }
    }

    // Update charts
    window.cpuChart.update();
    window.memoryChart.update();
    window.diskChart.update();
    window.networkChart.update();
}

// ML Classification update function
function updateMLClassification() {
    fetch('/api/classification')
        .then(response => response.json())
        .then(data => {
            const workloadElement = document.getElementById('ml-workload');
            const workloadSubtitle = document.getElementById('ml-workload-subtitle');

            if (data.error || data.service_status === 'unavailable') {
                workloadElement.textContent = 'Service Offline';
                workloadSubtitle.textContent = data.error || 'Classification service not running';
                document.getElementById('ml-status').textContent = 'Offline';
                document.getElementById('ml-status-detail').textContent = 'Start classifier service on port 5000';
                document.getElementById('ml-confidence').textContent = '--';
                document.getElementById('ml-confidence-bar').style.width = '0%';
                document.getElementById('ml-timestamp').textContent = '--';
                return;
            }

            workloadElement.textContent = formatWorkloadName(data.workload_type);
            workloadSubtitle.textContent = getWorkloadDescription(data.workload_type);

            const confidence = (data.confidence * 100).toFixed(1);
            document.getElementById('ml-confidence').textContent = confidence + '%';
            document.getElementById('ml-confidence-bar').style.width = confidence + '%';

            document.getElementById('ml-status').textContent = 'Running';
            document.getElementById('ml-status-detail').textContent = 'Classifying every 3 seconds';

            const timestamp = data.timestamp ? new Date(data.timestamp).toLocaleTimeString() : '--';
            document.getElementById('ml-timestamp').textContent = timestamp;
        })
        .catch(error => {
            console.log('ML classification fetch error:', error);
            document.getElementById('ml-status').textContent = 'Error';
            document.getElementById('ml-status-detail').textContent = 'Failed to fetch classification';
        });
}

function formatWorkloadName(workload) {
    if (workload === 'insufficient_confidence') {
        return 'Unable to Classify';
    }
    const names = {
        'transformer_training': 'Transformer Training',
        'transformer_inference': 'Transformer Inference',
        'cnn_training': 'CNN Training',
        'cnn_inference': 'CNN Inference',
        'unknown': 'Unknown Workload'
    };
    return names[workload] || workload;
}

function getWorkloadDescription(workload) {
    if (workload === 'insufficient_confidence') {
        return 'Confidence below 40% or GPU idle';
    }
    const descriptions = {
        'transformer_training': 'High GPU utilization, memory growth pattern',
        'transformer_inference': 'Medium GPU utilization, stable memory',
        'cnn_training': 'Very high GPU utilization, stable memory',
        'cnn_inference': 'High GPU utilization, stable memory',
        'unknown': 'No active ML workload detected'
    };
    return descriptions[workload] || 'Unknown workload pattern';
}

// Connect to SSE stream
const eventSource = new EventSource('/ws');

eventSource.onmessage = function(event) {
    const data = JSON.parse(event.data);
    updateMetrics(data);
};

eventSource.onerror = function(error) {
    console.error('SSE error:', error);
    document.getElementById('last-update').textContent = 'Disconnected';
};

// Initial load
fetch('/api/metrics')
    .then(response => response.json())
    .then(data => updateMetrics(data))
    .catch(error => console.error('Failed to fetch initial metrics:', error));

// Poll for ML classifications every 4 seconds
setInterval(updateMLClassification, 4000);
updateMLClassification();

// Guardian Events functions
function updateGuardianEvents() {
    fetch('/api/guardian/events')
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                console.log('Guardian events not available:', data.error);
                return;
            }

            const events = data.events || [];

            // Count events by type
            const deadlockDetections = events.filter(e => e.type === 'deadlock' && e.event_type === 'detected').length;
            const anomalyDetections = events.filter(e => e.type === 'anomaly' && e.event_type === 'detected').length;
            const healings = events.filter(e => e.event_type === 'healed').length;

            document.getElementById('events-deadlocks').textContent = deadlockDetections;
            document.getElementById('events-anomalies').textContent = anomalyDetections;
            document.getElementById('events-healings').textContent = healings;

            // Update table
            const table = document.getElementById('guardian-events-table').getElementsByTagName('tbody')[0];
            table.innerHTML = '';

            if (events.length === 0) {
                const row = table.insertRow();
                row.innerHTML = '<td colspan="6" class="events-empty"><i class="ph ph-shield-warning"></i><br>No guardian events recorded yet</td>';
                return;
            }

            // Show last 20 events in reverse chronological order
            const recentEvents = events.slice(-20).reverse();
            recentEvents.forEach(event => {
                const row = table.insertRow();

                // Format timestamp
                const timestamp = event.timestamp.split(',')[1]?.trim() || event.timestamp;

                // Type badge with severity dot
                const severityClass = event.severity || 'warning';
                const typeBadge = `<span class="type-badge ${event.type}">${event.type.toUpperCase()}</span>`;

                // Event badge
                const eventBadge = `<span class="event-badge ${event.event_type}">${event.event_type.toUpperCase()}</span>`;

                // Details with severity indicator
                const details = `<span class="severity-dot ${severityClass}"></span><span class="event-details">${event.details || event.description}</span>`;

                // PIDs with styling
                let pidsHtml = '';
                if (event.pids && event.pids.length > 0) {
                    const displayPids = event.pids.slice(0, 5);
                    pidsHtml = '<div class="pid-list">' +
                        displayPids.map(pid => `<span class="pid-item">${pid}</span>`).join('') +
                        (event.pids.length > 5 ? `<span class="pid-item">+${event.pids.length - 5}</span>` : '') +
                        '</div>';
                } else {
                    pidsHtml = '<span class="event-timestamp">--</span>';
                }

                // Resolution with conditional styling
                const resolutionClass = event.resolution ? 'event-resolution' : 'event-resolution none';
                const resolutionText = event.resolution || '--';

                row.innerHTML =
                    `<td><span class="event-timestamp">${timestamp}</span></td>` +
                    `<td>${typeBadge}</td>` +
                    `<td>${eventBadge}</td>` +
                    `<td>${details}</td>` +
                    `<td>${pidsHtml}</td>` +
                    `<td><span class="${resolutionClass}">${resolutionText}</span></td>`;
            });
        })
        .catch(error => {
            console.log('Guardian events fetch error:', error);
        });
}

// Poll for guardian events every 5 seconds
setInterval(updateGuardianEvents, 5000);
updateGuardianEvents();
