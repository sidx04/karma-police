package ui

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"sync"
	"time"

	"github.com/teamovercooked/system-eye/internal/metrics"
)

// WebServer provides a web-based user interface
type WebServer struct {
	collector metrics.MetricsCollector
	port      string
	server    *http.Server

	// WebSocket clients
	clients    map[*WebSocketClient]bool
	clientsMux sync.RWMutex
	broadcast  chan *metrics.SystemMetrics
	ctx        context.Context
	cancel     context.CancelFunc
}

// WebSocketClient represents a connected WebSocket client
type WebSocketClient struct {
	conn     chan []byte
	done     chan struct{}
}

// NewWebServer creates a new web server
func NewWebServer(collector metrics.MetricsCollector, port string) *WebServer {
	ctx, cancel := context.WithCancel(context.Background())
	return &WebServer{
		collector: collector,
		port:      port,
		clients:   make(map[*WebSocketClient]bool),
		broadcast: make(chan *metrics.SystemMetrics, 10),
		ctx:       ctx,
		cancel:    cancel,
	}
}

// Run starts the web server
func (ws *WebServer) Run(ctx context.Context) error {
	mux := http.NewServeMux()

	// Serve static files from web/static directory
	fs := http.FileServer(http.Dir("web/static"))
	mux.Handle("/static/", http.StripPrefix("/static/", fs))

	// Main dashboard page
	mux.HandleFunc("/", ws.handleHome)

	// API endpoints
	mux.HandleFunc("/api/metrics", ws.handleMetrics)
	mux.HandleFunc("/api/metrics/history", ws.handleMetricsHistory)
	mux.HandleFunc("/api/health", ws.handleHealth)
	mux.HandleFunc("/api/system/info", ws.handleSystemInfo)
	mux.HandleFunc("/api/classification", ws.handleClassification)
	mux.HandleFunc("/api/chat", ws.handleChatProxy)
	mux.HandleFunc("/api/guardian/events", ws.handleGuardianEvents)

	// WebSocket endpoint for real-time updates (using SSE)
	mux.HandleFunc("/ws", ws.handleWebSocket)

	ws.server = &http.Server{
		Addr:    ":" + ws.port,
		Handler: mux,
	}

	// Start metrics broadcaster
	go ws.startBroadcaster()

	// Start server in goroutine
	go func() {
		log.Printf("Web server starting on port %s", ws.port)
		log.Printf("Dashboard available at http://localhost:%s", ws.port)
		if err := ws.server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Printf("Web server error: %v", err)
		}
	}()

	// Wait for context cancellation
	<-ctx.Done()

	// Graceful shutdown
	log.Println("Web server shutting down...")
	ws.cancel()
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	return ws.server.Shutdown(shutdownCtx)
}

// startBroadcaster continuously collects metrics and broadcasts to clients
func (ws *WebServer) startBroadcaster() {
	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ws.ctx.Done():
			return
		case <-ticker.C:
			metrics, err := ws.collector.Collect()
			if err != nil {
				log.Printf("Error collecting metrics: %v", err)
				continue
			}

			// Broadcast to all connected clients
			ws.clientsMux.RLock()
			if len(ws.clients) > 0 {
				data, err := json.Marshal(metrics)
				if err == nil {
					for client := range ws.clients {
						select {
						case client.conn <- data:
						default:
							// Client channel full, skip
						}
					}
				}
			}
			ws.clientsMux.RUnlock()
		}
	}
}

// handleHome serves the main dashboard page
func (ws *WebServer) handleHome(w http.ResponseWriter, r *http.Request) {
	http.ServeFile(w, r, "web/templates/index.html")
}

// handleMetrics serves current metrics as JSON
func (ws *WebServer) handleMetrics(w http.ResponseWriter, r *http.Request) {
	metrics, err := ws.collector.Collect()
	if err != nil {
		http.Error(w, fmt.Sprintf("Failed to collect metrics: %v", err), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*") // Enable CORS for development

	// Use JSON encoder for proper serialization including GPU metrics
	if err := json.NewEncoder(w).Encode(metrics); err != nil {
		http.Error(w, fmt.Sprintf("Failed to encode metrics: %v", err), http.StatusInternalServerError)
	}
}

// handleHealth serves health check
func (ws *WebServer) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	fmt.Fprint(w, `{"status": "healthy", "service": "system-eye"}`)
}

// handleSystemInfo serves system information
func (ws *WebServer) handleSystemInfo(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")

	// Get system info from collector if it implements the method
	info := map[string]interface{}{
		"service": "system-eye",
		"version": "1.0.0",
	}

	json.NewEncoder(w).Encode(info)
}

// handleMetricsHistory serves historical metrics
func (ws *WebServer) handleMetricsHistory(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")

	// For now, return empty history - could be enhanced with actual history tracking
	fmt.Fprint(w, `{"samples": []}`)
}

// handleWebSocket handles WebSocket connections for real-time updates
func (ws *WebServer) handleWebSocket(w http.ResponseWriter, r *http.Request) {
	// Simple SSE (Server-Sent Events) implementation as we don't have a WebSocket library
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("Access-Control-Allow-Origin", "*")

	// Create new client
	client := &WebSocketClient{
		conn: make(chan []byte, 10),
		done: make(chan struct{}),
	}

	// Register client
	ws.clientsMux.Lock()
	ws.clients[client] = true
	ws.clientsMux.Unlock()

	// Cleanup on disconnect
	defer func() {
		ws.clientsMux.Lock()
		delete(ws.clients, client)
		ws.clientsMux.Unlock()
		close(client.done)
	}()

	// Send metrics to client
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "Streaming not supported", http.StatusInternalServerError)
		return
	}

	for {
		select {
		case <-r.Context().Done():
			return
		case <-client.done:
			return
		case data := <-client.conn:
			fmt.Fprintf(w, "data: %s\n\n", data)
			flusher.Flush()
		}
	}
}

// handleClassification proxies requests to the classification service
func (ws *WebServer) handleClassification(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")

	// Try to fetch from classification service (running on port 5000)
	classifierURL := "http://system-brain:5000/api/classification"

	client := &http.Client{Timeout: 2 * time.Second}
	resp, err := client.Get(classifierURL)

	if err != nil {
		// Classification service not available
		fmt.Fprintf(w, `{
  "service_status": "unavailable",
  "error": "Classification service not running",
  "workload_type": "unknown",
  "confidence": 0.0
}`)
		return
	}
	defer resp.Body.Close()

	// Forward the response
	w.WriteHeader(resp.StatusCode)
	fmt.Fprint(w, "\n")
	// Copy response body
	buffer := make([]byte, 4096)
	for {
		n, err := resp.Body.Read(buffer)
		if n > 0 {
			w.Write(buffer[:n])
		}
		if err != nil {
			break
		}
	}
}

// handleChatProxy proxies chat requests to the Node.js chat server
func (ws *WebServer) handleChatProxy(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")

	// Only allow POST requests
	if r.Method != http.MethodPost {
		http.Error(w, `{"error":"Method not allowed"}`, http.StatusMethodNotAllowed)
		return
	}

	// Proxy request to Node.js chat server on port 3000
	chatURL := "http://chat-server:3000/api/chat"

	// Read request body
	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, `{"error":"Failed to read request"}`, http.StatusBadRequest)
		return
	}
	defer r.Body.Close()

	// Create new request to chat server
	chatReq, err := http.NewRequest("POST", chatURL, bytes.NewReader(body))
	if err != nil {
		http.Error(w, `{"error":"Failed to create request"}`, http.StatusInternalServerError)
		return
	}

	// Copy headers
	chatReq.Header.Set("Content-Type", "application/json")

	// Send request to chat server
	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(chatReq)
	if err != nil {
		log.Printf("Chat proxy error: %v", err)
		http.Error(w, `{"error":"Chat service unavailable"}`, http.StatusServiceUnavailable)
		return
	}
	defer resp.Body.Close()

	// Forward response
	w.WriteHeader(resp.StatusCode)
	io.Copy(w, resp.Body)
}

// Guardian Event represents a detection or healing event
type GuardianEvent struct {
	Timestamp   string   `json:"timestamp"`
	Type        string   `json:"type"` // "deadlock" or "anomaly"
	EventType   string   `json:"event_type"` // "detected" or "healed"
	Severity    string   `json:"severity"`
	Description string   `json:"description"`
	PIDs        []int    `json:"pids"`
	Details     string   `json:"details"`
	Resolution  string   `json:"resolution,omitempty"`
}

// handleGuardianEvents fetches recent deadlock and anomaly events from guardian log
func (ws *WebServer) handleGuardianEvents(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")

	// Read only last 2000 lines of guardian log (more efficient than reading 220MB file)
	logPath := "/tmp/system-guardian.log"

	// Check if log file exists
	if _, err := os.Stat(logPath); os.IsNotExist(err) {
		json.NewEncoder(w).Encode(map[string]interface{}{
			"events": []GuardianEvent{},
			"error": "Guardian log not available",
		})
		return
	}

	// Use tail to read last 2000 lines instead of entire file
	cmd := exec.Command("tail", "-n", "2000", logPath)
	output, err := cmd.Output()
	if err != nil {
		log.Printf("Error reading guardian log: %v", err)
		json.NewEncoder(w).Encode(map[string]interface{}{
			"events": []GuardianEvent{},
			"error": "Failed to read guardian log",
		})
		return
	}

	// Parse events from log
	events := parseGuardianEvents(string(output))

	// Return last 100 events
	maxEvents := 100
	if len(events) > maxEvents {
		events = events[len(events)-maxEvents:]
	}

	json.NewEncoder(w).Encode(map[string]interface{}{
		"events": events,
		"count":  len(events),
	})
}

// parseGuardianEvents extracts events from guardian log content
func parseGuardianEvents(content string) []GuardianEvent {
	var events []GuardianEvent
	lines := bytes.Split([]byte(content), []byte("\n"))

	for _, line := range lines {
		if len(line) == 0 {
			continue
		}
		lineStr := string(line)

		// Parse deadlock detections
		if bytes.Contains(line, []byte("[DEADLOCK_DETECTED]")) {
			event := GuardianEvent{
				Type:      "deadlock",
				EventType: "detected",
				Severity:  "critical",
			}

			// Extract timestamp (first part before first " - ")
			parts := bytes.SplitN(line, []byte(" - "), 2)
			if len(parts) >= 1 {
				event.Timestamp = string(parts[0])
			}

			// Extract description after the marker
			markerIdx := bytes.Index(line, []byte("[DEADLOCK_DETECTED]"))
			if markerIdx >= 0 {
				afterMarker := line[markerIdx+len("[DEADLOCK_DETECTED]"):]
				afterMarker = bytes.TrimSpace(afterMarker)
				// Get everything up to next " - " if exists, otherwise all
				descParts := bytes.SplitN(afterMarker, []byte(" - "), 2)
				event.Description = string(descParts[0])
			}

			event.PIDs = extractPIDs(lineStr)
			events = append(events, event)
		}

		// Parse deadlock healings
		if bytes.Contains(line, []byte("[DEADLOCK_HEALED]")) {
			event := GuardianEvent{
				Type:      "deadlock",
				EventType: "healed",
				Severity:  "info",
				Details:   "Deadlock Resolved",
			}

			parts := bytes.SplitN(line, []byte(" - "), 2)
			if len(parts) >= 1 {
				event.Timestamp = string(parts[0])
			}

			// Extract description after marker
			markerIdx := bytes.Index(line, []byte("[DEADLOCK_HEALED]"))
			if markerIdx >= 0 {
				afterMarker := line[markerIdx+len("[DEADLOCK_HEALED]"):]
				afterMarker = bytes.TrimSpace(afterMarker)
				descParts := bytes.SplitN(afterMarker, []byte(" - "), 2)
				event.Description = string(descParts[0])
			}

			// Extract resolution action
			if bytes.Contains(line, []byte("kill_process")) {
				event.Resolution = "Killed victim process"
			} else if bytes.Contains(line, []byte("log_only")) {
				event.Resolution = "Logged only"
			} else if bytes.Contains(line, []byte("actions successful")) {
				event.Resolution = "Healing actions applied"
			}

			event.PIDs = extractPIDs(lineStr)
			events = append(events, event)
		}

		// Parse anomaly detections
		if bytes.Contains(line, []byte("[ANOMALY_DETECTED]")) {
			event := GuardianEvent{
				Type:      "anomaly",
				EventType: "detected",
				Severity:  "warning",
			}

			parts := bytes.SplitN(line, []byte(" - "), 2)
			if len(parts) >= 1 {
				event.Timestamp = string(parts[0])
			}

			// Determine anomaly type
			if bytes.Contains(line, []byte("CPU_SPIKE")) || bytes.Contains(line, []byte("cpu_spike")) {
				event.Details = "CPU Spike"
			} else if bytes.Contains(line, []byte("MEMORY_LEAK")) || bytes.Contains(line, []byte("memory_leak")) {
				event.Details = "Memory Leak"
			} else if bytes.Contains(line, []byte("IO_WAIT")) {
				event.Details = "High I/O Wait"
			}

			// Extract description after marker
			markerIdx := bytes.Index(line, []byte("[ANOMALY_DETECTED]"))
			if markerIdx >= 0 {
				afterMarker := line[markerIdx+len("[ANOMALY_DETECTED]"):]
				afterMarker = bytes.TrimSpace(afterMarker)
				// Get first part before " - "
				descParts := bytes.SplitN(afterMarker, []byte(" - "), 2)
				event.Description = string(descParts[0])
			}

			event.PIDs = extractPIDs(lineStr)
			events = append(events, event)
		}

		// Parse anomaly healings
		if bytes.Contains(line, []byte("[ANOMALY_HEALED]")) {
			event := GuardianEvent{
				Type:      "anomaly",
				EventType: "healed",
				Severity:  "info",
			}

			parts := bytes.SplitN(line, []byte(" - "), 2)
			if len(parts) >= 1 {
				event.Timestamp = string(parts[0])
			}

			// Determine anomaly type
			if bytes.Contains(line, []byte("CPU_SPIKE")) {
				event.Details = "CPU Spike Resolved"
			} else if bytes.Contains(line, []byte("MEMORY_LEAK")) {
				event.Details = "Memory Leak Resolved"
			} else {
				event.Details = "Anomaly Resolved"
			}

			// Extract resolution action
			if bytes.Contains(line, []byte("nice_process")) {
				event.Resolution = "Reduced process priority"
			} else if bytes.Contains(line, []byte("kill_process")) {
				event.Resolution = "Killed processes"
			} else if bytes.Contains(line, []byte("restart_process")) {
				event.Resolution = "Restarted processes"
			} else if bytes.Contains(line, []byte("0/0 actions successful")) {
				event.Resolution = "Resolved automatically"
			} else if bytes.Contains(line, []byte("actions successful")) {
				event.Resolution = "Mitigation actions applied"
			}

			// Extract description after marker
			markerIdx := bytes.Index(line, []byte("[ANOMALY_HEALED]"))
			if markerIdx >= 0 {
				afterMarker := line[markerIdx+len("[ANOMALY_HEALED]"):]
				afterMarker = bytes.TrimSpace(afterMarker)
				descParts := bytes.SplitN(afterMarker, []byte(" - "), 2)
				event.Description = string(descParts[0])
			}

			event.PIDs = extractPIDs(lineStr)
			events = append(events, event)
		}
	}

	return events
}

// extractPIDs extracts PID numbers from log line
func extractPIDs(line string) []int {
	var pids []int

	// Look for array pattern after keywords like "PIDs:" or "processes:" or "affecting PIDs"
	// Common patterns: "Affected PIDs: [123, 456]", "processes: [123, 456]", "PIDs [123, 456]"
	pidKeywords := []string{"processes:", "PIDs:", "PIDs", "affecting PIDs"}

	for _, keyword := range pidKeywords {
		keywordIdx := bytes.Index([]byte(line), []byte(keyword))
		if keywordIdx >= 0 {
			// Look for [ and ] after the keyword
			afterKeyword := line[keywordIdx:]
			startIdx := bytes.Index([]byte(afterKeyword), []byte("["))
			endIdx := bytes.Index([]byte(afterKeyword), []byte("]"))

			if startIdx >= 0 && endIdx > startIdx {
				// Extract the content between [ and ]
				arrayContent := afterKeyword[startIdx+1 : endIdx]
				// Split by comma and parse each number
				numStrs := bytes.Split([]byte(arrayContent), []byte(","))
				for _, numStr := range numStrs {
					numStr = bytes.TrimSpace(numStr)
					var pid int
					if _, err := fmt.Sscanf(string(numStr), "%d", &pid); err == nil {
						pids = append(pids, pid)
					}
				}

				// Found PIDs, return them
				if len(pids) > 0 {
					return pids
				}
			}
		}
	}

	// Also look for single PID pattern like "PID 123"
	if len(pids) == 0 {
		parts := bytes.Fields([]byte(line))
		for i, part := range parts {
			partStr := string(part)
			if (partStr == "PID" || partStr == "PID:") && i+1 < len(parts) {
				var pid int
				if _, err := fmt.Sscanf(string(parts[i+1]), "%d", &pid); err == nil {
					pids = append(pids, pid)
				}
			}
		}
	}

	return pids
}
