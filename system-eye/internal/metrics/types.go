package metrics

import (
	"time"
)

// SystemMetrics represents a complete snapshot of system metrics
type SystemMetrics struct {
	Timestamp time.Time       `json:"timestamp"`
	CPU       CPUMetrics      `json:"cpu"`
	Memory    MemoryMetrics   `json:"memory"`
	Disk      DiskMetrics     `json:"disk"`
	Network   NetworkMetrics  `json:"network"`
	GPU       *GPUMetrics     `json:"gpu,omitempty"`       // Optional - only if GPU available
	Processes *ProcessMetrics `json:"processes,omitempty"` // Optional - includes deadlock detection fields
	// NOTE: Deadlock detection (cycle detection) is performed in Python (system-guardian)
	// ProcessInfo contains all necessary raw fields for deadlock detection
}

// CPUMetrics contains CPU usage information
type CPUMetrics struct {
	Overall         float64   `json:"overall"`     // Overall CPU usage percentage
	PerCore         []float64 `json:"per_core"`    // Per-core usage percentages
	LoadAvg1        float64   `json:"load_avg_1"`  // 1-minute load average
	LoadAvg5        float64   `json:"load_avg_5"`  // 5-minute load average
	LoadAvg15       float64   `json:"load_avg_15"` // 15-minute load average
	ContextSwitches uint64    `json:"context_switches"`
	Interrupts      uint64    `json:"interrupts"`
}

// MemoryMetrics contains memory usage information
type MemoryMetrics struct {
	Total        uint64  `json:"total"`         // Total memory in bytes
	Used         uint64  `json:"used"`          // Used memory in bytes
	Free         uint64  `json:"free"`          // Free memory in bytes
	Available    uint64  `json:"available"`     // Available memory in bytes
	Cached       uint64  `json:"cached"`        // Cached memory in bytes
	Buffers      uint64  `json:"buffers"`       // Buffer memory in bytes
	UsagePercent float64 `json:"usage_percent"` // Memory usage as percentage
}

// DiskMetrics contains disk I/O information
type DiskMetrics struct {
	Devices []DiskDevice `json:"devices"`
	Total   DiskIO       `json:"total"` // Aggregated across all devices
}

// DiskDevice represents a single disk device
type DiskDevice struct {
	Name string `json:"name"` // Device name (e.g., "sda", "nvme0n1")
	IO   DiskIO `json:"io"`   // I/O statistics
}

// DiskIO contains I/O statistics
type DiskIO struct {
	ReadBytes    uint64  `json:"read_bytes"`              // Bytes read
	WriteBytes   uint64  `json:"write_bytes"`             // Bytes written
	ReadOps      uint64  `json:"read_ops"`                // Read operations
	WriteOps     uint64  `json:"write_ops"`               // Write operations
	ReadLatency  float64 `json:"read_latency,omitempty"`  // Average read latency (ms)
	WriteLatency float64 `json:"write_latency,omitempty"` // Average write latency (ms)
}

// NetworkMetrics contains network interface information
type NetworkMetrics struct {
	Interfaces []NetworkInterface `json:"interfaces"`
	Total      NetworkTraffic     `json:"total"` // Aggregated across all interfaces
}

// NetworkInterface represents a single network interface
type NetworkInterface struct {
	Name    string         `json:"name"`    // Interface name (e.g., "eth0", "lo")
	Traffic NetworkTraffic `json:"traffic"` // Traffic statistics
}

// NetworkTraffic contains network traffic statistics
type NetworkTraffic struct {
	RxBytes   uint64 `json:"rx_bytes"`   // Bytes received
	TxBytes   uint64 `json:"tx_bytes"`   // Bytes transmitted
	RxPackets uint64 `json:"rx_packets"` // Packets received
	TxPackets uint64 `json:"tx_packets"` // Packets transmitted
	RxErrors  uint64 `json:"rx_errors"`  // Receive errors
	TxErrors  uint64 `json:"tx_errors"`  // Transmit errors
	RxDropped uint64 `json:"rx_dropped"` // Received packets dropped
	TxDropped uint64 `json:"tx_dropped"` // Transmitted packets dropped
}

// GPUMetrics contains GPU information (NVIDIA via NVML)
type GPUMetrics struct {
	Devices []GPUDevice `json:"devices"`
}

// GPUDevice represents a single GPU device
type GPUDevice struct {
	Index       int       `json:"index"`       // GPU index
	Name        string    `json:"name"`        // GPU name/model
	UUID        string    `json:"uuid"`        // GPU UUID
	Utilization int       `json:"utilization"` // GPU utilization percentage
	Memory      GPUMemory `json:"memory"`      // Memory information
	Temperature int       `json:"temperature"` // GPU temperature (°C)
	PowerUsage  int       `json:"power_usage"` // Power usage (watts)
	FanSpeed    int       `json:"fan_speed"`   // Fan speed percentage
}

// GPUMemory contains GPU memory information
type GPUMemory struct {
	Total        uint64  `json:"total"`         // Total memory in bytes
	Used         uint64  `json:"used"`          // Used memory in bytes
	Free         uint64  `json:"free"`          // Free memory in bytes
	UsagePercent float64 `json:"usage_percent"` // Memory usage percentage
}

// MetricsCollector interface defines the contract for metric collection
type MetricsCollector interface {
	Collect() (*SystemMetrics, error)
	Close() error
}

// CollectorConfig holds configuration for metrics collection
type CollectorConfig struct {
	EnableEBPF              bool          `json:"enable_ebpf"`               // Enable eBPF monitoring
	EnableGPU               bool          `json:"enable_gpu"`                // Enable GPU monitoring
	EnableDeadlockDetection bool          `json:"enable_deadlock_detection"` // Enable deadlock detection
	SampleRate              time.Duration `json:"sample_rate"`               // How often to collect metrics
	HistorySize             int           `json:"history_size"`              // Number of historical samples to keep
}

// MetricsHistory maintains a rolling window of metrics
type MetricsHistory struct {
	Samples []SystemMetrics `json:"samples"`
	MaxSize int             `json:"max_size"`
}

// Add appends a new metric sample and maintains the rolling window
func (h *MetricsHistory) Add(metrics SystemMetrics) {
	h.Samples = append(h.Samples, metrics)
	if len(h.Samples) > h.MaxSize {
		h.Samples = h.Samples[1:] // Remove oldest sample
	}
}

// Latest returns the most recent metrics sample
func (h *MetricsHistory) Latest() *SystemMetrics {
	if len(h.Samples) == 0 {
		return nil
	}
	return &h.Samples[len(h.Samples)-1]
}

// GetRange returns metrics samples within a time range
func (h *MetricsHistory) GetRange(start, end time.Time) []SystemMetrics {
	var result []SystemMetrics
	for _, sample := range h.Samples {
		if sample.Timestamp.After(start) && sample.Timestamp.Before(end) {
			result = append(result, sample)
		}
	}
	return result
}

// ProcessMetrics contains per-process monitoring data
type ProcessMetrics struct {
	Processes []ProcessInfo `json:"processes"`
	TopCPU    []ProcessInfo `json:"top_cpu"`           // Top CPU consuming processes
	TopMemory []ProcessInfo `json:"top_memory"`        // Top memory consuming processes
	TopGPU    []ProcessInfo `json:"top_gpu,omitempty"` // Top GPU using processes
	Total     int           `json:"total_count"`
	Running   int           `json:"running_count"`
	Sleeping  int           `json:"sleeping_count"`
}

// ProcessInfo contains information about a single process
type ProcessInfo struct {
	PID             uint32  `json:"pid"`
	PPID            uint32  `json:"ppid"`
	Name            string  `json:"name"`
	Command         string  `json:"command,omitempty"`
	State           string  `json:"state"`
	CPUPercent      float64 `json:"cpu_percent"`
	MemoryPercent   float64 `json:"memory_percent"`
	MemoryRSS       uint64  `json:"memory_rss"`     // Resident Set Size in bytes
	MemoryVirtual   uint64  `json:"memory_virtual"` // Virtual memory size in bytes
	NumThreads      uint32  `json:"num_threads"`
	ReadBytes       uint64  `json:"read_bytes"`
	WriteBytes      uint64  `json:"write_bytes"`
	OpenFiles       uint32  `json:"open_files,omitempty"`
	SyscallCount    uint64  `json:"syscall_count,omitempty"`
	ContextSwitches uint64  `json:"context_switches,omitempty"`
	GPUMemory       uint64  `json:"gpu_memory,omitempty"`      // GPU memory usage if applicable
	GPUUtilization  float64 `json:"gpu_utilization,omitempty"` // GPU utilization if applicable

	// Deadlock detection fields
	WaitChannel            string     `json:"wait_channel,omitempty"`             // Kernel wait channel (wchan)
	WaitState              string     `json:"wait_state,omitempty"`               // D, S, R, Z, T
	WaitTimeMs             uint64     `json:"wait_time_ms,omitempty"`             // Time in current state
	HeldLocks              []FileLock `json:"held_locks,omitempty"`               // Locks held by this process
	WaitingLocks           []FileLock `json:"waiting_locks,omitempty"`            // Locks being waited for
	BlockedThreads         uint32     `json:"blocked_threads,omitempty"`          // Number of blocked threads
	VoluntaryCtxSwitches   uint64     `json:"voluntary_ctx_switches,omitempty"`   // Voluntary context switches
	InvoluntaryCtxSwitches uint64     `json:"involuntary_ctx_switches,omitempty"` // Involuntary context switches
	WaitingOnPID           uint32     `json:"waiting_on_pid,omitempty"`           // PID this process is blocked by
	BlockingPIDs           []uint32   `json:"blocking_pids,omitempty"`            // PIDs blocked by this process
	CurrentSyscall         string     `json:"current_syscall,omitempty"`          // Current system call
	SyscallDurationMs      uint64     `json:"syscall_duration_ms,omitempty"`      // Syscall execution time
	IOWaitTimeMs           uint64     `json:"io_wait_time_ms,omitempty"`          // Time waiting for I/O
	CPUTimeDelta           uint64     `json:"cpu_time_delta,omitempty"`           // CPU time change since last sample
}

// FileLock represents a file lock held or waited for by a process
type FileLock struct {
	Path     string `json:"path"`      // File path
	LockType string `json:"lock_type"` // READ, WRITE, ADVISORY, MANDATORY
	Inode    uint64 `json:"inode"`     // File inode
	PID      uint32 `json:"pid"`       // Process ID
}

// ============================================================================
// Deadlock Detection Types (Used by Python - system-guardian)
// ============================================================================
// NOTE: These types are kept for compatibility with protobuf definitions and
// Python code. Go (system-eye) only collects RAW metrics in ProcessInfo fields.
// Python (system-guardian) performs actual deadlock detection (cycle detection).
// ============================================================================

// DeadlockInfo contains deadlock detection results (computed in Python)
type DeadlockInfo struct {
	Detected     bool              `json:"detected"`      // Whether a deadlock was detected
	InvolvedPIDs []uint32          `json:"involved_pids"` // PIDs in the deadlock cycle
	DeadlockType string            `json:"deadlock_type"` // file_lock, futex, pipe, socket, unknown
	Description  string            `json:"description"`   // Human-readable explanation
	WaitGraph    []ProcessWaitEdge `json:"wait_graph"`    // Dependency graph
	DetectedAt   time.Time         `json:"detected_at"`   // When deadlock was first detected
	DurationMs   uint64            `json:"duration_ms"`   // How long deadlock has persisted
}

// ProcessWaitEdge represents a wait dependency between processes
type ProcessWaitEdge struct {
	FromPID      uint32 `json:"from_pid"`      // Process waiting
	ToPID        uint32 `json:"to_pid"`        // Process being waited on
	Resource     string `json:"resource"`      // Resource being contended (file path, pipe, etc.)
	ResourceType string `json:"resource_type"` // file, pipe, socket, futex, semaphore
}
