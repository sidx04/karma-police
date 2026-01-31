package monitor

import (
	"bufio"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/teamovercooked/system-eye/internal/metrics"
)

// ProcessMetricsEnricher enriches process info with data needed for deadlock detection in Python
type ProcessMetricsEnricher struct {
	previousState map[uint32]*processState // Track process state over time for delta calculations
	mu            sync.RWMutex              // Mutex to protect previousState map
}

// processState tracks historical state for calculating time deltas
type processState struct {
	pid            uint32
	cpuTime        uint64
	state          string
	waitChannel    string
	stateStartTime time.Time // When the current state first started
	lastCheckTime  time.Time // When we last checked (for interval calculations)
}

// NewProcessMetricsEnricher creates a new process metrics enricher
func NewProcessMetricsEnricher() *ProcessMetricsEnricher {
	return &ProcessMetricsEnricher{
		previousState: make(map[uint32]*processState),
	}
}

// EnrichProcessInfo adds raw metrics data needed for deadlock detection in Python
func (e *ProcessMetricsEnricher) EnrichProcessInfo(proc *metrics.ProcessInfo) error {
	pidStr := strconv.FormatUint(uint64(proc.PID), 10)

	// Read wait channel (what the process is waiting on)
	wchan, err := e.readWaitChannel(pidStr)
	if err == nil && wchan != "" {
		proc.WaitChannel = wchan
	}

	// Read context switches from /proc/[pid]/status
	volCtx, involCtx, err := e.readContextSwitches(pidStr)
	if err == nil {
		proc.VoluntaryCtxSwitches = volCtx
		proc.InvoluntaryCtxSwitches = involCtx
	}

	// Set wait state (Python will interpret this)
	proc.WaitState = proc.State

	// Calculate wait time and CPU time delta
	e.calculateTimings(proc)

	// Read blocked threads count
	blockedThreads, err := e.countBlockedThreads(pidStr)
	if err == nil {
		proc.BlockedThreads = blockedThreads
	}

	return nil
}

// readWaitChannel reads the kernel wait channel for a process
func (e *ProcessMetricsEnricher) readWaitChannel(pid string) (string, error) {
	data, err := os.ReadFile(filepath.Join("/proc", pid, "wchan"))
	if err != nil {
		return "", err
	}

	wchan := strings.TrimSpace(string(data))
	if wchan == "0" {
		return "", nil
	}

	return wchan, nil
}

// readContextSwitches reads voluntary and involuntary context switches
func (e *ProcessMetricsEnricher) readContextSwitches(pid string) (uint64, uint64, error) {
	file, err := os.Open(filepath.Join("/proc", pid, "status"))
	if err != nil {
		return 0, 0, err
	}
	defer file.Close()

	var voluntary, involuntary uint64
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		fields := strings.Fields(line)
		if len(fields) < 2 {
			continue
		}

		switch fields[0] {
		case "voluntary_ctxt_switches:":
			voluntary, _ = strconv.ParseUint(fields[1], 10, 64)
		case "nonvoluntary_ctxt_switches:":
			involuntary, _ = strconv.ParseUint(fields[1], 10, 64)
		}
	}

	return voluntary, involuntary, nil
}

// calculateTimings calculates wait time and CPU time delta
func (e *ProcessMetricsEnricher) calculateTimings(proc *metrics.ProcessInfo) {
	now := time.Now()

	e.mu.Lock()
	defer e.mu.Unlock()

	prevState, exists := e.previousState[proc.PID]
	if !exists {
		// First time seeing this process
		e.previousState[proc.PID] = &processState{
			pid:            proc.PID,
			cpuTime:        uint64(proc.CPUPercent),
			state:          proc.State,
			waitChannel:    proc.WaitChannel,
			stateStartTime: now,
			lastCheckTime:  now,
		}
		return
	}

	// Calculate CPU time delta
	currentCPUTime := uint64(proc.CPUPercent * 1000) // Convert to milliseconds
	proc.CPUTimeDelta = currentCPUTime - prevState.cpuTime

	// Calculate wait time based on state changes
	if proc.State == prevState.state && proc.WaitChannel == prevState.waitChannel {
		// State unchanged - calculate total time in this state
		waitDuration := now.Sub(prevState.stateStartTime)
		proc.WaitTimeMs = uint64(waitDuration.Milliseconds())
	} else {
		// State changed - reset to 0 and start tracking new state
		proc.WaitTimeMs = 0
		prevState.stateStartTime = now
	}

	// Update previous state
	prevState.cpuTime = currentCPUTime
	prevState.state = proc.State
	prevState.waitChannel = proc.WaitChannel
	prevState.lastCheckTime = now
}

// countBlockedThreads counts threads in blocked states
func (e *ProcessMetricsEnricher) countBlockedThreads(pid string) (uint32, error) {
	taskDir := filepath.Join("/proc", pid, "task")
	entries, err := os.ReadDir(taskDir)
	if err != nil {
		return 0, err
	}

	var blocked uint32
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}

		statFile := filepath.Join(taskDir, entry.Name(), "stat")
		data, err := os.ReadFile(statFile)
		if err != nil {
			continue
		}

		// Parse state from stat file (3rd field)
		parts := strings.Fields(string(data))
		if len(parts) > 2 {
			state := strings.Trim(parts[2], "()")
			if state == "D" || state == "S" {
				blocked++
			}
		}
	}

	return blocked, nil
}

// ReadSystemLocks reads all file locks from /proc/locks
func (e *ProcessMetricsEnricher) ReadSystemLocks() (map[uint32][]metrics.FileLock, error) {
	file, err := os.Open("/proc/locks")
	if err != nil {
		return nil, err
	}
	defer file.Close()

	locksByPID := make(map[uint32][]metrics.FileLock)
	scanner := bufio.NewScanner(file)

	for scanner.Scan() {
		line := scanner.Text()
		fields := strings.Fields(line)
		if len(fields) < 8 {
			continue
		}

		// Parse lock information
		// Format: ordinal POSIX/FLOCK ADVISORY/MANDATORY READ/WRITE pid major:minor:inode start end
		lockType := fields[3] // READ or WRITE
		pidStr := fields[4]
		inodeStr := fields[7]

		pid, err := strconv.ParseUint(pidStr, 10, 32)
		if err != nil {
			continue
		}

		inode, err := strconv.ParseUint(inodeStr, 10, 64)
		if err != nil {
			continue
		}

		lock := metrics.FileLock{
			LockType: lockType,
			Inode:    inode,
			PID:      uint32(pid),
			Path:     "", // Would need to resolve inode to path (expensive)
		}

		locksByPID[uint32(pid)] = append(locksByPID[uint32(pid)], lock)
	}

	return locksByPID, nil
}

// Close cleans up resources
func (e *ProcessMetricsEnricher) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.previousState = nil
	return nil
}
