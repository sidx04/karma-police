package monitor

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/teamovercooked/system-eye/internal/metrics"
)

// ProcessMonitorProc monitors processes using /proc filesystem
type ProcessMonitorProc struct {
	pageSize        int64
	totalMemory     uint64
	lastCPUStats    map[uint32]*CPUTime
	mu              sync.RWMutex             // Protects lastCPUStats map
	systemCPUTime   uint64
	lastUpdate      time.Time
	clkTck          float64                  // Clock ticks per second (typically 100)
	numCPUs         int                      // Number of CPU cores for normalization
	metricsEnricher *ProcessMetricsEnricher  // Enriches process metrics with deadlock-detection fields
}

// CPUTime holds CPU usage information for a process
type CPUTime struct {
	Utime uint64
	Stime uint64
	Total uint64
}

// processInfoInternal extends ProcessInfo with internal fields
type processInfoInternal struct {
	metrics.ProcessInfo
	cpuTime *CPUTime
}

// NewProcessMonitor creates a new process monitor using /proc
func NewProcessMonitor() *ProcessMonitorProc {
	pageSize := int64(os.Getpagesize())
	totalMem := getTotalMemory()
	numCPUs := getNumCPUs()

	return &ProcessMonitorProc{
		pageSize:        pageSize,
		totalMemory:     totalMem,
		lastCPUStats:    make(map[uint32]*CPUTime),
		clkTck:          100.0, // Linux standard clock ticks per second
		numCPUs:         numCPUs,
		metricsEnricher: NewProcessMetricsEnricher(),
	}
}

// GetProcessMetrics collects metrics for all processes
func (pm *ProcessMonitorProc) GetProcessMetrics() (*metrics.ProcessMetrics, error) {
	procMetrics := &metrics.ProcessMetrics{
		Processes: make([]metrics.ProcessInfo, 0),
	}

	// Get current system CPU time for percentage calculation
	systemCPU, err := pm.getSystemCPUTime()
	if err != nil {
		return nil, fmt.Errorf("failed to get system CPU time: %w", err)
	}

	// Read all process directories from /proc
	procDirs, err := os.ReadDir("/proc")
	if err != nil {
		return nil, fmt.Errorf("failed to read /proc: %w", err)
	}

	processes := make([]metrics.ProcessInfo, 0)

	for _, entry := range procDirs {
		// Skip non-process directories
		if !entry.IsDir() {
			continue
		}

		// Process directories are numeric PIDs
		pid, err := strconv.ParseUint(entry.Name(), 10, 32)
		if err != nil {
			continue // Not a PID directory
		}

		// Read process info
		info, err := pm.readProcessInfo(uint32(pid))
		if err != nil {
			continue // Process might have exited
		}

		// Calculate CPU percentage if we have previous data
		pm.mu.Lock()
		if pm.lastCPUStats[uint32(pid)] != nil && pm.lastUpdate.Unix() > 0 {
			deltaTime := time.Since(pm.lastUpdate).Seconds()
			if deltaTime > 0 {
				lastCPU := pm.lastCPUStats[uint32(pid)]
				deltaCPU := float64(info.cpuTime.Total - lastCPU.Total)
				// CPU% = (ticks / ticks_per_second) / seconds * 100
				// Normalize by number of CPUs to match overall CPU scale
				cpuPercent := (deltaCPU / pm.clkTck) / deltaTime * 100.0
				if pm.numCPUs > 0 {
					info.CPUPercent = cpuPercent / float64(pm.numCPUs)
				} else {
					info.CPUPercent = cpuPercent
				}
			}
		}

		// Store current CPU time for next calculation
		pm.lastCPUStats[uint32(pid)] = info.cpuTime
		pm.mu.Unlock()

		// Enrich with raw metrics needed for deadlock detection in Python
		if pm.metricsEnricher != nil {
			pm.metricsEnricher.EnrichProcessInfo(&info.ProcessInfo)
		}

		// Track process states
		switch info.State {
		case "R":
			procMetrics.Running++
		case "S", "I":
			procMetrics.Sleeping++
		}

		processes = append(processes, info.ProcessInfo)
	}

	// Update timing
	pm.lastUpdate = time.Now()
	pm.systemCPUTime = systemCPU

	// Sort by CPU usage for top consumers
	sort.Slice(processes, func(i, j int) bool {
		return processes[i].CPUPercent > processes[j].CPUPercent
	})

	// Get top CPU consumers (up to 10)
	for i := 0; i < len(processes) && i < 10; i++ {
		procMetrics.TopCPU = append(procMetrics.TopCPU, processes[i])
	}

	// Sort by memory usage for top consumers
	sort.Slice(processes, func(i, j int) bool {
		return processes[i].MemoryPercent > processes[j].MemoryPercent
	})

	// Get top memory consumers (up to 10)
	for i := 0; i < len(processes) && i < 10; i++ {
		procMetrics.TopMemory = append(procMetrics.TopMemory, processes[i])
	}

	// Add GPU process info if available (from nvidia-smi)
	pm.addGPUProcessInfo(processes, procMetrics)

	procMetrics.Processes = processes
	procMetrics.Total = len(processes)

	return procMetrics, nil
}

// readProcessInfo reads information for a single process
func (pm *ProcessMonitorProc) readProcessInfo(pid uint32) (*processInfoInternal, error) {
	procPath := fmt.Sprintf("/proc/%d", pid)

	info := &processInfoInternal{
		ProcessInfo: metrics.ProcessInfo{
			PID: pid,
		},
	}

	// Read /proc/[pid]/stat for basic info
	statData, err := os.ReadFile(filepath.Join(procPath, "stat"))
	if err != nil {
		return nil, err
	}

	if err := pm.parseStat(string(statData), info); err != nil {
		return nil, err
	}

	// Read /proc/[pid]/status for more details
	statusData, err := os.ReadFile(filepath.Join(procPath, "status"))
	if err == nil {
		pm.parseStatus(string(statusData), info)
	}

	// Read /proc/[pid]/cmdline for command
	cmdlineData, err := os.ReadFile(filepath.Join(procPath, "cmdline"))
	if err == nil && len(cmdlineData) > 0 {
		info.Command = strings.ReplaceAll(string(cmdlineData), "\x00", " ")
		info.Command = strings.TrimSpace(info.Command)
	}

	// Read /proc/[pid]/io for I/O stats
	ioData, err := os.ReadFile(filepath.Join(procPath, "io"))
	if err == nil {
		pm.parseIO(string(ioData), info)
	}

	// Count open file descriptors
	fdPath := filepath.Join(procPath, "fd")
	if fds, err := os.ReadDir(fdPath); err == nil {
		info.OpenFiles = uint32(len(fds))
	}

	// Calculate memory percentage
	if pm.totalMemory > 0 {
		info.MemoryPercent = float64(info.MemoryRSS) * 100.0 / float64(pm.totalMemory)
	}

	return info, nil
}

// parseStat parses /proc/[pid]/stat file
func (pm *ProcessMonitorProc) parseStat(data string, info *processInfoInternal) error {
	// Find the last ) to handle process names with parentheses
	lastParen := strings.LastIndex(data, ")")
	if lastParen == -1 {
		return fmt.Errorf("invalid stat format")
	}

	// Extract process name
	firstParen := strings.Index(data, "(")
	if firstParen != -1 && lastParen > firstParen {
		info.Name = data[firstParen+1 : lastParen]
	}

	// Parse fields after the command name
	fields := strings.Fields(data[lastParen+2:])
	if len(fields) < 20 {
		return fmt.Errorf("insufficient stat fields")
	}

	// State (field 3)
	info.State = fields[0]

	// PPID (field 4)
	if ppid, err := strconv.ParseUint(fields[1], 10, 32); err == nil {
		info.PPID = uint32(ppid)
	}

	// Number of threads (field 20)
	if threads, err := strconv.ParseUint(fields[17], 10, 32); err == nil {
		info.NumThreads = uint32(threads)
	}

	// CPU times (fields 14-15)
	info.cpuTime = &CPUTime{}
	if utime, err := strconv.ParseUint(fields[11], 10, 64); err == nil {
		info.cpuTime.Utime = utime
	}
	if stime, err := strconv.ParseUint(fields[12], 10, 64); err == nil {
		info.cpuTime.Stime = stime
	}
	info.cpuTime.Total = info.cpuTime.Utime + info.cpuTime.Stime

	// Virtual memory size (field 23)
	if vsize, err := strconv.ParseUint(fields[20], 10, 64); err == nil {
		info.MemoryVirtual = vsize
	}

	// RSS (field 24) - in pages
	if rss, err := strconv.ParseInt(fields[21], 10, 64); err == nil {
		info.MemoryRSS = uint64(rss * pm.pageSize)
	}

	return nil
}

// parseStatus parses /proc/[pid]/status file
func (pm *ProcessMonitorProc) parseStatus(data string, info *processInfoInternal) {
	scanner := bufio.NewScanner(strings.NewReader(data))
	for scanner.Scan() {
		line := scanner.Text()
		parts := strings.Fields(line)
		if len(parts) < 2 {
			continue
		}

		switch parts[0] {
		case "VmRSS:":
			if rss, err := strconv.ParseUint(parts[1], 10, 64); err == nil {
				info.MemoryRSS = rss * 1024 // Convert from KB to bytes
			}
		case "VmSize:":
			if vsize, err := strconv.ParseUint(parts[1], 10, 64); err == nil {
				info.MemoryVirtual = vsize * 1024 // Convert from KB to bytes
			}
		case "voluntary_ctxt_switches:":
			if switches, err := strconv.ParseUint(parts[1], 10, 64); err == nil {
				info.ContextSwitches = switches
			}
		case "nonvoluntary_ctxt_switches:":
			if switches, err := strconv.ParseUint(parts[1], 10, 64); err == nil {
				info.ContextSwitches += switches
			}
		}
	}
}

// parseIO parses /proc/[pid]/io file
func (pm *ProcessMonitorProc) parseIO(data string, info *processInfoInternal) {
	scanner := bufio.NewScanner(strings.NewReader(data))
	for scanner.Scan() {
		line := scanner.Text()
		parts := strings.Fields(line)
		if len(parts) < 2 {
			continue
		}

		switch parts[0] {
		case "read_bytes:":
			if bytes, err := strconv.ParseUint(parts[1], 10, 64); err == nil {
				info.ReadBytes = bytes
			}
		case "write_bytes:":
			if bytes, err := strconv.ParseUint(parts[1], 10, 64); err == nil {
				info.WriteBytes = bytes
			}
		case "syscr:":
			if count, err := strconv.ParseUint(parts[1], 10, 64); err == nil {
				info.SyscallCount = count
			}
		case "syscw:":
			if count, err := strconv.ParseUint(parts[1], 10, 64); err == nil {
				info.SyscallCount += count
			}
		}
	}
}

// addGPUProcessInfo adds GPU usage information from nvidia-smi
func (pm *ProcessMonitorProc) addGPUProcessInfo(processes []metrics.ProcessInfo, procMetrics *metrics.ProcessMetrics) {
	// Try to get GPU process info using nvidia-smi
	// This is a simplified version - could be expanded
	// For now, just mark that we tried
	procMetrics.TopGPU = []metrics.ProcessInfo{}
}

// getSystemCPUTime gets total system CPU time
func (pm *ProcessMonitorProc) getSystemCPUTime() (uint64, error) {
	data, err := os.ReadFile("/proc/stat")
	if err != nil {
		return 0, err
	}

	lines := strings.Split(string(data), "\n")
	for _, line := range lines {
		if strings.HasPrefix(line, "cpu ") {
			fields := strings.Fields(line)
			if len(fields) < 8 {
				return 0, fmt.Errorf("invalid cpu stat line")
			}

			var total uint64
			for i := 1; i < 8; i++ {
				if val, err := strconv.ParseUint(fields[i], 10, 64); err == nil {
					total += val
				}
			}
			return total, nil
		}
	}

	return 0, fmt.Errorf("cpu stat line not found")
}

// IsAvailable returns whether process monitoring is available
func (pm *ProcessMonitorProc) IsAvailable() bool {
	// /proc is always available on Linux
	_, err := os.Stat("/proc")
	return err == nil
}

// Close cleans up resources
func (pm *ProcessMonitorProc) Close() error {
	if pm.metricsEnricher != nil {
		pm.metricsEnricher.Close()
	}
	return nil
}

// Helper function to get total memory
func getTotalMemory() uint64 {
	data, err := os.ReadFile("/proc/meminfo")
	if err != nil {
		return 0
	}

	lines := strings.Split(string(data), "\n")
	for _, line := range lines {
		if strings.HasPrefix(line, "MemTotal:") {
			fields := strings.Fields(line)
			if len(fields) >= 2 {
				if total, err := strconv.ParseUint(fields[1], 10, 64); err == nil {
					return total * 1024 // Convert KB to bytes
				}
			}
		}
	}
	return 0
}

// Helper function to get number of CPUs
func getNumCPUs() int {
	data, err := os.ReadFile("/proc/stat")
	if err != nil {
		return 1 // Default to 1 if can't read
	}

	count := 0
	lines := strings.Split(string(data), "\n")
	for _, line := range lines {
		if strings.HasPrefix(line, "cpu") && len(line) > 3 && line[3] >= '0' && line[3] <= '9' {
			count++
		}
	}

	if count == 0 {
		return 1 // Default to 1 if no CPUs found
	}
	return count
}