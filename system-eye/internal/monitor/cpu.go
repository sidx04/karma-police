package monitor

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/teamovercooked/system-eye/internal/metrics"
)

// CPUMonitor handles CPU metrics collection
type CPUMonitor struct {
	lastCPUStats map[string]CPUStat
	lastUpdate   time.Time
}

// CPUStat represents raw CPU statistics from /proc/stat
type CPUStat struct {
	User   uint64
	Nice   uint64
	System uint64
	Idle   uint64
	IOWait uint64
	IRQ    uint64
	SoftIRQ uint64
	Steal   uint64
	Guest   uint64
	GuestNice uint64
}

// NewCPUMonitor creates a new CPU monitor
func NewCPUMonitor() *CPUMonitor {
	return &CPUMonitor{
		lastCPUStats: make(map[string]CPUStat),
	}
}

// GetCPUMetrics collects CPU metrics from /proc/stat and /proc/loadavg
func (m *CPUMonitor) GetCPUMetrics() (metrics.CPUMetrics, error) {
	cpuMetrics := metrics.CPUMetrics{}

	// Parse /proc/stat for CPU usage
	stats, err := m.parseProcStat()
	if err != nil {
		return cpuMetrics, fmt.Errorf("failed to parse /proc/stat: %w", err)
	}

	// Calculate CPU usage percentages
	now := time.Now()
	if !m.lastUpdate.IsZero() {
		cpuMetrics.Overall = m.calculateCPUUsage("cpu", stats["cpu"], m.lastCPUStats["cpu"])

		// Per-core usage
		coreCount := 0
		for name, stat := range stats {
			if strings.HasPrefix(name, "cpu") && name != "cpu" {
				usage := m.calculateCPUUsage(name, stat, m.lastCPUStats[name])
				cpuMetrics.PerCore = append(cpuMetrics.PerCore, usage)
				coreCount++
			}
		}
	}

	// Store current stats for next calculation
	m.lastCPUStats = stats
	m.lastUpdate = now

	// Parse /proc/loadavg for load averages
	loadAvg, err := m.parseLoadAvg()
	if err != nil {
		return cpuMetrics, fmt.Errorf("failed to parse load average: %w", err)
	}

	cpuMetrics.LoadAvg1 = loadAvg[0]
	cpuMetrics.LoadAvg5 = loadAvg[1]
	cpuMetrics.LoadAvg15 = loadAvg[2]

	// Get context switches and interrupts from /proc/stat
	if interrupts, ok := stats["intr"]; ok {
		cpuMetrics.Interrupts = interrupts.User // Total interrupts stored in User field
	}
	if ctxt, ok := stats["ctxt"]; ok {
		cpuMetrics.ContextSwitches = ctxt.User // Total context switches stored in User field
	}

	return cpuMetrics, nil
}

// parseProcStat parses /proc/stat file
func (m *CPUMonitor) parseProcStat() (map[string]CPUStat, error) {
	file, err := os.Open("/proc/stat")
	if err != nil {
		return nil, err
	}
	defer file.Close()

	stats := make(map[string]CPUStat)
	scanner := bufio.NewScanner(file)

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		fields := strings.Fields(line)

		if len(fields) < 2 {
			continue
		}

		name := fields[0]

		// Handle CPU lines (cpu, cpu0, cpu1, ...)
		if strings.HasPrefix(name, "cpu") && len(fields) >= 8 {
			stat := CPUStat{}
			values := make([]uint64, 0, 10)

			// Parse numeric values
			for i := 1; i < len(fields) && i <= 10; i++ {
				val, err := strconv.ParseUint(fields[i], 10, 64)
				if err != nil {
					continue
				}
				values = append(values, val)
			}

			// Map values to struct fields
			if len(values) >= 4 {
				stat.User = values[0]
				stat.Nice = values[1]
				stat.System = values[2]
				stat.Idle = values[3]
			}
			if len(values) >= 5 {
				stat.IOWait = values[4]
			}
			if len(values) >= 6 {
				stat.IRQ = values[5]
			}
			if len(values) >= 7 {
				stat.SoftIRQ = values[6]
			}
			if len(values) >= 8 {
				stat.Steal = values[7]
			}
			if len(values) >= 9 {
				stat.Guest = values[8]
			}
			if len(values) >= 10 {
				stat.GuestNice = values[9]
			}

			stats[name] = stat
		}

		// Handle interrupts line
		if name == "intr" && len(fields) >= 2 {
			if total, err := strconv.ParseUint(fields[1], 10, 64); err == nil {
				stats["intr"] = CPUStat{User: total}
			}
		}

		// Handle context switches line
		if name == "ctxt" && len(fields) >= 2 {
			if total, err := strconv.ParseUint(fields[1], 10, 64); err == nil {
				stats["ctxt"] = CPUStat{User: total}
			}
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return stats, nil
}

// calculateCPUUsage calculates CPU usage percentage between two measurements
func (m *CPUMonitor) calculateCPUUsage(name string, current, last CPUStat) float64 {
	// Calculate total time for current and last measurements
	currentTotal := current.User + current.Nice + current.System + current.Idle +
		current.IOWait + current.IRQ + current.SoftIRQ + current.Steal

	lastTotal := last.User + last.Nice + last.System + last.Idle +
		last.IOWait + last.IRQ + last.SoftIRQ + last.Steal

	// Calculate idle time
	currentIdle := current.Idle + current.IOWait
	lastIdle := last.Idle + last.IOWait

	// Calculate differences
	totalDiff := currentTotal - lastTotal
	idleDiff := currentIdle - lastIdle

	if totalDiff <= 0 {
		return 0.0
	}

	// CPU usage percentage
	usage := (float64(totalDiff-idleDiff) / float64(totalDiff)) * 100.0

	// Ensure we don't return negative values or values > 100%
	if usage < 0 {
		usage = 0
	} else if usage > 100 {
		usage = 100
	}

	return usage
}

// parseLoadAvg parses /proc/loadavg for load averages
func (m *CPUMonitor) parseLoadAvg() ([3]float64, error) {
	var loadAvg [3]float64

	data, err := os.ReadFile("/proc/loadavg")
	if err != nil {
		return loadAvg, err
	}

	line := strings.TrimSpace(string(data))
	fields := strings.Fields(line)

	if len(fields) < 3 {
		return loadAvg, fmt.Errorf("invalid /proc/loadavg format: %s", line)
	}

	for i := 0; i < 3; i++ {
		val, err := strconv.ParseFloat(fields[i], 64)
		if err != nil {
			return loadAvg, fmt.Errorf("failed to parse load average %d: %w", i+1, err)
		}
		loadAvg[i] = val
	}

	return loadAvg, nil
}