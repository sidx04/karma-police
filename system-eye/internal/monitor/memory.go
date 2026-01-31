package monitor

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/teamovercooked/system-eye/internal/metrics"
)

// MemoryMonitor handles memory metrics collection
type MemoryMonitor struct{}

// NewMemoryMonitor creates a new memory monitor
func NewMemoryMonitor() *MemoryMonitor {
	return &MemoryMonitor{}
}

// GetMemoryMetrics collects memory metrics from /proc/meminfo
func (m *MemoryMonitor) GetMemoryMetrics() (metrics.MemoryMetrics, error) {
	memInfo, err := m.parseProcMeminfo()
	if err != nil {
		return metrics.MemoryMetrics{}, fmt.Errorf("failed to parse /proc/meminfo: %w", err)
	}

	memMetrics := metrics.MemoryMetrics{
		Total:     memInfo["MemTotal"] * 1024,     // Convert from KB to bytes
		Free:      memInfo["MemFree"] * 1024,      // Convert from KB to bytes
		Available: memInfo["MemAvailable"] * 1024, // Convert from KB to bytes
		Cached:    memInfo["Cached"] * 1024,       // Convert from KB to bytes
		Buffers:   memInfo["Buffers"] * 1024,      // Convert from KB to bytes
	}

	// Calculate used memory
	// Used = Total - Free - Buffers - Cached (simplified calculation)
	memMetrics.Used = memMetrics.Total - memMetrics.Available

	// Calculate usage percentage
	if memMetrics.Total > 0 {
		memMetrics.UsagePercent = (float64(memMetrics.Used) / float64(memMetrics.Total)) * 100.0
	}

	return memMetrics, nil
}

// parseProcMeminfo parses /proc/meminfo file and returns memory information in KB
func (m *MemoryMonitor) parseProcMeminfo() (map[string]uint64, error) {
	file, err := os.Open("/proc/meminfo")
	if err != nil {
		return nil, err
	}
	defer file.Close()

	memInfo := make(map[string]uint64)
	scanner := bufio.NewScanner(file)

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		// Parse line format: "MemTotal:    16384000 kB"
		parts := strings.Fields(line)
		if len(parts) < 2 {
			continue
		}

		// Extract key (remove colon)
		key := strings.TrimSuffix(parts[0], ":")

		// Extract value (should be in KB)
		valueStr := parts[1]
		value, err := strconv.ParseUint(valueStr, 10, 64)
		if err != nil {
			// Skip lines we can't parse
			continue
		}

		memInfo[key] = value
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	// Verify we have essential fields
	requiredFields := []string{"MemTotal", "MemFree", "MemAvailable"}
	for _, field := range requiredFields {
		if _, exists := memInfo[field]; !exists {
			return nil, fmt.Errorf("missing required field: %s", field)
		}
	}

	// Set defaults for optional fields if they don't exist
	if _, exists := memInfo["Cached"]; !exists {
		memInfo["Cached"] = 0
	}
	if _, exists := memInfo["Buffers"]; !exists {
		memInfo["Buffers"] = 0
	}

	return memInfo, nil
}

// GetMemoryDetails returns additional memory details for debugging/verbose output
func (m *MemoryMonitor) GetMemoryDetails() (map[string]uint64, error) {
	return m.parseProcMeminfo()
}