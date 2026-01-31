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

// DiskMonitor handles disk I/O metrics collection
type DiskMonitor struct {
	lastDiskStats map[string]DiskStat
	lastUpdate    time.Time
}

// DiskStat represents raw disk statistics from /proc/diskstats
type DiskStat struct {
	ReadsCompleted  uint64 // Number of reads completed
	ReadsMerged     uint64 // Number of reads merged
	SectorsRead     uint64 // Number of sectors read
	ReadTimeMs      uint64 // Time spent reading (ms)
	WritesCompleted uint64 // Number of writes completed
	WritesMerged    uint64 // Number of writes merged
	SectorsWritten  uint64 // Number of sectors written
	WriteTimeMs     uint64 // Time spent writing (ms)
	IOsInProgress   uint64 // I/Os currently in progress
	IOTimeMs        uint64 // Time spent doing I/Os (ms)
	WeightedIOTimeMs uint64 // Weighted time spent doing I/Os (ms)
}

const sectorSize = 512 // Standard sector size in bytes

// NewDiskMonitor creates a new disk monitor
func NewDiskMonitor() *DiskMonitor {
	return &DiskMonitor{
		lastDiskStats: make(map[string]DiskStat),
	}
}

// GetDiskMetrics collects disk I/O metrics from /proc/diskstats
func (m *DiskMonitor) GetDiskMetrics() (metrics.DiskMetrics, error) {
	diskStats, err := m.parseProcDiskstats()
	if err != nil {
		return metrics.DiskMetrics{}, fmt.Errorf("failed to parse /proc/diskstats: %w", err)
	}

	now := time.Now()
	timeDelta := now.Sub(m.lastUpdate).Seconds()

	var devices []metrics.DiskDevice
	var totalIO metrics.DiskIO

	for name, current := range diskStats {
		// Skip devices we don't want to monitor (loop devices, ram devices, etc.)
		if m.shouldSkipDevice(name) {
			continue
		}

		device := metrics.DiskDevice{
			Name: name,
		}

		if last, exists := m.lastDiskStats[name]; exists && timeDelta > 0 {
			// Calculate deltas
			device.IO = m.calculateDiskIO(current, last, timeDelta)
		} else {
			// First measurement - just store current values without rates
			device.IO = metrics.DiskIO{
				ReadBytes:  current.SectorsRead * sectorSize,
				WriteBytes: current.SectorsWritten * sectorSize,
				ReadOps:    current.ReadsCompleted,
				WriteOps:   current.WritesCompleted,
			}
		}

		devices = append(devices, device)

		// Add to totals
		totalIO.ReadBytes += device.IO.ReadBytes
		totalIO.WriteBytes += device.IO.WriteBytes
		totalIO.ReadOps += device.IO.ReadOps
		totalIO.WriteOps += device.IO.WriteOps
	}

	// Store current stats for next calculation
	m.lastDiskStats = diskStats
	m.lastUpdate = now

	return metrics.DiskMetrics{
		Devices: devices,
		Total:   totalIO,
	}, nil
}

// parseProcDiskstats parses /proc/diskstats file
func (m *DiskMonitor) parseProcDiskstats() (map[string]DiskStat, error) {
	file, err := os.Open("/proc/diskstats")
	if err != nil {
		return nil, err
	}
	defer file.Close()

	stats := make(map[string]DiskStat)
	scanner := bufio.NewScanner(file)

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		fields := strings.Fields(line)

		// /proc/diskstats format:
		// major minor name reads_completed reads_merged sectors_read read_time_ms
		// writes_completed writes_merged sectors_written write_time_ms ios_in_progress
		// io_time_ms weighted_io_time_ms
		if len(fields) < 14 {
			continue
		}

		name := fields[2]
		stat := DiskStat{}

		// Parse numeric fields
		values := []uint64{}
		for i := 3; i < len(fields) && i < 14; i++ {
			val, err := strconv.ParseUint(fields[i], 10, 64)
			if err != nil {
				// Skip malformed lines
				continue
			}
			values = append(values, val)
		}

		if len(values) < 11 {
			continue
		}

		// Map values to struct fields
		stat.ReadsCompleted = values[0]
		stat.ReadsMerged = values[1]
		stat.SectorsRead = values[2]
		stat.ReadTimeMs = values[3]
		stat.WritesCompleted = values[4]
		stat.WritesMerged = values[5]
		stat.SectorsWritten = values[6]
		stat.WriteTimeMs = values[7]
		stat.IOsInProgress = values[8]
		stat.IOTimeMs = values[9]
		stat.WeightedIOTimeMs = values[10]

		stats[name] = stat
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return stats, nil
}

// calculateDiskIO calculates disk I/O metrics between two measurements
func (m *DiskMonitor) calculateDiskIO(current, last DiskStat, timeDelta float64) metrics.DiskIO {
	// Calculate differences (rates per second)
	readBytes := (current.SectorsRead - last.SectorsRead) * sectorSize
	writeBytes := (current.SectorsWritten - last.SectorsWritten) * sectorSize
	readOps := current.ReadsCompleted - last.ReadsCompleted
	writeOps := current.WritesCompleted - last.WritesCompleted

	io := metrics.DiskIO{
		ReadBytes:  readBytes,
		WriteBytes: writeBytes,
		ReadOps:    readOps,
		WriteOps:   writeOps,
	}

	// Calculate latencies (average per operation)
	readTimeDiff := current.ReadTimeMs - last.ReadTimeMs
	writeTimeDiff := current.WriteTimeMs - last.WriteTimeMs

	if readOps > 0 {
		io.ReadLatency = float64(readTimeDiff) / float64(readOps)
	}

	if writeOps > 0 {
		io.WriteLatency = float64(writeTimeDiff) / float64(writeOps)
	}

	return io
}

// shouldSkipDevice determines if a device should be skipped from monitoring
func (m *DiskMonitor) shouldSkipDevice(name string) bool {
	// Skip loop devices
	if strings.HasPrefix(name, "loop") {
		return true
	}

	// Skip ram devices
	if strings.HasPrefix(name, "ram") {
		return true
	}

	// Skip device mapper metadata devices
	if strings.HasSuffix(name, "_mlog") || strings.HasSuffix(name, "_mimage") {
		return true
	}

	// Skip partitions of devices we're already monitoring
	// Keep only whole devices (sda, nvme0n1) and their numbered partitions
	// but skip if it's just a partition and we have the whole device
	return false
}

// GetDiskList returns a list of available disk devices
func (m *DiskMonitor) GetDiskList() ([]string, error) {
	diskStats, err := m.parseProcDiskstats()
	if err != nil {
		return nil, err
	}

	var devices []string
	for name := range diskStats {
		if !m.shouldSkipDevice(name) {
			devices = append(devices, name)
		}
	}

	return devices, nil
}