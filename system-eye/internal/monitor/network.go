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

// NetworkMonitor handles network metrics collection
type NetworkMonitor struct {
	lastNetStats map[string]NetworkStat
	lastUpdate   time.Time
}

// NetworkStat represents raw network statistics from /proc/net/dev
type NetworkStat struct {
	RxBytes   uint64 // Bytes received
	RxPackets uint64 // Packets received
	RxErrors  uint64 // Receive errors
	RxDropped uint64 // Received packets dropped
	TxBytes   uint64 // Bytes transmitted
	TxPackets uint64 // Packets transmitted
	TxErrors  uint64 // Transmit errors
	TxDropped uint64 // Transmitted packets dropped
}

// NewNetworkMonitor creates a new network monitor
func NewNetworkMonitor() *NetworkMonitor {
	return &NetworkMonitor{
		lastNetStats: make(map[string]NetworkStat),
	}
}

// GetNetworkMetrics collects network metrics from /proc/net/dev
func (m *NetworkMonitor) GetNetworkMetrics() (metrics.NetworkMetrics, error) {
	netStats, err := m.parseProcNetDev()
	if err != nil {
		return metrics.NetworkMetrics{}, fmt.Errorf("failed to parse /proc/net/dev: %w", err)
	}

	now := time.Now()
	timeDelta := now.Sub(m.lastUpdate).Seconds()

	var interfaces []metrics.NetworkInterface
	var totalTraffic metrics.NetworkTraffic

	for name, current := range netStats {
		// Skip loopback and other virtual interfaces for totals
		shouldIncludeInTotal := !m.shouldSkipFromTotal(name)

		iface := metrics.NetworkInterface{
			Name: name,
		}

		if last, exists := m.lastNetStats[name]; exists && timeDelta > 0 {
			// Calculate rates (bytes/packets per second during the interval)
			iface.Traffic = m.calculateNetworkTraffic(current, last, timeDelta)
		} else {
			// First measurement - use cumulative values
			iface.Traffic = metrics.NetworkTraffic{
				RxBytes:   current.RxBytes,
				TxBytes:   current.TxBytes,
				RxPackets: current.RxPackets,
				TxPackets: current.TxPackets,
				RxErrors:  current.RxErrors,
				TxErrors:  current.TxErrors,
				RxDropped: current.RxDropped,
				TxDropped: current.TxDropped,
			}
		}

		interfaces = append(interfaces, iface)

		// Add to totals (skip virtual/loopback interfaces)
		if shouldIncludeInTotal {
			totalTraffic.RxBytes += iface.Traffic.RxBytes
			totalTraffic.TxBytes += iface.Traffic.TxBytes
			totalTraffic.RxPackets += iface.Traffic.RxPackets
			totalTraffic.TxPackets += iface.Traffic.TxPackets
			totalTraffic.RxErrors += iface.Traffic.RxErrors
			totalTraffic.TxErrors += iface.Traffic.TxErrors
			totalTraffic.RxDropped += iface.Traffic.RxDropped
			totalTraffic.TxDropped += iface.Traffic.TxDropped
		}
	}

	// Store current stats for next calculation
	m.lastNetStats = netStats
	m.lastUpdate = now

	return metrics.NetworkMetrics{
		Interfaces: interfaces,
		Total:      totalTraffic,
	}, nil
}

// parseProcNetDev parses /proc/net/dev file
func (m *NetworkMonitor) parseProcNetDev() (map[string]NetworkStat, error) {
	file, err := os.Open("/proc/net/dev")
	if err != nil {
		return nil, err
	}
	defer file.Close()

	stats := make(map[string]NetworkStat)
	scanner := bufio.NewScanner(file)
	lineCount := 0

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		lineCount++

		// Skip header lines
		if lineCount <= 2 {
			continue
		}

		// Parse interface line
		// Format: "eth0: 1234 5678 ... 9012 3456 ..."
		colonIndex := strings.Index(line, ":")
		if colonIndex == -1 {
			continue
		}

		name := strings.TrimSpace(line[:colonIndex])
		valuesStr := strings.TrimSpace(line[colonIndex+1:])
		fields := strings.Fields(valuesStr)

		// We need at least 16 fields for complete statistics
		if len(fields) < 16 {
			continue
		}

		stat := NetworkStat{}

		// Parse receive statistics (first 8 fields)
		if rxBytes, err := strconv.ParseUint(fields[0], 10, 64); err == nil {
			stat.RxBytes = rxBytes
		}
		if rxPackets, err := strconv.ParseUint(fields[1], 10, 64); err == nil {
			stat.RxPackets = rxPackets
		}
		if rxErrors, err := strconv.ParseUint(fields[2], 10, 64); err == nil {
			stat.RxErrors = rxErrors
		}
		if rxDropped, err := strconv.ParseUint(fields[3], 10, 64); err == nil {
			stat.RxDropped = rxDropped
		}

		// Parse transmit statistics (fields 8-15)
		if len(fields) >= 9 {
			if txBytes, err := strconv.ParseUint(fields[8], 10, 64); err == nil {
				stat.TxBytes = txBytes
			}
		}
		if len(fields) >= 10 {
			if txPackets, err := strconv.ParseUint(fields[9], 10, 64); err == nil {
				stat.TxPackets = txPackets
			}
		}
		if len(fields) >= 11 {
			if txErrors, err := strconv.ParseUint(fields[10], 10, 64); err == nil {
				stat.TxErrors = txErrors
			}
		}
		if len(fields) >= 12 {
			if txDropped, err := strconv.ParseUint(fields[11], 10, 64); err == nil {
				stat.TxDropped = txDropped
			}
		}

		stats[name] = stat
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return stats, nil
}

// calculateNetworkTraffic calculates network traffic rates between two measurements
func (m *NetworkMonitor) calculateNetworkTraffic(current, last NetworkStat, timeDelta float64) metrics.NetworkTraffic {
	// Calculate differences (note: these are cumulative counters, so we calculate rates)
	rxBytes := current.RxBytes - last.RxBytes
	txBytes := current.TxBytes - last.TxBytes
	rxPackets := current.RxPackets - last.RxPackets
	txPackets := current.TxPackets - last.TxPackets

	// For error counters, we typically want the current total rather than rate
	return metrics.NetworkTraffic{
		RxBytes:   rxBytes,
		TxBytes:   txBytes,
		RxPackets: rxPackets,
		TxPackets: txPackets,
		RxErrors:  current.RxErrors,
		TxErrors:  current.TxErrors,
		RxDropped: current.RxDropped,
		TxDropped: current.TxDropped,
	}
}

// shouldSkipFromTotal determines if an interface should be excluded from total calculations
func (m *NetworkMonitor) shouldSkipFromTotal(name string) bool {
	// Skip loopback interface
	if name == "lo" {
		return true
	}

	// Skip Docker interfaces
	if strings.HasPrefix(name, "docker") {
		return true
	}

	// Skip virtual interfaces
	if strings.HasPrefix(name, "veth") {
		return true
	}

	// Skip bridge interfaces
	if strings.HasPrefix(name, "br-") {
		return true
	}

	// Skip tunnel interfaces
	if strings.HasPrefix(name, "tun") || strings.HasPrefix(name, "tap") {
		return true
	}

	return false
}

// GetNetworkInterfaces returns a list of available network interfaces
func (m *NetworkMonitor) GetNetworkInterfaces() ([]string, error) {
	netStats, err := m.parseProcNetDev()
	if err != nil {
		return nil, err
	}

	var interfaces []string
	for name := range netStats {
		interfaces = append(interfaces, name)
	}

	return interfaces, nil
}

// GetInterfaceInfo returns detailed information about a specific network interface
func (m *NetworkMonitor) GetInterfaceInfo(interfaceName string) (*NetworkStat, error) {
	netStats, err := m.parseProcNetDev()
	if err != nil {
		return nil, err
	}

	if stat, exists := netStats[interfaceName]; exists {
		return &stat, nil
	}

	return nil, fmt.Errorf("interface %s not found", interfaceName)
}