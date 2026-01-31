package ui

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/teamovercooked/system-eye/internal/metrics"
)

// TerminalUI provides a terminal-based user interface
type TerminalUI struct {
	collector metrics.MetricsCollector
}

// NewTerminalUI creates a new terminal UI
func NewTerminalUI(collector metrics.MetricsCollector) *TerminalUI {
	return &TerminalUI{
		collector: collector,
	}
}

// Run starts the terminal UI
func (ui *TerminalUI) Run(ctx context.Context) error {
	log.Println("Starting terminal UI (stub implementation)")

	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			log.Println("Terminal UI shutting down")
			return nil

		case <-ticker.C:
			// Collect metrics
			systemMetrics, err := ui.collector.Collect()
			if err != nil {
				log.Printf("Error collecting metrics: %v", err)
				continue
			}

			// Display metrics (simple text output for now)
			ui.displayMetrics(systemMetrics)
		}
	}
}

// displayMetrics shows metrics in a simple text format
func (ui *TerminalUI) displayMetrics(metrics *metrics.SystemMetrics) {
	// Clear screen (ANSI escape code)
	fmt.Print("\033[2J\033[H")

	fmt.Printf("=== System's Eye System Monitor ===\n")
	fmt.Printf("Timestamp: %s\n\n", metrics.Timestamp.Format("2006-01-02 15:04:05"))

	// CPU metrics
	fmt.Printf("CPU:\n")
	fmt.Printf("  Overall Usage: %.2f%%\n", metrics.CPU.Overall)
	fmt.Printf("  Load Average:  %.2f (1m) %.2f (5m) %.2f (15m)\n",
		metrics.CPU.LoadAvg1, metrics.CPU.LoadAvg5, metrics.CPU.LoadAvg15)
	if len(metrics.CPU.PerCore) > 0 {
		fmt.Printf("  Per-Core:     ")
		for i, usage := range metrics.CPU.PerCore {
			fmt.Printf("CPU%d:%.1f%% ", i, usage)
			if (i+1)%4 == 0 {
				fmt.Printf("\n                ")
			}
		}
		fmt.Println()
	}
	fmt.Printf("  Context Switches: %d\n", metrics.CPU.ContextSwitches)
	fmt.Printf("  Interrupts:       %d\n\n", metrics.CPU.Interrupts)

	// Memory metrics
	fmt.Printf("Memory:\n")
	fmt.Printf("  Total:     %s\n", formatBytes(metrics.Memory.Total))
	fmt.Printf("  Used:      %s (%.1f%%)\n", formatBytes(metrics.Memory.Used), metrics.Memory.UsagePercent)
	fmt.Printf("  Available: %s\n", formatBytes(metrics.Memory.Available))
	fmt.Printf("  Cached:    %s\n", formatBytes(metrics.Memory.Cached))
	fmt.Printf("  Buffers:   %s\n\n", formatBytes(metrics.Memory.Buffers))

	// Disk metrics
	fmt.Printf("Disk I/O:\n")
	fmt.Printf("  Total Read:  %s (%d ops)\n",
		formatBytes(metrics.Disk.Total.ReadBytes), metrics.Disk.Total.ReadOps)
	fmt.Printf("  Total Write: %s (%d ops)\n\n",
		formatBytes(metrics.Disk.Total.WriteBytes), metrics.Disk.Total.WriteOps)

	// Network metrics
	fmt.Printf("Network:\n")
	fmt.Printf("  Total RX: %s (%d packets)\n",
		formatBytes(metrics.Network.Total.RxBytes), metrics.Network.Total.RxPackets)
	fmt.Printf("  Total TX: %s (%d packets)\n",
		formatBytes(metrics.Network.Total.TxBytes), metrics.Network.Total.TxPackets)
	if metrics.Network.Total.RxErrors > 0 || metrics.Network.Total.TxErrors > 0 {
		fmt.Printf("  Errors:   RX:%d TX:%d\n", metrics.Network.Total.RxErrors, metrics.Network.Total.TxErrors)
	}
	fmt.Println()

	// GPU metrics (if available)
	if metrics.GPU != nil && len(metrics.GPU.Devices) > 0 {
		fmt.Printf("GPU:\n")
		for i, device := range metrics.GPU.Devices {
			fmt.Printf("  GPU %d (%s):\n", i, device.Name)
			fmt.Printf("    Utilization: %d%%\n", device.Utilization)
			fmt.Printf("    Memory:      %s / %s (%.1f%%)\n",
				formatBytes(device.Memory.Used),
				formatBytes(device.Memory.Total),
				device.Memory.UsagePercent)
			fmt.Printf("    Temperature: %d°C\n", device.Temperature)
			fmt.Printf("    Power:       %dW\n", device.PowerUsage)
			fmt.Printf("    Fan Speed:   %d%%\n", device.FanSpeed)
		}
		fmt.Println()
	}

	// Show top disk devices
	if len(metrics.Disk.Devices) > 0 {
		fmt.Printf("Top Disk Devices:\n")
		for _, device := range metrics.Disk.Devices[:min(5, len(metrics.Disk.Devices))] {
			if device.IO.ReadBytes > 0 || device.IO.WriteBytes > 0 {
				fmt.Printf("  %s: R:%s W:%s\n",
					device.Name,
					formatBytes(device.IO.ReadBytes),
					formatBytes(device.IO.WriteBytes))
			}
		}
		fmt.Println()
	}

	// Show active network interfaces
	if len(metrics.Network.Interfaces) > 0 {
		fmt.Printf("Active Network Interfaces:\n")
		for _, iface := range metrics.Network.Interfaces {
			if iface.Traffic.RxBytes > 0 || iface.Traffic.TxBytes > 0 {
				fmt.Printf("  %s: RX:%s TX:%s\n",
					iface.Name,
					formatBytes(iface.Traffic.RxBytes),
					formatBytes(iface.Traffic.TxBytes))
			}
		}
		fmt.Println()
	}

	fmt.Printf("Press Ctrl+C to exit\n")
}

// formatBytes formats bytes into human-readable format
func formatBytes(bytes uint64) string {
	const unit = 1024
	if bytes < unit {
		return fmt.Sprintf("%d B", bytes)
	}

	div, exp := uint64(unit), 0
	for n := bytes / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}

	units := []string{"B", "KB", "MB", "GB", "TB", "PB"}
	return fmt.Sprintf("%.1f %s", float64(bytes)/float64(div), units[exp])
}

// min returns the minimum of two integers
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
