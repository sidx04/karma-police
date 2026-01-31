package export

import (
	"encoding/csv"
	"fmt"
	"os"
	"strconv"
	"time"

	"github.com/teamovercooked/system-eye/internal/metrics"
)

// CSVExporter exports metrics to CSV format
type CSVExporter struct {
	filePath string
}

// NewCSVExporter creates a new CSV exporter
func NewCSVExporter(filePath string) *CSVExporter {
	return &CSVExporter{
		filePath: filePath,
	}
}

// Export writes metrics to CSV file
func (e *CSVExporter) Export(samples []metrics.SystemMetrics) error {
	if len(samples) == 0 {
		return fmt.Errorf("no samples to export")
	}

	file, err := os.Create(e.filePath)
	if err != nil {
		return fmt.Errorf("failed to create file %s: %w", e.filePath, err)
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	// Write header
	header := e.getCSVHeader(samples[0])
	if err := writer.Write(header); err != nil {
		return fmt.Errorf("failed to write CSV header: %w", err)
	}

	// Write data rows
	for _, sample := range samples {
		row := e.sampleToCSVRow(sample)
		if err := writer.Write(row); err != nil {
			return fmt.Errorf("failed to write CSV row: %w", err)
		}
	}

	return nil
}

// getCSVHeader returns the CSV header based on available metrics
func (e *CSVExporter) getCSVHeader(sample metrics.SystemMetrics) []string {
	header := []string{
		"timestamp",
		"timestamp_unix",
		// CPU metrics
		"cpu_overall",
		"cpu_load_1m",
		"cpu_load_5m",
		"cpu_load_15m",
		"cpu_context_switches",
		"cpu_interrupts",
		// Memory metrics
		"memory_total",
		"memory_used",
		"memory_free",
		"memory_available",
		"memory_cached",
		"memory_buffers",
		"memory_usage_percent",
		// Disk metrics
		"disk_total_read_bytes",
		"disk_total_write_bytes",
		"disk_total_read_ops",
		"disk_total_write_ops",
		// Network metrics
		"network_total_rx_bytes",
		"network_total_tx_bytes",
		"network_total_rx_packets",
		"network_total_tx_packets",
		"network_total_rx_errors",
		"network_total_tx_errors",
	}

	// Add per-core CPU metrics
	for i := 0; i < len(sample.CPU.PerCore); i++ {
		header = append(header, fmt.Sprintf("cpu_core_%d", i))
	}

	// Add per-device disk metrics
	for _, device := range sample.Disk.Devices {
		prefix := fmt.Sprintf("disk_%s", device.Name)
		header = append(header,
			prefix+"_read_bytes",
			prefix+"_write_bytes",
			prefix+"_read_ops",
			prefix+"_write_ops",
			prefix+"_read_latency",
			prefix+"_write_latency",
		)
	}

	// Add per-interface network metrics
	for _, iface := range sample.Network.Interfaces {
		prefix := fmt.Sprintf("net_%s", iface.Name)
		header = append(header,
			prefix+"_rx_bytes",
			prefix+"_tx_bytes",
			prefix+"_rx_packets",
			prefix+"_tx_packets",
		)
	}

	// Add GPU metrics if available
	if sample.GPU != nil {
		for i := range sample.GPU.Devices {
			prefix := fmt.Sprintf("gpu_%d", i)
			header = append(header,
				prefix+"_name",
				prefix+"_utilization",
				prefix+"_memory_total",
				prefix+"_memory_used",
				prefix+"_memory_usage_percent",
				prefix+"_temperature",
				prefix+"_power_usage",
				prefix+"_fan_speed",
			)
		}
	}

	return header
}

// sampleToCSVRow converts a metrics sample to a CSV row
func (e *CSVExporter) sampleToCSVRow(sample metrics.SystemMetrics) []string {
	row := []string{
		sample.Timestamp.Format(time.RFC3339),
		strconv.FormatInt(sample.Timestamp.Unix(), 10),
		// CPU metrics
		formatFloat(sample.CPU.Overall, 2),
		formatFloat(sample.CPU.LoadAvg1, 2),
		formatFloat(sample.CPU.LoadAvg5, 2),
		formatFloat(sample.CPU.LoadAvg15, 2),
		formatUint64(sample.CPU.ContextSwitches),
		formatUint64(sample.CPU.Interrupts),
		// Memory metrics
		formatUint64(sample.Memory.Total),
		formatUint64(sample.Memory.Used),
		formatUint64(sample.Memory.Free),
		formatUint64(sample.Memory.Available),
		formatUint64(sample.Memory.Cached),
		formatUint64(sample.Memory.Buffers),
		formatFloat(sample.Memory.UsagePercent, 2),
		// Disk metrics
		formatUint64(sample.Disk.Total.ReadBytes),
		formatUint64(sample.Disk.Total.WriteBytes),
		formatUint64(sample.Disk.Total.ReadOps),
		formatUint64(sample.Disk.Total.WriteOps),
		// Network metrics
		formatUint64(sample.Network.Total.RxBytes),
		formatUint64(sample.Network.Total.TxBytes),
		formatUint64(sample.Network.Total.RxPackets),
		formatUint64(sample.Network.Total.TxPackets),
		formatUint64(sample.Network.Total.RxErrors),
		formatUint64(sample.Network.Total.TxErrors),
	}

	// Add per-core CPU metrics
	for _, coreUsage := range sample.CPU.PerCore {
		row = append(row, formatFloat(coreUsage, 2))
	}

	// Add per-device disk metrics
	for _, device := range sample.Disk.Devices {
		row = append(row,
			formatUint64(device.IO.ReadBytes),
			formatUint64(device.IO.WriteBytes),
			formatUint64(device.IO.ReadOps),
			formatUint64(device.IO.WriteOps),
			formatFloat(device.IO.ReadLatency, 3),
			formatFloat(device.IO.WriteLatency, 3),
		)
	}

	// Add per-interface network metrics
	for _, iface := range sample.Network.Interfaces {
		row = append(row,
			formatUint64(iface.Traffic.RxBytes),
			formatUint64(iface.Traffic.TxBytes),
			formatUint64(iface.Traffic.RxPackets),
			formatUint64(iface.Traffic.TxPackets),
		)
	}

	// Add GPU metrics if available
	if sample.GPU != nil {
		for _, device := range sample.GPU.Devices {
			row = append(row,
				device.Name,
				strconv.Itoa(device.Utilization),
				formatUint64(device.Memory.Total),
				formatUint64(device.Memory.Used),
				formatFloat(device.Memory.UsagePercent, 2),
				strconv.Itoa(device.Temperature),
				strconv.Itoa(device.PowerUsage),
				strconv.Itoa(device.FanSpeed),
			)
		}
	}

	return row
}

// ExportSingle exports a single metrics sample to CSV
func (e *CSVExporter) ExportSingle(sample *metrics.SystemMetrics) error {
	file, err := os.Create(e.filePath)
	if err != nil {
		return fmt.Errorf("failed to create file %s: %w", e.filePath, err)
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	// Write header
	header := e.getCSVHeader(*sample)
	if err := writer.Write(header); err != nil {
		return fmt.Errorf("failed to write CSV header: %w", err)
	}

	// Write data row
	row := e.sampleToCSVRow(*sample)
	if err := writer.Write(row); err != nil {
		return fmt.Errorf("failed to write CSV row: %w", err)
	}

	return nil
}

// ExportAppend appends a single sample to an existing CSV file
func (e *CSVExporter) ExportAppend(sample *metrics.SystemMetrics) error {
	// Check if file exists to determine if we need to write header
	writeHeader := false
	if _, err := os.Stat(e.filePath); os.IsNotExist(err) {
		writeHeader = true
	}

	file, err := os.OpenFile(e.filePath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err != nil {
		return fmt.Errorf("failed to open file %s: %w", e.filePath, err)
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	// Write header if this is a new file
	if writeHeader {
		header := e.getCSVHeader(*sample)
		if err := writer.Write(header); err != nil {
			return fmt.Errorf("failed to write CSV header: %w", err)
		}
	}

	// Write data row
	row := e.sampleToCSVRow(*sample)
	if err := writer.Write(row); err != nil {
		return fmt.Errorf("failed to write CSV row: %w", err)
	}

	return nil
}

// formatFloat formats a float64 to string with specified precision
func formatFloat(f float64, precision int) string {
	return strconv.FormatFloat(f, 'f', precision, 64)
}

// formatUint64 formats a uint64 to string
func formatUint64(u uint64) string {
	return strconv.FormatUint(u, 10)
}

// formatInt formats an int to string
func formatInt(i int) string {
	return strconv.Itoa(i)
}
