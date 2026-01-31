package export

import (
	"encoding/json"
	"fmt"
	"os"
	"time"

	"github.com/teamovercooked/system-eye/internal/metrics"
)

// JSONExporter exports metrics to JSON format
type JSONExporter struct {
	filePath string
}

// NewJSONExporter creates a new JSON exporter
func NewJSONExporter(filePath string) *JSONExporter {
	return &JSONExporter{
		filePath: filePath,
	}
}

// Export writes metrics to JSON file
func (e *JSONExporter) Export(samples []metrics.SystemMetrics) error {
	if len(samples) == 0 {
		return fmt.Errorf("no samples to export")
	}

	// Create export structure with metadata
	export := JSONExport{
		Metadata: ExportMetadata{
			Version:     "1.0.0",
			ExportTime:  time.Now(),
			SampleCount: len(samples),
			StartTime:   samples[0].Timestamp,
			EndTime:     samples[len(samples)-1].Timestamp,
			Duration:    samples[len(samples)-1].Timestamp.Sub(samples[0].Timestamp),
		},
		Samples: samples,
	}

	// Calculate summary statistics
	export.Summary = e.calculateSummary(samples)

	// Marshal to JSON with pretty printing
	data, err := json.MarshalIndent(export, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal JSON: %w", err)
	}

	// Write to file
	file, err := os.Create(e.filePath)
	if err != nil {
		return fmt.Errorf("failed to create file %s: %w", e.filePath, err)
	}
	defer file.Close()

	if _, err := file.Write(data); err != nil {
		return fmt.Errorf("failed to write JSON data: %w", err)
	}

	return nil
}

// JSONExport represents the complete JSON export structure
type JSONExport struct {
	Metadata ExportMetadata          `json:"metadata"`
	Summary  SummaryStatistics       `json:"summary"`
	Samples  []metrics.SystemMetrics `json:"samples"`
}

// ExportMetadata contains information about the export
type ExportMetadata struct {
	Version     string        `json:"version"`
	ExportTime  time.Time     `json:"export_time"`
	SampleCount int           `json:"sample_count"`
	StartTime   time.Time     `json:"start_time"`
	EndTime     time.Time     `json:"end_time"`
	Duration    time.Duration `json:"duration"`
}

// SummaryStatistics contains summary statistics for the exported data
type SummaryStatistics struct {
	CPU     CPUSummary     `json:"cpu"`
	Memory  MemorySummary  `json:"memory"`
	Disk    DiskSummary    `json:"disk"`
	Network NetworkSummary `json:"network"`
	GPU     *GPUSummary    `json:"gpu,omitempty"`
}

// CPUSummary contains CPU summary statistics
type CPUSummary struct {
	OverallUsage struct {
		Average float64 `json:"average"`
		Min     float64 `json:"min"`
		Max     float64 `json:"max"`
	} `json:"overall_usage"`
	LoadAverage struct {
		Avg1Min  float64 `json:"avg_1min"`
		Avg5Min  float64 `json:"avg_5min"`
		Avg15Min float64 `json:"avg_15min"`
	} `json:"load_average"`
}

// MemorySummary contains memory summary statistics
type MemorySummary struct {
	Usage struct {
		Average float64 `json:"average"`
		Min     float64 `json:"min"`
		Max     float64 `json:"max"`
	} `json:"usage_percent"`
	TotalBytes uint64 `json:"total_bytes"`
}

// DiskSummary contains disk I/O summary statistics
type DiskSummary struct {
	TotalReadBytes  uint64  `json:"total_read_bytes"`
	TotalWriteBytes uint64  `json:"total_write_bytes"`
	TotalReadOps    uint64  `json:"total_read_ops"`
	TotalWriteOps   uint64  `json:"total_write_ops"`
	AvgReadLatency  float64 `json:"avg_read_latency"`
	AvgWriteLatency float64 `json:"avg_write_latency"`
}

// NetworkSummary contains network summary statistics
type NetworkSummary struct {
	TotalRxBytes   uint64 `json:"total_rx_bytes"`
	TotalTxBytes   uint64 `json:"total_tx_bytes"`
	TotalRxPackets uint64 `json:"total_rx_packets"`
	TotalTxPackets uint64 `json:"total_tx_packets"`
	TotalErrors    uint64 `json:"total_errors"`
}

// GPUSummary contains GPU summary statistics
type GPUSummary struct {
	DeviceCount      int     `json:"device_count"`
	AverageUtil      float64 `json:"average_utilization"`
	AverageMemUsage  float64 `json:"average_memory_usage"`
	AverageTemp      float64 `json:"average_temperature"`
	AveragePower     float64 `json:"average_power"`
	MaxTemp          int     `json:"max_temperature"`
	MaxPower         int     `json:"max_power"`
}

// calculateSummary computes summary statistics from samples
func (e *JSONExporter) calculateSummary(samples []metrics.SystemMetrics) SummaryStatistics {
	summary := SummaryStatistics{}

	if len(samples) == 0 {
		return summary
	}

	// Initialize min/max values
	summary.CPU.OverallUsage.Min = samples[0].CPU.Overall
	summary.CPU.OverallUsage.Max = samples[0].CPU.Overall
	summary.Memory.Usage.Min = samples[0].Memory.UsagePercent
	summary.Memory.Usage.Max = samples[0].Memory.UsagePercent
	summary.Memory.TotalBytes = samples[0].Memory.Total

	// Accumulate values for averages
	var cpuSum, memorySum float64
	var loadSum1, loadSum5, loadSum15 float64

	for _, sample := range samples {
		// CPU statistics
		cpuSum += sample.CPU.Overall
		if sample.CPU.Overall < summary.CPU.OverallUsage.Min {
			summary.CPU.OverallUsage.Min = sample.CPU.Overall
		}
		if sample.CPU.Overall > summary.CPU.OverallUsage.Max {
			summary.CPU.OverallUsage.Max = sample.CPU.Overall
		}

		loadSum1 += sample.CPU.LoadAvg1
		loadSum5 += sample.CPU.LoadAvg5
		loadSum15 += sample.CPU.LoadAvg15

		// Memory statistics
		memorySum += sample.Memory.UsagePercent
		if sample.Memory.UsagePercent < summary.Memory.Usage.Min {
			summary.Memory.Usage.Min = sample.Memory.UsagePercent
		}
		if sample.Memory.UsagePercent > summary.Memory.Usage.Max {
			summary.Memory.Usage.Max = sample.Memory.UsagePercent
		}

		// Disk I/O statistics
		summary.Disk.TotalReadBytes += sample.Disk.Total.ReadBytes
		summary.Disk.TotalWriteBytes += sample.Disk.Total.WriteBytes
		summary.Disk.TotalReadOps += sample.Disk.Total.ReadOps
		summary.Disk.TotalWriteOps += sample.Disk.Total.WriteOps

		// Network statistics
		summary.Network.TotalRxBytes += sample.Network.Total.RxBytes
		summary.Network.TotalTxBytes += sample.Network.Total.TxBytes
		summary.Network.TotalRxPackets += sample.Network.Total.RxPackets
		summary.Network.TotalTxPackets += sample.Network.Total.TxPackets
		summary.Network.TotalErrors += sample.Network.Total.RxErrors + sample.Network.Total.TxErrors

		// GPU statistics (if available)
		if sample.GPU != nil && len(sample.GPU.Devices) > 0 {
			if summary.GPU == nil {
				summary.GPU = &GPUSummary{
					DeviceCount: len(sample.GPU.Devices),
				}
			}

			for _, device := range sample.GPU.Devices {
				summary.GPU.AverageUtil += float64(device.Utilization)
				summary.GPU.AverageMemUsage += device.Memory.UsagePercent
				summary.GPU.AverageTemp += float64(device.Temperature)
				summary.GPU.AveragePower += float64(device.PowerUsage)

				if device.Temperature > summary.GPU.MaxTemp {
					summary.GPU.MaxTemp = device.Temperature
				}
				if device.PowerUsage > summary.GPU.MaxPower {
					summary.GPU.MaxPower = device.PowerUsage
				}
			}
		}
	}

	// Calculate averages
	sampleCount := float64(len(samples))
	summary.CPU.OverallUsage.Average = cpuSum / sampleCount
	summary.CPU.LoadAverage.Avg1Min = loadSum1 / sampleCount
	summary.CPU.LoadAverage.Avg5Min = loadSum5 / sampleCount
	summary.CPU.LoadAverage.Avg15Min = loadSum15 / sampleCount
	summary.Memory.Usage.Average = memorySum / sampleCount

	// Calculate disk latencies
	if summary.Disk.TotalReadOps > 0 {
		summary.Disk.AvgReadLatency = 0 // Would need to track latency sum
	}
	if summary.Disk.TotalWriteOps > 0 {
		summary.Disk.AvgWriteLatency = 0 // Would need to track latency sum
	}

	// Calculate GPU averages
	if summary.GPU != nil && sampleCount > 0 {
		deviceCount := float64(summary.GPU.DeviceCount)
		totalSamples := sampleCount * deviceCount

		summary.GPU.AverageUtil /= totalSamples
		summary.GPU.AverageMemUsage /= totalSamples
		summary.GPU.AverageTemp /= totalSamples
		summary.GPU.AveragePower /= totalSamples
	}

	return summary
}

// ExportSingle exports a single metrics sample to JSON
func (e *JSONExporter) ExportSingle(sample *metrics.SystemMetrics) error {
	data, err := json.MarshalIndent(sample, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal JSON: %w", err)
	}

	file, err := os.Create(e.filePath)
	if err != nil {
		return fmt.Errorf("failed to create file %s: %w", e.filePath, err)
	}
	defer file.Close()

	if _, err := file.Write(data); err != nil {
		return fmt.Errorf("failed to write JSON data: %w", err)
	}

	return nil
}