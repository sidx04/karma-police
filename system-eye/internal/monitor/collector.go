package monitor

import (
	"context"
	"fmt"
	"log"
	"os"
	"sync"
	"time"

	"github.com/teamovercooked/system-eye/internal/metrics"
)

// Collector coordinates all monitoring components
type Collector struct {
	config *metrics.CollectorConfig

	// Monitors
	cpuMonitor     *CPUMonitor
	memoryMonitor  *MemoryMonitor
	diskMonitor    *DiskMonitor
	networkMonitor *NetworkMonitor
	gpuMonitor     GPUMonitor // Interface to be implemented
	processMonitor ProcessMonitor // Interface for process monitoring

	// State
	mu     sync.RWMutex
	closed bool
}

// GPUMonitor interface for GPU monitoring (will be implemented with NVML)
type GPUMonitor interface {
	GetGPUMetrics() (*metrics.GPUMetrics, error)
	IsAvailable() bool
	Close() error
}

// ProcessMonitor interface for process monitoring (eBPF-based)
type ProcessMonitor interface {
	GetProcessMetrics() (*metrics.ProcessMetrics, error)
	IsAvailable() bool
	Close() error
}

// NewCollector creates a new metrics collector
func NewCollector(config *metrics.CollectorConfig) (*Collector, error) {
	if config == nil {
		config = &metrics.CollectorConfig{
			EnableEBPF:              true,
			EnableGPU:               true,
			EnableDeadlockDetection: true,
			SampleRate:              1 * time.Second,
			HistorySize:             300,
		}
	}

	collector := &Collector{
		config:         config,
		cpuMonitor:     NewCPUMonitor(),
		memoryMonitor:  NewMemoryMonitor(),
		diskMonitor:    NewDiskMonitor(),
		networkMonitor: NewNetworkMonitor(),
	}

	// Initialize GPU monitoring if enabled
	if config.EnableGPU {
		// Use a separate goroutine and recover from panics
		func() {
			defer func() {
				if r := recover(); r != nil {
					log.Printf("Warning: GPU monitoring initialization panicked: %v", r)
				}
			}()

			gpuMonitor, err := NewGPUMonitorNVML()
			if err != nil {
				log.Printf("Warning: GPU monitoring not available: %v", err)
				// Continue without GPU monitoring
			} else {
				collector.gpuMonitor = gpuMonitor
				log.Println("GPU monitoring enabled")
			}
		}()
	}

	// Initialize process monitoring
	// Using /proc filesystem instead of eBPF for better compatibility
	processMonitor := NewProcessMonitor()
	if processMonitor.IsAvailable() {
		collector.processMonitor = processMonitor
		log.Println("Process monitoring enabled (/proc)")
	} else {
		log.Println("Warning: Process monitoring not available")
	}

	log.Printf("Metrics collector initialized (eBPF: %v, GPU: %v)",
		config.EnableEBPF, config.EnableGPU && collector.gpuMonitor != nil)

	return collector, nil
}

// Collect gathers metrics from all enabled monitors
func (c *Collector) Collect() (*metrics.SystemMetrics, error) {
	c.mu.RLock()
	if c.closed {
		c.mu.RUnlock()
		return nil, fmt.Errorf("collector is closed")
	}
	c.mu.RUnlock()

	systemMetrics := &metrics.SystemMetrics{
		Timestamp: time.Now(),
	}

	// Collect CPU metrics
	if cpuMetrics, err := c.cpuMonitor.GetCPUMetrics(); err != nil {
		log.Printf("Warning: failed to collect CPU metrics: %v", err)
	} else {
		systemMetrics.CPU = cpuMetrics
	}

	// Collect memory metrics
	if memMetrics, err := c.memoryMonitor.GetMemoryMetrics(); err != nil {
		log.Printf("Warning: failed to collect memory metrics: %v", err)
	} else {
		systemMetrics.Memory = memMetrics
	}

	// Collect disk metrics
	if diskMetrics, err := c.diskMonitor.GetDiskMetrics(); err != nil {
		log.Printf("Warning: failed to collect disk metrics: %v", err)
	} else {
		systemMetrics.Disk = diskMetrics
	}

	// Collect network metrics
	if netMetrics, err := c.networkMonitor.GetNetworkMetrics(); err != nil {
		log.Printf("Warning: failed to collect network metrics: %v", err)
	} else {
		systemMetrics.Network = netMetrics
	}

	// Collect GPU metrics if available
	if c.gpuMonitor != nil && c.gpuMonitor.IsAvailable() {
		if gpuMetrics, err := c.gpuMonitor.GetGPUMetrics(); err != nil {
			log.Printf("Warning: failed to collect GPU metrics: %v", err)
		} else {
			systemMetrics.GPU = gpuMetrics
		}
	}

	// Collect process metrics if available
	if c.processMonitor != nil && c.processMonitor.IsAvailable() {
		if procMetrics, err := c.processMonitor.GetProcessMetrics(); err != nil {
			log.Printf("Warning: failed to collect process metrics: %v", err)
		} else {
			systemMetrics.Processes = procMetrics
			// Raw metrics are enriched with deadlock detection fields
			// Python (system-guardian) will perform the actual deadlock detection
		}
	}

	return systemMetrics, nil
}

// Close shuts down the collector and releases resources
func (c *Collector) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return nil
	}

	c.closed = true

	// Close GPU monitor if available
	if c.gpuMonitor != nil {
		if err := c.gpuMonitor.Close(); err != nil {
			log.Printf("Warning: error closing GPU monitor: %v", err)
		}
	}

	// Close process monitor if available
	if c.processMonitor != nil {
		if err := c.processMonitor.Close(); err != nil {
			log.Printf("Warning: error closing process monitor: %v", err)
		}
	}

	log.Println("Metrics collector closed")
	return nil
}

// GetConfig returns the collector configuration
func (c *Collector) GetConfig() *metrics.CollectorConfig {
	return c.config
}

// GetSystemInfo returns basic system information
func (c *Collector) GetSystemInfo() map[string]interface{} {
	info := map[string]interface{}{
		"ebpf_enabled": c.config.EnableEBPF,
		"gpu_enabled":  c.config.EnableGPU && c.gpuMonitor != nil,
		"sample_rate":  c.config.SampleRate.String(),
		"history_size": c.config.HistorySize,
	}

	// Add GPU info if available
	if c.gpuMonitor != nil && c.gpuMonitor.IsAvailable() {
		info["gpu_available"] = true
		// Could add more detailed GPU info here
	} else {
		info["gpu_available"] = false
	}

	// Add system capabilities
	info["capabilities"] = c.getSystemCapabilities()

	return info
}

// getSystemCapabilities checks what monitoring capabilities are available
func (c *Collector) getSystemCapabilities() map[string]bool {
	capabilities := map[string]bool{}

	// Check /proc filesystem access
	capabilities["proc_stat"] = c.canReadFile("/proc/stat")
	capabilities["proc_meminfo"] = c.canReadFile("/proc/meminfo")
	capabilities["proc_diskstats"] = c.canReadFile("/proc/diskstats")
	capabilities["proc_net_dev"] = c.canReadFile("/proc/net/dev")
	capabilities["proc_loadavg"] = c.canReadFile("/proc/loadavg")

	// Check eBPF capabilities (would need more sophisticated checking)
	capabilities["ebpf"] = c.config.EnableEBPF // Simplified for now

	// GPU capabilities
	capabilities["gpu"] = c.gpuMonitor != nil && c.gpuMonitor.IsAvailable()

	return capabilities
}

// canReadFile checks if a file can be read
func (c *Collector) canReadFile(path string) bool {
	// Try to read first few bytes
	file, err := os.Open(path)
	if err != nil {
		return false
	}
	defer file.Close()

	buf := make([]byte, 1)
	_, err = file.Read(buf)
	return err == nil
}

// CollectContinuous starts continuous metrics collection
func (c *Collector) CollectContinuous(ctx context.Context, interval time.Duration) (<-chan *metrics.SystemMetrics, <-chan error) {
	metricsChan := make(chan *metrics.SystemMetrics, 10)
	errorChan := make(chan error, 10)

	go func() {
		defer close(metricsChan)
		defer close(errorChan)

		ticker := time.NewTicker(interval)
		defer ticker.Stop()

		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				metrics, err := c.Collect()
				if err != nil {
					select {
					case errorChan <- err:
					default: // Don't block if error channel is full
					}
					continue
				}

				select {
				case metricsChan <- metrics:
				case <-ctx.Done():
					return
				default:
					// Don't block if metrics channel is full
					log.Println("Warning: metrics channel full, dropping sample")
				}
			}
		}
	}()

	return metricsChan, errorChan
}