// internal/monitor/gpu_smi.go
package monitor

import (
	"bytes"
	"encoding/csv"
	"fmt"
	"log"
	"os/exec"
	"strconv"
	"strings"

	"github.com/teamovercooked/system-eye/internal/metrics"
)

// GPUMonitorNVML implements GPU monitoring using nvidia-smi
type GPUMonitorNVML struct {
	available bool
	devices   []string
}

// NewGPUMonitorNVML creates a new NVIDIA GPU monitor using nvidia-smi
func NewGPUMonitorNVML() (*GPUMonitorNVML, error) {
	monitor := &GPUMonitorNVML{
		available: false,
		devices:   make([]string, 0),
	}

	// Check if nvidia-smi is available
	if err := monitor.checkNVIDIASMI(); err != nil {
		log.Printf("[GPU] nvidia-smi not available: %v", err)
		return monitor, nil // Return monitor but disabled
	}

	// Get device list
	if err := monitor.initDevices(); err != nil {
		log.Printf("[GPU] Failed to initialize devices: %v", err)
		return monitor, nil
	}

	monitor.available = true
	log.Printf("[GPU] GPU monitoring initialized with %d devices (via nvidia-smi)", len(monitor.devices))
	for i, name := range monitor.devices {
		log.Printf("[GPU]   GPU %d: %s", i, name)
	}

	return monitor, nil
}

// checkNVIDIASMI verifies nvidia-smi is available
func (m *GPUMonitorNVML) checkNVIDIASMI() error {
	cmd := exec.Command("nvidia-smi", "--version")
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("nvidia-smi not found or not executable")
	}
	return nil
}

// initDevices gets the list of GPU devices
func (m *GPUMonitorNVML) initDevices() error {
	cmd := exec.Command("nvidia-smi", "-L")
	output, err := cmd.Output()
	if err != nil {
		return fmt.Errorf("failed to list GPUs: %w", err)
	}

	// Parse output like: "GPU 0: NVIDIA GeForce RTX 3070 ... (UUID: ...)"
	lines := strings.Split(string(output), "\n")
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		// Extract GPU name between "GPU X: " and " (UUID"
		if idx := strings.Index(line, ": "); idx > 0 {
			name := line[idx+2:]
			if uuidIdx := strings.Index(name, " (UUID"); uuidIdx > 0 {
				name = name[:uuidIdx]
			}
			m.devices = append(m.devices, name)
		}
	}

	if len(m.devices) == 0 {
		return fmt.Errorf("no GPU devices found")
	}

	return nil
}

// GetGPUMetrics collects GPU metrics using nvidia-smi
func (m *GPUMonitorNVML) GetGPUMetrics() (*metrics.GPUMetrics, error) {
	if !m.available {
		return &metrics.GPUMetrics{Devices: []metrics.GPUDevice{}}, nil
	}

	cmd := exec.Command(
		"nvidia-smi",
		"--query-gpu=index,name,uuid,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,fan.speed",
		"--format=csv,noheader,nounits",
	)

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		log.Printf("[GPU] nvidia-smi error: %v, stderr: %s", err, stderr.String())
		return &metrics.GPUMetrics{Devices: []metrics.GPUDevice{}}, nil
	}

	// Parse CSV output
	reader := csv.NewReader(strings.NewReader(stdout.String()))
	records, err := reader.ReadAll()
	if err != nil {
		log.Printf("[GPU] Failed to parse nvidia-smi output: %v", err)
		return &metrics.GPUMetrics{Devices: []metrics.GPUDevice{}}, nil
	}

	var devices []metrics.GPUDevice
	for _, record := range records {
		if len(record) < 9 {
			continue
		}

		device := metrics.GPUDevice{}

		// Index
		if idx, err := strconv.Atoi(strings.TrimSpace(record[0])); err == nil {
			device.Index = idx
		}

		// Name
		device.Name = strings.TrimSpace(record[1])

		// UUID
		device.UUID = strings.TrimSpace(record[2])

		// Utilization
		if util, err := strconv.Atoi(strings.TrimSpace(record[3])); err == nil {
			device.Utilization = util
		}

		// Memory
		memUsed, _ := strconv.ParseUint(strings.TrimSpace(record[4]), 10, 64)
		memTotal, _ := strconv.ParseUint(strings.TrimSpace(record[5]), 10, 64)

		device.Memory = metrics.GPUMemory{
			Used:  memUsed * 1024 * 1024, // Convert MiB to bytes
			Total: memTotal * 1024 * 1024,
			Free:  (memTotal - memUsed) * 1024 * 1024,
		}
		if memTotal > 0 {
			device.Memory.UsagePercent = float64(memUsed) / float64(memTotal) * 100.0
		}

		// Temperature
		if temp, err := strconv.Atoi(strings.TrimSpace(record[6])); err == nil {
			device.Temperature = temp
		}

		// Power (convert to watts)
		if power, err := strconv.ParseFloat(strings.TrimSpace(record[7]), 64); err == nil {
			device.PowerUsage = int(power)
		}

		// Fan speed
		fanStr := strings.TrimSpace(record[8])
		if fanStr != "[N/A]" && fanStr != "" {
			if fan, err := strconv.Atoi(fanStr); err == nil {
				device.FanSpeed = fan
			}
		}

		devices = append(devices, device)
	}

	return &metrics.GPUMetrics{
		Devices: devices,
	}, nil
}

// IsAvailable returns whether GPU monitoring is available
func (m *GPUMonitorNVML) IsAvailable() bool {
	return m.available
}

// Close is a no-op for nvidia-smi based monitoring
func (m *GPUMonitorNVML) Close() error {
	log.Println("[GPU] GPU monitor closed")
	return nil
}

// GetDeviceCount returns the number of available GPU devices
func (m *GPUMonitorNVML) GetDeviceCount() int {
	return len(m.devices)
}

// GetDeviceNames returns the names of available GPU devices
func (m *GPUMonitorNVML) GetDeviceNames() []string {
	return m.devices
}
