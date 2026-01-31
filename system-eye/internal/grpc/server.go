package grpc

import (
	"context"
	"fmt"
	"log"
	"time"

	pb "github.com/teamovercooked/system-eye/api/proto"
	"github.com/teamovercooked/system-eye/internal/metrics"
	"github.com/teamovercooked/system-eye/internal/monitor"
	"google.golang.org/protobuf/types/known/timestamppb"
)

// MetricsServer implements the gRPC MetricsService
type MetricsServer struct {
	pb.UnimplementedMetricsServiceServer
	collector *monitor.Collector
}

// NewMetricsServer creates a new gRPC metrics server
func NewMetricsServer(collector *monitor.Collector) *MetricsServer {
	return &MetricsServer{
		collector: collector,
	}
}

// GetMetrics implements the unary RPC for getting a single metrics snapshot
func (s *MetricsServer) GetMetrics(ctx context.Context, req *pb.MetricsRequest) (*pb.SystemMetrics, error) {
	// Collect metrics
	metrics, err := s.collector.Collect()
	if err != nil {
		return nil, fmt.Errorf("failed to collect metrics: %w", err)
	}

	// Convert to protobuf
	pbMetrics := convertMetricsToProto(metrics, req.IncludeGpu, req.IncludeProcesses)
	return pbMetrics, nil
}

// StreamMetrics implements the server streaming RPC for continuous metrics
func (s *MetricsServer) StreamMetrics(req *pb.StreamRequest, stream pb.MetricsService_StreamMetricsServer) error {
	// Default interval if not specified
	interval := time.Duration(req.IntervalMs) * time.Millisecond
	if interval == 0 {
		interval = 1 * time.Second
	}

	log.Printf("Starting metrics stream with interval: %v", interval)

	// Use the collector's continuous collection
	ctx := stream.Context()
	metricsChan, errorChan := s.collector.CollectContinuous(ctx, interval)

	for {
		select {
		case <-ctx.Done():
			log.Println("Client disconnected from metrics stream")
			return ctx.Err()

		case err := <-errorChan:
			if err != nil {
				log.Printf("Error in metrics collection: %v", err)
				// Continue streaming despite errors
			}

		case metricsData, ok := <-metricsChan:
			if !ok {
				// Channel closed
				return nil
			}

			// Convert to protobuf
			pbMetrics := convertMetricsToProto(metricsData, req.IncludeGpu, req.IncludeProcesses)

			// Send to client
			if err := stream.Send(pbMetrics); err != nil {
				log.Printf("Error sending metrics: %v", err)
				return err
			}
		}
	}
}

// convertMetricsToProto converts internal metrics to protobuf format
func convertMetricsToProto(m *metrics.SystemMetrics, includeGPU, includeProcesses bool) *pb.SystemMetrics {
	pbMetrics := &pb.SystemMetrics{
		Timestamp: timestamppb.New(m.Timestamp),
		Cpu:       convertCPUMetrics(&m.CPU),
		Memory:    convertMemoryMetrics(&m.Memory),
		Disk:      convertDiskMetrics(&m.Disk),
		Network:   convertNetworkMetrics(&m.Network),
	}

	// Optional GPU metrics
	if includeGPU && m.GPU != nil {
		pbMetrics.Gpu = convertGPUMetrics(m.GPU)
	}

	// Optional Process metrics (includes deadlock detection fields)
	if includeProcesses && m.Processes != nil {
		pbMetrics.Processes = convertProcessMetrics(m.Processes)
	}

	return pbMetrics
}

func convertCPUMetrics(cpu *metrics.CPUMetrics) *pb.CPUMetrics {
	return &pb.CPUMetrics{
		Overall:         cpu.Overall,
		PerCore:         cpu.PerCore,
		LoadAvg_1:       cpu.LoadAvg1,
		LoadAvg_5:       cpu.LoadAvg5,
		LoadAvg_15:      cpu.LoadAvg15,
		ContextSwitches: cpu.ContextSwitches,
		Interrupts:      cpu.Interrupts,
	}
}

func convertMemoryMetrics(mem *metrics.MemoryMetrics) *pb.MemoryMetrics {
	return &pb.MemoryMetrics{
		Total:        mem.Total,
		Used:         mem.Used,
		Free:         mem.Free,
		Available:    mem.Available,
		Cached:       mem.Cached,
		Buffers:      mem.Buffers,
		UsagePercent: mem.UsagePercent,
	}
}

func convertDiskMetrics(disk *metrics.DiskMetrics) *pb.DiskMetrics {
	devices := make([]*pb.DiskDevice, len(disk.Devices))
	for i, dev := range disk.Devices {
		devices[i] = &pb.DiskDevice{
			Name: dev.Name,
			Io:   convertDiskIO(&dev.IO),
		}
	}

	return &pb.DiskMetrics{
		Devices: devices,
		Total:   convertDiskIO(&disk.Total),
	}
}

func convertDiskIO(io *metrics.DiskIO) *pb.DiskIO {
	pbIO := &pb.DiskIO{
		ReadBytes:  io.ReadBytes,
		WriteBytes: io.WriteBytes,
		ReadOps:    io.ReadOps,
		WriteOps:   io.WriteOps,
	}

	if io.ReadLatency > 0 {
		pbIO.ReadLatency = &io.ReadLatency
	}
	if io.WriteLatency > 0 {
		pbIO.WriteLatency = &io.WriteLatency
	}

	return pbIO
}

func convertNetworkMetrics(net *metrics.NetworkMetrics) *pb.NetworkMetrics {
	interfaces := make([]*pb.NetworkInterface, len(net.Interfaces))
	for i, iface := range net.Interfaces {
		interfaces[i] = &pb.NetworkInterface{
			Name:    iface.Name,
			Traffic: convertNetworkTraffic(&iface.Traffic),
		}
	}

	return &pb.NetworkMetrics{
		Interfaces: interfaces,
		Total:      convertNetworkTraffic(&net.Total),
	}
}

func convertNetworkTraffic(traffic *metrics.NetworkTraffic) *pb.NetworkTraffic {
	return &pb.NetworkTraffic{
		RxBytes:   traffic.RxBytes,
		TxBytes:   traffic.TxBytes,
		RxPackets: traffic.RxPackets,
		TxPackets: traffic.TxPackets,
		RxErrors:  traffic.RxErrors,
		TxErrors:  traffic.TxErrors,
		RxDropped: traffic.RxDropped,
		TxDropped: traffic.TxDropped,
	}
}

func convertGPUMetrics(gpu *metrics.GPUMetrics) *pb.GPUMetrics {
	devices := make([]*pb.GPUDevice, len(gpu.Devices))
	for i, dev := range gpu.Devices {
		devices[i] = &pb.GPUDevice{
			Index:       int32(dev.Index),
			Name:        dev.Name,
			Uuid:        dev.UUID,
			Utilization: int32(dev.Utilization),
			Memory: &pb.GPUMemory{
				Total:        dev.Memory.Total,
				Used:         dev.Memory.Used,
				Free:         dev.Memory.Free,
				UsagePercent: dev.Memory.UsagePercent,
			},
			Temperature: int32(dev.Temperature),
			PowerUsage:  int32(dev.PowerUsage),
			FanSpeed:    int32(dev.FanSpeed),
		}
	}

	return &pb.GPUMetrics{
		Devices: devices,
	}
}

func convertProcessMetrics(proc *metrics.ProcessMetrics) *pb.ProcessMetrics {
	processes := make([]*pb.ProcessInfo, len(proc.Processes))
	for i, p := range proc.Processes {
		processes[i] = convertProcessInfo(&p)
	}

	topCPU := make([]*pb.ProcessInfo, len(proc.TopCPU))
	for i, p := range proc.TopCPU {
		topCPU[i] = convertProcessInfo(&p)
	}

	topMemory := make([]*pb.ProcessInfo, len(proc.TopMemory))
	for i, p := range proc.TopMemory {
		topMemory[i] = convertProcessInfo(&p)
	}

	var topGPU []*pb.ProcessInfo
	if proc.TopGPU != nil {
		topGPU = make([]*pb.ProcessInfo, len(proc.TopGPU))
		for i, p := range proc.TopGPU {
			topGPU[i] = convertProcessInfo(&p)
		}
	}

	return &pb.ProcessMetrics{
		Processes:     processes,
		TopCpu:        topCPU,
		TopMemory:     topMemory,
		TopGpu:        topGPU,
		TotalCount:    int32(proc.Total),
		RunningCount:  int32(proc.Running),
		SleepingCount: int32(proc.Sleeping),
	}
}

func convertProcessInfo(p *metrics.ProcessInfo) *pb.ProcessInfo {
	pbProc := &pb.ProcessInfo{
		Pid:           p.PID,
		Ppid:          p.PPID,
		Name:          p.Name,
		State:         p.State,
		CpuPercent:    p.CPUPercent,
		MemoryPercent: p.MemoryPercent,
		MemoryRss:     p.MemoryRSS,
		MemoryVirtual: p.MemoryVirtual,
		NumThreads:    p.NumThreads,
		ReadBytes:     p.ReadBytes,
		WriteBytes:    p.WriteBytes,
	}

	if p.Command != "" {
		pbProc.Command = &p.Command
	}
	if p.OpenFiles > 0 {
		pbProc.OpenFiles = &p.OpenFiles
	}
	if p.SyscallCount > 0 {
		pbProc.SyscallCount = &p.SyscallCount
	}
	if p.ContextSwitches > 0 {
		pbProc.ContextSwitches = &p.ContextSwitches
	}
	if p.GPUMemory > 0 {
		pbProc.GpuMemory = &p.GPUMemory
	}
	if p.GPUUtilization > 0 {
		pbProc.GpuUtilization = &p.GPUUtilization
	}

	// Deadlock detection fields
	if p.WaitChannel != "" {
		pbProc.WaitChannel = &p.WaitChannel
	}
	if p.WaitState != "" {
		pbProc.WaitState = &p.WaitState
	}
	if p.WaitTimeMs > 0 {
		pbProc.WaitTimeMs = &p.WaitTimeMs
	}
	if len(p.HeldLocks) > 0 {
		pbProc.HeldLocks = convertFileLocks(p.HeldLocks)
	}
	if len(p.WaitingLocks) > 0 {
		pbProc.WaitingLocks = convertFileLocks(p.WaitingLocks)
	}
	if p.BlockedThreads > 0 {
		pbProc.BlockedThreads = &p.BlockedThreads
	}
	if p.VoluntaryCtxSwitches > 0 {
		pbProc.VoluntaryCtxSwitches = &p.VoluntaryCtxSwitches
	}
	if p.InvoluntaryCtxSwitches > 0 {
		pbProc.InvoluntaryCtxSwitches = &p.InvoluntaryCtxSwitches
	}
	if p.WaitingOnPID > 0 {
		pbProc.WaitingOnPid = &p.WaitingOnPID
	}
	if len(p.BlockingPIDs) > 0 {
		pbProc.BlockingPids = p.BlockingPIDs
	}
	if p.CurrentSyscall != "" {
		pbProc.CurrentSyscall = &p.CurrentSyscall
	}
	if p.SyscallDurationMs > 0 {
		pbProc.SyscallDurationMs = &p.SyscallDurationMs
	}
	if p.IOWaitTimeMs > 0 {
		pbProc.IoWaitTimeMs = &p.IOWaitTimeMs
	}
	if p.CPUTimeDelta > 0 {
		pbProc.CpuTimeDelta = &p.CPUTimeDelta
	}

	return pbProc
}

func convertFileLocks(locks []metrics.FileLock) []*pb.FileLock {
	pbLocks := make([]*pb.FileLock, len(locks))
	for i, lock := range locks {
		pbLocks[i] = &pb.FileLock{
			Path:     lock.Path,
			LockType: lock.LockType,
			Inode:    lock.Inode,
			Pid:      lock.PID,
		}
	}
	return pbLocks
}
