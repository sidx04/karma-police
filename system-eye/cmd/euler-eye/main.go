package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"log"
	"net"
	"os"
	"os/signal"
	"syscall"
	"time"

	pb "github.com/teamovercooked/system-eye/api/proto"
	"github.com/teamovercooked/system-eye/internal/export"
	grpcServer "github.com/teamovercooked/system-eye/internal/grpc"
	"github.com/teamovercooked/system-eye/internal/metrics"
	"github.com/teamovercooked/system-eye/internal/monitor"
	"github.com/teamovercooked/system-eye/internal/ui"
	"google.golang.org/grpc"
)

const (
	appName    = "System's Eye"
	appVersion = "1.0.0"
)

func main() {
	// Command line flags
	var (
		showVersion    = flag.Bool("version", false, "Show version information")
		webMode        = flag.Bool("web", false, "Start web interface")
		webPort        = flag.String("port", "8080", "Web interface port")
		grpcMode       = flag.Bool("grpc", false, "Start gRPC server")
		grpcPort       = flag.String("grpc-port", "50051", "gRPC server port")
		exportPath     = flag.String("export", "", "Export metrics to file (JSON/CSV)")
		sampleRate     = flag.Duration("rate", 1*time.Second, "Metrics collection rate")
		enableEBPF     = flag.Bool("ebpf", true, "Enable eBPF monitoring (fallback to /proc if unavailable)")
		enableGPU      = flag.Bool("gpu", true, "Enable GPU monitoring")
		enableDeadlock = flag.Bool("deadlock", true, "Enable deadlock detection")
		historySize    = flag.Int("history", 300, "Number of historical samples to keep")
		duration       = flag.Duration("duration", 0, "Run for specific duration (0 = run forever)")
		verbose        = flag.Bool("verbose", false, "Enable verbose logging")
	)

	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "%s v%s - System Monitoring Tool\n", appName, appVersion)
		fmt.Fprintf(os.Stderr, "Usage: %s [options]\n\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "Options:\n")
		flag.PrintDefaults()
		fmt.Fprintf(os.Stderr, "\nExamples:\n")
		fmt.Fprintf(os.Stderr, "  %s                    # Terminal UI (default)\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  %s --web             # Web interface on port 8080\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  %s --export data     # Export to data.json and data.csv\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  %s --rate 500ms      # Collect metrics every 500ms\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  %s --duration 1h     # Run for 1 hour then exit\n", os.Args[0])
	}

	flag.Parse()

	if *showVersion {
		fmt.Printf("%s v%s\n", appName, appVersion)
		fmt.Printf("Built for openSystem OS with eBPF and NVML support\n")
		os.Exit(0)
	}

	// Setup logging
	if !*verbose {
		log.SetOutput(io.Discard)
	}

	log.Printf("Starting %s v%s", appName, appVersion)

	// Create metrics collector configuration
	config := &metrics.CollectorConfig{
		EnableEBPF:              *enableEBPF,
		EnableGPU:               *enableGPU,
		EnableDeadlockDetection: *enableDeadlock,
		SampleRate:              *sampleRate,
		HistorySize:             *historySize,
	}

	// Initialize metrics collector
	collector, err := monitor.NewCollector(config)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error initializing metrics collector: %v\n", err)
		os.Exit(1)
	}
	defer collector.Close()

	// Setup context for graceful shutdown
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Handle shutdown signals
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		<-sigChan
		log.Println("Shutdown signal received")
		cancel()
	}()

	// Add duration timeout if specified
	if *duration > 0 {
		go func() {
			time.Sleep(*duration)
			log.Printf("Duration timeout reached (%v)", *duration)
			cancel()
		}()
	}

	// Determine run mode
	if *exportPath != "" {
		// Export mode - collect data and write to files (exclusive mode)
		err = runExportMode(ctx, collector, *exportPath, *sampleRate, *duration)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}
	} else if *webMode || *grpcMode {
		// Server modes - web and gRPC can run together
		errChan := make(chan error, 2)

		// Start gRPC server if requested
		if *grpcMode {
			go func() {
				if err := runGRPCMode(ctx, collector, *grpcPort); err != nil {
					errChan <- fmt.Errorf("gRPC error: %w", err)
				}
			}()
		}

		// Start web server if requested
		if *webMode {
			go func() {
				if err := runWebMode(ctx, collector, *webPort); err != nil {
					errChan <- fmt.Errorf("web error: %w", err)
				}
			}()
		}

		// Wait for shutdown or error
		select {
		case <-ctx.Done():
			log.Println("Shutdown initiated")
		case err := <-errChan:
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}
	} else {
		// Terminal mode - interactive TUI (default)
		err = runTerminalMode(ctx, collector)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}
	}

	log.Println("System's Eye shutdown complete")
}

func runTerminalMode(ctx context.Context, collector metrics.MetricsCollector) error {
	log.Println("Starting terminal UI mode")

	terminalUI := ui.NewTerminalUI(collector)
	return terminalUI.Run(ctx)
}

func runWebMode(ctx context.Context, collector metrics.MetricsCollector, port string) error {
	log.Printf("Starting web server on port %s", port)

	webServer := ui.NewWebServer(collector, port)
	return webServer.Run(ctx)
}

func runGRPCMode(ctx context.Context, collector metrics.MetricsCollector, port string) error {
	log.Printf("Starting gRPC server on port %s", port)

	// Create TCP listener
	lis, err := net.Listen("tcp", ":"+port)
	if err != nil {
		return fmt.Errorf("failed to listen on port %s: %w", port, err)
	}

	// Create gRPC server
	s := grpc.NewServer()

	// Create and register metrics service
	metricsServer := grpcServer.NewMetricsServer(collector.(*monitor.Collector))
	pb.RegisterMetricsServiceServer(s, metricsServer)

	log.Printf("gRPC server listening on :%s", port)
	log.Println("Clients can connect using: localhost:" + port)

	// Start serving in a goroutine
	errChan := make(chan error, 1)
	go func() {
		if err := s.Serve(lis); err != nil {
			errChan <- fmt.Errorf("gRPC server error: %w", err)
		}
	}()

	// Wait for shutdown signal or error
	select {
	case <-ctx.Done():
		log.Println("Shutting down gRPC server...")
		s.GracefulStop()
		log.Println("gRPC server stopped")
		return nil
	case err := <-errChan:
		return err
	}
}

func runExportMode(ctx context.Context, collector metrics.MetricsCollector, basePath string, sampleRate, duration time.Duration) error {
	log.Printf("Starting export mode - writing to %s", basePath)

	// Create exporters
	jsonExporter := export.NewJSONExporter(basePath + ".json")
	csvExporter := export.NewCSVExporter(basePath + ".csv")

	// Initialize history
	history := &metrics.MetricsHistory{
		MaxSize: int(duration / sampleRate), // Size based on duration and sample rate
	}

	ticker := time.NewTicker(sampleRate)
	defer ticker.Stop()

	startTime := time.Now()
	sampleCount := 0

	for {
		select {
		case <-ctx.Done():
			log.Printf("Export completed: %d samples collected in %v", sampleCount, time.Since(startTime))

			// Export collected data
			if err := jsonExporter.Export(history.Samples); err != nil {
				return fmt.Errorf("JSON export failed: %w", err)
			}

			if err := csvExporter.Export(history.Samples); err != nil {
				return fmt.Errorf("CSV export failed: %w", err)
			}

			fmt.Printf("Data exported successfully:\n")
			fmt.Printf("  JSON: %s.json\n", basePath)
			fmt.Printf("  CSV:  %s.csv\n", basePath)
			fmt.Printf("  Samples: %d\n", sampleCount)
			fmt.Printf("  Duration: %v\n", time.Since(startTime))

			return nil

		case <-ticker.C:
			// Collect metrics
			systemMetrics, err := collector.Collect()
			if err != nil {
				log.Printf("Warning: failed to collect metrics: %v", err)
				continue
			}

			history.Add(*systemMetrics)
			sampleCount++

			if sampleCount%10 == 0 { // Log progress every 10 samples
				fmt.Printf("Collected %d samples...\n", sampleCount)
			}
		}
	}
}
