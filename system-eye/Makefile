# System's Eye Makefile
.PHONY: all build test clean deps run-terminal run-web docker-build docker-run

BINARY_NAME=system-eye
MAIN_PATH=./cmd/system-eye
BUILD_FLAGS=-ldflags "-X main.appVersion=$(shell git describe --tags --always --dirty 2>/dev/null || echo 'dev')"

all: deps build

# Build the binary
build:
	@echo "Building $(BINARY_NAME)..."
	go build $(BUILD_FLAGS) -o $(BINARY_NAME) $(MAIN_PATH)

# Install dependencies
deps:
	@echo "Installing dependencies..."
	go mod download
	go mod tidy

# Run tests
test:
	@echo "Running tests..."
	go test -v ./...

# Run with coverage
test-coverage:
	@echo "Running tests with coverage..."
	go test -v -coverprofile=coverage.out ./...
	go tool cover -html=coverage.out -o coverage.html

# Clean build artifacts
clean:
	@echo "Cleaning..."
	rm -f $(BINARY_NAME)
	rm -f *.json *.csv
	rm -f coverage.out coverage.html

# Development shortcuts
run-terminal: build
	@echo "Running terminal UI..."
	./$(BINARY_NAME)

run-web: build
	@echo "Running web interface on port 8080..."
	./$(BINARY_NAME) --web

run-export: build
	@echo "Exporting metrics for 10 seconds..."
	./$(BINARY_NAME) --export test_data --duration 10s --rate 100ms

# Docker commands
docker-build:
	@echo "Building Docker image..."
	docker build -t system-eye .

docker-run: docker-build
	@echo "Running in Docker container..."
	docker run --rm -it --privileged \
		--cap-add=SYS_ADMIN --cap-add=BPF --cap-add=NET_ADMIN \
		--cap-add=PERFMON --cap-add=SYS_PTRACE \
		-v /sys/kernel/btf:/sys/kernel/btf:ro \
		-v /sys/fs/bpf:/sys/fs/bpf \
		-v /sys/kernel/debug:/sys/kernel/debug \
		-v /proc:/host/proc:ro \
		-p 8080:8080 \
		system-eye

# Development helpers
fmt:
	@echo "Formatting code..."
	go fmt ./...

lint:
	@echo "Linting code..."
	golangci-lint run ./...

# Install development tools
dev-tools:
	@echo "Installing development tools..."
	go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest

# Generate mock files for testing
generate:
	@echo "Generating code..."
	go generate ./...

# Cross-compilation targets
build-linux:
	@echo "Building for Linux..."
	GOOS=linux GOARCH=amd64 go build $(BUILD_FLAGS) -o $(BINARY_NAME)-linux-amd64 $(MAIN_PATH)

build-arm64:
	@echo "Building for ARM64..."
	GOOS=linux GOARCH=arm64 go build $(BUILD_FLAGS) -o $(BINARY_NAME)-linux-arm64 $(MAIN_PATH)

# Help
help:
	@echo "Available targets:"
	@echo "  all          - Install deps and build"
	@echo "  build        - Build the binary"
	@echo "  deps         - Install dependencies"
	@echo "  test         - Run tests"
	@echo "  clean        - Clean build artifacts"
	@echo "  run-terminal - Run with terminal UI"
	@echo "  run-web      - Run with web interface"
	@echo "  run-export   - Export sample data"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run   - Run in Docker"
	@echo "  fmt          - Format code"
	@echo "  lint         - Lint code"
	@echo "  help         - Show this help"