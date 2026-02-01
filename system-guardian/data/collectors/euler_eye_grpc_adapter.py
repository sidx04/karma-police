"""
gRPC Adapter for consuming real-time metrics from system-eye Go telemetry system.
"""

import grpc
import logging
from typing import Iterator, Optional
from dataclasses import dataclass
from datetime import datetime

# Import generated protobuf modules
try:
    from api.proto import metrics_pb2, metrics_pb2_grpc
except ImportError:
    import sys
    import os

    sys.path.insert(0, os.path.dirname(__file__))
    from api.proto import metrics_pb2

    metrics_pb2_grpc = None

logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """Python representation of system metrics from system-eye"""

    timestamp: datetime
    cpu_overall: float
    cpu_per_core: list
    memory_used: int
    memory_total: int
    memory_percent: float
    disk_read_bytes: int
    disk_write_bytes: int
    network_rx_bytes: int
    network_tx_bytes: int
    process_count: int = 0
    raw_proto: object = None  # Store raw protobuf for detailed access


class SystemEyeGRPCAdapter:
    """
    gRPC client adapter for consuming real-time metrics from system-eye.

    Example usage:
        adapter = SystemEyeGRPCAdapter("localhost:50051")
        adapter.connect()

        # Stream metrics
        for metrics in adapter.stream_metrics(interval_ms=1000):
            print(f"CPU: {metrics.cpu_overall}%")
    """

    def __init__(self, server_address: str = "localhost:50051"):
        self.server_address = server_address
        self.channel: Optional[grpc.Channel] = None
        self.stub = None

    def connect(self) -> bool:
        """Establish connection to the system-eye gRPC server."""
        try:
            self.channel = grpc.insecure_channel(self.server_address)
            grpc.channel_ready_future(self.channel).result(timeout=5)

            if metrics_pb2_grpc:
                self.stub = metrics_pb2_grpc.MetricsServiceStub(self.channel)
            else:
                logger.warning(
                    "gRPC service stub not available, using manual implementation"
                )
                self._create_manual_stub()

            logger.info(f"Connected to system-eye at {self.server_address}")
            return True

        except grpc.FutureTimeoutError:
            logger.error(f"Connection timeout to {self.server_address}")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to {self.server_address}: {e}")
            return False

    def _create_manual_stub(self):
        """Fallback if grpc stub isn't available."""
        self.stub = self.channel

    def get_metrics(
        self, include_gpu: bool = False, include_processes: bool = True
    ) -> Optional[SystemMetrics]:
        """Get a single snapshot of system metrics (unary RPC)."""
        if not self.channel:
            logger.error("Not connected. Call connect() first.")
            return None

        try:
            request = metrics_pb2.MetricsRequest(
                include_gpu=include_gpu, include_processes=include_processes
            )

            if metrics_pb2_grpc and self.stub:
                response = self.stub.GetMetrics(request, timeout=10)
            else:
                response = self.channel.unary_unary(
                    "/metrics.MetricsService/GetMetrics",
                    request_serializer=metrics_pb2.MetricsRequest.SerializeToString,
                    response_deserializer=metrics_pb2.SystemMetrics.FromString,
                )(request, timeout=10)

            return self._convert_proto_to_metrics(response)

        except grpc.RpcError as e:
            logger.error(f"RPC failed: {e.code()} - {e.details()}")
            return None
        except Exception as e:
            logger.error(f"Failed to get metrics: {e}")
            return None

    def stream_metrics(
        self,
        interval_ms: int = 1000,
        include_gpu: bool = False,
        include_processes: bool = True,
    ) -> Iterator[SystemMetrics]:
        """Stream metrics continuously from system-eye (server streaming RPC)."""
        if not self.channel:
            logger.error("Not connected. Call connect() first.")
            return

        try:
            request = metrics_pb2.StreamRequest(
                interval_ms=interval_ms,
                include_gpu=include_gpu,
                include_processes=include_processes,
            )

            logger.info(f"Starting metrics stream (interval: {interval_ms}ms)")

            if metrics_pb2_grpc and self.stub:
                response_stream = self.stub.StreamMetrics(request)
            else:
                response_stream = self.channel.unary_stream(
                    "/metrics.MetricsService/StreamMetrics",
                    request_serializer=metrics_pb2.StreamRequest.SerializeToString,
                    response_deserializer=metrics_pb2.SystemMetrics.FromString,
                )(request)

            for proto_metrics in response_stream:
                yield self._convert_proto_to_metrics(proto_metrics)

        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.CANCELLED:
                logger.info("Stream cancelled by client")
            else:
                logger.error(f"Stream RPC failed: {e.code()} - {e.details()}")
        except Exception as e:
            logger.error(f"Stream error: {e}")

    def _convert_proto_to_metrics(
        self, proto: metrics_pb2.SystemMetrics
    ) -> SystemMetrics:
        """Convert protobuf SystemMetrics to Python dataclass."""
        metrics = SystemMetrics(
            timestamp=proto.timestamp.ToDatetime(),
            cpu_overall=proto.cpu.overall,
            cpu_per_core=list(proto.cpu.per_core),
            memory_used=proto.memory.used,
            memory_total=proto.memory.total,
            memory_percent=proto.memory.usage_percent,
            disk_read_bytes=proto.disk.total.read_bytes,
            disk_write_bytes=proto.disk.total.write_bytes,
            network_rx_bytes=proto.network.total.rx_bytes,
            network_tx_bytes=proto.network.total.tx_bytes,
            raw_proto=proto,
        )

        if proto.HasField("processes"):
            metrics.process_count = proto.processes.total_count

        return metrics

    def close(self):
        """Close the gRPC channel."""
        if self.channel:
            self.channel.close()
            logger.info("Connection closed")


# Usage example
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    adapter = SystemEyeGRPCAdapter("localhost:50051")

    if not adapter.connect():
        print("Failed to connect to system-eye. Make sure the server is running.")
        exit(1)

    try:
        print("Streaming metrics from system-eye... (Press Ctrl+C to stop)")
        for metrics in adapter.stream_metrics(interval_ms=1000, include_processes=True):
            print(
                f"[{metrics.timestamp.strftime('%H:%M:%S')}] "
                f"CPU: {metrics.cpu_overall:5.1f}% | "
                f"Memory: {metrics.memory_percent:5.1f}% "
                f"({metrics.memory_used / 1024**3:.1f}GB / {metrics.memory_total / 1024**3:.1f}GB) | "
                f"Processes: {metrics.process_count}"
            )

            # Access raw process metrics if needed:
            # for p in metrics.raw_proto.processes.processes:
            #     print(p.wait_channel, p.held_locks, ...)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        adapter.close()
