#!/usr/bin/env python3
"""
Comprehensive System's Eye-style telemetry collection with per-process monitoring.
"""
import json
import time
import subprocess
import os
import psutil
from datetime import datetime

def collect_telemetry():
    """Comprehensive telemetry collection with per-process data"""
    print("[TELEMETRY] Starting comprehensive telemetry collection...")
    sample_count = 0
    all_samples = []
    max_samples = 300  # Keep last 5 minutes at 1 second intervals
    start_time = time.time()

    while True:
        sample_count += 1
        sample = {
            'timestamp': datetime.now().isoformat(),
            'cpu': {'overall': 0, 'per_core': []},
            'memory': {'usage_percent': 0, 'used_gb': 0, 'total_gb': 0},
            'gpu': {'devices': []},
            'processes': {'top_cpu': [], 'top_memory': [], 'gpu_processes': []},
            'system': {'load_avg': [], 'uptime': 0}
        }

        # Get comprehensive system metrics using psutil
        try:
            # CPU metrics
            sample['cpu']['overall'] = psutil.cpu_percent(interval=0.1)
            sample['cpu']['per_core'] = psutil.cpu_percent(interval=0.1, percpu=True)

            # Memory metrics
            mem = psutil.virtual_memory()
            sample['memory']['usage_percent'] = mem.percent
            sample['memory']['used_gb'] = round(mem.used / (1024**3), 2)
            sample['memory']['total_gb'] = round(mem.total / (1024**3), 2)

            # System metrics
            sample['system']['load_avg'] = list(os.getloadavg())
            sample['system']['uptime'] = round(time.time() - psutil.boot_time())

        except Exception as e:
            print(f"[ERROR] System metrics: {e}")

        # Get GPU metrics with process info
        try:
            # GPU utilization and memory
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )

            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    parts = line.split(',')
                    if len(parts) >= 6:
                        gpu_data = {
                            'index': int(parts[0]),
                            'utilization': float(parts[1].strip()),
                            'memory': {
                                'used': float(parts[2].strip()),
                                'total': float(parts[3].strip()),
                                'usage_percent': (float(parts[2].strip()) / float(parts[3].strip())) * 100 if float(parts[3].strip()) > 0 else 0
                            },
                            'temperature': float(parts[4].strip()),
                            'power_usage': float(parts[5].strip())
                        }
                        sample['gpu']['devices'].append(gpu_data)

            # GPU process information
            gpu_proc_result = subprocess.run(
                ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )

            if gpu_proc_result.returncode == 0 and gpu_proc_result.stdout.strip():
                for line in gpu_proc_result.stdout.strip().split('\n'):
                    parts = line.split(',')
                    if len(parts) >= 3:
                        gpu_proc = {
                            'pid': int(parts[0].strip()),
                            'name': parts[1].strip(),
                            'gpu_memory_mb': int(parts[2].strip()) if parts[2].strip() != '[Not Supported]' else 0
                        }
                        sample['processes']['gpu_processes'].append(gpu_proc)

        except Exception as e:
            print(f"[ERROR] GPU metrics: {e}")

        # Get top processes by CPU and memory
        try:
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'cmdline']):
                try:
                    proc_info = proc.info
                    if proc_info['cpu_percent'] is not None and proc_info['memory_percent'] is not None:
                        proc_info['cmdline'] = ' '.join(proc_info['cmdline'][:3]) if proc_info['cmdline'] else ''
                        processes.append(proc_info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            # Sort by CPU and memory usage
            top_cpu = sorted(processes, key=lambda x: x['cpu_percent'], reverse=True)[:10]
            top_memory = sorted(processes, key=lambda x: x['memory_percent'], reverse=True)[:10]

            sample['processes']['top_cpu'] = top_cpu
            sample['processes']['top_memory'] = top_memory

        except Exception as e:
            print(f"[ERROR] Process metrics: {e}")

        # Add sample to rolling window
        all_samples.append(sample)

        # Maintain rolling window of max_samples
        if len(all_samples) > max_samples:
            all_samples = all_samples[-max_samples:]

        # Prepare output with accumulated samples
        current_time = time.time()
        duration = current_time - start_time

        output = {
            'metadata': {
                'export_time': datetime.now().isoformat(),
                'sample_count': len(all_samples),
                'duration': round(duration, 2),
                'rate_hz': round(len(all_samples) / max(duration, 1), 2)
            },
            'samples': all_samples
        }

        # Use atomic write to prevent corruption
        temp_file = '/tmp/system_eye_metrics.json.tmp'
        try:
            with open(temp_file, 'w') as f:
                json.dump(output, f, indent=2)
            # Atomic move
            import os
            os.rename(temp_file, '/tmp/system_eye_metrics.json')
        except Exception as e:
            print(f"[TELEMETRY ERROR] Failed to write metrics: {e}")
            # Clean up temp file
            try:
                os.remove(temp_file)
            except:
                pass

        if sample_count % 10 == 0:
            print(f"[TELEMETRY] Collected {sample_count} samples, window: {len(all_samples)}, duration: {duration:.1f}s")

        time.sleep(1)

if __name__ == '__main__':
    collect_telemetry()