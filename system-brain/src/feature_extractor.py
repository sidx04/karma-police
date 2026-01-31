#!/usr/bin/env python3
"""
Feature extraction for ML workload classification.
Extracts meaningful patterns from telemetry data.
"""

import numpy as np


def extract_features(telemetry_data):
    """
    Extract comprehensive features from telemetry samples for ML classification.
    Uses all available telemetry data including per-core CPU, process info, and system metrics.

    Args:
        telemetry_data: Dict with 'samples' containing comprehensive telemetry metrics

    Returns:
        numpy array of features (fixed size: 120)
    """
    samples = telemetry_data.get('samples', [])
    if not samples:
        return np.zeros(120)

    features = []

    # Extract basic time series metrics (backward compatible)
    gpu_utils = []
    mem_utils = []
    gpu_mem_utils = []
    temps = []
    powers = []
    cpu_utils = []

    for s in samples:
        if 'gpu' in s and s['gpu']['devices']:
            gpu_utils.append(s['gpu']['devices'][0]['utilization'])
            gpu_mem_utils.append(s['gpu']['devices'][0]['memory']['usage_percent'])
            temps.append(s['gpu']['devices'][0]['temperature'])
            powers.append(s['gpu']['devices'][0]['power_usage'])
        if 'memory' in s:
            mem_utils.append(s['memory']['usage_percent'])
        if 'cpu' in s:
            cpu_utils.append(s['cpu']['overall'])

    # 1. BASIC METRICS (48 features - 6 metrics × 8 stats each)
    basic_series = [gpu_utils, mem_utils, gpu_mem_utils, temps, powers, cpu_utils]
    for series in basic_series:
        if series:
            arr = np.array(series)
            features.extend([
                np.mean(arr), np.std(arr), np.min(arr), np.max(arr),
                np.median(arr), np.percentile(arr, 25), np.percentile(arr, 75)
            ])
            # Trend analysis
            if len(series) > 1:
                x = np.arange(len(series))
                trend = np.polyfit(x, series, 1)[0]
                features.append(trend)
            else:
                features.append(0)
        else:
            features.extend([0] * 8)

    # 2. PER-CORE CPU ANALYSIS (8 features)
    per_core_data = []
    for s in samples:
        if 'cpu' in s and 'per_core' in s['cpu'] and s['cpu']['per_core']:
            per_core_data.append(s['cpu']['per_core'])

    if per_core_data:
        per_core_array = np.array(per_core_data)
        # CPU core utilization patterns
        features.extend([
            np.mean(per_core_array),  # Average across all cores/samples
            np.std(per_core_array),   # Variance in core usage
            np.max(per_core_array),   # Peak core usage
            np.mean(np.std(per_core_array, axis=1)),  # Core imbalance per sample
            np.mean(np.max(per_core_array, axis=1)),  # Max core per sample
            np.mean(np.min(per_core_array, axis=1)),  # Min core per sample
            len(per_core_array[0]) if per_core_array.size > 0 else 0,  # Number of cores
            np.mean([np.count_nonzero(cores) for cores in per_core_array])  # Active cores
        ])
    else:
        features.extend([0] * 8)

    # 3. PROCESS-LEVEL ANALYSIS (16 features)
    top_cpu_usage = []
    top_mem_usage = []
    gpu_process_count = []
    gpu_process_memory = []

    for s in samples:
        if 'processes' in s:
            # Top CPU processes
            if 'top_cpu' in s['processes'] and s['processes']['top_cpu']:
                cpu_procs = s['processes']['top_cpu'][:5]  # Top 5
                top_cpu_usage.extend([p['cpu_percent'] for p in cpu_procs])

            # Top memory processes
            if 'top_memory' in s['processes'] and s['processes']['top_memory']:
                mem_procs = s['processes']['top_memory'][:5]  # Top 5
                top_mem_usage.extend([p['memory_percent'] for p in mem_procs])

            # GPU processes
            if 'gpu_processes' in s['processes']:
                gpu_procs = s['processes']['gpu_processes']
                gpu_process_count.append(len(gpu_procs))
                gpu_mem = sum([p.get('gpu_memory_mb', 0) for p in gpu_procs])
                gpu_process_memory.append(gpu_mem)

    # Process statistics
    process_features = []
    for data in [top_cpu_usage, top_mem_usage, gpu_process_count, gpu_process_memory]:
        if data:
            arr = np.array(data)
            process_features.extend([np.mean(arr), np.max(arr), np.std(arr), len(data)])
        else:
            process_features.extend([0, 0, 0, 0])

    features.extend(process_features)  # 16 features

    # 4. SYSTEM-LEVEL METRICS (8 features)
    load_avgs = []
    memory_gb_used = []
    memory_gb_total = []

    for s in samples:
        if 'system' in s and 'load_avg' in s['system']:
            load_avgs.extend(s['system']['load_avg'])
        if 'memory' in s:
            if 'used_gb' in s['memory']:
                memory_gb_used.append(s['memory']['used_gb'])
            if 'total_gb' in s['memory']:
                memory_gb_total.append(s['memory']['total_gb'])

    # System statistics
    system_features = []
    for data in [load_avgs, memory_gb_used]:
        if data:
            arr = np.array(data)
            system_features.extend([np.mean(arr), np.max(arr)])
        else:
            system_features.extend([0, 0])

    # Memory capacity info
    if memory_gb_total:
        system_features.extend([np.mean(memory_gb_total), max(memory_gb_used)/max(memory_gb_total) if memory_gb_total and memory_gb_used else 0])
    else:
        system_features.extend([0, 0])

    # Sample frequency and duration
    if samples:
        duration = len(samples)
        system_features.extend([duration, duration/30.0])  # samples and rate
    else:
        system_features.extend([0, 0])

    features.extend(system_features)  # 8 features

    # 5. ADVANCED PATTERN DETECTION (40 features)

    # Memory growth patterns (training vs inference signature)
    if mem_utils and len(mem_utils) > 10:
        quarter = len(mem_utils) // 4
        if quarter > 0:
            start_avg = np.mean(mem_utils[:quarter])
            end_avg = np.mean(mem_utils[-quarter:])
            memory_growth = (end_avg - start_avg) / max(start_avg, 1)

            # More detailed memory analysis
            memory_slope = np.polyfit(range(len(mem_utils)), mem_utils, 1)[0]
            memory_volatility = np.std(np.diff(mem_utils))
            memory_peaks = len([i for i in range(1, len(mem_utils)-1)
                              if mem_utils[i] > mem_utils[i-1] and mem_utils[i] > mem_utils[i+1]])

            features.extend([memory_growth, memory_slope, memory_volatility, memory_peaks])
        else:
            features.extend([0, 0, 0, 0])
    else:
        features.extend([0, 0, 0, 0])

    # GPU utilization patterns
    if gpu_utils and len(gpu_utils) > 2:
        gpu_variance = np.var(gpu_utils)
        gpu_stability = 1.0 / (1.0 + gpu_variance)  # Stable = inference, unstable = training
        gpu_duty_cycle = len([x for x in gpu_utils if x > 50]) / len(gpu_utils)
        gpu_peak_frequency = len([i for i in range(1, len(gpu_utils)-1)
                                if gpu_utils[i] > gpu_utils[i-1] and gpu_utils[i] > gpu_utils[i+1]])
        features.extend([gpu_variance, gpu_stability, gpu_duty_cycle, gpu_peak_frequency])
    else:
        features.extend([0, 0, 0, 0])

    # Cross-metric correlations (important for workload identification)
    if len(gpu_utils) == len(mem_utils) and len(gpu_utils) > 3:
        gpu_mem_corr = np.corrcoef(gpu_utils, mem_utils)[0,1] if not np.isnan(np.corrcoef(gpu_utils, mem_utils)[0,1]) else 0
        features.append(gpu_mem_corr)
    else:
        features.append(0)

    if len(gpu_utils) == len(cpu_utils) and len(gpu_utils) > 3:
        gpu_cpu_corr = np.corrcoef(gpu_utils, cpu_utils)[0,1] if not np.isnan(np.corrcoef(gpu_utils, cpu_utils)[0,1]) else 0
        features.append(gpu_cpu_corr)
    else:
        features.append(0)

    # Workload intensity signatures
    if gpu_utils and powers:
        avg_efficiency = np.mean(gpu_utils) / (np.mean(powers) + 1)  # GPU util per watt
        features.append(avg_efficiency)
    else:
        features.append(0)

    # Thermal behavior patterns
    if temps and gpu_utils and len(temps) == len(gpu_utils) and len(temps) > 3:
        temp_gpu_corr = np.corrcoef(temps, gpu_utils)[0,1] if not np.isnan(np.corrcoef(temps, gpu_utils)[0,1]) else 0
        temp_range = max(temps) - min(temps)
        features.extend([temp_gpu_corr, temp_range])
    else:
        features.extend([0, 0])

    # Fill remaining advanced features
    while len(features) < 120:
        features.append(0)

    return np.array(features[:120])