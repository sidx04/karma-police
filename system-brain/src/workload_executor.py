#!/usr/bin/env python3
"""
Real workload executor for generating training data with actual GPU utilization.
Executes real PyTorch models and captures system telemetry.
"""

import os
import json
import time
import subprocess
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime


class RealWorkloadExecutor:
    """Executes real ML workloads and captures telemetry"""

    def __init__(self):
        self.workload_configs = {
            'transformer_training': {
                'model_type': 'transformer',
                'mode': 'training',
                'batch_size': 8,  # Memory-safe batch for training intensity
                'sequence_length': 1024,  # Moderate sequences for training patterns
                'hidden_size': 1024,  # Reasonable model size
                'num_layers': 16,  # Sufficient layers for training signature
                'learning_rate': 0.001,
                'gradient_accumulation': 8,  # Higher accumulation for training intensity
                'description': 'Training Large Transformer Language Model'
            },
            'cnn_training': {
                'model_type': 'cnn',
                'mode': 'training',
                'batch_size': 32,  # Moderate training batch
                'image_size': 256,  # Medium images for training differentiation
                'num_classes': 1000,
                'learning_rate': 0.01,
                'description': 'Training High-Resolution CNN Image Classifier'
            },
            'transformer_inference': {
                'model_type': 'transformer',
                'mode': 'inference',
                'batch_size': 64,  # High throughput for inference signature
                'sequence_length': 128,  # Short sequences for fast inference
                'hidden_size': 512,  # Efficient smaller model
                'num_layers': 6,  # Minimal layers for speed
                'temperature': 0.7,  # Inference parameter
                'description': 'Fast Transformer Text Generation'
            },
            'cnn_inference': {
                'model_type': 'cnn',
                'mode': 'inference',
                'batch_size': 256,  # Very high throughput inference
                'image_size': 128,  # Small images for fast inference
                'num_classes': 1000,
                'description': 'High-Throughput CNN Image Classification'
            }
        }

    def create_transformer_model(self, config):
        """Create a transformer model"""
        class SimpleTransformer(nn.Module):
            def __init__(self, vocab_size=50000, hidden_size=768, num_layers=12, num_heads=16):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, hidden_size)
                self.pos_encoding = nn.Parameter(torch.randn(2048, hidden_size))
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=hidden_size, nhead=num_heads, batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                self.output = nn.Linear(hidden_size, vocab_size)

            def forward(self, x):
                seq_len = x.size(1)
                x = self.embedding(x) + self.pos_encoding[:seq_len]
                x = self.transformer(x)
                return self.output(x)

        return SimpleTransformer(
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            num_heads=16
        )

    def create_cnn_model(self, config):
        """Create a CNN model"""
        class SimpleCNN(nn.Module):
            def __init__(self, num_classes=1000):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(128, 256, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(256, 512, 3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((7, 7))
                )
                self.classifier = nn.Sequential(
                    nn.Linear(512 * 7 * 7, 4096),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(4096, 4096),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(4096, num_classes)
                )

            def forward(self, x):
                x = self.features(x)
                x = x.view(x.size(0), -1)
                return self.classifier(x)

        return SimpleCNN(num_classes=config['num_classes'])

    def get_system_metrics(self):
        """Get system metrics from System's Eye ONLY"""
        # MUST use System's Eye for telemetry
        system_eye_data = self._get_system_eye_metrics()
        if system_eye_data:
            return system_eye_data

        # If System's Eye data is not available, raise an error
        raise RuntimeError("[ERROR] System's Eye metrics not available! Ensure System's Eye is running and exporting to /tmp/system_eye_metrics.json")

    def _get_system_eye_metrics(self):
        """Get metrics from System's Eye export file"""
        try:
            import os
            import json
            metrics_file = '/tmp/system_eye_metrics.json'

            # Check if metrics file exists (telemetry collection should be running)
            if not os.path.exists(metrics_file):
                print(f"[WARN] Telemetry file not found: {metrics_file}")
                print("[INFO] Waiting for telemetry collection to start...")
                time.sleep(2)

            if os.path.exists(metrics_file):
                # Retry JSON reading with small delays to handle concurrent writes
                for retry in range(3):
                    try:
                        with open(metrics_file, 'r') as f:
                            content = f.read().strip()
                            if not content:
                                time.sleep(0.1)
                                continue
                            data = json.loads(content)
                            # Handle both full export format and simple samples array
                            if 'samples' in data:
                                samples = data['samples']
                                if len(samples) > 0:
                                    return samples[-1]  # Return latest sample
                            elif isinstance(data, list) and len(data) > 0:
                                return data[-1]
                            break
                    except (json.JSONDecodeError, ValueError) as json_err:
                        print(f"[WARN] JSON parse error (retry {retry+1}/3): {json_err}")
                        time.sleep(0.1)
                        continue
        except Exception as e:
            print(f"[ERROR] Failed to get System's Eye metrics: {e}")
        return None


    def _get_process_metrics(self):
        """Get process metrics from System's Eye (includes process-level data)"""
        # Process metrics should come from System's Eye's process monitoring
        # This is now handled by System's Eye itself
        return None

    def execute_real_workload(self, workload_type, duration_seconds=60):
        """Execute real ML workload and capture telemetry"""
        config = self.workload_configs[workload_type]
        samples = []

        print(f"Starting {config['description']}...")
        print(f"  Workload Type: {workload_type}")
        print(f"  Mode: {config['mode']}")
        print(f"  Duration: {duration_seconds}s")

        # Create model
        if config['model_type'] == 'transformer':
            model = self.create_transformer_model(config)
        else:
            model = self.create_cnn_model(config)

        # Compute parameter count for the created model
        try:
            parameter_count = sum(p.numel() for p in model.parameters())
        except Exception:
            parameter_count = 0

        def _human_readable_params(n):
            # Return a human friendly string like '20B' or '3.2M'
            if n >= 10**11:
                return f"{n/10**9:.0f}B"
            if n >= 10**9:
                # show one decimal for billions if needed
                val = n/10**9
                if abs(val - round(val)) < 1e-6:
                    return f"{int(round(val))}B"
                return f"{val:.1f}B"
            if n >= 10**6:
                return f"{n/10**6:.1f}M".rstrip('0').rstrip('.')
            return str(n)

        model_info = {
            'parameter_count': int(parameter_count),
            'parameter_count_human': _human_readable_params(parameter_count),
            'phase': config.get('mode', 'unknown')
        }

        # Move to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        print(f"  Device: {device}")

        # Setup for training or inference
        if config['mode'] == 'training':
            lr = config.get('learning_rate', 0.001)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()
            model.train()
            gradient_accumulation = config.get('gradient_accumulation', 1)
            print(f"  Training: lr={lr}, grad_accum={gradient_accumulation}")
        else:
            model.eval()
            print(f"  Inference: eval mode, no gradients")

        start_time = time.time()
        iteration = 0
        accumulated_loss = 0

        while time.time() - start_time < duration_seconds:
            # Create input data
            if config['model_type'] == 'transformer':
                batch_size = config['batch_size']
                seq_len = config['sequence_length']
                inputs = torch.randint(0, 50000, (batch_size, seq_len)).to(device)
                if config['mode'] == 'training':
                    targets = torch.randint(0, 50000, (batch_size, seq_len)).to(device)
            else:
                batch_size = config['batch_size']
                img_size = config['image_size']
                inputs = torch.randn(batch_size, 3, img_size, img_size).to(device)
                if config['mode'] == 'training':
                    targets = torch.randint(0, config['num_classes'], (batch_size,)).to(device)

            # Execute forward/backward pass
            if config['mode'] == 'training':
                # Enhanced training with gradient accumulation
                if iteration % gradient_accumulation == 0:
                    optimizer.zero_grad()

                outputs = model(inputs)
                if config['model_type'] == 'transformer':
                    loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                else:
                    loss = criterion(outputs, targets)

                # Scale loss for gradient accumulation
                loss = loss / gradient_accumulation
                loss.backward()
                accumulated_loss += loss.item()

                # Update weights every gradient_accumulation steps
                if (iteration + 1) % gradient_accumulation == 0:
                    optimizer.step()

                # Additional intensive operations for training signature
                if iteration % 10 == 0:
                    # Simulate additional training operations
                    _ = torch.norm(model.parameters().__next__())  # Parameter norm calculation

            else:
                # Optimized inference with no_grad
                with torch.no_grad():
                    outputs = model(inputs)
                    # Simulate inference post-processing
                    if config['model_type'] == 'transformer' and 'temperature' in config:
                        outputs = outputs / config['temperature']

            # Ensure GPU work completes
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # Capture metrics
            metrics = self.get_system_metrics()
            samples.append(metrics)

            iteration += 1

            # Progress update
            if iteration % 100 == 0:
                gpu_util = metrics['gpu']['devices'][0]['utilization']
                print(f"  Iteration {iteration}: GPU {gpu_util:.1f}%")

        print(f"Completed {iteration} iterations")
        return {'samples': samples}