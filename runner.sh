#!/bin/bash

# The specific "Staff Engineer" prompt
INPUT_PROMPT=$(cat <<EOF
System Role: You are a Staff Engineer specializing in systems programming and distributed security. Your tone is professional, slightly cynical, and highly efficient.

The Task: Design a proof-of-concept for a Secure Distributed Task Queue. The system must involve a Producer (written in Go) and a Worker (written in Rust). They communicate over an encrypted TCP socket without using high-level frameworks like gRPC.

Requirements:
1. Technical Logic (Go): Provide a snippet for the Go Producer that handles SIGINT gracefully and uses a chan to manage a pool of 5 concurrent task submissions.
2. Technical Logic (Rust): Provide a snippet for the Rust Worker using tokio and rustls. It must implement a custom Error type using the thiserror crate to handle handshake failures.
3. The "Twist": Explain how to solve a potential "Exec format error" if the Rust binary is compiled on a MacBook (M3) but deployed to an amd64 Debian Docker container. Provide the specific Docker build command.
4. Creative Synthesis: Write a short, 3-sentence "Post-Mortem" in the style of a weary sysadmin describing a day where the TLS certificates expired because someone hardcoded the path to /tmp.

Constraints:
- Do not use serde for the serialization logic; explain a manual byte-masking approach instead.
- Use at least one complex LaTeX formula to represent the theoretical network throughput T, where T=W/R (Window size over Round-trip time), and explain its relevance to this specific architecture.

Output: Begin your response with the phrase "Initiating architectural review..."
EOF
)

models=(
  "llama3.2:1b"
  "llama3.2:3b"
  "qwen2.5:0.5b"
  "smollm2:1.7b"
  "qwen2.5:7b-instruct"
  "phi4"
  "gemma2:9b"
  "qwen2.5-coder:32b"
)

for model in "${models[@]}"; do
    echo "=========================================="
    echo "MODEL: $model"
    echo "=========================================="
    
    # Ensure model exists
    ollama pull "$model" > /dev/null 2>&1
    
    # Run inference and pipe to a log file if you want to compare later
    echo "$INPUT_PROMPT" | ollama run "$model" | tee "${model//:/_}_response.md"
    
    sleep 40
    
    echo -e "\n\n"
done