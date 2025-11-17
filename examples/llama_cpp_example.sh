#!/bin/bash

# llama.cpp usage example for Galena-2B (GGUF format)
#
# Prerequisites:
# 1. Build llama.cpp from source:
#    git clone https://github.com/ggerganov/llama.cpp.git
#    cd llama.cpp
#    cmake -B build -DGGML_CUDA=ON  # For GPU support
#    cmake --build build --config Release
#
# 2. Adjust the paths below to match your setup

# Path to llama.cpp executable (adjust as needed)
LLAMA_CLI="./llama.cpp/build/bin/llama-cli"

# Path to the GGUF model file
MODEL_PATH="models/math-physics/gguf/granite-math-physics-f16.gguf"

# Example 1: Simple prompt
echo "Example 1: Simple mathematical question"
echo "========================================"
$LLAMA_CLI \
  -m "$MODEL_PATH" \
  -p "Calculate the integral of f(x) = 3x^2 + 2x - 1 from x=0 to x=5." \
  -n 256 \
  --temp 0.7 \
  --top-p 0.95 \
  --repeat-penalty 1.1 \
  -ngl 35  # Number of GPU layers (adjust based on your GPU VRAM)

echo -e "\n\n"

# Example 2: Physics question
echo "Example 2: Physics concept explanation"
echo "======================================="
$LLAMA_CLI \
  -m "$MODEL_PATH" \
  -p "Explain the concept of gravitational time dilation in simple terms." \
  -n 384 \
  --temp 0.7 \
  --top-p 0.95 \
  --repeat-penalty 1.1 \
  -ngl 35

echo -e "\n\n"

# Example 3: Interactive chat mode
echo "Example 3: Interactive chat (type /bye to exit)"
echo "================================================"
$LLAMA_CLI \
  -m "$MODEL_PATH" \
  -i \
  --temp 0.7 \
  --top-p 0.95 \
  --repeat-penalty 1.1 \
  -ngl 35 \
  --color \
  --ctx-size 4096

# Additional llama.cpp options you might find useful:
#
# -ngl N          : Number of layers to offload to GPU (0 = CPU only)
# -c N, --ctx-size N : Context size (default: 512, max: 131072 for this model)
# -n N            : Number of tokens to generate (default: -1 = infinite)
# --temp N        : Temperature for sampling (default: 0.8)
# --top-p N       : Top-p sampling (default: 0.9)
# --top-k N       : Top-k sampling (default: 40)
# --repeat-penalty N : Penalize repetition (default: 1.1)
# -b N            : Batch size for prompt processing (default: 512)
# -t N            : Number of threads (default: auto)
# --color         : Colorize output
# -i, --interactive : Run in interactive mode
# -ins            : Instruction mode (more suitable for Q&A)
# --mlock         : Lock model in RAM to prevent swapping
# --no-mmap       : Don't memory-map the model file
# -f FILE         : Read prompt from file
# --log-disable   : Disable logging
# --verbose-prompt : Print prompt tokens before generation

# For more options, run:
# $LLAMA_CLI --help
