# Galena-2B Usage Examples

This directory contains example scripts demonstrating how to use the Galena-2B model for various tasks.

## Prerequisites

### For Python Examples

Install the required dependencies:

```bash
pip install -r ../requirements.txt
```

For GPU acceleration (recommended):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers>=4.44.0 accelerate>=0.33.0
```

### For llama.cpp Example

1. Clone and build llama.cpp:

```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# For CPU-only build
cmake -B build
cmake --build build --config Release

# For GPU build (CUDA)
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release

# For GPU build (ROCm/AMD)
cmake -B build -DGGML_HIPBLAS=ON
cmake --build build --config Release
```

2. Update the paths in `llama_cpp_example.sh` to match your installation.

## Examples

### 1. Basic Usage (`basic_usage.py`)

Demonstrates simple model loading and inference on predefined prompts.

**Usage:**

```bash
python examples/basic_usage.py
```

**Features:**
- Loads the model with automatic device placement
- Runs inference on multiple example prompts
- Shows basic response generation with sensible defaults

**Expected output:** Responses to math and physics questions

---

### 2. Interactive Chat (`chat_example.py`)

Provides an interactive chat interface for conversing with the model.

**Usage:**

```bash
python examples/chat_example.py
```

**Features:**
- Interactive REPL-style interface
- Maintains conversation history
- Supports multi-turn conversations
- Commands:
  - `exit`, `quit`, `q` - End the chat
  - `clear` - Reset conversation history

**Example session:**

```
You: What is the second law of thermodynamics?
Assistant: [Model explains entropy and thermodynamics]

You: Can you give me an example?
Assistant: [Model provides example using conversation context]
```

---

### 3. llama.cpp Usage (`llama_cpp_example.sh`)

Demonstrates how to use the GGUF model format with llama.cpp for efficient inference.

**Usage:**

```bash
chmod +x examples/llama_cpp_example.sh
./examples/llama_cpp_example.sh
```

**Features:**
- Three example modes:
  1. Simple mathematical question
  2. Physics concept explanation
  3. Interactive chat mode
- GPU offloading for faster inference
- Customizable generation parameters

**Customization:**

Edit the script to adjust:
- `LLAMA_CLI` - Path to llama-cli executable
- `MODEL_PATH` - Path to GGUF model file
- `-ngl` - Number of GPU layers (higher = more GPU usage)
- `--ctx-size` - Context window size
- `--temp` - Sampling temperature
- Other generation parameters

---

## Tips for Best Results

### Model Configuration

**For general use:**
```python
temperature=0.7,    # Balanced creativity
top_p=0.95,         # Nucleus sampling
repetition_penalty=1.1  # Reduce repetition
```

**For more factual/precise responses:**
```python
temperature=0.3,    # Lower randomness
top_p=0.9,
do_sample=True
```

**For creative explanations:**
```python
temperature=0.9,    # Higher creativity
top_p=0.95,
top_k=50
```

### Hardware Recommendations

| Hardware | Configuration | Expected Performance |
|----------|--------------|---------------------|
| **RTX 3060 (12GB)** | Full model in GPU | ~30-50 tokens/sec |
| **RTX 4060 (8GB)** | Full model in GPU | ~25-40 tokens/sec |
| **CPU (16GB RAM)** | CPU inference | ~2-8 tokens/sec |
| **Apple M1/M2** | CPU inference | ~10-20 tokens/sec |

### Memory Usage

- **Transformers (bf16):** ~5 GB VRAM/RAM
- **GGUF (F16):** ~4.7 GB RAM
- **With quantization (Q4_K_M):** ~1.5-2 GB RAM (not included, needs conversion)

### Prompt Engineering

For best results with math/physics questions:

**Good prompts:**
- "Calculate the integral of..."
- "Explain the concept of..."
- "Derive the formula for..."
- "What is the relationship between X and Y?"

**Less effective:**
- Overly vague questions
- Questions outside math/physics domain
- Requests requiring real-time data
- Multi-part questions without clear structure

## Troubleshooting

### Common Issues

**1. Out of memory errors**

```python
# Solution: Use CPU offloading
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    max_memory={0: "6GB", "cpu": "16GB"}  # Adjust to your hardware
)
```

**2. Slow inference on CPU**

- Use GGUF format with llama.cpp for faster CPU inference
- Reduce `max_new_tokens` to generate shorter responses
- Consider quantization (requires converting model to Q4/Q5 GGUF)

**3. Model not loading**

- Ensure the model artifacts are present: `python scripts/download_artifacts.py --artifact all`
- Check you have sufficient disk space (~10 GB free)
- Verify Python version (3.10+)

**4. Import errors**

```bash
# Reinstall transformers
pip install --upgrade transformers torch
```

## Advanced Usage

### Batch Inference

```python
prompts = ["Question 1", "Question 2", "Question 3"]
inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256)
responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
```

### Custom Generation Parameters

```python
generation_config = {
    "max_new_tokens": 512,
    "min_new_tokens": 50,
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 50,
    "repetition_penalty": 1.1,
    "no_repeat_ngram_size": 3,
    "do_sample": True,
    "num_beams": 1,
    "early_stopping": True
}

outputs = model.generate(inputs, **generation_config)
```

### Streaming Generation

```python
from transformers import TextIteratorStreamer
from threading import Thread

streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)
generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=256)

thread = Thread(target=model.generate, kwargs=generation_kwargs)
thread.start()

for text in streamer:
    print(text, end="", flush=True)
```

## Contributing

Found a useful pattern or example? Consider contributing it back to this repository!

## License

These examples are released under the same Apache 2.0 license as the model.
