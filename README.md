# Galena-2B: Granite 3.3 Math & Physics Model

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Model](https://img.shields.io/badge/Model-Granite%203.3--2B-green.svg)](https://huggingface.co/ibm-granite/granite-3.3-2b-instruct)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

A specialized 2B parameter language model fine-tuned on advanced mathematics and physics datasets. Built on IBM's Granite 3.3-2B Instruct base model with LoRA fine-tuning on 26k instruction-response pairs covering advanced calculations and physics concepts.

## Download Model Artifacts

The HF checkpoint and GGUF exports are hosted externally (e.g., Hugging Face) and
are **not** stored inside this repository. Fetch them before running the
examples:

```bash
python scripts/download_artifacts.py --artifact all
```

- `--source huggingface` (default) pulls from `xJoepec/galena-2b-math-physics`.
- `--source mirror --hf-url ... --gguf-url ...` lets you point to release assets/CDN downloads instead.

Artifacts install under `models/math-physics/{hf,gguf}` and are ignored by Git.

## Quick Start

### Using Hugging Face Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "models/math-physics/hf",
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("models/math-physics/hf")

# Generate response
prompt = "Explain the relationship between energy and momentum in special relativity."
messages = [{"role": "user", "content": prompt}]
inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(model.device)

outputs = model.generate(inputs, max_new_tokens=256, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### Using llama.cpp (GGUF)

```bash
# Requires llama.cpp build and downloaded GGUF artifact
./llama.cpp/build/bin/llama-cli \
  -m models/math-physics/gguf/granite-math-physics-f16.gguf \
  -p "Calculate the escape velocity from Earth's surface." \
  -n 256 \
  --temp 0.7
```

## Model Details

- **Base Model**: [ibm-granite/granite-3.3-2b-instruct](https://huggingface.co/ibm-granite/granite-3.3-2b-instruct)
- **Parameters**: 2.0B
- **Architecture**: GraniteForCausalLM (40 layers, 2048 hidden size, 32 attention heads)
- **Context Length**: 131,072 tokens (128k)
- **Training Method**: QLoRA (4-bit quantization with Low-Rank Adaptation)
- **Fine-tuning Data**: 26k examples blending:
  - **nvidia/Nemotron-RL-math-advanced_calculations** - Advanced calculator tasks with tool reasoning traces
  - **camel-ai/physics** - Physics dialogue pairs with topic/subtopic metadata

### Model Formats

| Format | Location (after download) | Size | Use Case |
|--------|---------------------------|------|----------|
| **Hugging Face** | `models/math-physics/hf/` | ~5.0 GB | PyTorch, Transformers, vLLM, further fine-tuning |
| **GGUF (F16)** | `models/math-physics/gguf/` | ~4.7 GB | llama.cpp, Ollama, LM Studio, on-device inference |

## Installation

### Prerequisites

- Python 3.10 or higher
- CUDA 12.1+ (for GPU acceleration)
- `huggingface_hub` (installed via `pip install -r requirements.txt`) for scripted downloads

### For Transformers Usage

```bash
# Clone repository
git clone <repository-url>
cd galena-2B

# Install dependencies
pip install -r requirements.txt

# Download artifacts (Hugging Face by default)
python scripts/download_artifacts.py --artifact hf
```

### For llama.cpp Usage

```bash
# Clone llama.cpp (if not already available)
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# Build with CUDA support (Linux/WSL)
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release

# Run inference
python scripts/download_artifacts.py --artifact gguf
./build/bin/llama-cli -m ../galena-2B/models/math-physics/gguf/granite-math-physics-f16.gguf
```

## Usage Examples

See the [`examples/`](examples/) directory for detailed usage demonstrations:

- **[basic_usage.py](examples/basic_usage.py)** - Simple model loading and inference
- **[chat_example.py](examples/chat_example.py)** - Interactive chat session
- **[llama_cpp_example.sh](examples/llama_cpp_example.sh)** - GGUF inference with llama.cpp

## Training Details

The model was fine-tuned using the following configuration:

```bash
# LoRA fine-tuning
python scripts/train_lora.py \
  --base_model ibm-granite/granite-3.3-2b-instruct \
  --dataset_path data/math_physics.jsonl \
  --output_dir outputs/granite-math-physics-lora \
  --use_4bit --gradient_checkpointing \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --num_train_epochs 1 \
  --max_steps 500 \
  --batching_strategy padding \
  --max_seq_length 512 \
  --bf16 \
  --trust_remote_code
```

For detailed training methodology and dataset preparation, see [MODEL_CARD.md](MODEL_CARD.md).

## Performance & Limitations

**Strengths:**
- Advanced mathematical calculations and reasoning
- Physics concepts and problem-solving
- Tool-augmented reasoning for complex calculations
- Efficient 2B parameter footprint suitable for edge deployment

**Limitations:**
- Specialized for math/physics; may underperform on general tasks
- 500-step fine-tune optimized for domain knowledge, not extensive generalization
- Inherits base model biases and constraints
- Best suited for educational and research applications

## Citation

If you use this model in your research, please cite:

```bibtex
@software{galena_2b_2024,
  title = {Galena-2B: Granite 3.3 Math & Physics Model},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/galena-2B},
  note = {Fine-tuned from IBM Granite 3.3-2B Instruct}
}
```

Also cite the base Granite model:

```bibtex
@software{granite_3_3_2024,
  title = {Granite 3.3: IBM's Open Foundation Models},
  author = {IBM Research},
  year = {2024},
  url = {https://www.ibm.com/granite}
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

The base Granite 3.3 model is also released under Apache 2.0 by IBM.

## Acknowledgments

- **IBM Research** for the Granite 3.3 foundation models
- **NVIDIA** for the Nemotron-RL-math dataset
- **CAMEL-AI** for the physics dialogue dataset
- **Hugging Face** for the Transformers library and model hosting infrastructure
- **llama.cpp** project for efficient GGUF inference

## Links

- [IBM Granite Models](https://www.ibm.com/granite)
- [Base Model: granite-3.3-2b-instruct](https://huggingface.co/ibm-granite/granite-3.3-2b-instruct)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)

## Support

For issues, questions, or contributions, please open an issue in this repository's issue tracker.
