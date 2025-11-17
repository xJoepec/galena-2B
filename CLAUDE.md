# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Purpose

This is a **model artifact distribution repository** for Granite 3.3-2B fine-tuned on math and physics datasets (nicknamed "Galena-2B"). It contains trained model checkpoints in multiple formats, comprehensive documentation, and usage examples.

The actual training/development environment exists in the parent `GRANITE/` directory.

## Repository Structure

```
galena-2B/
├── README.md              # Main documentation with quick start
├── LICENSE               # Apache 2.0 license
├── MODEL_CARD.md         # Detailed model card (architecture, training, limitations)
├── CITATION.cff          # Citation file for academic use
├── requirements.txt      # Minimal runtime dependencies
├── .gitignore           # Git ignore patterns
├── .gitattributes       # Git LFS configuration for large files
├── examples/            # Usage examples
│   ├── README.md        # Examples documentation
│   ├── basic_usage.py   # Simple inference example
│   ├── chat_example.py  # Interactive chat interface
│   └── llama_cpp_example.sh  # GGUF/llama.cpp usage
└── models/
    └── math-physics/
        ├── README.md    # Model-specific documentation
        ├── hf/          # Hugging Face format (~5GB)
        └── gguf/        # GGUF format (~4.7GB)
```

## Model Details

- **Base Model**: `ibm-granite/granite-3.3-2b-instruct`
- **Fine-tuning Datasets**:
  - `nvidia/Nemotron-RL-math-advanced_calculations` (advanced calculator tasks with tool reasoning)
  - `camel-ai/physics` (physics dialogue pairs)
- **Training Data**: 26k blended instruction/response examples in Granite dialog format
- **Architecture**: GraniteForCausalLM (2B parameters, 40 layers, 2048 hidden size)

## Model Artifacts

| Path | Format | Usage |
|------|--------|-------|
| `models/math-physics/hf/` | Hugging Face | Transformers, vLLM, or further fine-tuning |
| `models/math-physics/gguf/granite-math-physics-f16.gguf` | GGUF (F16) | llama.cpp inference |

## Using the Model

### Hugging Face Transformers / vLLM

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "models/math-physics/hf",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("models/math-physics/hf")
```

### llama.cpp (GGUF)

Requires building llama.cpp from `third_party/llama.cpp` in the GRANITE directory:

```bash
# From GRANITE/third_party/llama.cpp
cmake -B build
cmake --build build --config Release

# Run inference
./build/bin/llama-cli -m ../../galena-2B/models/math-physics/gguf/granite-math-physics-f16.gguf -i
```

## Training Provenance

The model was trained using LoRA adapters and merged back into the base weights. Training scripts referenced in `models/math-physics/README.md` are located in the parent GRANITE directory:

- `scripts/prepare_math_physics.py` - Dataset preparation
- `scripts/train_lora.py` - LoRA fine-tuning
- `scripts/merge_lora.py` - Adapter merging
- `third_party/llama.cpp/convert_hf_to_gguf.py` - GGUF conversion

Training outputs and logs are stored in `GRANITE/outputs/granite-math-physics-lora/`.

## Documentation Files

- **README.md**: Main entry point with quick start guide, installation instructions, and overview
- **MODEL_CARD.md**: Comprehensive model card including:
  - Architecture details and specifications
  - Training configuration and hyperparameters
  - Evaluation results and limitations
  - Ethical considerations and bias information
  - Technical specifications and system requirements
- **CITATION.cff**: Machine-readable citation file in Citation File Format
- **examples/README.md**: Detailed examples documentation with troubleshooting

## Common Commands

Since this is a distribution repository, there are no build or test commands. However, you can:

### Run Examples
```bash
# Basic inference example
python examples/basic_usage.py

# Interactive chat
python examples/chat_example.py

# llama.cpp (requires building llama.cpp separately)
./examples/llama_cpp_example.sh
```

### Install Dependencies
```bash
pip install -r requirements.txt

# Optional: GPU support
pip install accelerate bitsandbytes
```

### Download / Verify Model Files
```bash
# Fetch both the HF checkpoint and GGUF export
python scripts/download_artifacts.py --artifact all

# Hugging Face only
python scripts/download_artifacts.py --artifact hf
```

## Development Context

This repository is for **model distribution only**. For development tasks:

### Training & Fine-tuning
- Location: `GRANITE/` directory
- Scripts: `scripts/train_lora.py`, `scripts/merge_lora.py`
- Data preparation: `scripts/prepare_math_physics.py`

### Model Conversion
- GGUF conversion: `GRANITE/third_party/llama.cpp/convert_hf_to_gguf.py`
- Core ML export: `GRANITE/mobile/export_coreml.py`

### Working with This Repository

When helping users with this repository:

1. **For usage questions**: Refer to `README.md` and `examples/README.md`
2. **For model details**: Check `MODEL_CARD.md` for architecture, training, and limitations
3. **For integration**: Use code from `examples/` directory as templates
4. **For training/modification**: Direct users to the parent `GRANITE/` directory
5. **For citations**: Reference `CITATION.cff` and citation blocks in README.md

## Git LFS Notes

Large binaries are no longer committed to Git; they live on Hugging Face or
release/CDN storage and are fetched via `scripts/download_artifacts.py`.
`.gitattributes` keeps the LFS rules so future weight drops (if any) still use
LFS, but cloning this repo does **not** require Git LFS anymore.
