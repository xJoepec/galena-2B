# Model Card: Galena-2B (Granite 3.3 Math & Physics)

## Model Description

**Galena-2B** is a specialized 2-billion parameter language model optimized for mathematical reasoning and physics problem-solving. It is derived from IBM's Granite 3.3-2B Instruct base model through parameter-efficient fine-tuning (LoRA) on curated datasets focused on advanced calculations and physics concepts.

- **Developed by:** [Your Name/Organization]
- **Base Model:** [IBM Granite 3.3-2B Instruct](https://huggingface.co/ibm-granite/granite-3.3-2b-instruct)
- **Model Type:** Causal Language Model (Decoder-only Transformer)
- **Language:** English
- **License:** Apache 2.0
- **Fine-tuned from:** ibm-granite/granite-3.3-2b-instruct

## Model Architecture

- **Architecture:** GraniteForCausalLM
- **Parameters:** 2.0B
- **Layers:** 40
- **Hidden Size:** 2048
- **Attention Heads:** 32 (query) / 8 (key-value, GQA)
- **Intermediate Size:** 8192
- **Vocabulary Size:** 49,159 tokens
- **Context Window:** 131,072 tokens (128k)
- **Precision:** bfloat16 (training & inference)
- **Activation Function:** SiLU (Swish)

### Key Features

- **Grouped Query Attention (GQA)** for efficient inference
- **RoPE Embeddings** with extended context support (theta=10M)
- **Attention & Logits Scaling** for training stability
- **Embedding Multiplier** (12.0) and Residual Multiplier (0.22)

## Intended Use

### Primary Use Cases

- **Educational Applications:** Teaching and learning advanced mathematics and physics
- **Research Tools:** Assisting with physics problem formulation and mathematical reasoning
- **Conversational AI:** Domain-specific chatbots for STEM topics
- **Tool-Augmented Reasoning:** Integration with calculators and symbolic math engines

### Out-of-Scope Use

- **Critical Decision Making:** Not suitable for medical, legal, or safety-critical applications
- **General-Purpose Conversational AI:** Optimized for math/physics; may underperform on general topics
- **Production Systems:** This is a research/educational model without production guarantees
- **Factual Information Retrieval:** May hallucinate; always verify outputs

## Training Data

The model was fine-tuned on a carefully curated dataset of 26,000 instruction-response pairs blending two specialized datasets:

### 1. NVIDIA Nemotron-RL-Math (Advanced Calculations)

- **Source:** `nvidia/Nemotron-RL-math-advanced_calculations`
- **Content:** Complex mathematical problems with step-by-step reasoning traces
- **Features:** Tool-augmented reasoning, calculator integration, multi-step problem decomposition
- **Format:** Instruction-following with detailed solution traces

### 2. CAMEL-AI Physics Dataset

- **Source:** `camel-ai/physics`
- **Content:** Physics dialogue pairs covering diverse topics and subtopics
- **Features:** Conceptual explanations, problem-solving, physics principles
- **Metadata:** Topic and subtopic categorization for structured learning

### Data Preparation

- **Preprocessing:** `scripts/prepare_math_physics.py` in parent GRANITE repository
- **Format Conversion:** Unified into Granite's chat format (`<|user|>`/`<|assistant|>` tags)
- **Output:** `data/math_physics.jsonl` (26k examples)
- **Token Length:** Max sequence length capped at 512 tokens during training

## Training Procedure

### Training Hyperparameters

- **Method:** QLoRA (Quantized Low-Rank Adaptation)
- **Base Model Precision:** 4-bit quantization (NF4)
- **LoRA Rank:** Default (typically 8-16)
- **LoRA Alpha:** Default
- **Target Modules:** Query, Key, Value, Output projections
- **Gradient Checkpointing:** Enabled
- **Mixed Precision:** bfloat16

### Training Configuration

```python
{
    "base_model": "ibm-granite/granite-3.3-2b-instruct",
    "dataset_path": "data/math_physics.jsonl",
    "output_dir": "outputs/granite-math-physics-lora",
    "use_4bit": true,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "effective_batch_size": 4,
    "num_train_epochs": 1,
    "max_steps": 500,
    "max_seq_length": 512,
    "learning_rate": "2e-4 (default)",
    "batching_strategy": "padding",
    "optimizer": "paged_adamw_8bit",
    "bf16": true
}
```

### Training Infrastructure

- **Hardware:** NVIDIA GeForce RTX 4060 (8GB VRAM)
- **Software Stack:**
  - PyTorch 2.x
  - Hugging Face Transformers 4.44+
  - PEFT 0.11+
  - bitsandbytes 0.43+
  - CUDA 12.1
- **Training Time:** ~500 steps (1 epoch over 26k examples with batch size 4)
- **Checkpointing:** LoRA adapters saved every N steps

### Post-Training

1. **Adapter Merging:** LoRA adapters merged back into base weights using `scripts/merge_lora.py`
2. **GGUF Conversion:** Exported to F16 GGUF format via `llama.cpp/convert_hf_to_gguf.py`
3. **Formats Produced:**
   - Hugging Face Transformers (safetensors)
   - GGUF F16 (llama.cpp compatible)

## Evaluation

### Qualitative Assessment

The model demonstrates improved performance on:

- Multi-step mathematical reasoning
- Physics problem explanation
- Calculator-augmented computation tasks
- Domain-specific terminology and notation

### Limitations

- **Limited Training Steps:** Only 500 training steps; longer training may improve performance
- **Domain Specialization:** May sacrifice general capabilities for math/physics expertise
- **Hallucination Risk:** Can generate plausible but incorrect solutions
- **Tool Integration:** Expects calculator tools in reasoning traces; standalone performance may vary
- **Context Window:** Fine-tuned on 512-token sequences; full 128k context not extensively tested

## Bias, Risks, and Limitations

### Known Limitations

1. **Domain Specificity:** Optimized for math/physics; general knowledge may be limited
2. **Factual Accuracy:** No guarantee of correctness; outputs should be verified
3. **Training Data Bias:** Inherits biases from Nemotron and CAMEL-AI datasets
4. **Base Model Limitations:** Retains all limitations of Granite 3.3-2B Instruct
5. **Small Training Set:** 26k examples may not cover all edge cases

### Ethical Considerations

- **Educational Use:** Should supplement, not replace, human instruction
- **Verification Required:** Always validate mathematical and scientific outputs
- **Accessibility:** May use technical jargon inaccessible to beginners
- **Dataset Provenance:** Users should review source dataset licenses and terms

### Recommendations

- Use as an educational aid, not a source of truth
- Implement output validation for critical applications
- Combine with symbolic computation tools for verification
- Monitor for hallucinations and incorrect reasoning
- Consider fine-tuning on domain-specific data for production use

## Environmental Impact

- **Hardware:** NVIDIA RTX 4060 (8GB VRAM)
- **Training Duration:** ~500 steps (estimated 1-2 hours)
- **Energy Consumption:** Estimated <1 kWh for training
- **Carbon Footprint:** Minimal due to efficient LoRA training

## Technical Specifications

### Model Formats

| Format | Precision | Size | Compatible Frameworks |
|--------|-----------|------|-----------------------|
| Hugging Face Transformers | bfloat16 | ~5.0 GB | PyTorch, Transformers, vLLM, TGI |
| GGUF F16 | float16 | ~4.7 GB | llama.cpp, Ollama, LM Studio |

### System Requirements

**Minimum (CPU Inference):**
- RAM: 8 GB
- Storage: 10 GB free space
- CPU: Modern x86-64 with AVX2 support

**Recommended (GPU Inference):**
- GPU: 6+ GB VRAM (RTX 3060, A4000, or better)
- RAM: 16 GB
- CUDA 12.1+ or ROCm 5.7+

### Loading & Inference

Before running inference, pull the artifacts into `models/math-physics/`:

```bash
python scripts/download_artifacts.py --artifact all
```

**Transformers (Python):**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "models/math-physics/hf",
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained("models/math-physics/hf")
```

**llama.cpp (Command Line):**
```bash
./llama-cli -m granite-math-physics-f16.gguf -p "Your prompt" -n 256
```

## Citation

```bibtex
@software{galena_2b_2024,
  title = {Galena-2B: Granite 3.3 Math & Physics Model},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/galena-2B},
  note = {Fine-tuned from IBM Granite 3.3-2B Instruct on math and physics datasets}
}
```

## Acknowledgments

- IBM Research for the Granite 3.3 foundation model
- NVIDIA for the Nemotron-RL-Math dataset
- CAMEL-AI for the physics dialogue dataset
- Hugging Face for training infrastructure and libraries

## Contact

For questions, issues, or contributions:
- **Repository:** [GitHub Issues](https://github.com/yourusername/galena-2B/issues)
- **Email:** your.email@example.com

## Changelog

### Version 1.0 (2024-11-17)

- Initial release
- Fine-tuned on 26k math/physics examples
- 500 training steps with QLoRA
- Hugging Face and GGUF formats released
