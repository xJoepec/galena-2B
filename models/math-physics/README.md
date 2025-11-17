# Granite 3.3 Math & Physics Model

This directory packages the Granite 3.3-2B instruct checkpoint fine-tuned on:

- **nvidia/Nemotron-RL-math-advanced_calculations** (advanced calculator tasks with tool reasoning traces)
- **camel-ai/physics** (physics dialogue pairs with topic/subtopic metadata)

The dataset was prepared via `scripts/prepare_math_physics.py`, producing `data/math_physics.jsonl`
(~26k blended instruction/response examples in Granite dialog format).

## Downloading the Artifacts

Large binaries are hosted externally. Pull the desired artifact(s) into this directory before use:

```bash
python scripts/download_artifacts.py --artifact all
```

- Default behavior downloads from `xJoepec/galena-2b-math-physics` on Hugging Face.
- Use `--repo-id <namespace/repo>` to point to your fork of the snapshot.
- Or provide `--source mirror --hf-url <...> --gguf-url <...>` if you host tarballs/zip files on releases or a CDN.

After download the layout becomes:

| Path | Description |
| ---- | ----------- |
| `hf/` | Full Hugging Face-format checkpoint (config, tokenizer, merged LoRA weights) ready for `transformers`, vLLM, or further fine-tuning. |
| `gguf/granite-math-physics-f16.gguf` | F16 export for llama.cpp / GGUF consumers. Build `third_party/llama.cpp` and run `./build/bin/llama-cli -m gguf/granite-math-physics-f16.gguf ...` to chat locally. |

## Training Recipe

```
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

Post-training, adapters were merged via:

```
python scripts/merge_lora.py \
  --base_model ibm-granite/granite-3.3-2b-instruct \
  --adapter_path outputs/granite-math-physics-lora \
  --output_dir models/math-physics/hf \
  --dtype bfloat16 \
  --trust_remote_code
```

and exported to GGUF:

```
python third_party/llama.cpp/convert_hf_to_gguf.py \
  models/math-physics/hf \
  --outfile models/math-physics/gguf/granite-math-physics-f16.gguf \
  --outtype f16 --verbose
```

## Usage Highlights

- **Transformers / vLLM**: point `--model` to `models/math-physics/hf` after downloading.
- **GGUF**: `./build/bin/llama-cli -m models/math-physics/gguf/granite-math-physics-f16.gguf -i ...`.
- **Provenance**: inherits IBM Granite 3.3-2B instruct base weights; fine-tune logs & adapter checkpoints remain under `outputs/granite-math-physics-lora/`.
