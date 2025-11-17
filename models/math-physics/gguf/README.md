# Galena-2B GGUF Artifact

The GGUF export(s) live outside of Git and must be downloaded before llama.cpp
examples can run:

```bash
python scripts/download_artifacts.py --artifact gguf
```

The helper script fetches `gguf/*.gguf` from the Galena-2B snapshot on Hugging Face
(or from a release/CDN archive when `--source mirror` is used). After download,
`models/math-physics/gguf/granite-math-physics-f16.gguf` becomes available for
`llama-cli`, LM Studio, Ollama, etc.
