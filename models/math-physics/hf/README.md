# Galena-2B Hugging Face Checkpoint

The Hugging Face-format checkpoint (config, tokenizer, safetensors) is not stored in this Git repository.
Download it before running any of the Python examples:

```bash
python scripts/download_artifacts.py --artifact hf
```

By default the script pulls the snapshot from `xJoepec/galena-2b-math-physics` on
Hugging Face. Provide `--repo-id <namespace/repo>` if you host the files under a
different account, or `--source mirror --hf-url <tarball>` if you prefer a
release/CDN download.

Once downloaded, the Transformers examples continue to load from
`models/math-physics/hf/`.
