#!/usr/bin/env python3
"""Upload Galena-2B model artifacts to Hugging Face."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from huggingface_hub import HfApi, create_repo
except ImportError:
    print(
        "huggingface_hub is required. Install it via 'pip install huggingface_hub'.",
        file=sys.stderr,
    )
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload Galena-2B model artifacts to Hugging Face."
    )
    parser.add_argument(
        "--repo-id",
        default="xJoepec/galena-2b-math-physics",
        help="Hugging Face repository ID (namespace/repo-name).",
    )
    parser.add_argument(
        "--token",
        required=True,
        help="Hugging Face API token with write access.",
    )
    parser.add_argument(
        "--model-dir",
        default="models/math-physics",
        help="Local directory containing hf/ and gguf/ subdirectories.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repository private.",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Branch/revision to upload to.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_dir = Path(args.model_dir).resolve()
    hf_dir = model_dir / "hf"
    gguf_dir = model_dir / "gguf"

    # Validate directories exist
    if not hf_dir.exists():
        print(f"Error: HF model directory not found: {hf_dir}", file=sys.stderr)
        sys.exit(1)

    if not gguf_dir.exists():
        print(f"Error: GGUF directory not found: {gguf_dir}", file=sys.stderr)
        sys.exit(1)

    # Initialize API
    api = HfApi(token=args.token)

    # Create repository if it doesn't exist
    print(f"Creating/verifying repository: {args.repo_id}")
    try:
        create_repo(
            repo_id=args.repo_id,
            token=args.token,
            private=args.private,
            exist_ok=True,
            repo_type="model",
        )
        print(f"[OK] Repository ready: {args.repo_id}")
    except Exception as e:
        print(f"Error creating repository: {e}", file=sys.stderr)
        sys.exit(1)

    # Upload HF model files
    print(f"\nUploading HF model files from {hf_dir}...")
    try:
        api.upload_folder(
            folder_path=str(hf_dir),
            repo_id=args.repo_id,
            path_in_repo="hf",
            revision=args.revision,
            commit_message="Upload Hugging Face model checkpoint",
        )
        print("[OK] HF model files uploaded successfully")
    except Exception as e:
        print(f"Error uploading HF files: {e}", file=sys.stderr)
        sys.exit(1)

    # Upload GGUF files
    print(f"\nUploading GGUF files from {gguf_dir}...")
    try:
        api.upload_folder(
            folder_path=str(gguf_dir),
            repo_id=args.repo_id,
            path_in_repo="gguf",
            revision=args.revision,
            commit_message="Upload GGUF model exports",
        )
        print("[OK] GGUF files uploaded successfully")
    except Exception as e:
        print(f"Error uploading GGUF files: {e}", file=sys.stderr)
        sys.exit(1)

    # Upload documentation files
    print("\nUploading documentation files...")
    doc_files = ["README.md", "MODEL_CARD.md", "LICENSE", "CITATION.cff"]
    for doc_file in doc_files:
        doc_path = Path(doc_file)
        if doc_path.exists():
            try:
                api.upload_file(
                    path_or_fileobj=str(doc_path),
                    path_in_repo=doc_file,
                    repo_id=args.repo_id,
                    revision=args.revision,
                    commit_message=f"Upload {doc_file}",
                )
                print(f"[OK] Uploaded {doc_file}")
            except Exception as e:
                print(f"Warning: Could not upload {doc_file}: {e}")
        else:
            print(f"[SKIP] {doc_file} (not found)")

    print(f"\n[SUCCESS] Upload complete! View your model at:")
    print(f"   https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
