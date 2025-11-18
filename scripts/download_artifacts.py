
#!/usr/bin/env python3
"""Download Galena-2B model artifacts from Hugging Face or a custom mirror."""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import List, Sequence
from urllib import request
from urllib.parse import urlparse
import zipfile

try:  # Optional dependency when pulling from Hugging Face
    from huggingface_hub import snapshot_download
except Exception:  # pragma: no cover - fallback when package missing
    snapshot_download = None

ARTIFACT_CHOICES = ("hf", "gguf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the Galena-2B Hugging Face checkpoint and/or GGUF exports "
            "without storing the large binaries inside Git."
        )
    )
    parser.add_argument(
        "--artifact",
        choices=(*ARTIFACT_CHOICES, "all"),
        default="all",
        help="Which artifact to download (default: both).",
    )
    parser.add_argument(
        "--output-dir",
        default="models/math-physics",
        help="Destination directory that will receive the artifacts.",
    )
    parser.add_argument(
        "--source",
        choices=("huggingface", "mirror"),
        default="huggingface",
        help="Where to download from.",
    )
    parser.add_argument(
        "--repo-id",
        default="xJoePec/galena-2b-math-physics",
        help="Hugging Face repo id that hosts the snapshot.",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Revision/branch/tag to download from on Hugging Face.",
    )
    parser.add_argument(
        "--hf-url",
        help="Direct archive URL for the HF-format checkpoint when using --source mirror.",
    )
    parser.add_argument(
        "--gguf-url",
        help="Direct archive URL for the GGUF exports when using --source mirror.",
    )
    return parser.parse_args()


def ensure_dependency() -> None:
    if snapshot_download is None:
        print(
            "huggingface_hub is required for --source huggingface. Install it via 'pip install huggingface_hub'.",
            file=sys.stderr,
        )
        sys.exit(1)


def selected_artifacts(selection: str) -> List[str]:
    if selection == "all":
        return list(ARTIFACT_CHOICES)
    return [selection]


def download_from_hf(repo_id: str, revision: str, artifacts: Sequence[str], base_dir: Path) -> None:
    ensure_dependency()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        allow_patterns = [f"{artifact}/*" for artifact in artifacts]
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=tmp_path,
            allow_patterns=allow_patterns,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        for artifact in artifacts:
            src = locate_artifact(tmp_path, artifact)
            copy_artifact(src, base_dir / artifact)


def locate_artifact(root: Path, artifact: str) -> Path:
    candidates = [root / artifact, root / "models" / "math-physics" / artifact]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = [p for p in root.rglob(artifact) if p.is_dir()]
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not find '{artifact}' inside downloaded snapshot at {root}")


def copy_artifact(src: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dest)
    try:
        label = dest.relative_to(Path.cwd())
    except ValueError:
        label = dest
    print(f"[done] Installed {label}")


def download_from_mirror(url: str, artifact: str, base_dir: Path) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        parsed = urlparse(url)
        suffix = Path(parsed.path).suffix or ".bin"
        archive_path = tmp_path / f"{artifact}{suffix}"
        print(f"Downloading {artifact} from {url} ...")
        with request.urlopen(url) as response, open(archive_path, "wb") as handle:
            shutil.copyfileobj(response, handle)
        extract_root = tmp_path / "extract"
        extract_root.mkdir()
        extract_archive(archive_path, extract_root)
        src = locate_artifact(extract_root, artifact)
        copy_artifact(src, base_dir / artifact)


def extract_archive(archive_path: Path, destination: Path) -> None:
    path_str = str(archive_path).lower()
    if path_str.endswith((".tar.gz", ".tgz", ".tar")):
        mode = "r:gz" if path_str.endswith((".tar.gz", ".tgz")) else "r:"
        with tarfile.open(archive_path, mode) as tar:
            tar.extractall(destination)
    elif path_str.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(destination)
    else:
        raise ValueError(
            "Mirror archives must be .zip, .tar, .tar.gz, or .tgz so they can be extracted automatically."
        )


def main() -> None:
    args = parse_args()
    artifacts = selected_artifacts(args.artifact)
    base_dir = Path(args.output_dir).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    if args.source == "mirror":
        for artifact in artifacts:
            url = args.hf_url if artifact == "hf" else args.gguf_url
            if not url:
                raise SystemExit(f"--source mirror requires a URL for the '{artifact}' artifact")
            download_from_mirror(url, artifact, base_dir)
    else:
        download_from_hf(args.repo_id, args.revision, artifacts, base_dir)


if __name__ == "__main__":
    main()
