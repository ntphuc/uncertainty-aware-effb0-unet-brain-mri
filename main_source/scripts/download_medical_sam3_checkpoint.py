#!/usr/bin/env python3
"""Download Medical-SAM3 checkpoint_3D.pt into this source tree.

The file is approximately 10.3 GB. The Hugging Face client supports resumable
cached downloads and repositories backed by Xet storage.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

REPO_ID = "ChongCong/Medical-SAM3"
FILENAME = "checkpoint_3D.pt"
EXPECTED_SHA256 = "6e40bbaa739ac44e3e47dc6355ef6dedc560a30411377ad891f8af9e6df0dbd6"


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    default_dir = project_root / "checkpoints" / "medical_sam3"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_dir,
        help=f"Destination directory (default: {default_dir})",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="Optional Hugging Face token. Public repo normally does not need one.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if the file already exists.",
    )
    parser.add_argument(
        "--skip-sha256",
        action="store_true",
        help="Skip SHA256 verification after download.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / FILENAME

    if target.exists() and not args.force:
        print(f"[INFO] Checkpoint already exists: {target}")
    else:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            print(
                "[ERROR] Missing huggingface_hub. Run:\n"
                "        pip install -U huggingface_hub hf_xet",
                file=sys.stderr,
            )
            raise SystemExit(2) from exc

        print(f"[INFO] Downloading {REPO_ID}/{FILENAME}")
        print(f"[INFO] Destination: {args.output_dir}")
        downloaded = hf_hub_download(
            repo_id=REPO_ID,
            filename=FILENAME,
            repo_type="model",
            local_dir=str(args.output_dir),
            token=args.token,
            force_download=args.force,
        )
        target = Path(downloaded).resolve()
        print(f"[INFO] Download completed: {target}")

    if not target.exists():
        print(f"[ERROR] Download finished but file was not found: {target}", file=sys.stderr)
        return 3

    size_gb = target.stat().st_size / (1024 ** 3)
    print(f"[INFO] File size: {size_gb:.2f} GiB")

    if not args.skip_sha256:
        print("[INFO] Verifying SHA256. This can take several minutes for a 10 GB file...")
        actual = sha256_file(target)
        if actual.lower() != EXPECTED_SHA256.lower():
            print("[ERROR] SHA256 mismatch!", file=sys.stderr)
            print(f"        expected: {EXPECTED_SHA256}", file=sys.stderr)
            print(f"        actual:   {actual}", file=sys.stderr)
            return 4
        print(f"[OK] SHA256 verified: {actual}")

    relative_hint = Path("checkpoints") / "medical_sam3" / FILENAME
    print("\n[READY] Use this checkpoint in YAML configs:")
    print(f"teacher:\n  checkpoint: {relative_hint.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
