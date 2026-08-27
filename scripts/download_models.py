#!/usr/bin/env python3
"""Download and hash-verify every production checkpoint in the lock file."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


CHUNK_BYTES = 8 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def matches(path: Path, entry: dict[str, Any]) -> bool:
    if not path.is_file():
        return False
    expected_bytes = entry.get("bytes")
    if expected_bytes is not None and path.stat().st_size != expected_bytes:
        return False
    return sha256_file(path) == entry["sha256"]


def run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def download_url(url: str, partial: Path) -> None:
    curl = shutil.which("curl")
    if curl is None:
        raise RuntimeError("curl is required for checkpoint downloads")
    run(
        [
            curl,
            "--fail",
            "--location",
            "--retry",
            "8",
            "--retry-all-errors",
            "--connect-timeout",
            "30",
            "--continue-at",
            "-",
            "--output",
            str(partial),
            url,
        ]
    )


def download_huggingface(entry: dict[str, Any], partial: Path) -> None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if entry.get("gated") and not token:
        raise RuntimeError(
            f"{entry['name']} is gated. Request access to "
            f"https://huggingface.co/{entry['huggingface_repo']} and rerun with "
            "HF_TOKEN set. The token is read from the environment and is never saved."
        )
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise RuntimeError("huggingface-hub is not installed") from error
    cached = Path(
        hf_hub_download(
            repo_id=entry["huggingface_repo"],
            filename=entry["huggingface_filename"],
            token=token,
        )
    )
    shutil.copyfile(cached, partial)


def quarantine_bad_file(path: Path) -> Path:
    suffix = 0
    while True:
        candidate = path.with_name(f"{path.name}.sha256-mismatch.{suffix}")
        if not candidate.exists():
            path.replace(candidate)
            return candidate
        suffix += 1


def ensure_checkpoint(
    entry: dict[str, Any], third_party_root: Path, *, offline: bool
) -> None:
    destination = third_party_root / entry["destination"]
    if matches(destination, entry):
        print(f"verified {entry['name']}: {destination}")
        return
    if destination.exists():
        quarantined = quarantine_bad_file(destination)
        print(f"preserved invalid file as {quarantined}", file=sys.stderr)
    partial = destination.with_name(destination.name + ".part")
    if offline:
        raise RuntimeError(f"missing valid checkpoint in offline mode: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if "url" in entry:
        download_url(entry["url"], partial)
    else:
        download_huggingface(entry, partial)
    if not matches(partial, entry):
        actual = sha256_file(partial) if partial.is_file() else "missing"
        raise RuntimeError(
            f"checkpoint verification failed for {entry['name']}: "
            f"expected {entry['sha256']}, got {actual}"
        )
    partial.replace(destination)
    print(f"installed {entry['name']}: {destination}")


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=project / "third_party.lock.json"
    )
    parser.add_argument("--third-party-root", type=Path, required=True)
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--list", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text())
    entries = [
        entry
        for entry in manifest["checkpoints"]
        if not args.only or entry["name"] in set(args.only)
    ]
    if args.list:
        for entry in entries:
            print(
                json.dumps(
                    {
                        "name": entry["name"],
                        "destination": str(
                            args.third_party_root / entry["destination"]
                        ),
                        "bytes": entry.get("bytes"),
                        "gated": bool(entry.get("gated")),
                    },
                    sort_keys=True,
                )
            )
        return
    for entry in entries:
        if entry.get("required", True):
            ensure_checkpoint(entry, args.third_party_root, offline=args.offline)


if __name__ == "__main__":
    main()
