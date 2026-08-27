#!/usr/bin/env python3
"""Install pinned source trees and model files into one runtime directory."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any


RETRIES = 4
GIT_TIMEOUT_SECONDS = int(os.environ.get("EGODEX_GIT_TIMEOUT", "180"))


def git_environment() -> dict[str, str]:
    """Make unattended Git operations fail and retry instead of hanging forever."""
    environment = dict(os.environ)
    environment.update(
        {
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_HTTP_LOW_SPEED_LIMIT": "1024",
            "GIT_HTTP_LOW_SPEED_TIME": "30",
        }
    )
    return environment


def run(
    command: list[str],
    *,
    environment: dict[str, str] | None = None,
    dry_run: bool = False,
    timeout: int | None = None,
) -> None:
    print("+ " + " ".join(command), flush=True)
    if dry_run:
        return
    process = subprocess.Popen(
        command,
        env=environment,
        start_new_session=timeout is not None,
    )
    try:
        return_code = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        raise
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def retry(
    command: list[str],
    *,
    environment: dict[str, str] | None = None,
    dry_run: bool = False,
) -> None:
    if dry_run:
        run(command, environment=environment, dry_run=True)
        return
    for attempt in range(1, RETRIES + 1):
        try:
            run(command, environment=environment, timeout=GIT_TIMEOUT_SECONDS)
            return
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            if attempt == RETRIES:
                raise
            delay = attempt * 3
            print(
                f"{type(error).__name__}; retrying in {delay}s "
                f"({attempt}/{RETRIES})",
                file=sys.stderr,
            )
            time.sleep(delay)


def git_output(path: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), *arguments], text=True
    ).strip()


def install_repository(
    entry: dict[str, Any],
    *,
    third_party_root: Path,
    python: Path,
    offline: bool,
    dry_run: bool,
) -> None:
    path = third_party_root / entry["name"]
    if not path.exists():
        if offline:
            raise RuntimeError(f"missing source tree in offline mode: {path}")
        run(["git", "init", str(path)], dry_run=dry_run)
        run(
            ["git", "-C", str(path), "remote", "add", "origin", entry["url"]],
            dry_run=dry_run,
        )
    elif not dry_run and not (path / ".git").is_dir():
        raise RuntimeError(f"refusing to replace non-git directory: {path}")

    if not dry_run:
        dirty = git_output(path, "status", "--porcelain", "--untracked-files=no")
        if dirty:
            raise RuntimeError(f"refusing to change modified source tree: {path}")
        origin = git_output(path, "remote", "get-url", "origin")
        if origin != entry["url"]:
            raise RuntimeError(
                f"unexpected origin for {entry['name']}: {origin}"
            )
    if not offline:
        retry(
            [
                "git",
                "-C",
                str(path),
                "fetch",
                "--depth",
                "1",
                "origin",
                entry["commit"],
            ],
            environment=git_environment(),
            dry_run=dry_run,
        )
    run(
        ["git", "-C", str(path), "checkout", "--detach", entry["commit"]],
        dry_run=dry_run,
    )
    if entry.get("recursive"):
        retry(
            [
                "git",
                "-C",
                str(path),
                "submodule",
                "update",
                "--init",
                "--recursive",
                "--depth",
                "1",
                "--jobs",
                "1",
            ],
            environment=git_environment(),
            dry_run=dry_run,
        )
    if not dry_run and git_output(path, "rev-parse", "HEAD") != entry["commit"]:
        raise RuntimeError(f"failed to pin {entry['name']} to {entry['commit']}")
    if entry.get("editable_install"):
        environment = dict(os.environ)
        environment.update(entry.get("environment", {}))
        run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--no-build-isolation",
                "-e",
                str(path),
            ],
            environment=environment,
            dry_run=dry_run,
        )


def ensure_project_link(runtime: Path, project: Path, *, dry_run: bool) -> None:
    link = runtime / "project"
    if link.resolve(strict=False) == project.resolve():
        return
    if link.exists() or link.is_symlink():
        raise RuntimeError(
            f"{link} already exists and does not resolve to {project.resolve()}"
        )
    print(f"+ ln -s {project.resolve()} {link}")
    if not dry_run:
        link.symlink_to(project.resolve(), target_is_directory=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--without-models", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = args.runtime.resolve()
    project = args.project.resolve()
    manifest_path = project / "third_party.lock.json"
    manifest = json.loads(manifest_path.read_text())
    if not args.dry_run:
        runtime.mkdir(parents=True, exist_ok=True)
    ensure_project_link(runtime, project, dry_run=args.dry_run)
    third_party = runtime / "third_party"
    if not args.dry_run:
        third_party.mkdir(parents=True, exist_ok=True)
    for entry in manifest["repositories"]:
        install_repository(
            entry,
            third_party_root=third_party,
            python=args.python,
            offline=args.offline,
            dry_run=args.dry_run,
        )
    if not args.without_models:
        command = [
            str(args.python),
            str(project / "scripts/download_models.py"),
            "--manifest",
            str(manifest_path),
            "--third-party-root",
            str(third_party),
        ]
        if args.offline:
            command.append("--offline")
        run(command, dry_run=args.dry_run)
    print(f"runtime ready at {runtime}")


if __name__ == "__main__":
    main()
