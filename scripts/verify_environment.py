#!/usr/bin/env python3
"""Verify source, package, checkpoint, CUDA, and Vulkan runtime provenance."""

from __future__ import annotations

import argparse
import hashlib
from importlib import import_module, metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


MODULE_DISTRIBUTIONS = {
    "cv2": "opencv-python",
    "dex_retargeting": "dex-retargeting",
    "PIL": "pillow",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def module_version(module_name: str) -> str:
    module = import_module(module_name)
    version = getattr(module, "__version__", None)
    if version is not None:
        return str(version)
    distribution = MODULE_DISTRIBUTIONS.get(module_name, module_name)
    return metadata.version(distribution)


def locked_requirements(path: Path) -> dict[str, str]:
    result = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        name, separator, version = line.partition("==")
        if separator != "==" or not name or not version:
            raise RuntimeError(f"non-exact requirement in {path}: {line}")
        result[name] = version
    return result


def installed_lock_versions(path: Path) -> tuple[dict[str, str], list[str]]:
    actual = {}
    errors = []
    for name, expected in locked_requirements(path).items():
        try:
            version = metadata.version(name)
            actual[name] = version
            if version != expected:
                errors.append(f"{name}: expected {expected}, got {version}")
        except metadata.PackageNotFoundError:
            errors.append(f"locked package is missing: {name}=={expected}")
    return actual, errors


def version_tuple(value: str) -> tuple[int, ...]:
    pieces = []
    for item in value.split("."):
        digits = "".join(character for character in item if character.isdigit())
        if not digits:
            break
        pieces.append(int(digits))
    return tuple(pieces)


def git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def verify_sidecar(python: Path, expected: dict[str, str]) -> dict[str, str]:
    script = (
        "import json,mediapipe,numpy,cv2;"
        "print(json.dumps({'mediapipe':mediapipe.__version__,"
        "'numpy':numpy.__version__,'opencv':cv2.__version__}))"
    )
    result = subprocess.check_output([str(python), "-c", script], text=True)
    actual = json.loads(result)
    for name in ("mediapipe", "numpy", "opencv"):
        if actual[name] != expected[name]:
            raise RuntimeError(
                f"MediaPipe sidecar {name}: expected {expected[name]}, "
                f"got {actual[name]}"
            )
    return actual


def verify_sidecar_lock(python: Path, lock: Path) -> dict[str, str]:
    script = (
        "import json,sys;from importlib import metadata;"
        "expected=json.load(sys.stdin);"
        "print(json.dumps({name:metadata.version(name) for name in expected}))"
    )
    result = subprocess.run(
        [str(python), "-c", script],
        input=json.dumps(locked_requirements(lock)),
        check=True,
        text=True,
        capture_output=True,
    )
    actual = json.loads(result.stdout)
    expected = locked_requirements(lock)
    mismatches = [
        f"{name}: expected {expected[name]}, got {actual[name]}"
        for name in expected
        if actual[name] != expected[name]
    ]
    if mismatches:
        raise RuntimeError("; ".join(mismatches))
    return actual


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--project", type=Path, default=project)
    parser.add_argument("--without-models", action="store_true")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--render-device")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = args.runtime.resolve()
    project = args.project.resolve()
    environment_lock = json.loads((project / "environment.lock.json").read_text())
    third_party_lock = json.loads((project / "third_party.lock.json").read_text())
    errors: list[str] = []
    report: dict[str, Any] = {
        "runtime": str(runtime),
        "project": str(project),
        "python": sys.version.split()[0],
        "packages": {},
        "repositories": {},
        "checkpoints": {},
    }

    if sys.version_info[:2] != (3, 10):
        errors.append(f"Python 3.10 required, got {sys.version.split()[0]}")
    for command in ("git", "curl", "ffmpeg"):
        if shutil.which(command) is None:
            errors.append(f"required command is missing: {command}")

    for module_name, expected in environment_lock["critical_imports"].items():
        try:
            actual = module_version(module_name)
            report["packages"][module_name] = actual
            if actual != expected:
                errors.append(
                    f"{module_name}: expected {expected}, got {actual}"
                )
        except Exception as error:  # noqa: BLE001 - aggregate all setup failures
            errors.append(f"could not import {module_name}: {error}")

    report["locked_packages"] = {}
    for lock_name in ("python_lock", "pytorch_cuda_lock"):
        lock_path = project / environment_lock[lock_name]
        actual, lock_errors = installed_lock_versions(lock_path)
        report["locked_packages"][lock_name] = actual
        errors.extend(lock_errors)
    for name, expected in environment_lock["build_tools"].items():
        try:
            actual = metadata.version(name)
            report["locked_packages"][name] = actual
            if actual != expected:
                errors.append(f"{name}: expected {expected}, got {actual}")
        except metadata.PackageNotFoundError:
            errors.append(f"locked build tool is missing: {name}=={expected}")

    third_party = runtime / "third_party"
    for entry in third_party_lock["repositories"]:
        path = third_party / entry["name"]
        try:
            actual = git_head(path)
            report["repositories"][entry["name"]] = actual
            if actual != entry["commit"]:
                errors.append(
                    f"{entry['name']}: expected {entry['commit']}, got {actual}"
                )
            for submodule in entry.get("submodules", []):
                submodule_name = f"{entry['name']}/{submodule['path']}"
                submodule_path = path / submodule["path"]
                submodule_head = git_head(submodule_path)
                report["repositories"][submodule_name] = submodule_head
                if submodule_head != submodule["commit"]:
                    errors.append(
                        f"{submodule_name}: expected {submodule['commit']}, "
                        f"got {submodule_head}"
                    )
        except Exception as error:  # noqa: BLE001
            errors.append(f"could not verify {entry['name']}: {error}")

    required_paths = (
        third_party
        / "dex-retargeting/assets/robots/assembly/ur5e_shadow/ur5e_shadow_left_hand_glb.urdf",
        third_party
        / "dex-retargeting/assets/robots/assembly/ur5e_shadow/ur5e_shadow_right_hand_glb.urdf",
        third_party / "sam2/sam2/sam2_video_predictor.py",
        third_party / "sam3/sam3/model_builder.py",
        third_party / "ProPainter/inference_propainter.py",
    )
    for path in required_paths:
        if not path.is_file():
            errors.append(f"required runtime asset is missing: {path}")

    if not args.without_models:
        for entry in third_party_lock["checkpoints"]:
            if not entry.get("required", True):
                continue
            path = third_party / entry["destination"]
            if not path.is_file():
                errors.append(f"checkpoint is missing: {path}")
                continue
            actual = sha256_file(path)
            report["checkpoints"][entry["name"]] = actual
            if actual != entry["sha256"]:
                errors.append(
                    f"{entry['name']}: expected {entry['sha256']}, got {actual}"
                )

    sidecar_python = runtime / ".venv-mediapipe/bin/python"
    try:
        report["mediapipe_sidecar"] = verify_sidecar(
            sidecar_python, environment_lock["mediapipe_sidecar"]
        )
        report["mediapipe_locked_packages"] = verify_sidecar_lock(
            sidecar_python, project / environment_lock["mediapipe_lock"]
        )
        sidecar_tools_script = (
            "import json;from importlib import metadata;"
            "print(json.dumps({name:metadata.version(name) "
            "for name in ('pip','setuptools','wheel')}))"
        )
        sidecar_tools = json.loads(
            subprocess.check_output(
                [str(sidecar_python), "-c", sidecar_tools_script], text=True
            )
        )
        report["mediapipe_build_tools"] = sidecar_tools
        for name, expected in environment_lock["build_tools"].items():
            if sidecar_tools[name] != expected:
                errors.append(
                    f"MediaPipe sidecar {name}: expected {expected}, "
                    f"got {sidecar_tools[name]}"
                )
    except Exception as error:  # noqa: BLE001
        errors.append(f"could not verify MediaPipe sidecar: {error}")

    try:
        import torch

        report["torch_cuda"] = torch.version.cuda
        if torch.version.cuda != environment_lock["validated_platform"]["cuda_runtime"]:
            errors.append(
                f"PyTorch CUDA runtime: expected "
                f"{environment_lock['validated_platform']['cuda_runtime']}, "
                f"got {torch.version.cuda}"
            )
        report["cuda_available"] = bool(torch.cuda.is_available())
        if args.require_gpu and not torch.cuda.is_available():
            errors.append("CUDA GPU is required but torch.cuda.is_available() is false")
    except Exception as error:  # noqa: BLE001
        errors.append(f"could not verify CUDA runtime: {error}")

    if shutil.which("nvidia-smi"):
        driver = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            text=True,
        ).splitlines()[0]
        report["nvidia_driver"] = driver
        minimum = environment_lock["validated_platform"]["minimum_nvidia_driver"]
        if version_tuple(driver) < version_tuple(minimum):
            errors.append(f"NVIDIA driver {driver} is older than required {minimum}")
    elif args.require_gpu:
        errors.append("nvidia-smi is missing")

    if args.render_device:
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    str(project / "scripts/preflight_sapien_vulkan.py"),
                    "--device",
                    args.render_device,
                ],
                check=True,
                text=True,
                capture_output=True,
                timeout=90,
            )
            report["vulkan_preflight"] = json.loads(result.stdout)
        except Exception as error:  # noqa: BLE001
            errors.append(f"SAPIEN/Vulkan preflight failed: {error}")

    report["ok"] = not errors
    report["errors"] = errors
    print(json.dumps(report, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
