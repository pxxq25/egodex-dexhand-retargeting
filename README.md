# EgoDex DexHand Retargeting

Training-free egocentric video retargeting from human arms and hands to a
bimanual UR5e + Shadow Hand embodiment.

The pipeline uses measured 3-D hand/arm trajectories for robot retargeting and
image-aligned landmarks for removal masks. It supports intervals where only the
left hand, only the right hand, or both hands are visible. Side-specific masks
are tracked independently and unioned only for human removal, which avoids
left/right identity swaps during crossings.

## Pipeline

```text
egocentric video + EgoDex-style HDF5
        │
        ├─ dex-retargeting POSITION optimizer ── Shadow joint trajectories
        ├─ UR5e inverse kinematics ────────────── arm trajectories
        ├─ SAM3.1 direct side tracking ────────── human arm/hand masks
        └─ ProPainter ─────────────────────────── human-removed background
                                                        │
SAPIEN robot render + alpha/motion blur ────────────────┘
        │
        └─ temporally smoothed composite MP4
```

The frozen production backend is `sam3.1_direct_geometry_refined_v1`:

```text
HTS camera-coordinate keypoints
  → per-side SAM3.1 direct geometry prompts
  → appearance refinement and interval-local temporal stabilization
  → ProPainter removal
  → continuity-preserving UR5e + Shadow compositing
```

`scripts/run_egoquest_sam3_recording.py` orchestrates that exact path. It
stops the legacy CLI after robot rendering, generates the approved SAM3 masks,
recomposes each segment, and assembles the complete recording. Production mode
omits only raw-mask copies and per-chunk comparison-video encoding; the
stabilized masks consumed by ProPainter are unchanged.
It also keeps one SAM3 model loaded per assigned GPU across segment jobs. This
removes repeated checkpoint startup; the reuse path is accepted only when its
stabilized masks are byte-identical to isolated-job inference.

`scripts/run_interact_batch.py` records the backend, checkpoint, prompt mode,
prompt stride, and executable-source fingerprint in an immutable manifest.
Completion validation rejects missing per-segment SAM3 provenance or a backend
mismatch, preventing SAM2 and SAM3 results from being mixed silently.

This repository contains orchestration, retargeting, rendering, masking,
inpainting integration, compositing, verification, EgoQuest conversion tools,
and tests. It deliberately does **not** contain datasets, checkpoints, model
weights, generated videos, or third-party robot assets.

## Requirements

- Linux x86_64 with an NVIDIA GPU and driver 560.35.03 or newer
- Python 3.10 (the validated host used 3.10.12)
- About 12 GiB of free space for Python wheels, source trees, and checkpoints
- Hugging Face access to the gated `facebook/sam3` checkpoint

ProPainter is noncommercial research software. UR5e graphical assets also have
separate use restrictions. Review every upstream license before distributing
assets or outputs.

## One-command installation

After Hugging Face grants access to `facebook/sam3`, export a read-only token
and run the bootstrap script from the repository root:

```bash
HF_TOKEN="${HF_TOKEN}" ./bootstrap.sh --render-device cuda:0
```

The token is read only by `huggingface-hub`; it is never printed, written to a
configuration file, copied into the repository, or committed. Standard
`HTTPS_PROXY`/`HTTP_PROXY` variables are honored for installations behind a
VPN or proxy.

The command is idempotent and performs the complete setup:

1. Creates isolated Python 3.10 environments for the production pipeline and
   MediaPipe RGB detector.
2. Installs the exact Python, PyTorch 2.7.1, CUDA 12.6, and cuDNN 9.5 wheel set
   from the checked-in lock files.
3. Clones every production third-party repository at its exact tested commit,
   including the `dex-retargeting` robot-asset submodule.
4. Downloads SAM2.1, direct-video SAM3, RAFT, flow-completion, and ProPainter
   checkpoints and rejects any SHA-256 mismatch.
5. Verifies imports, repository commits, robot assets, model hashes, NVIDIA
   driver compatibility, CUDA availability, and the requested SAPIEN/Vulkan
   device.

Activate the resulting runtime with:

```bash
source ./activate.sh
```

Useful setup modes:

```bash
# Print third-party/model actions without changing anything.
./bootstrap.sh --dry-run

# Code-only setup for CI; no multi-gigabyte checkpoints.
./bootstrap.sh --without-models

# Verify an already-cached installation without network access.
./bootstrap.sh --offline --render-device cuda:0
```

The authoritative machine-readable locks are:

- [`environment.lock.json`](environment.lock.json): OS, Python, CUDA, driver,
  container, and critical-import versions
- [`requirements/pytorch-cu126.lock.txt`](requirements/pytorch-cu126.lock.txt):
  PyTorch and every CUDA runtime wheel
- [`requirements/runtime-cu126.lock.txt`](requirements/runtime-cu126.lock.txt):
  complete primary Python dependency closure
- [`requirements/mediapipe.lock.txt`](requirements/mediapipe.lock.txt): isolated
  NumPy 1.26 MediaPipe environment
- [`third_party.lock.json`](third_party.lock.json): source URLs, commits,
  checkpoint locations, sizes, and SHA-256 values

MediaPipe is intentionally isolated: version 0.10.21 requires NumPy below 2,
while `dex-retargeting` and the approved production runtime use NumPy 2.2.6.
Combining them in one environment is not reproducible.

The pinned SAM3 source declares an older NumPy upper bound, but the approved
pipeline was validated end-to-end with NumPy 2.2.6. Its editable install is
therefore performed with dependency resolution disabled, and the verifier
checks the tested runtime versions directly instead of silently downgrading the
environment.

### Conda and Docker

[`environment.yml`](environment.yml) provides a pinned Conda entry point for
the primary Python environment. The bootstrap script remains the authoritative
full installer because it also creates the MediaPipe sidecar, clones licensed
assets, downloads gated models, and performs hardware validation.

The Docker image uses a digest-pinned CUDA 12.6.3/cuDNN/Ubuntu 22.04 base and a
BuildKit secret, so the Hugging Face token does not enter an image layer:

```bash
DOCKER_BUILDKIT=1 docker build \
  --secret id=hf_token,env=HF_TOKEN \
  -t egodex-dexhand:cu126 .

docker run --rm -it --gpus all --ipc=host \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  egodex-dexhand:cu126
```

## Runtime layout

Bootstrap creates this layout by default:

```text
runtime/
├── .venv/
├── .venv-mediapipe/
├── project -> egodex-dexhand-retargeting/
├── egodex-dexhand-retargeting/   # this repository
└── third_party/
    ├── dex-retargeting/
    ├── sam2/
    │   └── checkpoints/sam2.1_hiera_small.pt
    ├── sam3/
    │   └── checkpoints/sam3.pt
    └── ProPainter/
        └── weights/{ProPainter,raft-things,recurrent_flow_completion}.pth
```

The locations can be overridden with `EGODEX_DEXHAND_ROOT`,
`EGODEX_VENV_ROOT`, and `EGODEX_THIRD_PARTY_ROOT`.

## Run the bimanual UR5e + Shadow pipeline

```bash
export EGODEX_DEXHAND_ROOT=/path/to/runtime

./egodex-dexhand-retargeting/scripts/run_bimanual_shadow.sh \
  /path/to/camera.mp4 \
  /path/to/trajectory.hdf5 \
  /path/to/output/run_001 \
  --force
```

The complete stage sequence is:

```text
prepare → retarget → render → segment → inpaint → compose
```

Runs can be resumed without recomputing prior stages:

```bash
./scripts/run_bimanual_shadow.sh VIDEO HDF5 OUTPUT \
  --start-stage segment --stop-stage compose
```

The final run directory contains the composite video, retargeted trajectories,
side-specific and union masks, the inpainted background, derived URDFs,
provenance with checkpoint hashes, and verification metrics.

For direct control of every parameter:

```bash
egodex-bimanual-shadow --help
egodex-ur5e-shadow --help
egodex-dexhand --help
egodex-whole-arm --help
```

## EgoQuest input

Convert an EgoQuest recording into the HDF5 schema used by the pipeline:

```bash
python scripts/convert_egoquest_to_egodex_hdf5.py \
  /path/to/recording \
  /path/to/trajectory.hdf5 \
  --start-frame 0 \
  --end-frame 540 \
  --active-hand both
```

For difficult lighting or projection mismatch, the repository also includes:

- `build_rgb_mask_landmarks.py` for image-aligned masking landmarks
- `align_egoquest_3d_to_rgb.py` for smoothed 3-D/RGB alignment
- `blend_render_variants.py` for temporally blended render variants

Pass the image-aligned result to `egodex-bimanual-shadow --mask-hdf5 ...` so
masking can be corrected without changing robot motion.

### Adaptive whole-recording processing

Use the adaptive runner when a recording alternates between left-hand,
right-hand, and bimanual activity:

```bash
python scripts/run_egoquest_adaptive_recording.py \
  --recording /path/to/egoquest/recording \
  --workspace /path/to/runtime \
  --run-name adaptive_run
```

The default processing policy reserves eight frames before and after a
projected hand interval for segmentation and inpainting. Final compositing is
controlled separately, frame by frame: unaligned world-coordinate landmarks
are projected into the camera, direct RGB landmark detections recover
calibration-edge cases, and each side must also contain at least 16 rendered
robot-mask pixels. Padded frames therefore help SAM3 and ProPainter without
making a robot appear early. A single valid projected landmark is enough to
open an interval, and intervals as short as six frames are retained. Long
intervals are split into balanced chunks so they cannot leave an undersized
tail. The runner also retains the Shadow Hand's distal `forearm` mesh while
hiding the proximal UR5e links. Override the processing reserve with
`--minimum-visible-landmarks`, `--entry-padding-frames`,
`--exit-padding-frames`, and `--min-frames` when needed.

Every segment writes `final/visibility_gate.json`, including human-visible,
robot-visible, composited, and mismatch frame indices for each side.

### Production multi-episode batch

Prepare an immutable manifest before launching the dataset-wide run:

```bash
export DATASET_ROOT=/path/to/interact_dataset/interact
export WORKSPACE=/path/to/egodex_dexhand_pipeline
export SCRATCH_ROOT=/path/to/local_nvme/egodex_batch

python scripts/run_interact_batch.py prepare \
  --dataset-root "$DATASET_ROOT" \
  --workspace "$WORKSPACE" \
  --project "$PWD" \
  --batch-name interact_all_sam3_v1 \
  --episodes 1-100
```

First prepare each compute host without starting an episode. This stages the
SAM2 compatibility runtime, ProPainter, robot assets, SAM3 code, and only the
selected SAM3.1 checkpoint to a host-local cache shared across batch versions.
It precomputes model hashes once and runs a real one-frame SAPIEN/Vulkan capture
on each eligible GPU:

```bash
python scripts/run_interact_batch.py host \
  --manifest "$WORKSPACE/batches/interact_all_sam3_v1/manifest.json" \
  --scratch-root "$SCRATCH_ROOT" \
  --minimum-gpu-free-gib 36 --maximum-gpus 4 --check-only
```

Remove `--check-only` to consume the shared queue. The host launcher creates
one independent episode lane per healthy GPU. This overlaps the chronological
CPU trajectory phase of later episodes with segmentation, rendering, and
inpainting on other GPUs instead of placing the entire host behind one
episode-level barrier. Atomic claims let multiple hosts safely consume the
same queue.

Each episode is staged to local NVMe and processed with an explicitly assigned,
preflighted Vulkan GPU. Model weights and robot assets are read from the
host-local cache, and provenance hashes are reused instead of re-reading about
380 MB of immutable checkpoints for every chunk. The worker requires zero
skipped segments, validates
the exact frame count of every segment and full-length MP4, generates the
keypoint/mask/retarget diagnostic, copies the validated episode to shared
storage, validates the promoted copy, and only then clears local scratch.
Partial runs are checkpointed to shared storage every five minutes, so another
host can resume after interruption. These changes do not alter masking, IK,
continuity, rendering, compositing, or inpainting parameters. Inspect progress
with:

```bash
python scripts/run_interact_batch.py status \
  --manifest "$WORKSPACE/batches/interact_all_sam3_v1/manifest.json"
```

## Tests

```bash
python -m pytest -q
```

The tests cover setup-lock integrity, credential safety, bootstrap dry-runs,
mask/inpaint alignment, render compositing, and temporal smoothing. Full
environment verification is also available independently:

```bash
python scripts/verify_environment.py \
  --runtime "$EGODEX_DEXHAND_ROOT" \
  --require-gpu --render-device cuda:0
```

## Current limitations

- There is no contact-aware object occlusion reconstruction yet.
- Heavy hand/object occlusion can still require corrected image-space prompts.
- ProPainter may leave ghosts when removal masks miss the first or last visible
  frames; the pipeline therefore expands and temporally smooths tracked masks.
- Robot arm base placement is fitted to the observed trajectory and may need a
  side-specific reference posture for unusual camera motion.

## Licensing

No repository-level open-source license has been granted yet. Third-party code,
checkpoints, datasets, and robot assets retain their own licenses and are not
redistributed here. See
[THIRD_PARTY.lock.md](THIRD_PARTY.lock.md).
