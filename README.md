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
        ├─ SAM2 side-specific tracking ────────── human arm/hand masks
        └─ ProPainter ─────────────────────────── human-removed background
                                                        │
SAPIEN robot render + alpha/motion blur ────────────────┘
        │
        └─ temporally smoothed composite MP4
```

This repository contains orchestration, retargeting, rendering, masking,
inpainting integration, compositing, verification, EgoQuest conversion tools,
and tests. It deliberately does **not** contain datasets, checkpoints, model
weights, generated videos, or third-party robot assets.

## Requirements

- Linux with an NVIDIA GPU (the validated environment used CUDA)
- Python 3.10+
- FFmpeg
- The pinned third-party revisions in [THIRD_PARTY.lock.md](THIRD_PARTY.lock.md)
- SAM2 and ProPainter checkpoints listed in that lock file
- SAPIEN, Pinocchio, PyTorch, and `dex-retargeting`

ProPainter is noncommercial research software. UR5e graphical assets also have
separate use restrictions. Review every upstream license before distributing
assets or outputs.

## Runtime layout

The convenience runner expects this layout by default:

```text
runtime/
├── .venv/
├── egodex-dexhand-retargeting/   # this repository
└── third_party/
    ├── dex-retargeting/
    ├── sam2/
    │   └── checkpoints/sam2.1_hiera_small.pt
    └── ProPainter/
        └── weights/
```

The locations can be overridden with `EGODEX_DEXHAND_ROOT`,
`EGODEX_VENV_ROOT`, and `EGODEX_THIRD_PARTY_ROOT`.

## Installation

Create the environment, install this package, and then install each pinned
third-party project according to its upstream instructions:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e './egodex-dexhand-retargeting[test,egoquest]'
```

`egoquest` is optional; it installs MediaPipe and PyArrow for the conversion
utilities. Model downloads remain explicit so checkpoint hashes can be checked
against [THIRD_PARTY.lock.md](THIRD_PARTY.lock.md).

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

The default visibility policy starts removal eight frames before a projected
hand first becomes reliable and keeps it active for four frames after the last
detection. A single valid projected landmark is enough to open an interval,
and intervals as short as six frames are retained. These defaults prevent the
human hand or sleeve from appearing briefly before inpainting begins. The
runner also retains the Shadow Hand's distal `forearm` mesh while hiding the
proximal UR5e links. Override the policy with
`--minimum-visible-landmarks`, `--entry-padding-frames`,
`--exit-padding-frames`, and `--min-frames` when needed.

## Tests

```bash
python -m pytest -q
```

The tests cover mask/inpaint alignment, render compositing, and temporal
smoothing. Full integration verification additionally requires the pinned
third-party models and a sample episode.

## Current limitations

- There is no contact-aware object occlusion reconstruction yet.
- Heavy hand/object occlusion can still require corrected image-space prompts.
- ProPainter may leave ghosts when removal masks miss the first or last visible
  frames; the pipeline therefore expands and temporally smooths tracked masks.
- Robot arm base placement is fitted to the observed trajectory and may need a
  side-specific reference posture for unusual camera motion.

## Licensing

No repository-level open-source license has been granted yet. The repository is
private by default. Third-party code, checkpoints, datasets, and robot assets
retain their own licenses and are not redistributed here. See
[THIRD_PARTY.lock.md](THIRD_PARTY.lock.md).
