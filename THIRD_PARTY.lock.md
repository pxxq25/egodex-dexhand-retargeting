# Tested third-party revisions

[`third_party.lock.json`](third_party.lock.json) is the machine-readable source
used by `bootstrap.sh`; this document records the corresponding human-readable
license and provenance notes.

| Component | Revision/checkpoint | SHA-256 when applicable |
|---|---|---|
| dex-retargeting | `3f56141bc8bd2760d5e452e382937269554ebb21` | — |
| dex-urdf asset submodule | `7304c7fb59214dab870eca02cf26f76e944e12df` | — |
| DexUMI | `acddb8f8a89a8f0186868bbec44306eb7808114a` | — |
| SAM2 | `2b90b9f5ceec907a1c18123530e92e794ad901a4` | — |
| SAM3 | `8f0b7f4d4e7eda2ed606ebde6702c93359ad01da` | — |
| ProPainter | `e870e79321c31b733e2031af5aa2fb1fe3ac7eec` | — |
| `sam2.1_hiera_small.pt` | checkpoint | `6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38` |
| `sam3.pt` | direct-video checkpoint | `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e` |
| `ProPainter.pth` | checkpoint | `12c070c4b48f374c91d8a2a17851140b85c159621080989f9e191bbc18bd6591` |
| `raft-things.pth` | checkpoint | `fcfa4125d6418f4de95d84aec20a3c5f4e205101715a79f193243c186ac9a7e1` |
| `recurrent_flow_completion.pth` | checkpoint | `22939a1a7900da878dbe1ccd011d646b1bfb30b8290039d8ff0e0c2fefbfd283` |

DexUMI remains in this historical provenance table because it informed the
pipeline design, but no production code or assets are imported from it. The
machine-readable runtime lock therefore installs dex-retargeting, SAM2, SAM3,
and ProPainter only.

Bootstrap clones the original repositories and their license files into the
runtime's `third_party/` directory. The licenses are not uniform; notably,
ProPainter is limited to noncommercial use. Checkpoints are downloaded from
their official Meta or ProPainter release locations and accepted only after
their SHA-256 values match the lock.

The integrated UR5e + Shadow run uses nested assets from the pinned
`dex-retargeting` tree:

- `assets/robots/hands/shadow_hand/LICENSE` is Apache-2.0.
- `assets/robots/arms/ur5e/LICENSE` contains Universal Robots' Terms and
  Conditions for Use of Graphical Documentation. Those terms include
  purpose- and use-specific restrictions; they are not an unrestricted
  open-source license.

Keep both notices with any redistributed model assets or derived URDF. The
pipeline writes its absolute-path/collision-reference repair into the run
directory and does not modify the vendored source assembly.
