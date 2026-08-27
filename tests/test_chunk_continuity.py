from __future__ import annotations

import numpy as np

from egodex_dexhand.chunk_continuity import (
    last_state,
    load_chunk_continuity_state,
    save_chunk_continuity_state,
    smooth_wrapped_joint_boundary,
    trailing_context,
    wrapped_joint_distance,
)


def test_continuity_state_round_trip_preserves_trajectory_tail(tmp_path) -> None:
    hand = np.arange(15, dtype=np.float32).reshape(3, 5)
    arm = np.arange(18, dtype=np.float32).reshape(3, 6)
    path = tmp_path / "left.npz"
    save_chunk_continuity_state(
        path,
        hand_qpos=hand,
        arm_qpos=arm,
        base_translation_world=np.ones(3),
        base_rotation_world=np.eye(3),
        source_frame=41,
    )
    state = load_chunk_continuity_state(path)
    assert state is not None
    np.testing.assert_array_equal(state.hand_qpos, hand)
    np.testing.assert_array_equal(last_state(state.arm_qpos), arm[-1])
    assert state.source_frame == 41


def test_wrapped_distance_treats_two_pi_as_same_branch() -> None:
    assert wrapped_joint_distance(np.array([np.pi]), np.array([-np.pi])) < 1e-12


def test_boundary_smoothing_uses_previous_margin_and_reduces_jump() -> None:
    previous = np.zeros((6, 2), dtype=np.float32)
    current = np.ones((12, 2), dtype=np.float32)
    smoothed = smooth_wrapped_joint_boundary(
        previous,
        current,
        margin_frames=5,
        lower=np.full(2, -2.0),
        upper=np.full(2, 2.0),
    )
    assert np.max(np.abs(smoothed[0] - previous[-1])) < 1.0
    assert np.max(np.abs(smoothed[-1] - current[-1])) < 0.05


def test_trailing_context_keeps_requested_cross_chunk_margin() -> None:
    previous = np.arange(6, dtype=np.float32).reshape(3, 2)
    current = np.arange(6, 14, dtype=np.float32).reshape(4, 2)
    result = trailing_context(previous, current, 5)
    np.testing.assert_array_equal(result, np.concatenate([previous, current])[-5:])
