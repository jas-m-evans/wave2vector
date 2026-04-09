"""Tests for the rule-based commentary engine."""

from __future__ import annotations

import pytest

from app.commentary import (
    CommentaryResult,
    _confidence_label,
    _top_contrast,
    _top_shared,
    generate_commentary,
)
from app.recommender import BridgeResult, CandidateScores, Knobs

# ── Fixtures ───────────────────────────────────────────────────────────

PROFILE_A = {
    "energy": 0.85,
    "valence": 0.75,
    "danceability": 0.80,
    "acousticness": 0.10,
    "instrumentalness": 0.08,
    "speechiness": 0.12,
    "tempo_norm": 0.78,
}

PROFILE_B = {
    "energy": 0.22,
    "valence": 0.28,
    "danceability": 0.35,
    "acousticness": 0.88,
    "instrumentalness": 0.52,
    "speechiness": 0.05,
    "tempo_norm": 0.22,
}

_BRIDGE_SCORES = CandidateScores(
    sim_a=0.72,
    sim_b=0.69,
    sim_target=0.74,
    bridge_quality=0.705,
    fairness=0.97,
    novelty=0.28,
    total=0.71,
)

BRIDGE_RESULT = BridgeResult(
    track_id="track_bridge",
    track_name="The Middle",
    artist="Bridge Artist",
    album="Bridge Album",
    scores=_BRIDGE_SCORES,
    target_profile={
        "energy": 0.54,
        "valence": 0.52,
        "danceability": 0.58,
        "acousticness": 0.45,
        "instrumentalness": 0.28,
        "speechiness": 0.08,
        "tempo_norm": 0.50,
    },
    knobs_applied=Knobs().to_dict(),
)


# ── _top_shared ────────────────────────────────────────────────────────

def test_top_shared_returns_n_items():
    shared = _top_shared(PROFILE_A, PROFILE_B, n=2)
    assert len(shared) == 2


def test_top_shared_sorted_by_closeness():
    shared = _top_shared(PROFILE_A, PROFILE_B, n=3)
    diffs = [abs(a - b) for _, a, b in shared]
    assert diffs == sorted(diffs)


def test_top_shared_identical_profiles():
    shared = _top_shared(PROFILE_A, PROFILE_A, n=2)
    # All diffs are 0 — should still return n items
    assert len(shared) == 2
    for _, a, b in shared:
        assert a == b


# ── _top_contrast ──────────────────────────────────────────────────────

def test_top_contrast_returns_biggest_diff():
    feat, val_a, val_b = _top_contrast(PROFILE_A, PROFILE_B)
    all_diffs = {
        k: abs(PROFILE_A[k] - PROFILE_B[k]) for k in PROFILE_A
    }
    expected_feat = max(all_diffs, key=all_diffs.get)
    assert feat == expected_feat


def test_top_contrast_values_match_profiles():
    feat, val_a, val_b = _top_contrast(PROFILE_A, PROFILE_B)
    assert val_a == PROFILE_A[feat]
    assert val_b == PROFILE_B[feat]


# ── _confidence_label ──────────────────────────────────────────────────

@pytest.mark.parametrize("score,expected", [
    (0.80, "Strong"),
    (0.75, "Strong"),
    (0.65, "Good"),
    (0.60, "Good"),
    (0.50, "Moderate"),
    (0.45, "Moderate"),
    (0.30, "Low"),
    (0.0,  "Low"),
])
def test_confidence_label(score, expected):
    assert _confidence_label(score) == expected


# ── generate_commentary ────────────────────────────────────────────────

def test_generate_commentary_returns_dataclass():
    result = generate_commentary(PROFILE_A, PROFILE_B, "Alex", "Taylor", BRIDGE_RESULT)
    assert isinstance(result, CommentaryResult)


def test_shared_vibe_mentions_common_ground():
    result = generate_commentary(PROFILE_A, PROFILE_B, "Alex", "Taylor", BRIDGE_RESULT)
    text = result.shared_vibe.lower()
    assert "both" in text


def test_biggest_contrast_mentions_names():
    result = generate_commentary(PROFILE_A, PROFILE_B, "Alex", "Taylor", BRIDGE_RESULT)
    assert "Alex" in result.biggest_contrast or "Taylor" in result.biggest_contrast


def test_bridge_reason_mentions_track():
    result = generate_commentary(PROFILE_A, PROFILE_B, "Alex", "Taylor", BRIDGE_RESULT)
    assert "The Middle" in result.bridge_reason


def test_confidence_matches_bridge_quality():
    result = generate_commentary(PROFILE_A, PROFILE_B, "Alex", "Taylor", BRIDGE_RESULT)
    assert abs(result.confidence - _BRIDGE_SCORES.bridge_quality) < 1e-6


def test_confidence_label_is_valid():
    result = generate_commentary(PROFILE_A, PROFILE_B, "Alex", "Taylor", BRIDGE_RESULT)
    assert result.confidence_label in ("Low", "Moderate", "Good", "Strong")


def test_raw_scores_populated():
    result = generate_commentary(PROFILE_A, PROFILE_B, "Alex", "Taylor", BRIDGE_RESULT)
    assert "bridge_quality" in result.raw_scores
    assert "sim_a" in result.raw_scores
    assert "total" in result.raw_scores


def test_commentary_deterministic():
    """Same inputs always produce identical commentary strings."""
    r1 = generate_commentary(PROFILE_A, PROFILE_B, "Alex", "Taylor", BRIDGE_RESULT)
    r2 = generate_commentary(PROFILE_A, PROFILE_B, "Alex", "Taylor", BRIDGE_RESULT)
    assert r1.shared_vibe == r2.shared_vibe
    assert r1.biggest_contrast == r2.biggest_contrast
    assert r1.bridge_reason == r2.bridge_reason


def test_commentary_identical_profiles():
    """No crash when both users have identical taste profiles."""
    result = generate_commentary(PROFILE_A, PROFILE_A, "Alex", "Taylor", BRIDGE_RESULT)
    assert isinstance(result, CommentaryResult)
    assert len(result.shared_vibe) > 0


def test_commentary_works_with_missing_features():
    """Gracefully handles profiles that are missing some feature keys."""
    sparse_a = {"energy": 0.8, "valence": 0.7}
    sparse_b = {"energy": 0.3, "valence": 0.2}
    result = generate_commentary(sparse_a, sparse_b, "Alex", "Taylor", BRIDGE_RESULT)
    assert isinstance(result, CommentaryResult)
