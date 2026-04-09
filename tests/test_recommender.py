"""Tests for the deterministic bridge recommendation engine."""

from __future__ import annotations

import pytest

from app.recommender import (
    FEATURE_KEYS,
    BridgeResult,
    Knobs,
    apply_slash_command,
    compute_target_profile,
    parse_natural_language_nudge,
    recommend_bridge,
    score_candidate,
)

# ── Fixtures ───────────────────────────────────────────────────────────

PROFILE_ENERGETIC = {
    "energy": 0.9,
    "valence": 0.7,
    "danceability": 0.8,
    "acousticness": 0.1,
    "instrumentalness": 0.1,
    "speechiness": 0.1,
    "tempo_norm": 0.8,
}

PROFILE_MELLOW = {
    "energy": 0.2,
    "valence": 0.3,
    "danceability": 0.35,
    "acousticness": 0.85,
    "instrumentalness": 0.5,
    "speechiness": 0.05,
    "tempo_norm": 0.25,
}

MOCK_TRACKS = [
    {
        "track_id": "track_a",
        "track_name": "Energetic Anthem",
        "artist": "High Octane",
        "energy": 0.88,
        "valence": 0.72,
        "danceability": 0.82,
        "acousticness": 0.08,
        "instrumentalness": 0.05,
        "speechiness": 0.09,
        "tempo_norm": 0.78,
    },
    {
        "track_id": "track_b",
        "track_name": "Calm Waters",
        "artist": "Mellow Sound",
        "energy": 0.22,
        "valence": 0.32,
        "danceability": 0.38,
        "acousticness": 0.88,
        "instrumentalness": 0.45,
        "speechiness": 0.04,
        "tempo_norm": 0.22,
    },
    {
        "track_id": "track_bridge",
        "track_name": "The Middle",
        "artist": "Bridge Artist",
        "energy": 0.55,
        "valence": 0.50,
        "danceability": 0.58,
        "acousticness": 0.45,
        "instrumentalness": 0.28,
        "speechiness": 0.07,
        "tempo_norm": 0.52,
    },
]


# ── compute_target_profile ─────────────────────────────────────────────

def test_target_balanced_is_midpoint():
    knobs = Knobs(balanceBias=0.5, noveltyBias=0.0, energyBias=0.5, moodBias=0.5)
    target = compute_target_profile(PROFILE_ENERGETIC, PROFILE_MELLOW, knobs)
    expected_energy = (PROFILE_ENERGETIC["energy"] + PROFILE_MELLOW["energy"]) / 2
    # Energy bias of 0.5 pulls 40% toward 0.5 from the blended midpoint
    assert abs(target["energy"] - expected_energy * 0.6 - 0.5 * 0.4) < 1e-6


def test_target_balance_bias_favors_a():
    knobs = Knobs(balanceBias=0.0, noveltyBias=0.0, energyBias=0.5, moodBias=0.5)
    target = compute_target_profile(PROFILE_ENERGETIC, PROFILE_MELLOW, knobs)
    # balanceBias=0 should pull heavily toward profile A (energetic)
    assert target["acousticness"] < 0.35  # energetic profile has low acousticness


def test_target_balance_bias_favors_b():
    knobs = Knobs(balanceBias=1.0, noveltyBias=0.0, energyBias=0.5, moodBias=0.5)
    target = compute_target_profile(PROFILE_ENERGETIC, PROFILE_MELLOW, knobs)
    # balanceBias=1 pulls toward profile B (mellow)
    assert target["acousticness"] > 0.6  # mellow profile has high acousticness


def test_target_values_clamped():
    knobs = Knobs(balanceBias=0.5, noveltyBias=0.0, energyBias=1.0, moodBias=1.0)
    target = compute_target_profile(PROFILE_MELLOW, PROFILE_MELLOW, knobs)
    for v in target.values():
        assert 0.0 <= v <= 1.0


def test_target_all_features_present():
    target = compute_target_profile(PROFILE_ENERGETIC, PROFILE_MELLOW, Knobs())
    for key in FEATURE_KEYS:
        assert key in target


# ── recommend_bridge ───────────────────────────────────────────────────

def test_recommend_bridge_returns_result():
    result = recommend_bridge(PROFILE_ENERGETIC, PROFILE_MELLOW, MOCK_TRACKS, Knobs())
    assert isinstance(result, BridgeResult)


def test_recommend_bridge_deterministic():
    """Same inputs → same output regardless of call order."""
    r1 = recommend_bridge(PROFILE_ENERGETIC, PROFILE_MELLOW, MOCK_TRACKS, Knobs(), seed=42)
    r2 = recommend_bridge(PROFILE_ENERGETIC, PROFILE_MELLOW, MOCK_TRACKS, Knobs(), seed=42)
    assert r1.track_id == r2.track_id
    assert r1.scores.total == r2.scores.total


def test_recommend_bridge_prefers_bridge_track():
    """The track closest to the midpoint should score well under balanced knobs."""
    result = recommend_bridge(PROFILE_ENERGETIC, PROFILE_MELLOW, MOCK_TRACKS, Knobs())
    # 'track_bridge' sits between both profiles — expect it to win on bridge quality
    assert result.track_id == "track_bridge"


def test_recommend_bridge_returns_none_for_empty_candidates():
    assert recommend_bridge(PROFILE_ENERGETIC, PROFILE_MELLOW, [], Knobs()) is None


def test_recommend_bridge_diversity_penalty():
    """Repeating the previous track should not be the top result when /next is used."""
    result_first = recommend_bridge(
        PROFILE_ENERGETIC, PROFILE_MELLOW, MOCK_TRACKS, Knobs()
    )
    assert result_first is not None
    result_next = recommend_bridge(
        PROFILE_ENERGETIC,
        PROFILE_MELLOW,
        MOCK_TRACKS,
        Knobs(),
        previous_id=result_first.track_id,
    )
    assert result_next is not None
    # With penalty applied, the same track shouldn't win again
    assert result_next.track_id != result_first.track_id


def test_recommend_bridge_score_in_range():
    result = recommend_bridge(PROFILE_ENERGETIC, PROFILE_MELLOW, MOCK_TRACKS, Knobs())
    assert result is not None
    s = result.scores
    assert 0.0 <= s.total <= 1.0
    assert 0.0 <= s.bridge_quality <= 1.0
    assert 0.0 <= s.fairness <= 1.0


# ── apply_slash_command ────────────────────────────────────────────────

def test_slash_balanced_resets_knobs():
    knobs = Knobs(balanceBias=0.1, noveltyBias=0.9)
    updated = apply_slash_command(knobs, "/balanced")
    assert updated.balanceBias == 0.5
    assert updated.noveltyBias == 0.3


def test_slash_more_energy_increases_energy():
    knobs = Knobs(energyBias=0.4)
    updated = apply_slash_command(knobs, "/more-energy")
    assert updated.energyBias > knobs.energyBias


def test_slash_weirder_increases_novelty():
    knobs = Knobs(noveltyBias=0.3)
    updated = apply_slash_command(knobs, "/weirder")
    assert updated.noveltyBias > knobs.noveltyBias


def test_slash_safer_decreases_novelty():
    knobs = Knobs(noveltyBias=0.7)
    updated = apply_slash_command(knobs, "/safer")
    assert updated.noveltyBias < knobs.noveltyBias


def test_slash_more_me_shifts_balance_toward_a():
    knobs = Knobs(balanceBias=0.5)
    updated = apply_slash_command(knobs, "/more-me")
    assert updated.balanceBias < 0.5


def test_slash_more_them_shifts_balance_toward_b():
    knobs = Knobs(balanceBias=0.5)
    updated = apply_slash_command(knobs, "/more-them")
    assert updated.balanceBias > 0.5


def test_slash_why_and_next_do_not_change_knobs():
    knobs = Knobs(balanceBias=0.6, noveltyBias=0.4)
    for cmd in ["/why", "/next"]:
        updated = apply_slash_command(knobs, cmd)
        assert updated.balanceBias == knobs.balanceBias
        assert updated.noveltyBias == knobs.noveltyBias


def test_slash_knobs_always_clamped():
    knobs = Knobs(energyBias=0.95, noveltyBias=0.95)
    updated = apply_slash_command(knobs, "/more-energy")
    assert updated.energyBias <= 1.0
    updated2 = apply_slash_command(knobs, "/weirder")
    assert updated2.noveltyBias <= 1.0


def test_slash_knobs_immutable():
    """apply_slash_command must not mutate the input Knobs instance."""
    knobs = Knobs(balanceBias=0.5)
    apply_slash_command(knobs, "/more-me")
    assert knobs.balanceBias == 0.5


# ── parse_natural_language_nudge ───────────────────────────────────────

def test_nlp_energy_up():
    adj = parse_natural_language_nudge("give us more energy please")
    assert adj.get("energyBias", 0) >= 0.6


def test_nlp_energy_down():
    adj = parse_natural_language_nudge("something more chill and relaxed")
    assert adj.get("energyBias", 1) <= 0.4


def test_nlp_darker():
    adj = parse_natural_language_nudge("make it darker and more emotional")
    assert adj.get("moodBias", 1) <= 0.4


def test_nlp_weirder():
    adj = parse_natural_language_nudge("give us a left turn — something unexpected")
    assert adj.get("noveltyBias", 0) >= 0.6


def test_nlp_no_match_returns_empty():
    adj = parse_natural_language_nudge("hello there")
    assert adj == {}


# ── Knobs dataclass ────────────────────────────────────────────────────

def test_knobs_from_dict_roundtrip():
    d = {"balanceBias": 0.3, "noveltyBias": 0.7, "energyBias": 0.6, "moodBias": 0.4}
    k = Knobs.from_dict(d)
    assert k.to_dict() == d


def test_knobs_from_dict_ignores_unknown_keys():
    k = Knobs.from_dict({"balanceBias": 0.2, "unknownKey": 99})
    assert k.balanceBias == 0.2
