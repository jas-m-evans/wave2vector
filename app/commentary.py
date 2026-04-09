"""Rule-based commentary engine.

Generates human-readable commentary for the Taste Bridge experience:
  - shared vibe (what the users have in common)
  - biggest contrast (where they diverge most)
  - bridge reason (why the recommendation fits)
  - confidence label and score

All functions are pure — no side effects, easy to unit-test.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.recommender import FEATURE_KEYS, BridgeResult, CandidateScores

# ---------------------------------------------------------------------------
# Feature vocabulary
# ---------------------------------------------------------------------------

_FEATURE_HIGH: dict[str, list[str]] = {
    "energy": ["high-energy", "intense", "driving"],
    "valence": ["upbeat", "feel-good", "joyful"],
    "danceability": ["danceable", "groove-heavy", "body-moving"],
    "acousticness": ["acoustic", "organic", "raw"],
    "instrumentalness": ["instrumental", "wordless", "texture-first"],
    "speechiness": ["lyric-dense", "wordy", "verse-heavy"],
    "tempo_norm": ["fast-paced", "high-tempo", "racing"],
}

_FEATURE_LOW: dict[str, list[str]] = {
    "energy": ["mellow", "calm", "introspective"],
    "valence": ["melancholic", "brooding", "dark"],
    "danceability": ["non-dance", "complex", "listening-first"],
    "acousticness": ["produced", "electronic", "polished"],
    "instrumentalness": ["vocal-forward", "lyric-led", "singer-centric"],
    "speechiness": ["instrumental-leaning", "sparse vocals", "vibe-driven"],
    "tempo_norm": ["slow-burning", "laid-back", "patient"],
}

_FEATURE_DISPLAY: dict[str, str] = {
    "energy": "energy",
    "valence": "mood",
    "danceability": "danceability",
    "acousticness": "acoustic texture",
    "instrumentalness": "instrumentation",
    "speechiness": "vocal density",
    "tempo_norm": "tempo",
}


def _describe(feature: str, value: float, deterministic_seed: int = 0) -> str:
    """Return a descriptor word for a feature value, picking from vocab by seed."""
    vocab = _FEATURE_HIGH[feature] if value >= 0.5 else _FEATURE_LOW[feature]
    return vocab[deterministic_seed % len(vocab)]


def _top_shared(
    profile_a: dict[str, float],
    profile_b: dict[str, float],
    n: int = 2,
) -> list[tuple[str, float, float]]:
    """Return n features where both users score most similarly (lowest abs diff)."""
    diffs = [
        (k, profile_a.get(k, 0.5), profile_b.get(k, 0.5))
        for k in FEATURE_KEYS
    ]
    diffs.sort(key=lambda t: abs(t[1] - t[2]))
    return diffs[:n]


def _top_contrast(
    profile_a: dict[str, float],
    profile_b: dict[str, float],
) -> tuple[str, float, float]:
    """Return the feature where the two users diverge most."""
    diffs = [
        (k, profile_a.get(k, 0.5), profile_b.get(k, 0.5))
        for k in FEATURE_KEYS
    ]
    diffs.sort(key=lambda t: abs(t[1] - t[2]), reverse=True)
    return diffs[0]


# ---------------------------------------------------------------------------
# Exported dataclass + generator
# ---------------------------------------------------------------------------


@dataclass
class CommentaryResult:
    shared_vibe: str
    biggest_contrast: str
    bridge_reason: str
    confidence: float  # 0–1
    confidence_label: str  # "Low" | "Moderate" | "Good" | "Strong"
    raw_scores: dict[str, float]


def _confidence_label(score: float) -> str:
    if score >= 0.75:
        return "Strong"
    if score >= 0.60:
        return "Good"
    if score >= 0.45:
        return "Moderate"
    return "Low"


def generate_commentary(
    profile_a: dict[str, float],
    profile_b: dict[str, float],
    name_a: str,
    name_b: str,
    result: BridgeResult,
) -> CommentaryResult:
    """Generate all commentary strings for a bridge recommendation."""
    scores: CandidateScores = result.scores

    # ---- Shared vibe ----
    shared = _top_shared(profile_a, profile_b, n=2)
    shared_parts: list[str] = []
    for i, (feat, val_a, val_b) in enumerate(shared):
        avg = (val_a + val_b) / 2
        word = _describe(feat, avg, deterministic_seed=i)
        shared_parts.append(f"{word} {_FEATURE_DISPLAY[feat]}")

    if len(shared_parts) == 2:
        shared_vibe = (
            f"Both of you gravitate toward {shared_parts[0]} "
            f"and {shared_parts[1]}."
        )
    elif shared_parts:
        shared_vibe = f"Both of you share a taste for {shared_parts[0]}."
    else:
        shared_vibe = "You share more in common than you might think."

    # ---- Biggest contrast ----
    contrast_feat, val_a, val_b = _top_contrast(profile_a, profile_b)
    if val_a > val_b:
        word_a = _describe(contrast_feat, val_a, deterministic_seed=0)
        word_b = _describe(contrast_feat, val_b, deterministic_seed=1)
        biggest_contrast = (
            f"{name_a} leans toward {word_a} {_FEATURE_DISPLAY[contrast_feat]}, "
            f"while {name_b} prefers {word_b} sounds."
        )
    else:
        word_b = _describe(contrast_feat, val_b, deterministic_seed=0)
        word_a = _describe(contrast_feat, val_a, deterministic_seed=1)
        biggest_contrast = (
            f"{name_b} leans toward {word_b} {_FEATURE_DISPLAY[contrast_feat]}, "
            f"while {name_a} prefers {word_a} sounds."
        )

    # ---- Bridge reason ----
    # Find the feature where the recommendation sits closest to the midpoint of A and B.
    midpoints = {
        k: (profile_a.get(k, 0.5) + profile_b.get(k, 0.5)) / 2
        for k in FEATURE_KEYS
    }
    bridge_feature = min(
        FEATURE_KEYS,
        key=lambda k: abs(
            (result.target_profile.get(k, 0.5) or midpoints[k]) - midpoints[k]
        ),
    )
    reco_val = result.target_profile.get(bridge_feature, 0.5)
    bridge_word = _describe(bridge_feature, reco_val, deterministic_seed=2)
    bridge_reason = (
        f'"{result.track_name}" lands right in the sweet spot — its '
        f"{bridge_word} {_FEATURE_DISPLAY[bridge_feature]} sits between "
        f"both of your profiles."
    )

    # ---- Confidence ----
    confidence = round(scores.bridge_quality, 3)
    label = _confidence_label(confidence)

    return CommentaryResult(
        shared_vibe=shared_vibe,
        biggest_contrast=biggest_contrast,
        bridge_reason=bridge_reason,
        confidence=confidence,
        confidence_label=label,
        raw_scores={
            "sim_a": scores.sim_a,
            "sim_b": scores.sim_b,
            "bridge_quality": scores.bridge_quality,
            "fairness": scores.fairness,
            "novelty": scores.novelty,
            "total": scores.total,
        },
    )
