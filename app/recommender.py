"""Deterministic bridge recommendation engine.

All ranking is pure-functional and reproducible given the same inputs + seed.
Keeps the algorithm transparent and easy to explain in an interview.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Feature keys used for scoring (order matters for weight array).
FEATURE_KEYS: list[str] = [
    "energy",
    "valence",
    "danceability",
    "acousticness",
    "instrumentalness",
    "speechiness",
    "tempo_norm",
]

# Baseline weights for each audio dimension.
BASE_WEIGHTS: list[float] = [0.22, 0.22, 0.14, 0.12, 0.10, 0.08, 0.12]


@dataclass
class Knobs:
    """Preference dials that shift the target profile and scoring weights."""

    balanceBias: float = 0.5  # 0=favor user A, 1=favor user B
    noveltyBias: float = 0.3  # 0=familiar, 1=adventurous
    energyBias: float = 0.5  # 0=mellow, 1=high-energy
    moodBias: float = 0.5  # 0=dark/melancholic, 1=bright/upbeat

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> "Knobs":
        return cls(**{k: float(v) for k, v in d.items() if hasattr(cls, k)})

    def to_dict(self) -> dict[str, float]:
        return {
            "balanceBias": self.balanceBias,
            "noveltyBias": self.noveltyBias,
            "energyBias": self.energyBias,
            "moodBias": self.moodBias,
        }


@dataclass
class CandidateScores:
    sim_a: float = 0.0
    sim_b: float = 0.0
    sim_target: float = 0.0
    bridge_quality: float = 0.0
    fairness: float = 0.0
    novelty: float = 0.0
    total: float = 0.0


@dataclass
class BridgeResult:
    track_id: str
    track_name: str
    artist: str
    album: str = ""
    album_art: str | None = None
    preview_url: str | None = None
    scores: CandidateScores = field(default_factory=CandidateScores)
    target_profile: dict[str, float] = field(default_factory=dict)
    knobs_applied: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal helpers — all pure functions
# ---------------------------------------------------------------------------


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * _clamp(t)


def _cosine_similarity(v1: list[float], v2: list[float]) -> float:
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = sum(x * x for x in v1) ** 0.5
    norm2 = sum(x * x for x in v2) ** 0.5
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return max(-1.0, min(1.0, dot / (norm1 * norm2)))


def _profile_to_vector(profile: dict[str, float]) -> list[float]:
    return [profile.get(k, 0.5) for k in FEATURE_KEYS]


def _weighted_similarity(
    candidate: dict[str, float],
    target: dict[str, float],
    weights: list[float],
) -> float:
    """Weighted dot-product similarity normalised to [0, 1]."""
    score = 0.0
    for k, w in zip(FEATURE_KEYS, weights):
        cv = candidate.get(k, 0.5)
        tv = target.get(k, 0.5)
        score += w * (1.0 - abs(cv - tv))
    return _clamp(score / sum(weights))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_target_profile(
    profile_a: dict[str, float],
    profile_b: dict[str, float],
    knobs: Knobs,
) -> dict[str, float]:
    """Blend the two user profiles according to the current knob settings."""
    target: dict[str, float] = {}
    for k in FEATURE_KEYS:
        blended = _lerp(profile_a.get(k, 0.5), profile_b.get(k, 0.5), knobs.balanceBias)
        target[k] = blended

    # Apply energy + mood bias — pull up to 40% toward the desired pole.
    target["energy"] = _lerp(target["energy"], knobs.energyBias, 0.4)
    target["valence"] = _lerp(target["valence"], knobs.moodBias, 0.4)

    # Novelty bias: shift target slightly away from both users' centroids.
    if knobs.noveltyBias > 0.5:
        push = (knobs.noveltyBias - 0.5) * 0.5
        for k in FEATURE_KEYS:
            mid_a = profile_a.get(k, 0.5)
            mid_b = profile_b.get(k, 0.5)
            avg = (mid_a + mid_b) / 2
            # Push target away from average toward the opposite pole
            delta = target[k] - avg
            target[k] = _clamp(target[k] + delta * push)

    return {k: _clamp(v) for k, v in target.items()}


def score_candidate(
    candidate: dict[str, Any],
    profile_a: dict[str, float],
    profile_b: dict[str, float],
    target: dict[str, float],
    weights: list[float],
    novelty_bias: float,
) -> CandidateScores:
    """Score a single candidate track against both user profiles and the target."""
    sim_target = _weighted_similarity(candidate, target, weights)
    sim_a = _weighted_similarity(candidate, profile_a, weights)
    sim_b = _weighted_similarity(candidate, profile_b, weights)

    bridge_quality = (sim_a + sim_b) / 2
    fairness = 1.0 - abs(sim_a - sim_b)
    novelty = 1.0 - max(sim_a, sim_b)

    total = (
        0.35 * sim_target
        + 0.25 * bridge_quality
        + 0.20 * fairness
        + 0.20 * _lerp(0.0, novelty, novelty_bias)
    )

    return CandidateScores(
        sim_a=round(sim_a, 4),
        sim_b=round(sim_b, 4),
        sim_target=round(sim_target, 4),
        bridge_quality=round(bridge_quality, 4),
        fairness=round(fairness, 4),
        novelty=round(novelty, 4),
        total=round(total, 4),
    )


def recommend_bridge(
    profile_a: dict[str, float],
    profile_b: dict[str, float],
    candidates: list[dict[str, Any]],
    knobs: Knobs | None = None,
    previous_id: str | None = None,
    seed: int = 42,
) -> BridgeResult | None:
    """Return the best bridge recommendation.

    Deterministic: same inputs + same seed always produce the same result.
    Applies a 35% score penalty when a candidate matches the previous pick,
    encouraging diversity on /next.
    """
    if not candidates:
        return None

    knobs = knobs or Knobs()
    target = compute_target_profile(profile_a, profile_b, knobs)
    weights = list(BASE_WEIGHTS)  # copy; extendable per-knob in future

    scored: list[tuple[float, str, dict[str, Any], CandidateScores]] = []
    for candidate in candidates:
        scores = score_candidate(
            candidate, profile_a, profile_b, target, weights, knobs.noveltyBias
        )
        adj_total = scores.total * (0.65 if candidate.get("track_id") == previous_id else 1.0)
        # Tie-break deterministically by track_id so results are reproducible.
        scored.append((adj_total, candidate.get("track_id", ""), candidate, scores))

    # Sort descending by adjusted score, then by track_id ascending for tie-breaking.
    scored.sort(key=lambda x: (-x[0], x[1]))
    best = scored[0]

    _, _, candidate, scores = best
    return BridgeResult(
        track_id=candidate.get("track_id", ""),
        track_name=candidate.get("track_name", "Unknown"),
        artist=candidate.get("artist", "Unknown"),
        album=candidate.get("album", ""),
        album_art=candidate.get("album_art"),
        preview_url=candidate.get("preview_url"),
        scores=scores,
        target_profile=target,
        knobs_applied=knobs.to_dict(),
    )


def apply_slash_command(knobs: Knobs, command: str) -> Knobs:
    """Return updated knobs based on a slash command.  Pure — does not mutate input."""
    d = knobs.to_dict()

    adjustments: dict[str, dict[str, float]] = {
        "/balanced": {"balanceBias": 0.5, "noveltyBias": 0.3},
        "/more-me": {"balanceBias": max(0.0, d["balanceBias"] - 0.2)},
        "/more-them": {"balanceBias": min(1.0, d["balanceBias"] + 0.2)},
        "/weirder": {"noveltyBias": min(1.0, d["noveltyBias"] + 0.25)},
        "/safer": {"noveltyBias": max(0.0, d["noveltyBias"] - 0.25)},
        "/moodier": {"moodBias": max(0.0, d["moodBias"] - 0.2)},
        "/more-energy": {"energyBias": min(1.0, d["energyBias"] + 0.25)},
        "/next": {},  # diversity handled by previous_id penalty — no knob change
        "/why": {},  # commentary-only, no knob change
    }

    if command in adjustments:
        d.update(adjustments[command])
    return Knobs.from_dict(d)


def parse_natural_language_nudge(text: str) -> dict[str, float]:
    """Extract knob adjustments from a free-text message.  Returns a partial knobs dict."""
    text_lower = text.lower()
    adjustments: dict[str, float] = {}

    energy_up_words = ["more energy", "energetic", "upbeat", "pumped", "intense", "bangers"]
    energy_down_words = ["chill", "calm", "mellow", "relax", "quiet", "laid-back"]
    mood_up_words = ["happier", "brighter", "feel-good", "uplifting", "joyful"]
    mood_down_words = ["darker", "darker", "melancholic", "sad", "moody", "emotional", "brooding"]
    novelty_up_words = ["weirder", "unexpected", "left turn", "adventurous", "eclectic", "obscure"]
    novelty_down_words = ["safer", "familiar", "mainstream", "easy", "comfortable"]

    if any(w in text_lower for w in energy_up_words):
        adjustments["energyBias"] = 0.75
    elif any(w in text_lower for w in energy_down_words):
        adjustments["energyBias"] = 0.25

    if any(w in text_lower for w in mood_up_words):
        adjustments["moodBias"] = 0.75
    elif any(w in text_lower for w in mood_down_words):
        adjustments["moodBias"] = 0.25

    if any(w in text_lower for w in novelty_up_words):
        adjustments["noveltyBias"] = 0.75
    elif any(w in text_lower for w in novelty_down_words):
        adjustments["noveltyBias"] = 0.1

    if "both" in text_lower and any(w in text_lower for w in ["balanced", "middle", "together"]):
        adjustments["balanceBias"] = 0.5

    return adjustments
