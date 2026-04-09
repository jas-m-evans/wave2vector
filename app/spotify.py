"""Spotify OAuth2 PKCE + audio feature API.

When SPOTIFY_CLIENT_ID is not set the module falls back to deterministic demo
profiles so the Taste Bridge UI works without real credentials.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
import time
from typing import Any
from urllib.parse import urlencode

import httpx

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID", "")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "")
SPOTIFY_REDIRECT_URI = os.getenv(
    "SPOTIFY_REDIRECT_URI", "http://localhost:8000/auth/spotify/callback"
)

SPOTIFY_AUTH_URL = "https://accounts.spotify.com/authorize"
SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_API_BASE = "https://api.spotify.com/v1"

SCOPES = "user-read-private user-top-read user-read-email"

# ---------------------------------------------------------------------------
# PKCE helpers
# ---------------------------------------------------------------------------


def generate_pkce_pair() -> tuple[str, str]:
    """Return (code_verifier, code_challenge) for PKCE flow."""
    verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(verifier.encode()).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    return verifier, challenge


def get_auth_url(state: str, code_challenge: str) -> str:
    """Build the Spotify authorize URL."""
    params = {
        "response_type": "code",
        "client_id": SPOTIFY_CLIENT_ID,
        "scope": SCOPES,
        "redirect_uri": SPOTIFY_REDIRECT_URI,
        "state": state,
        "code_challenge_method": "S256",
        "code_challenge": code_challenge,
    }
    return f"{SPOTIFY_AUTH_URL}?{urlencode(params)}"


def exchange_code(code: str, code_verifier: str) -> dict[str, Any]:
    """Exchange an authorization code for access/refresh tokens."""
    resp = httpx.post(
        SPOTIFY_TOKEN_URL,
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": SPOTIFY_REDIRECT_URI,
            "client_id": SPOTIFY_CLIENT_ID,
            "code_verifier": code_verifier,
        },
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def refresh_access_token(refresh_token: str) -> dict[str, Any]:
    """Refresh an expired access token."""
    resp = httpx.post(
        SPOTIFY_TOKEN_URL,
        data={
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": SPOTIFY_CLIENT_ID,
        },
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# API calls
# ---------------------------------------------------------------------------


def get_profile(access_token: str) -> dict[str, Any]:
    resp = httpx.get(
        f"{SPOTIFY_API_BASE}/me",
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def get_top_tracks(
    access_token: str,
    time_range: str = "medium_term",
    limit: int = 20,
) -> list[dict[str, Any]]:
    resp = httpx.get(
        f"{SPOTIFY_API_BASE}/me/top/tracks",
        headers={"Authorization": f"Bearer {access_token}"},
        params={"time_range": time_range, "limit": limit},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json().get("items", [])


def get_audio_features(
    access_token: str, track_ids: list[str]
) -> list[dict[str, Any]]:
    """Fetch audio features for up to 100 track IDs."""
    if not track_ids:
        return []
    resp = httpx.get(
        f"{SPOTIFY_API_BASE}/audio-features",
        headers={"Authorization": f"Bearer {access_token}"},
        params={"ids": ",".join(track_ids[:100])},
        timeout=10,
    )
    resp.raise_for_status()
    return [f for f in resp.json().get("audio_features", []) if f]


def get_recommendations(
    access_token: str,
    seed_track_ids: list[str],
    target_features: dict[str, float],
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Fetch Spotify recommendations seeded by tracks and target audio features."""
    params: dict[str, Any] = {
        "seed_tracks": ",".join(seed_track_ids[:5]),
        "limit": limit,
    }
    # Map our feature keys to Spotify's target_* parameters
    feature_map = {
        "energy": "target_energy",
        "valence": "target_valence",
        "danceability": "target_danceability",
        "acousticness": "target_acousticness",
        "instrumentalness": "target_instrumentalness",
        "speechiness": "target_speechiness",
    }
    for key, spotify_key in feature_map.items():
        if key in target_features:
            params[spotify_key] = round(target_features[key], 3)
    resp = httpx.get(
        f"{SPOTIFY_API_BASE}/recommendations",
        headers={"Authorization": f"Bearer {access_token}"},
        params=params,
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json().get("tracks", [])


# ---------------------------------------------------------------------------
# Taste profile
# ---------------------------------------------------------------------------

_FEATURE_KEYS = [
    "energy",
    "valence",
    "danceability",
    "acousticness",
    "instrumentalness",
    "speechiness",
]


def normalize_tempo(bpm: float) -> float:
    """Map BPM (40–220) to a 0–1 scale."""
    return max(0.0, min(1.0, (bpm - 40) / 180))


def compute_taste_profile(features_list: list[dict[str, Any]]) -> dict[str, float]:
    """Average Spotify audio features across a list of tracks into a taste profile."""
    if not features_list:
        return {k: 0.5 for k in _FEATURE_KEYS}
    profile: dict[str, float] = {}
    for key in _FEATURE_KEYS:
        values = [f[key] for f in features_list if key in f and f[key] is not None]
        profile[key] = float(sum(values) / len(values)) if values else 0.5
    tempos = [f["tempo"] for f in features_list if "tempo" in f and f["tempo"]]
    profile["tempo_norm"] = normalize_tempo(
        float(sum(tempos) / len(tempos)) if tempos else 120.0
    )
    return profile


def compute_compatibility(
    profile_a: dict[str, float], profile_b: dict[str, float]
) -> float:
    """Cosine similarity between two taste profiles (0–1 scale)."""
    keys = list(profile_a.keys() & profile_b.keys())
    if not keys:
        return 0.5
    a = [profile_a[k] for k in keys]
    b = [profile_b[k] for k in keys]
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.5
    return max(0.0, min(1.0, dot / (norm_a * norm_b)))


# ---------------------------------------------------------------------------
# Demo mode — deterministic mock profiles when no Spotify credentials
# ---------------------------------------------------------------------------

# A curated set of mock candidate tracks (Spotify audio feature schema).
MOCK_CANDIDATES: list[dict[str, Any]] = [
    {
        "track_id": "mock_midnight_city",
        "track_name": "Midnight City",
        "artist": "M83",
        "album": "Hurry Up, We're Dreaming",
        "album_art": None,
        "preview_url": None,
        "energy": 0.821,
        "valence": 0.543,
        "danceability": 0.578,
        "acousticness": 0.003,
        "instrumentalness": 0.614,
        "speechiness": 0.044,
        "tempo_norm": normalize_tempo(105.0),
    },
    {
        "track_id": "mock_retrograde",
        "track_name": "Retrograde",
        "artist": "James Blake",
        "album": "Overgrown",
        "album_art": None,
        "preview_url": None,
        "energy": 0.374,
        "valence": 0.291,
        "danceability": 0.579,
        "acousticness": 0.302,
        "instrumentalness": 0.001,
        "speechiness": 0.035,
        "tempo_norm": normalize_tempo(80.0),
    },
    {
        "track_id": "mock_feels_like_summer",
        "track_name": "Feels Like Summer",
        "artist": "Childish Gambino",
        "album": "Summer Pack",
        "album_art": None,
        "preview_url": None,
        "energy": 0.434,
        "valence": 0.702,
        "danceability": 0.719,
        "acousticness": 0.169,
        "instrumentalness": 0.0,
        "speechiness": 0.089,
        "tempo_norm": normalize_tempo(86.0),
    },
    {
        "track_id": "mock_let_it_happen",
        "track_name": "Let It Happen",
        "artist": "Tame Impala",
        "album": "Currents",
        "album_art": None,
        "preview_url": None,
        "energy": 0.582,
        "valence": 0.454,
        "danceability": 0.612,
        "acousticness": 0.003,
        "instrumentalness": 0.202,
        "speechiness": 0.028,
        "tempo_norm": normalize_tempo(116.0),
    },
    {
        "track_id": "mock_electric_feel",
        "track_name": "Electric Feel",
        "artist": "MGMT",
        "album": "Oracular Spectacular",
        "album_art": None,
        "preview_url": None,
        "energy": 0.623,
        "valence": 0.773,
        "danceability": 0.721,
        "acousticness": 0.098,
        "instrumentalness": 0.032,
        "speechiness": 0.046,
        "tempo_norm": normalize_tempo(107.0),
    },
    {
        "track_id": "mock_bad_guy",
        "track_name": "bad guy",
        "artist": "Billie Eilish",
        "album": "When We All Fall Asleep",
        "album_art": None,
        "preview_url": None,
        "energy": 0.427,
        "valence": 0.559,
        "danceability": 0.702,
        "acousticness": 0.337,
        "instrumentalness": 0.0,
        "speechiness": 0.365,
        "tempo_norm": normalize_tempo(135.0),
    },
    {
        "track_id": "mock_redbone",
        "track_name": "Redbone",
        "artist": "Childish Gambino",
        "album": "Awaken, My Love!",
        "album_art": None,
        "preview_url": None,
        "energy": 0.452,
        "valence": 0.444,
        "danceability": 0.739,
        "acousticness": 0.196,
        "instrumentalness": 0.001,
        "speechiness": 0.043,
        "tempo_norm": normalize_tempo(95.0),
    },
    {
        "track_id": "mock_the_less_i_know",
        "track_name": "The Less I Know the Better",
        "artist": "Tame Impala",
        "album": "Currents",
        "album_art": None,
        "preview_url": None,
        "energy": 0.756,
        "valence": 0.709,
        "danceability": 0.758,
        "acousticness": 0.006,
        "instrumentalness": 0.049,
        "speechiness": 0.034,
        "tempo_norm": normalize_tempo(116.0),
    },
    {
        "track_id": "mock_breathe",
        "track_name": "Breathe (2AM)",
        "artist": "Anna Nalick",
        "album": "Wreck of the Day",
        "album_art": None,
        "preview_url": None,
        "energy": 0.348,
        "valence": 0.315,
        "danceability": 0.446,
        "acousticness": 0.738,
        "instrumentalness": 0.0,
        "speechiness": 0.026,
        "tempo_norm": normalize_tempo(74.0),
    },
    {
        "track_id": "mock_starboy",
        "track_name": "Starboy",
        "artist": "The Weeknd",
        "album": "Starboy",
        "album_art": None,
        "preview_url": None,
        "energy": 0.589,
        "valence": 0.482,
        "danceability": 0.679,
        "acousticness": 0.129,
        "instrumentalness": 0.0,
        "speechiness": 0.234,
        "tempo_norm": normalize_tempo(186.0),
    },
    {
        "track_id": "mock_mykonos",
        "track_name": "Mykonos",
        "artist": "Fleet Foxes",
        "album": "Sun Giant",
        "album_art": None,
        "preview_url": None,
        "energy": 0.421,
        "valence": 0.571,
        "danceability": 0.427,
        "acousticness": 0.912,
        "instrumentalness": 0.028,
        "speechiness": 0.038,
        "tempo_norm": normalize_tempo(150.0),
    },
    {
        "track_id": "mock_ribs",
        "track_name": "Ribs",
        "artist": "Lorde",
        "album": "Pure Heroine",
        "album_art": None,
        "preview_url": None,
        "energy": 0.385,
        "valence": 0.179,
        "danceability": 0.493,
        "acousticness": 0.017,
        "instrumentalness": 0.001,
        "speechiness": 0.031,
        "tempo_norm": normalize_tempo(92.0),
    },
]


def demo_profile_from_name(display_name: str) -> dict[str, float]:
    """Generate a deterministic mock taste profile from a display name."""
    seed = int(
        hmac.new(
            b"taste-bridge-demo",
            display_name.lower().encode(),
            hashlib.sha256,
        ).hexdigest(),
        16,
    ) % (2**32)
    # Lightweight deterministic RNG (xorshift32) from seed
    state = seed if seed != 0 else 1

    def next_float() -> float:
        nonlocal state
        state ^= state << 13
        state &= 0xFFFFFFFF
        state ^= state >> 17
        state &= 0xFFFFFFFF
        state ^= state << 5
        state &= 0xFFFFFFFF
        return (state & 0xFFFF) / 0xFFFF

    def ranged(lo: float, hi: float) -> float:
        return round(lo + next_float() * (hi - lo), 3)

    return {
        "energy": ranged(0.2, 0.92),
        "valence": ranged(0.18, 0.90),
        "danceability": ranged(0.28, 0.92),
        "acousticness": ranged(0.05, 0.85),
        "instrumentalness": ranged(0.0, 0.65),
        "speechiness": ranged(0.02, 0.28),
        "tempo_norm": ranged(0.25, 0.82),
    }


def is_demo_mode() -> bool:
    return not bool(SPOTIFY_CLIENT_ID)
