"""LiveKit JWT token generation.

Generates access tokens for LiveKit rooms. When LIVEKIT_API_KEY is not set,
tokens are generated with a demo key and the frontend connects to the default
local LiveKit URL, which is fine for portfolio demos.

LiveKit token format: standard HS256 JWT with a 'video' claim.
See https://docs.livekit.io/reference/server/generating-tokens/
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import jwt

LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "devkey")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "devsecret-at-least-32-chars-long!")
LIVEKIT_URL = os.getenv("LIVEKIT_URL", "ws://localhost:7880")

_TOKEN_TTL = 7200  # 2 hours


def create_token(
    room_name: str,
    identity: str,
    display_name: str,
    metadata: dict[str, Any] | None = None,
    attributes: dict[str, str] | None = None,
    ttl: int = _TOKEN_TTL,
) -> str:
    """Return a signed LiveKit access token for the given room and participant."""
    now = int(time.time())
    payload: dict[str, Any] = {
        "sub": identity,
        "iss": LIVEKIT_API_KEY,
        "nbf": now,
        "exp": now + ttl,
        "name": display_name,
        # LiveKit video grant
        "video": {
            "room": room_name,
            "roomJoin": True,
            "canPublish": True,
            "canSubscribe": True,
            "canPublishData": True,
        },
    }
    if metadata is not None:
        payload["metadata"] = json.dumps(metadata)
    if attributes is not None:
        payload["attributes"] = attributes

    return jwt.encode(payload, LIVEKIT_API_SECRET, algorithm="HS256")


def get_livekit_url() -> str:
    return LIVEKIT_URL
