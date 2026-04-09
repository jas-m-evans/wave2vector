from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel


class ClipCreateResponse(BaseModel):
    id: int
    filename: str
    path: str
    waveform_image_path: str
    spectrogram_image_path: str


class ClipDetail(BaseModel):
    id: int
    filename: str
    path: str
    sr: int
    duration: float
    created_at: datetime
    vector: list[float]
    waveform_image_path: str
    spectrogram_image_path: str


class ClipListItem(BaseModel):
    id: int
    filename: str
    duration: float
    created_at: datetime


class ClipNeighbor(BaseModel):
    id: int
    filename: str
    path: str
    distance: float


class ClipNeighborsResponse(BaseModel):
    clip_id: int
    k: int
    neighbors: list[ClipNeighbor]


# ---------------------------------------------------------------------------
# Taste Bridge schemas
# ---------------------------------------------------------------------------


class KnobsSchema(BaseModel):
    balanceBias: float = 0.5
    noveltyBias: float = 0.3
    energyBias: float = 0.5
    moodBias: float = 0.5


class AudioProfile(BaseModel):
    energy: float = 0.5
    valence: float = 0.5
    danceability: float = 0.5
    acousticness: float = 0.5
    instrumentalness: float = 0.5
    speechiness: float = 0.5
    tempo_norm: float = 0.5


class ParticipantInfo(BaseModel):
    identity: str
    display_name: str
    spotify_connected: bool = False
    taste_profile: Optional[AudioProfile] = None


class BridgeRecoSchema(BaseModel):
    track_id: str
    track_name: str
    artist: str
    album: str = ""
    album_art: Optional[str] = None
    preview_url: Optional[str] = None
    scores: dict[str, float]
    target_profile: dict[str, float]
    knobs_applied: dict[str, float]


class CommentarySchema(BaseModel):
    shared_vibe: str
    biggest_contrast: str
    bridge_reason: str
    confidence: float
    confidence_label: str
    raw_scores: dict[str, float]


class RoomStateSchema(BaseModel):
    room_id: str
    host_identity: str
    host_display_name: str
    guest_identity: Optional[str] = None
    guest_display_name: Optional[str] = None
    participants: list[ParticipantInfo] = []
    spotify_connected: bool = False
    taste_profile_ready: bool = False
    knobs: KnobsSchema = KnobsSchema()
    compatibility: float = 0.0
    bridge_reco: Optional[BridgeRecoSchema] = None
    commentary: Optional[CommentarySchema] = None
    previous_track_id: Optional[str] = None


class RoomCreateRequest(BaseModel):
    display_name: str


class RoomJoinRequest(BaseModel):
    display_name: str


class RoomTokenResponse(BaseModel):
    room_id: str
    livekit_token: str
    livekit_url: str
    restored: bool = False
    state: Optional[RoomStateSchema] = None


class KnobsUpdateRequest(BaseModel):
    knobs: KnobsSchema


class RecommendRequest(BaseModel):
    command: Optional[str] = None  # slash command e.g. "/balanced"
    free_text: Optional[str] = None  # natural language nudge


class ChatMessageSchema(BaseModel):
    id: int
    sender_identity: str
    display_name: str
    content: str
    timestamp: datetime
    is_system: bool


class PostChatRequest(BaseModel):
    content: str
    is_system: bool = False
    sender_identity: str = "system"
    display_name: str = "System"


class RecommendResponse(BaseModel):
    bridge_reco: BridgeRecoSchema
    commentary: CommentarySchema
    knobs: KnobsSchema
    system_message: str


class ProfileSetRequest(BaseModel):
    taste_profile: dict[str, Any]
    display_name: str
