from datetime import datetime
from typing import Optional

from sqlalchemy import JSON
from sqlmodel import Field, SQLModel


class ClipMetadata(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    filename: str
    path: str
    sr: int
    duration: float
    created_at: datetime
    vector: list[float] = Field(sa_type=JSON)
    waveform_image_path: str
    spectrogram_image_path: str


class TasteBridgeRoom(SQLModel, table=True):
    """Persisted room snapshot for state restoration on rejoin."""

    room_id: str = Field(primary_key=True)
    host_identity: str
    host_display_name: str
    guest_identity: Optional[str] = None
    guest_display_name: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    # Full room state snapshot serialised as JSON
    state: dict = Field(sa_type=JSON, default={})


class ChatMessage(SQLModel, table=True):
    """Persisted chat message — both user and system messages."""

    id: Optional[int] = Field(default=None, primary_key=True)
    room_id: str
    sender_identity: str  # user identity or "system"
    display_name: str
    content: str
    timestamp: datetime
    is_system: bool = False
