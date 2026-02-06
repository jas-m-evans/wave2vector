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
