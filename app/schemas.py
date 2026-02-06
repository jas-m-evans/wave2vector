from datetime import datetime

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
    distance: float


class ClipNeighborsResponse(BaseModel):
    clip_id: int
    k: int
    neighbors: list[ClipNeighbor]
