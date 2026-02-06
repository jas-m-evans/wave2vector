from typing import Optional

from sqlalchemy import JSON
from sqlmodel import Field, SQLModel


class ClipMetadata(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    filename: str
    image_path: str
    feature_vector: list[float] = Field(sa_type=JSON)
