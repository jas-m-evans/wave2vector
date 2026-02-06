from pydantic import BaseModel


class ClipCreateResponse(BaseModel):
    id: int
    filename: str
    image_path: str


class ClipDetail(BaseModel):
    id: int
    filename: str
    image_path: str
    feature_vector: list[float]
