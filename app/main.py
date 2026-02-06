from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlmodel import select
from starlette.requests import Request

from app.audio import extract_features, plot_features
from app.db import get_session, init_db
from app.models import ClipMetadata
from app.schemas import ClipCreateResponse, ClipDetail

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
IMAGE_DIR = DATA_DIR / "images"

app = FastAPI()
app.mount("/static", StaticFiles(directory=BASE_DIR / "app" / "static"), name="static")

templates = Jinja2Templates(directory=BASE_DIR / "app" / "templates")


def ensure_directories() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)


@app.on_event("startup")
def on_startup() -> None:
    ensure_directories()
    init_db()


def session_dep():
    with get_session() as session:
        yield session


SessionDep = Annotated[object, Depends(session_dep)]


@app.get("/", response_class=HTMLResponse)
async def index(request: Request, session: SessionDep):
    clips = session.exec(select(ClipMetadata)).all()
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "clips": clips},
    )


@app.get("/clips/{clip_id}", response_class=HTMLResponse)
async def clip_detail(request: Request, clip_id: int, session: SessionDep):
    clip = session.get(ClipMetadata, clip_id)
    if not clip:
        return templates.TemplateResponse(
            "clip_detail.html",
            {"request": request, "clip": None},
            status_code=404,
        )
    return templates.TemplateResponse(
        "clip_detail.html",
        {"request": request, "clip": clip},
    )


@app.post("/upload", response_model=ClipCreateResponse)
async def upload_clip(
    session: SessionDep,
    file: UploadFile = File(...),
    title: str = Form(""),
):
    ensure_directories()
    file_path = UPLOAD_DIR / file.filename
    with file_path.open("wb") as buffer:
        buffer.write(await file.read())

    vector = extract_features(file_path)
    image_path = IMAGE_DIR / f"{file_path.stem}.png"
    plot_features(vector, image_path)

    clip = ClipMetadata(
        filename=title or file.filename,
        image_path=f"/data/images/{image_path.name}",
        feature_vector=vector.tolist(),
    )
    session.add(clip)
    session.commit()
    session.refresh(clip)
    return ClipCreateResponse(
        id=clip.id,
        filename=clip.filename,
        image_path=clip.image_path,
    )


@app.get("/api/clips/{clip_id}", response_model=ClipDetail)
async def clip_detail_api(clip_id: int, session: SessionDep):
    clip = session.get(ClipMetadata, clip_id)
    if not clip:
        return ClipDetail(id=0, filename="", image_path="", feature_vector=[])
    return ClipDetail(
        id=clip.id,
        filename=clip.filename,
        image_path=clip.image_path,
        feature_vector=clip.feature_vector,
    )


app.mount("/data", StaticFiles(directory=DATA_DIR), name="data")
