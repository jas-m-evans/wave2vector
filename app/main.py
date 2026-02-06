from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated
from uuid import uuid4

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlmodel import select
from starlette.requests import Request

from app.db import get_session, init_db
from app.models import ClipMetadata
from app.schemas import ClipDetail, ClipListItem

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
IMAGE_DIR = DATA_DIR / "images"
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a"}

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
    clips = session.exec(select(ClipMetadata).order_by(ClipMetadata.created_at.desc())).all()
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "clips": clips},
    )


@app.get("/clips", response_model=list[ClipListItem])
async def clip_list(session: SessionDep):
    clips = session.exec(select(ClipMetadata).order_by(ClipMetadata.created_at.desc())).all()
    return [
        ClipListItem(
            id=clip.id,
            filename=clip.filename,
            duration=clip.duration,
            created_at=clip.created_at,
        )
        for clip in clips
    ]


@app.get("/clips/{clip_id}", response_class=HTMLResponse)
async def clip_detail(request: Request, clip_id: int, session: SessionDep):
    clip = session.get(ClipMetadata, clip_id)
    if not clip:
        return templates.TemplateResponse(
            "clip_detail.html",
            {"request": request, "clip": None},
            status_code=404,
        )
    vector_preview = clip.vector[:10] if clip.vector else []
    return templates.TemplateResponse(
        "clip_detail.html",
        {"request": request, "clip": clip, "vector_preview": vector_preview},
    )


@app.post("/upload")
async def upload_clip(
    session: SessionDep,
    file: UploadFile = File(...),
    title: str = Form(""),
):
    ensure_directories()
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")
    extension = Path(file.filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported file format.")

    clip_uuid = uuid4().hex
    file_path = UPLOAD_DIR / f"{clip_uuid}{extension}"
    with file_path.open("wb") as buffer:
        buffer.write(await file.read())

    try:
        audio, sr = librosa.load(path=str(file_path), sr=None, mono=True)
    except Exception as exc:  # pragma: no cover - defensive against decoder errors
        file_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail="Unable to decode audio file.",
        ) from exc

    duration_seconds = float(len(audio) / sr) if sr else 0.0
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)
    vector = np.concatenate([mfcc_mean, mfcc_std]).astype(float)

    waveform_path = IMAGE_DIR / f"{clip_uuid}_waveform.png"
    plt.figure(figsize=(8, 3))
    librosa.display.waveshow(audio, sr=sr)
    plt.title("Waveform")
    plt.tight_layout()
    plt.savefig(waveform_path)
    plt.close()

    spectrogram_path = IMAGE_DIR / f"{clip_uuid}_spectrogram.png"
    stft = librosa.stft(audio)
    spectrogram = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    plt.figure(figsize=(8, 3))
    librosa.display.specshow(spectrogram, sr=sr, x_axis="time", y_axis="hz")
    plt.title("Spectrogram")
    plt.colorbar(format="%+2.0f dB")
    plt.tight_layout()
    plt.savefig(spectrogram_path)
    plt.close()

    clip = ClipMetadata(
        filename=title or file.filename,
        path=f"/data/uploads/{file_path.name}",
        sr=sr,
        duration=duration_seconds,
        created_at=datetime.now(timezone.utc),
        vector=vector.tolist(),
        waveform_image_path=f"/data/images/{waveform_path.name}",
        spectrogram_image_path=f"/data/images/{spectrogram_path.name}",
    )
    session.add(clip)
    session.commit()
    session.refresh(clip)
    return RedirectResponse(url=f"/clips/{clip.id}", status_code=303)


@app.get("/api/clips/{clip_id}", response_model=ClipDetail)
async def clip_detail_api(clip_id: int, session: SessionDep):
    clip = session.get(ClipMetadata, clip_id)
    if not clip:
        return ClipDetail(
            id=0,
            filename="",
            path="",
            sr=0,
            duration=0.0,
            created_at=datetime.now(timezone.utc),
            vector=[],
            waveform_image_path="",
            spectrogram_image_path="",
        )
    return ClipDetail(
        id=clip.id,
        filename=clip.filename,
        path=clip.path,
        sr=clip.sr,
        duration=clip.duration,
        created_at=clip.created_at,
        vector=clip.vector,
        waveform_image_path=clip.waveform_image_path,
        spectrogram_image_path=clip.spectrogram_image_path,
    )


app.mount("/data", StaticFiles(directory=DATA_DIR), name="data")
