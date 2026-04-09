from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated
from uuid import uuid4

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlmodel import select
from starlette.requests import Request

from app.db import get_session, init_db
from app.models import ChatMessage, ClipMetadata, TasteBridgeRoom
from app.schemas import (
    ChatMessageSchema,
    ClipDetail,
    ClipListItem,
    ClipNeighbor,
    ClipNeighborsResponse,
    CommentarySchema,
    KnobsSchema,
    KnobsUpdateRequest,
    PostChatRequest,
    ProfileSetRequest,
    RecommendRequest,
    RecommendResponse,
    RoomCreateRequest,
    RoomJoinRequest,
    RoomStateSchema,
    RoomTokenResponse,
)

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


def cosine_distance(vector_a: list[float], vector_b: list[float]) -> float:
    array_a = np.array(vector_a, dtype=float)
    array_b = np.array(vector_b, dtype=float)
    if array_a.size == 0 or array_b.size == 0:
        return 1.0
    norm_product = np.linalg.norm(array_a) * np.linalg.norm(array_b)
    if norm_product == 0:
        return 1.0
    try:
        similarity = float(np.dot(array_a, array_b) / norm_product)
    except ValueError:
        return 1.0
    return float(1 - similarity)


def sample_rate_quality(sr: int) -> dict[str, str]:
    if sr >= 44100:
        return {"label": "High quality", "note": "Music-grade detail (44.1 kHz or above)."}
    if sr >= 22050:
        return {"label": "Usable", "note": "Good for analysis and casual playback."}
    if sr >= 16000:
        return {"label": "Speech-first", "note": "Typically acceptable for voice."}
    return {"label": "Low fidelity", "note": "May lose detail and affect similarity quality."}


def feature_name(index: int) -> str:
    if index < 20:
        return f"MFCC mean C{index + 1}"
    return f"MFCC std C{index - 19}"


def vector_summary(vector: list[float]) -> dict[str, object]:
    if not vector:
        return {
            "norm": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "top_components": [],
        }

    arr = np.array(vector, dtype=float)
    top_indices = np.argsort(np.abs(arr))[::-1][:5]
    max_abs = float(np.max(np.abs(arr))) if arr.size else 1.0
    scale = max(max_abs, 1.0)
    top_components = [
        {
            "index": int(idx),
            "name": feature_name(int(idx)),
            "value": float(arr[idx]),
            "magnitude_pct": (abs(float(arr[idx])) / scale) * 100,
        }
        for idx in top_indices
    ]

    return {
        "norm": float(np.linalg.norm(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "top_components": top_components,
    }


def analyze_audio(audio: np.ndarray, sr: int) -> tuple[float, np.ndarray]:
    duration_seconds = float(len(audio) / sr) if sr else 0.0
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)
    vector = np.concatenate([mfcc_mean, mfcc_std]).astype(float)
    return duration_seconds, vector


def save_plots(audio: np.ndarray, sr: int, clip_uuid: str) -> tuple[Path, Path]:
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
    return waveform_path, spectrogram_path


def create_clip_record(
    session: SessionDep,
    filename: str,
    file_path: Path,
    sr: int,
    duration_seconds: float,
    vector: np.ndarray,
    waveform_path: Path,
    spectrogram_path: Path,
) -> ClipMetadata:
    clip = ClipMetadata(
        filename=filename,
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
    return clip


@app.get("/", response_class=HTMLResponse)
async def index(request: Request, session: SessionDep):
    clips = session.exec(select(ClipMetadata).order_by(ClipMetadata.created_at.desc())).all()
    clip_rows = [
        {
            "clip": clip,
            "sample_rate": sample_rate_quality(clip.sr),
        }
        for clip in clips
    ]
    seed_status = request.query_params.get("seed")
    return templates.TemplateResponse(
        request,
        "index.html",
        {"clips": clips, "clip_rows": clip_rows, "seed_status": seed_status},
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
            request,
            "clip_detail.html",
            {"clip": None},
            status_code=404,
        )
    vector_preview = clip.vector[:10] if clip.vector else []
    summary = vector_summary(clip.vector or [])
    sample_rate = sample_rate_quality(clip.sr)
    return templates.TemplateResponse(
        request,
        "clip_detail.html",
        {
            "clip": clip,
            "vector_preview": vector_preview,
            "vector_summary": summary,
            "sample_rate": sample_rate,
        },
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
        supported = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension '{extension}'. Supported formats: {supported}.",
        )

    clip_uuid = uuid4().hex
    file_path = UPLOAD_DIR / f"{clip_uuid}{extension}"
    with file_path.open("wb") as buffer:
        buffer.write(await file.read())

    try:
        audio, sr = librosa.load(path=str(file_path), sr=None, mono=True)
    except Exception as exc:  # pragma: no cover - defensive against decoder errors
        file_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=422,
            detail=(
                "Unable to decode audio file. If you uploaded mp3/m4a, install ffmpeg "
                "and try again."
            ),
        ) from exc

    try:
        duration_seconds, vector = analyze_audio(audio, sr)
    except Exception as exc:  # pragma: no cover - defensive against analysis errors
        file_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500,
            detail=(
                "Audio decoding succeeded but feature analysis failed. Ensure ffmpeg is "
                "installed for mp3/m4a and try a WAV file to validate the pipeline."
            ),
        ) from exc

    waveform_path, spectrogram_path = save_plots(audio, sr, clip_uuid)
    clip = create_clip_record(
        session=session,
        filename=title or file.filename,
        file_path=file_path,
        sr=sr,
        duration_seconds=duration_seconds,
        vector=vector,
        waveform_path=waveform_path,
        spectrogram_path=spectrogram_path,
    )
    return RedirectResponse(url=f"/clips/{clip.id}", status_code=303)


@app.post("/seed-demo")
async def seed_demo(session: SessionDep):
    ensure_directories()
    demo_specs = [
        {"title": "Demo: mellow pad", "sr": 44100, "freqs": [220.0, 330.0, 440.0]},
        {"title": "Demo: bright lead", "sr": 22050, "freqs": [440.0, 660.0, 880.0]},
        {"title": "Demo: lo-fi pulse", "sr": 12000, "freqs": [130.81, 196.0, 261.63]},
    ]

    existing_names = {
        value[0] if isinstance(value, tuple) else value
        for value in session.exec(select(ClipMetadata.filename)).all()
    }
    created = 0

    for spec in demo_specs:
        if spec["title"] in existing_names:
            continue
        sr = int(spec["sr"])
        duration = 20.0
        t = np.linspace(0.0, duration, int(sr * duration), endpoint=False)
        waveform = np.zeros_like(t)
        for idx, freq in enumerate(spec["freqs"]):
            waveform += (0.22 / (idx + 1)) * np.sin(2 * np.pi * freq * t)
        envelope = np.linspace(0.4, 1.0, waveform.shape[0])
        waveform = np.clip(waveform * envelope, -1.0, 1.0).astype(np.float32)

        clip_uuid = uuid4().hex
        file_path = UPLOAD_DIR / f"{clip_uuid}.wav"
        sf.write(file_path, waveform, sr)

        duration_seconds, vector = analyze_audio(waveform, sr)
        waveform_path, spectrogram_path = save_plots(waveform, sr, clip_uuid)
        create_clip_record(
            session=session,
            filename=str(spec["title"]),
            file_path=file_path,
            sr=sr,
            duration_seconds=duration_seconds,
            vector=vector,
            waveform_path=waveform_path,
            spectrogram_path=spectrogram_path,
        )
        created += 1

    if created == 0:
        return RedirectResponse(url="/?seed=exists", status_code=303)
    return RedirectResponse(url="/?seed=created", status_code=303)


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


@app.get("/clips/{clip_id}/neighbors", response_model=ClipNeighborsResponse)
async def clip_neighbors(
    clip_id: int,
    session: SessionDep,
    k: int = Query(5, ge=1),
):
    target = session.get(ClipMetadata, clip_id)
    if not target:
        raise HTTPException(status_code=404, detail="Clip not found.")
    candidates = session.exec(
        select(ClipMetadata).where(ClipMetadata.id != clip_id),
    ).all()
    best_by_name: dict[str, ClipNeighbor] = {}
    for clip in candidates:
        if not clip.vector:
            continue
        distance = cosine_distance(target.vector or [], clip.vector)
        candidate = ClipNeighbor(
            id=clip.id,
            filename=clip.filename,
            path=clip.path,
            distance=distance,
        )
        existing = best_by_name.get(clip.filename)
        if not existing or candidate.distance < existing.distance:
            best_by_name[clip.filename] = candidate

    neighbors_sorted = sorted(best_by_name.values(), key=lambda item: item.distance)[:k]
    return ClipNeighborsResponse(clip_id=clip_id, k=k, neighbors=neighbors_sorted)


# ---------------------------------------------------------------------------
# Taste Bridge — room management
# ---------------------------------------------------------------------------

from app import commentary as commentary_mod  # noqa: E402
from app import recommender as rec_mod  # noqa: E402
from app import spotify as spotify_mod  # noqa: E402
from app.livekit_utils import create_token, get_livekit_url  # noqa: E402


def _room_state_from_db(room: TasteBridgeRoom) -> RoomStateSchema:
    """Reconstruct a RoomStateSchema from the DB snapshot."""
    s = room.state or {}
    knobs_data = s.get("knobs", {})
    bridge_data = s.get("bridge_reco")
    commentary_data = s.get("commentary")

    from app.schemas import BridgeRecoSchema  # local import avoids circular

    return RoomStateSchema(
        room_id=room.room_id,
        host_identity=room.host_identity,
        host_display_name=room.host_display_name,
        guest_identity=room.guest_identity,
        guest_display_name=room.guest_display_name,
        participants=s.get("participants", []),
        spotify_connected=s.get("spotify_connected", False),
        taste_profile_ready=s.get("taste_profile_ready", False),
        knobs=KnobsSchema(**knobs_data) if knobs_data else KnobsSchema(),
        compatibility=s.get("compatibility", 0.0),
        bridge_reco=BridgeRecoSchema(**bridge_data) if bridge_data else None,
        commentary=CommentarySchema(**commentary_data) if commentary_data else None,
        previous_track_id=s.get("previous_track_id"),
    )


def _save_room_state(session, room: TasteBridgeRoom, patch: dict) -> TasteBridgeRoom:
    """Merge patch into room.state and persist."""
    state = dict(room.state or {})
    state.update(patch)
    room.state = state
    room.updated_at = datetime.now(timezone.utc)
    session.add(room)
    session.commit()
    session.refresh(room)
    return room


def _build_recommend_and_commentary(
    room: TasteBridgeRoom,
    knobs: rec_mod.Knobs,
    command: str | None = None,
) -> tuple[rec_mod.BridgeResult | None, commentary_mod.CommentaryResult | None]:
    """Run the recommendation + commentary pipeline, returns (result, commentary)."""
    state = room.state or {}
    profile_a = state.get("profile_a", {})
    profile_b = state.get("profile_b", {})

    if not profile_a or not profile_b:
        # Fall back to demo profiles derived from display names
        profile_a = spotify_mod.demo_profile_from_name(room.host_display_name)
        if room.guest_display_name:
            profile_b = spotify_mod.demo_profile_from_name(room.guest_display_name)
        else:
            profile_b = {}

    if not profile_b:
        return None, None

    # Use MOCK_CANDIDATES for now; real Spotify candidates slot in here when
    # access tokens are present.
    candidates = spotify_mod.MOCK_CANDIDATES

    previous_id = state.get("previous_track_id")
    result = rec_mod.recommend_bridge(
        profile_a=profile_a,
        profile_b=profile_b,
        candidates=candidates,
        knobs=knobs,
        previous_id=previous_id,
    )
    if result is None:
        return None, None

    commentary = commentary_mod.generate_commentary(
        profile_a=profile_a,
        profile_b=profile_b,
        name_a=room.host_display_name,
        name_b=room.guest_display_name or "Guest",
        result=result,
    )
    return result, commentary


@app.post("/api/rooms", response_model=RoomTokenResponse)
async def create_room(body: RoomCreateRequest, session: SessionDep):
    """Create a new Taste Bridge room and return a LiveKit token for the host."""
    room_id = uuid4().hex[:12]
    identity = f"host-{uuid4().hex[:8]}"
    now = datetime.now(timezone.utc)

    # Seed demo profiles immediately so the room is interesting from the start.
    host_profile = spotify_mod.demo_profile_from_name(body.display_name)

    room = TasteBridgeRoom(
        room_id=room_id,
        host_identity=identity,
        host_display_name=body.display_name,
        created_at=now,
        updated_at=now,
        state={
            "profile_a": host_profile,
            "knobs": KnobsSchema().model_dump(),
            "compatibility": 0.0,
            "taste_profile_ready": False,
            "spotify_connected": False,
        },
    )
    session.add(room)
    session.commit()

    token = create_token(
        room_name=room_id,
        identity=identity,
        display_name=body.display_name,
        metadata={"role": "host", "room_id": room_id},
    )

    return RoomTokenResponse(
        room_id=room_id,
        livekit_token=token,
        livekit_url=get_livekit_url(),
        restored=False,
        state=_room_state_from_db(room),
    )


@app.post("/api/rooms/{room_id}/join", response_model=RoomTokenResponse)
async def join_room(room_id: str, body: RoomJoinRequest, session: SessionDep):
    """Join an existing room as guest; restores previous state if available."""
    room = session.get(TasteBridgeRoom, room_id)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found.")

    identity = f"guest-{uuid4().hex[:8]}"
    guest_profile = spotify_mod.demo_profile_from_name(body.display_name)

    # Calculate compatibility now that we have both profiles.
    profile_a = (room.state or {}).get("profile_a", {})
    compat = spotify_mod.compute_compatibility(profile_a, guest_profile) if profile_a else 0.0

    room.guest_identity = identity
    room.guest_display_name = body.display_name
    patch = {
        "profile_b": guest_profile,
        "compatibility": round(compat, 3),
        "taste_profile_ready": bool(profile_a),
    }
    room = _save_room_state(session, room, patch)

    was_restored = bool((room.state or {}).get("bridge_reco"))

    token = create_token(
        room_name=room_id,
        identity=identity,
        display_name=body.display_name,
        metadata={"role": "guest", "room_id": room_id},
    )

    return RoomTokenResponse(
        room_id=room_id,
        livekit_token=token,
        livekit_url=get_livekit_url(),
        restored=was_restored,
        state=_room_state_from_db(room),
    )


@app.get("/api/rooms/{room_id}", response_model=RoomStateSchema)
async def get_room_state(room_id: str, session: SessionDep):
    room = session.get(TasteBridgeRoom, room_id)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found.")
    return _room_state_from_db(room)


@app.put("/api/rooms/{room_id}/knobs")
async def update_knobs(room_id: str, body: KnobsUpdateRequest, session: SessionDep):
    """Update preference knobs and persist."""
    room = session.get(TasteBridgeRoom, room_id)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found.")
    _save_room_state(session, room, {"knobs": body.knobs.model_dump()})
    return {"ok": True}


@app.post("/api/rooms/{room_id}/recommend", response_model=RecommendResponse)
async def get_recommendation(room_id: str, body: RecommendRequest, session: SessionDep):
    """Recompute the bridge recommendation.  Handles slash commands + free text."""
    room = session.get(TasteBridgeRoom, room_id)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found.")

    state = room.state or {}
    knobs = rec_mod.Knobs.from_dict(state.get("knobs", {}))

    # Apply slash command knob adjustments.
    if body.command and body.command != "/why":
        knobs = rec_mod.apply_slash_command(knobs, body.command)

    # Apply natural language nudge.
    if body.free_text:
        nudge = rec_mod.parse_natural_language_nudge(body.free_text)
        if nudge:
            updated = knobs.to_dict()
            updated.update(nudge)
            knobs = rec_mod.Knobs.from_dict(updated)

    result, commentary = _build_recommend_and_commentary(room, knobs, body.command)
    if result is None:
        raise HTTPException(status_code=400, detail="Need at least two profiles to recommend.")

    from app.schemas import BridgeRecoSchema  # local import

    bridge_schema = BridgeRecoSchema(
        track_id=result.track_id,
        track_name=result.track_name,
        artist=result.artist,
        album=result.album,
        album_art=result.album_art,
        preview_url=result.preview_url,
        scores={k: v for k, v in result.scores.__dict__.items()},
        target_profile=result.target_profile,
        knobs_applied=result.knobs_applied,
    )
    commentary_schema = CommentarySchema(
        shared_vibe=commentary.shared_vibe,
        biggest_contrast=commentary.biggest_contrast,
        bridge_reason=commentary.bridge_reason,
        confidence=commentary.confidence,
        confidence_label=commentary.confidence_label,
        raw_scores=commentary.raw_scores,
    )

    # Persist updated state.
    _save_room_state(
        session,
        room,
        {
            "knobs": knobs.to_dict(),
            "bridge_reco": bridge_schema.model_dump(),
            "commentary": commentary_schema.model_dump(),
            "previous_track_id": result.track_id,
        },
    )

    # Build a friendly system message for the chat.
    cmd = body.command or ""
    if cmd == "/why":
        system_message = commentary.bridge_reason
    elif cmd == "/next":
        system_message = f"Shuffled! How about **{result.track_name}** by {result.artist}?"
    elif body.free_text:
        system_message = (
            f'Got it — nudging toward "{body.free_text}". '
            f"Trying **{result.track_name}** by {result.artist}."
        )
    else:
        system_message = (
            f"Bridge updated: **{result.track_name}** by {result.artist} "
            f"({commentary.confidence_label} match, {round(commentary.confidence * 100)}%)"
        )

    return RecommendResponse(
        bridge_reco=bridge_schema,
        commentary=commentary_schema,
        knobs=KnobsSchema(**knobs.to_dict()),
        system_message=system_message,
    )


@app.post("/api/rooms/{room_id}/profile")
async def set_profile(room_id: str, body: ProfileSetRequest, session: SessionDep):
    """Store a participant's taste profile (from Spotify or demo mode)."""
    room = session.get(TasteBridgeRoom, room_id)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found.")

    state = room.state or {}
    is_host = not room.guest_identity

    patch_key = "profile_a" if is_host else "profile_b"
    patch = {patch_key: body.taste_profile}

    # Recompute compatibility if we now have both profiles.
    if "profile_a" in state and "profile_b" in state:
        compat = spotify_mod.compute_compatibility(state["profile_a"], state["profile_b"])
        patch["compatibility"] = round(compat, 3)
        patch["taste_profile_ready"] = True

    _save_room_state(session, room, patch)
    return {"ok": True}


@app.post("/api/rooms/{room_id}/chat", response_model=ChatMessageSchema)
async def post_chat(room_id: str, body: PostChatRequest, session: SessionDep):
    """Persist a chat message (user or system)."""
    room = session.get(TasteBridgeRoom, room_id)
    if not room:
        raise HTTPException(status_code=404, detail="Room not found.")

    msg = ChatMessage(
        room_id=room_id,
        sender_identity=body.sender_identity,
        display_name=body.display_name,
        content=body.content,
        timestamp=datetime.now(timezone.utc),
        is_system=body.is_system,
    )
    session.add(msg)
    session.commit()
    session.refresh(msg)

    return ChatMessageSchema(
        id=msg.id,
        sender_identity=msg.sender_identity,
        display_name=msg.display_name,
        content=msg.content,
        timestamp=msg.timestamp,
        is_system=msg.is_system,
    )


@app.get("/api/rooms/{room_id}/chat", response_model=list[ChatMessageSchema])
async def get_chat(room_id: str, session: SessionDep, limit: int = Query(50, ge=1, le=200)):
    """Return recent chat history for a room."""
    msgs = session.exec(
        select(ChatMessage)
        .where(ChatMessage.room_id == room_id)
        .order_by(ChatMessage.timestamp.desc())
        .limit(limit)
    ).all()
    return [
        ChatMessageSchema(
            id=m.id,
            sender_identity=m.sender_identity,
            display_name=m.display_name,
            content=m.content,
            timestamp=m.timestamp,
            is_system=m.is_system,
        )
        for m in reversed(msgs)
    ]


# ---------------------------------------------------------------------------
# Room UI page
# ---------------------------------------------------------------------------


@app.get("/room/{room_id}", response_class=HTMLResponse)
async def room_page(request: Request, room_id: str, session: SessionDep):
    room = session.get(TasteBridgeRoom, room_id)
    if not room:
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "clips": [],
                "clip_rows": [],
                "seed_status": None,
                "error": f"Room '{room_id}' not found.",
            },
            status_code=404,
        )
    is_demo = spotify_mod.is_demo_mode()
    return templates.TemplateResponse(
        request,
        "room.html",
        {
            "room_id": room_id,
            "host_display_name": room.host_display_name,
            "livekit_url": get_livekit_url(),
            "is_demo": is_demo,
        },
    )


app.mount("/data", StaticFiles(directory=DATA_DIR, check_dir=False), name="data")
