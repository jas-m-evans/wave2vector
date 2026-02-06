from datetime import datetime, timezone

from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from app.main import app, cosine_distance, session_dep
from app.models import ClipMetadata


def test_neighbors_ordering_and_self_exclusion():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)

    def override_session():
        with Session(engine) as session:
            yield session

    app.dependency_overrides[session_dep] = override_session
    client = TestClient(app)

    with Session(engine) as session:
        target = ClipMetadata(
            filename="target.wav",
            path="/data/uploads/target.wav",
            sr=22050,
            duration=1.0,
            created_at=datetime.now(timezone.utc),
            vector=[1.0, 0.0],
            waveform_image_path="/data/images/target_wave.png",
            spectrogram_image_path="/data/images/target_spec.png",
        )
        neighbor_close = ClipMetadata(
            filename="neighbor_close.wav",
            path="/data/uploads/neighbor_close.wav",
            sr=22050,
            duration=1.0,
            created_at=datetime.now(timezone.utc),
            vector=[0.9, 0.1],
            waveform_image_path="/data/images/neighbor_close_wave.png",
            spectrogram_image_path="/data/images/neighbor_close_spec.png",
        )
        neighbor_far = ClipMetadata(
            filename="neighbor_far.wav",
            path="/data/uploads/neighbor_far.wav",
            sr=22050,
            duration=1.0,
            created_at=datetime.now(timezone.utc),
            vector=[0.0, 1.0],
            waveform_image_path="/data/images/neighbor_far_wave.png",
            spectrogram_image_path="/data/images/neighbor_far_spec.png",
        )
        session.add(target)
        session.add(neighbor_close)
        session.add(neighbor_far)
        session.commit()
        session.refresh(target)
        session.refresh(neighbor_close)
        session.refresh(neighbor_far)

    response = client.get(f"/clips/{target.id}/neighbors?k=2")
    assert response.status_code == 200

    payload = response.json()
    neighbor_ids = [neighbor["id"] for neighbor in payload["neighbors"]]
    assert target.id not in neighbor_ids

    distances = [neighbor["distance"] for neighbor in payload["neighbors"]]
    assert distances == sorted(distances)

    expected_close_distance = cosine_distance(target.vector, neighbor_close.vector)
    expected_far_distance = cosine_distance(target.vector, neighbor_far.vector)
    assert expected_close_distance < expected_far_distance
    assert neighbor_ids == [neighbor_close.id, neighbor_far.id]

    app.dependency_overrides.clear()
