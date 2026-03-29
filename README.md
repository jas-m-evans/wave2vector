# Wave2Vector Lab

Wave2Vector Lab is a research-friendly web app for uploading audio clips, extracting MFCC-based embeddings, and exploring nearest-neighbor similarity with an interactive UI.

## What it does

- Upload `.wav`, `.mp3`, or `.m4a` clips.
- Compute MFCC embeddings and save waveform/spectrogram plots.
- Show sample-rate quality labels (High quality, Usable, Speech-first, Low fidelity).
- Play clips directly in the app.
- Explore nearest neighbors with similarity cards, match bars, and inline playback.
- Use a one-click demo seeding flow to populate test clips.
- Visualize playback live on clip pages with a dynamic canvas and real-time metrics:
  - playback clock
  - RMS loudness
  - peak amplitude
  - spectral centroid

## Quick start

1. Clone and enter the repo.

```bash
git clone https://github.com/jas-m-evans/wave2vector.git
cd wave2vector
```

2. Install dependencies.

```bash
# Recommended: virtual environment
make install VENV=.venv
source .venv/bin/activate

# Alternative: current environment
make install
```

3. Run the app.

```bash
make run
```

4. Open the UI at `http://127.0.0.1:8000`.

## Web UI workflow

1. Upload a clip from the homepage.
2. Optionally click **Load demo clips** for a fast multi-clip setup.
3. Open clip details to inspect metadata, waveform, spectrogram, and vector summaries.
4. Press play to view the live visualizer and real-time audio metrics.
5. Use nearest neighbors to compare clips and audition matches.

## API examples

Upload a file:

```bash
curl -L -X POST http://127.0.0.1:8000/upload \
  -F "file=@/path/to/clip.wav" \
  -F "title=My demo clip"
```

Load demo clips:

```bash
curl -L -X POST http://127.0.0.1:8000/seed-demo
```

List clips:

```bash
curl http://127.0.0.1:8000/clips
```

Get neighbors for a clip:

```bash
curl "http://127.0.0.1:8000/clips/1/neighbors?k=5"
```

## Notes on demo seeding

- Demo seeding is idempotent by title.
- Repeated clicks do not keep adding duplicate demo names.
- Neighbor responses are deduplicated by filename and keep the closest match.

## Audio decoding notes

`librosa` may depend on external decoders for certain formats. MP3/M4A support often requires `ffmpeg`.

Install ffmpeg:

- macOS (Homebrew): `brew install ffmpeg`
- Ubuntu/Debian: `sudo apt-get update && sudo apt-get install -y ffmpeg`
- Windows (Chocolatey): `choco install ffmpeg`
- Windows (winget): `winget install --id=Gyan.FFmpeg`

## Troubleshooting

### Upload succeeds but decode fails

1. Install `ffmpeg`.
2. Retry with a `.wav` clip to isolate decoder issues.
3. Confirm extension is one of `.wav`, `.mp3`, `.m4a`.

### No neighbors found

You need more than one unique clip in the dataset. Upload another clip or use demo seeding.

### Images or audio do not load

Runtime assets are saved under `./data/`. Confirm the process can write to that directory.

## Tech stack

- FastAPI + Jinja templates
- SQLModel + SQLite
- Librosa + NumPy
- Matplotlib
