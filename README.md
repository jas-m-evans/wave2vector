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

---

## Taste Bridge Chat

Taste Bridge is a realtime shared listening experience built on top of the audio similarity engine.
Two users join a room, the system computes a **bridge recommendation** — the track that sits best
between both of their taste profiles — and they can steer it together through chat.

### Quick start

1. Open the home page and enter your name under **Taste Bridge Chat**.
2. Click **Start a room** — you'll land on a room page with a shareable link.
3. Send the link to a friend. Once they join, the taste bridge is computed automatically.
4. Use the chat panel or action buttons to steer the recommendation.

No Spotify credentials are required — demo mode generates deterministic taste profiles from
each user's display name. Connect Spotify (see below) for real recommendations.

### Chat commands

| Command | Effect |
|---|---|
| `/balanced` | Equal weight to both users' taste |
| `/more-me` | Lean the recommendation toward your taste |
| `/more-them` | Lean the recommendation toward your friend's taste |
| `/weirder` | More adventurous, less mainstream |
| `/safer` | More familiar, crowd-pleasing picks |
| `/moodier` | Darker, more emotional |
| `/more-energy` | Higher energy tracks |
| `/next` | Different track (diversity penalty applied) |
| `/why` | Explain the current recommendation |

You can also type natural language nudges:
- *"make it darker"*
- *"something both of us would actually save"*
- *"give us a left turn"*
- *"less electronic, more acoustic"*

### System console

Click **System Console** at the bottom of the room page to expand a real-time log of all
LiveKit events, API calls, and scoring results — useful for debugging or demos.

### Architecture

```
app/
├── spotify.py          Spotify OAuth2 PKCE + audio feature API (demo fallback included)
├── livekit_utils.py    LiveKit JWT token generation
├── recommender.py      Deterministic bridge recommendation engine + knob system
├── commentary.py       Rule-based commentary (shared vibe, contrast, confidence)
├── models.py           TasteBridgeRoom + ChatMessage DB tables
├── schemas.py          Pydantic request/response schemas
├── main.py             API routes: /api/rooms/*, /room/{id}
├── templates/
│   ├── room.html       Taste Bridge Chat UI
│   └── index.html      Home page with room entry
└── static/styles.css   Extended component styles
```

### Recommendation engine

The bridge score for each candidate track is:

```
score = 0.35 × sim_target      # matches knob-influenced target profile
      + 0.25 × bridge_quality  # fits both users on average
      + 0.20 × fairness        # equal fit for both users
      + 0.20 × novelty_term    # scaled by noveltyBias knob
```

All ranking is deterministic: identical inputs and seed always produce the same result.

### Connecting Spotify

Set the following environment variables before starting the server:

```bash
SPOTIFY_CLIENT_ID=your_client_id
SPOTIFY_CLIENT_SECRET=your_client_secret
SPOTIFY_REDIRECT_URI=http://localhost:8000/auth/spotify/callback
```

Register `http://localhost:8000/auth/spotify/callback` as a redirect URI in your
[Spotify Developer Dashboard](https://developer.spotify.com/dashboard).

### Connecting LiveKit

```bash
LIVEKIT_API_KEY=your_api_key
LIVEKIT_API_SECRET=your_api_secret
LIVEKIT_URL=wss://your-livekit-host.livekit.cloud
```

Without these variables the app runs with a local dev key (useful for demos without
a LiveKit Cloud instance).

