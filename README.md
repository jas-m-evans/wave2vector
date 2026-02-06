# Wave2Vector Lab

Wave2Vector Lab is a minimal research-friendly prototype for uploading short audio clips, extracting MFCC-based embeddings, and exploring similarity via cosine distance.

## Why this is useful (academic framing)

This project provides a compact, reproducible pipeline for turning raw audio into feature vectors and visualizations. It can be used to:

- Demonstrate signal processing concepts (waveforms, STFT spectrograms, MFCCs).
- Prototype retrieval experiments (nearest neighbor search in embedding space).
- Serve as a baseline for comparing handcrafted vs. learned audio embeddings.

## Quick start (step-by-step)

1. **Clone and enter the repo**

   ```bash
   git clone <repo-url>
   cd wave2vector
   ```

2. **Install dependencies**

   ```bash
   # Optional: create and use a virtual environment
   make install VENV=.venv
   source .venv/bin/activate

   # Or install into the current environment
   make install
   ```

3. **Run the app**

   ```bash
   make run
   ```

4. **Open the UI**

   Visit `http://127.0.0.1:8000` to upload a clip and explore similarity results.

## Using the web UI

1. Open the homepage and upload a `.wav`, `.mp3`, or `.m4a` file.
2. Optionally add a title for the clip.
3. Submit the form to compute MFCC features and save waveform/spectrogram images.
4. Open the clip detail page to see metadata, plots, and nearest neighbors.

## API examples (curl)

Upload a file:

```bash
curl -L -X POST http://127.0.0.1:8000/upload \
  -F "file=@/path/to/clip.wav" \
  -F "title=My demo clip"
```

List clips:

```bash
curl http://127.0.0.1:8000/clips
```

Get neighbors for a clip:

```bash
curl "http://127.0.0.1:8000/clips/1/neighbors?k=5"
```

## Audio decoding notes

`librosa` can rely on external decoders for certain formats. MP3/M4A support depends on which backend libraries are available in your environment, and many setups require `ffmpeg` for reliable decoding.

### Install ffmpeg

- **macOS (Homebrew):** `brew install ffmpeg`
- **Ubuntu/Debian:** `sudo apt-get update && sudo apt-get install -y ffmpeg`
- **Windows (Chocolatey):** `choco install ffmpeg`
- **Windows (winget):** `winget install --id=Gyan.FFmpeg`

## Possible next steps

- Add a learned embedding model (e.g., VGGish, OpenL3, or a small CNN) for comparison.
- Store multiple feature types and allow users to switch similarity metrics.
- Add evaluation utilities (precision@k, confusion matrices) for labeled datasets.
- Support batch uploads and dataset-level analytics.

## Troubleshooting

### File uploads but decode fails

If a file uploads successfully but you receive a decode or analysis error:

1. Install `ffmpeg` (see above).
2. Retry the upload, or test with a `.wav` file to confirm the pipeline.
3. Ensure the file extension is one of `.wav`, `.mp3`, or `.m4a`.

### API returns 404 for neighbors

If `/clips/{id}/neighbors` returns 404, ensure the clip ID exists by calling `/clips` and using one of the returned IDs.

### Images not rendering

Waveform and spectrogram images are saved under `./data/images`. If they are missing, confirm the server has permission to write to the `data/` directory and restart the app to recreate it.
