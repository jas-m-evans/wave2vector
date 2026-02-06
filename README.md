# wave2vector

Wave2vector lets you upload short audio clips, extract MFCC-based embeddings, and explore similar clips.

## Audio decoding notes

`librosa` can rely on external decoders for certain formats. MP3/M4A support depends on which backend libraries are available in your environment, and many setups require `ffmpeg` for reliable decoding.

### Install ffmpeg

- **macOS (Homebrew):** `brew install ffmpeg`
- **Ubuntu/Debian:** `sudo apt-get update && sudo apt-get install -y ffmpeg`
- **Windows (Chocolatey):** `choco install ffmpeg`
- **Windows (winget):** `winget install --id=Gyan.FFmpeg`

## Troubleshooting

### File uploads but decode fails

If a file uploads successfully but you receive a decode or analysis error:

1. Install `ffmpeg` (see above).
2. Retry the upload, or test with a `.wav` file to confirm the pipeline.
3. Ensure the file extension is one of `.wav`, `.mp3`, or `.m4a`.
