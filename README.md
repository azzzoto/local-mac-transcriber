# Local Mac Transcriber

Local pipeline to:

1. automatically split a lecture recording into 30-minute chunks;
2. transcribe each chunk with Whisper;
3. generate final Markdown notes with an Ollama model.

This project takes the notebook idea and turns it into a more professional workflow, configurable through `.env` and reusable from the terminal.

## Requirements

- Python 3.12+
- `ffmpeg` installed and available in `PATH`
- Ollama installed and running locally if you want to generate Markdown files

On macOS you can install `ffmpeg` with:

```bash
brew install ffmpeg
```

## Setup

1. Create `.env` from `.env.example`
2. Install dependencies:

```bash
uv sync
```

3. Put audio files in `input/` or configure `AUDIO_FILE` in `.env`

## `.env` configuration

Main options:

- `INPUT_DIR`: directory to read audio files from
- `OUTPUT_DIR`: directory where chunks, transcripts, and notes are written
- `AUDIO_FILE`: optional path to a single file to process
- `CHUNK_MINUTES=30`: duration of each segment
- `WHISPER_MODEL`: Whisper model to use, for example `large-v3`
- `WHISPER_LANGUAGE=en`: transcription language
- `OLLAMA_ENABLED=true`: enable or disable Markdown generation
- `OLLAMA_MODEL`: Ollama model used to create notes
- `KEEP_CHUNKS=true`: keep intermediate split audio files
- `OVERWRITE=false`: avoid regenerating outputs that already exist

## Usage

Process all files in `INPUT_DIR`:

```bash
uv run python main.py
```

Process one file:

```bash
uv run python main.py --file "/path/to/lecture.m4a"
```

Transcription only, without Markdown notes:

```bash
uv run python main.py --skip-notes
```

Force regeneration of all outputs:

```bash
uv run python main.py --force
```

## Generated output

For each lesson, a directory is created inside `output/` with this structure:

- `chunks/`: 30-minute audio segments
- `transcripts/`: per-chunk transcripts
- `transcript.txt`: aggregated full transcript
- `notes/`: intermediate chunk-by-chunk Ollama notes
- `lesson.md`: final lesson Markdown
- `metadata.json`: processing metadata

## Practical notes

- If Ollama is not running, the pipeline only fails during the Markdown generation step.
- Transcripts are reused if they already exist and `OVERWRITE=false`.
- The prompts are now in English, but you can change model and language from `.env`.
