# Local Mac Transcriber

Local pipeline to:

1. automatically split a lecture recording into 30-minute chunks;
2. transcribe each chunk with Whisper;
3. generate final Markdown notes with an Ollama model.

This project takes the notebook idea and turns it into a more professional workflow, configurable through `.env`, prompt templates in `config.yaml`, and reusable from the terminal.

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
- `CONFIG_PATH`: YAML file containing the system and user prompt templates
- `PROMPT_PROFILE`: prompt profile name to load from `config.yaml`
- `AUDIO_FILE`: optional path to a single file to process
- `CHUNK_MINUTES=30`: duration of each segment
- `WHISPER_MODEL`: Whisper model to use, for example `large-v3`
- `WHISPER_LANGUAGE=it`: transcription language
- `OLLAMA_ENABLED=true`: enable or disable Markdown generation
- `OLLAMA_MODEL`: Ollama model used to create notes
- `KEEP_CHUNKS=true`: keep intermediate split audio files
- `OVERWRITE=false`: avoid regenerating outputs that already exist

## `config.yaml`

Prompt customization lives in `config.yaml` under named profiles:

- `default_profile`
- `profiles.<name>.processing.ollama_max_input_chars`
- `profiles.<name>.prompts.system`
- `profiles.<name>.prompts.chunk_user`
- `profiles.<name>.prompts.final_user`

Available placeholders:

- chunk prompt: `{lesson_title}`, `{chunk_index}`, `{total_chunks}`, `{transcript}`
- final prompt: `{lesson_title}`, `{notes}`

The default config included in this project ships with multiple profiles, including a gastroenterology-focused Italian profile. You can switch profile with `PROMPT_PROFILE` or `--prompt-profile`.

If Ollama is slow on long transcripts, lower `profiles.<name>.processing.ollama_max_input_chars` to force additional sub-splitting of each transcript file before generation. For example, `4500` is much lighter than `6000`.

## Usage

Process all files in `INPUT_DIR`:

```bash
uv run python main.py
```

Process all files using a specific prompt profile:

```bash
uv run python main.py --prompt-profile gastro_it
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

Generate notes from existing transcript `.txt` files only, without running Whisper/ffmpeg:

```bash
uv run python main.py notes --transcripts-dir output/lez-gastro-16-04/transcripts
```

Generate notes from existing transcripts with a different prompt profile:

```bash
uv run python main.py notes --transcripts-dir output/lez-gastro-16-04/transcripts --prompt-profile medical_en
```

If a transcript file is still too heavy for Ollama, reduce `ollama_max_input_chars` in the selected profile and rerun with `--force` to regenerate the intermediate note files.

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
- Prompt behavior can be changed by editing `config.yaml` instead of modifying `main.py`.
- The `notes` command is the fastest way to test only the Ollama prompt/output stage starting from existing transcript files.
