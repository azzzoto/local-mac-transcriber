from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests
import whisper
from dotenv import load_dotenv
from tqdm import tqdm


class PipelineError(Exception):
    pass


def parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def slugify(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower())
    return normalized.strip("-") or "lesson"


@dataclass(slots=True)
class Settings:
    project_root: Path
    input_dir: Path
    output_dir: Path
    audio_file: Path | None
    whisper_model: str
    whisper_language: str | None
    chunk_minutes: int
    keep_chunks: bool
    overwrite: bool
    audio_extensions: tuple[str, ...]
    ollama_enabled: bool
    ollama_base_url: str
    ollama_model: str
    ollama_temperature: float
    ollama_timeout_seconds: int
    log_level: str

    @classmethod
    def from_env(cls) -> "Settings":
        project_root = Path(__file__).resolve().parent
        load_dotenv(project_root / ".env")

        def resolve_path(raw_value: str, default: str) -> Path:
            path = Path(raw_value or default)
            return path if path.is_absolute() else (project_root / path).resolve()

        extensions = tuple(
            part.strip().lower()
            for part in os.getenv(
                "AUDIO_EXTENSIONS",
                ".aac,.m4a,.mp3,.mp4,.mpeg,.mpga,.wav,.webm",
            ).split(",")
            if part.strip()
        )

        audio_file_value = os.getenv("AUDIO_FILE")

        return cls(
            project_root=project_root,
            input_dir=resolve_path(os.getenv("INPUT_DIR", "input"), "input"),
            output_dir=resolve_path(os.getenv("OUTPUT_DIR", "output"), "output"),
            audio_file=resolve_path(audio_file_value, audio_file_value) if audio_file_value else None,
            whisper_model=os.getenv("WHISPER_MODEL", "large-v3"),
            whisper_language=os.getenv("WHISPER_LANGUAGE", "it") or None,
            chunk_minutes=max(1, int(os.getenv("CHUNK_MINUTES", "30"))),
            keep_chunks=parse_bool(os.getenv("KEEP_CHUNKS"), True),
            overwrite=parse_bool(os.getenv("OVERWRITE"), False),
            audio_extensions=extensions,
            ollama_enabled=parse_bool(os.getenv("OLLAMA_ENABLED"), True),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/"),
            ollama_model=os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
            ollama_temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0.2")),
            ollama_timeout_seconds=max(30, int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "600"))),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        )


@dataclass(slots=True)
class LessonPaths:
    title: str
    slug: str
    root_dir: Path
    chunks_dir: Path
    transcripts_dir: Path
    notes_dir: Path
    transcript_path: Path
    markdown_path: Path
    metadata_path: Path


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(levelname)s | %(message)s",
    )


def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg"):
        return
    raise PipelineError(
        "ffmpeg is not installed or is not available in PATH. Install it before running the pipeline."
    )


def get_audio_files(settings: Settings, cli_file: str | None) -> list[Path]:
    if cli_file:
        target = Path(cli_file)
        return [(target if target.is_absolute() else (Path.cwd() / target)).resolve()]

    if settings.audio_file:
        return [settings.audio_file]

    if not settings.input_dir.exists():
        raise PipelineError(
            f"The input directory does not exist: {settings.input_dir}. Create it or set INPUT_DIR in the .env file."
        )

    files = [
        path
        for path in sorted(settings.input_dir.iterdir())
        if path.is_file() and path.suffix.lower() in settings.audio_extensions
    ]
    if not files:
        raise PipelineError(
            f"No audio files found in {settings.input_dir} with extensions {', '.join(settings.audio_extensions)}"
        )
    return files


def prepare_lesson_paths(audio_file: Path, settings: Settings) -> LessonPaths:
    title = audio_file.stem
    slug = slugify(title)
    root_dir = settings.output_dir / slug
    chunks_dir = root_dir / "chunks"
    transcripts_dir = root_dir / "transcripts"
    notes_dir = root_dir / "notes"

    for directory in (root_dir, chunks_dir, transcripts_dir, notes_dir):
        directory.mkdir(parents=True, exist_ok=True)

    return LessonPaths(
        title=title,
        slug=slug,
        root_dir=root_dir,
        chunks_dir=chunks_dir,
        transcripts_dir=transcripts_dir,
        notes_dir=notes_dir,
        transcript_path=root_dir / "transcript.txt",
        markdown_path=root_dir / "lesson.md",
        metadata_path=root_dir / "metadata.json",
    )


def clean_directory(directory: Path, pattern: str) -> None:
    for file_path in directory.glob(pattern):
        if file_path.is_file():
            file_path.unlink()


def split_audio(audio_file: Path, lesson_paths: LessonPaths, settings: Settings) -> list[Path]:
    existing_chunks = sorted(lesson_paths.chunks_dir.glob("*.mp3"))
    if existing_chunks and not settings.overwrite:
        logging.info("Chunks already exist for %s, reusing existing files.", audio_file.name)
        return existing_chunks

    clean_directory(lesson_paths.chunks_dir, "*.mp3")
    chunk_pattern = lesson_paths.chunks_dir / f"{lesson_paths.slug}_%03d.mp3"
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(audio_file),
        "-f",
        "segment",
        "-segment_time",
        str(settings.chunk_minutes * 60),
        "-c:a",
        "libmp3lame",
        "-q:a",
        "2",
        str(chunk_pattern),
    ]

    logging.info("Splitting %s into %s-minute chunks.", audio_file.name, settings.chunk_minutes)
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise PipelineError(
            f"ffmpeg failed for {audio_file.name}: {result.stderr.strip() or result.stdout.strip()}"
        )

    chunk_paths = sorted(lesson_paths.chunks_dir.glob("*.mp3"))
    if not chunk_paths:
        raise PipelineError(f"No chunks were produced for {audio_file.name}")
    return chunk_paths


def load_whisper_model(model_name: str):
    logging.info("Loading Whisper model %s", model_name)
    return whisper.load_model(model_name)


def transcribe_chunks(model, chunk_paths: Iterable[Path], lesson_paths: LessonPaths, settings: Settings) -> list[dict[str, str]]:
    transcripts: list[dict[str, str]] = []

    for chunk_path in tqdm(list(chunk_paths), desc=f"Transcribing {lesson_paths.slug}"):
        transcript_path = lesson_paths.transcripts_dir / f"{chunk_path.stem}.txt"

        if transcript_path.exists() and not settings.overwrite:
            text = transcript_path.read_text(encoding="utf-8").strip()
        else:
            result = model.transcribe(
                str(chunk_path),
                language=settings.whisper_language,
                verbose=False,
                fp16=False,
            )
            text = result["text"].strip()
            transcript_path.write_text(text + "\n", encoding="utf-8")

        transcripts.append({"chunk": chunk_path.name, "text": text})

    return transcripts


def write_combined_transcript(lesson_paths: LessonPaths, transcripts: list[dict[str, str]]) -> None:
    sections = []
    for index, item in enumerate(transcripts, start=1):
        sections.append(f"## Chunk {index:03d} - {item['chunk']}\n\n{item['text'].strip()}\n")
    lesson_paths.transcript_path.write_text("\n".join(sections).strip() + "\n", encoding="utf-8")


def write_metadata(audio_file: Path, lesson_paths: LessonPaths, transcripts: list[dict[str, str]], settings: Settings) -> None:
    payload = {
        "source_file": str(audio_file),
        "lesson_title": lesson_paths.title,
        "chunk_minutes": settings.chunk_minutes,
        "whisper_model": settings.whisper_model,
        "whisper_language": settings.whisper_language,
        "ollama_enabled": settings.ollama_enabled,
        "ollama_model": settings.ollama_model,
        "chunks": [item["chunk"] for item in transcripts],
        "outputs": {
            "transcript": str(lesson_paths.transcript_path),
            "markdown": str(lesson_paths.markdown_path),
        },
    }
    lesson_paths.metadata_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


class OllamaClient:
    def __init__(self, settings: Settings):
        self.settings = settings

    def chat(self, system_prompt: str, user_prompt: str) -> str:
        response = requests.post(
            f"{self.settings.ollama_base_url}/api/chat",
            json={
                "model": self.settings.ollama_model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "options": {"temperature": self.settings.ollama_temperature},
            },
            timeout=self.settings.ollama_timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        return payload["message"]["content"].strip()


def build_chunk_prompt(lesson_title: str, chunk_index: int, total_chunks: int, transcript: str) -> str:
    return textwrap.dedent(
        f"""
        Lesson: {lesson_title}
        Part: {chunk_index}/{total_chunks}

        Transcript:
        {transcript}

        Create professional lecture notes in English Markdown.
        Requirements:
        - use a clear section title for this part
        - extract the key concepts without inventing information
        - use bullet lists when helpful
        - include examples or definitions only if they appear in the transcript
        - if a concept is only mentioned but not explained, place it in a short "Mentioned concepts" list without adding definitions or guesses
        - add an "Open questions" section only if it truly emerges from the transcript
        - omit any section that does not have enough textual support
        - return Markdown only, using `#`, `##`, and `###` headings
        """
    ).strip()


def build_final_prompt(lesson_title: str, notes: str) -> str:
    return textwrap.dedent(
        f"""
        Lesson title: {lesson_title}

        Preliminary notes:
        {notes}

        Generate a polished final Markdown document in English.
        Required structure:
        - an H1 title with the lesson name
        - a short opening overview
        - H2/H3 sections organized by topic
        - a final list of key takeaways
        - when useful, a "Mentioned concepts" section for terms that are cited but not defined
        - an optional "Terminology" section only if technical terms are actually explained or contextualized

        Rules:
        - do not invent content that is not present in the notes
        - do not introduce speculative phrasing such as "it could be" or "this may be covered later"
        - omit any empty or unsupported section
        - do not use code blocks
        - use standard Markdown headings with `#`
        - return valid Markdown only
        """
    ).strip()


def generate_markdown(lesson_paths: LessonPaths, transcripts: list[dict[str, str]], settings: Settings) -> None:
    if not settings.ollama_enabled:
        logging.info("Ollama is disabled: skipping Markdown generation for %s.", lesson_paths.title)
        return

    client = OllamaClient(settings)
    system_prompt = (
        "You are an editorial assistant for university lectures. "
        "You turn transcripts into clear, accurate, concise Markdown notes. "
        "Do not invent facts that are not present in the provided material. "
        "If information is not supported by the text, omit it."
    )

    chunk_notes: list[str] = []
    total_chunks = len(transcripts)
    for index, item in enumerate(transcripts, start=1):
        note_path = lesson_paths.notes_dir / f"chunk_{index:03d}.md"
        if note_path.exists() and not settings.overwrite:
            chunk_markdown = note_path.read_text(encoding="utf-8").strip()
        else:
            chunk_markdown = client.chat(
                system_prompt=system_prompt,
                user_prompt=build_chunk_prompt(lesson_paths.title, index, total_chunks, item["text"]),
            )
            note_path.write_text(chunk_markdown + "\n", encoding="utf-8")
        chunk_notes.append(chunk_markdown)

    final_markdown = client.chat(
        system_prompt=system_prompt,
        user_prompt=build_final_prompt(lesson_paths.title, "\n\n".join(chunk_notes)),
    )
    lesson_paths.markdown_path.write_text(final_markdown + "\n", encoding="utf-8")


def cleanup_chunks(lesson_paths: LessonPaths, settings: Settings) -> None:
    if settings.keep_chunks:
        return
    clean_directory(lesson_paths.chunks_dir, "*.mp3")
    try:
        lesson_paths.chunks_dir.rmdir()
    except OSError:
        pass


def process_audio_file(audio_file: Path, model, settings: Settings, skip_notes: bool) -> None:
    if not audio_file.exists():
        raise PipelineError(f"The audio file does not exist: {audio_file}")

    lesson_paths = prepare_lesson_paths(audio_file, settings)
    chunk_paths = split_audio(audio_file, lesson_paths, settings)
    transcripts = transcribe_chunks(model, chunk_paths, lesson_paths, settings)
    write_combined_transcript(lesson_paths, transcripts)
    write_metadata(audio_file, lesson_paths, transcripts, settings)

    if not skip_notes:
        generate_markdown(lesson_paths, transcripts, settings)

    cleanup_chunks(lesson_paths, settings)
    logging.info("Completed: %s", lesson_paths.root_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transcribe audio files with Whisper and generate Markdown notes with Ollama.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=("run",),
        default="run",
        help="Command to execute. Currently available: run.",
    )
    parser.add_argument(
        "--file",
        dest="audio_file",
        help="Path to a specific audio file. If omitted, files in INPUT_DIR are processed.",
    )
    parser.add_argument(
        "--skip-notes",
        action="store_true",
        help="Run transcription only and skip Markdown generation with Ollama.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing chunks, transcripts, and notes for this run.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    settings = Settings.from_env()
    if args.force:
        settings.overwrite = True

    configure_logging(settings.log_level)
    ensure_ffmpeg()
    settings.output_dir.mkdir(parents=True, exist_ok=True)

    audio_files = get_audio_files(settings, args.audio_file)
    whisper_model = load_whisper_model(settings.whisper_model)

    for audio_file in audio_files:
        process_audio_file(audio_file=audio_file, model=whisper_model, settings=settings, skip_notes=args.skip_notes)


if __name__ == "__main__":
    try:
        main()
    except PipelineError as exc:
        raise SystemExit(f"Pipeline error: {exc}") from exc
    except requests.RequestException as exc:
        raise SystemExit(
            "Ollama error: unable to complete the HTTP request. "
            "Make sure Ollama is running and that OLLAMA_BASE_URL is correct."
        ) from exc
