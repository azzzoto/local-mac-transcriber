from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import textwrap
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests
import whisper
import yaml
from dotenv import load_dotenv
from tqdm import tqdm


class PipelineError(Exception):
    pass


DEFAULT_SYSTEM_PROMPT = textwrap.dedent(
    """
    Sei un assistente editoriale specializzato in sbobinature universitarie medico-scientifiche.
    Ricevi trascrizioni Whisper di lezioni in italiano e devi trasformarle in Markdown estremamente accurato,
    fedele e scientificamente rigoroso.

    Regole generali:
    - mantieni il contenuto il più vicino possibile alla trascrizione originale
    - correggi parole chiaramente trascritte male, refusi, punteggiatura e segmentazioni innaturali
    - conserva il significato didattico senza semplificazioni eccessive
    - non inventare nozioni, diagnosi, dati, farmaci, linee guida o dettagli non presenti
    - quando un termine appare evidentemente distorto dalla trascrizione, normalizzalo alla formulazione medico-scientifica più plausibile solo se il contesto lo rende chiaro
    - se un passaggio resta ambiguo, preferisci una formulazione prudente e aderente al testo invece di colmare i vuoti
    - scrivi in italiano, con lessico medico appropriato, stile scorrevole ma aderente al parlato accademico
    - l'obiettivo è produrre una sbobina ordinata, leggibile e affidabile, non appunti sintetici
    - restituisci solo Markdown valido, senza blocchi di codice
    """
).strip()


DEFAULT_CHUNK_PROMPT_TEMPLATE = textwrap.dedent(
    """
    Corso: Gastroenterologia e medicina interna, quarto anno
    Titolo lezione: {lesson_title}
    Parte: {chunk_index}/{total_chunks}

    Trascrizione grezza:
    {transcript}

    Produci una sbobina in Markdown per questa porzione di lezione.

    Requisiti:
    - usa un titolo di sezione chiaro e coerente con il contenuto del brano
    - riorganizza lievemente la sintassi per renderla leggibile, ma resta il più possibile fedele alla sequenza logica della spiegazione
    - correggi evidenti errori di trascrizione in termini medici, anatomici, fisiopatologici, farmacologici e procedurali quando il contesto è sufficiente
    - mantieni il livello tecnico universitario e un tono scientifico accurato
    - non trasformare il contenuto in riassunto: preserva spiegazioni, passaggi logici, precisazioni e nessi causali presenti nel parlato
    - usa elenchi puntati solo quando aiutano davvero la leggibilità; altrimenti preferisci paragrafi continui da sbobina
    - se il docente cita termini o concetti in modo frammentario, mantienili in forma prudente senza aggiungere spiegazioni non dette
    - ometti intestazioni vuote o sezioni artificiali
    - restituisci solo Markdown con titoli `##` e `###` quando servono
    """
).strip()


DEFAULT_FINAL_PROMPT_TEMPLATE = textwrap.dedent(
    """
    Corso: Gastroenterologia e medicina interna, quarto anno
    Titolo lezione: {lesson_title}

    Sbobine parziali:
    {notes}

    Unifica il materiale in un unico documento Markdown finale, in italiano, come sbobina completa di una lezione universitaria.

    Requisiti:
    - apri con un titolo H1 contenente il nome della lezione
    - organizza il testo in sezioni e sottosezioni coerenti, senza perdere contenuto rilevante
    - rendi il testo più scorrevole, correggendo refusi e ripetizioni spurie dovute alla trascrizione automatica
    - conserva il più possibile i dettagli tecnici e l'impostazione argomentativa del docente
    - non inventare parti mancanti e non aggiungere conoscenze esterne
    - se individui termini trascritti male ma chiaramente riconoscibili dal contesto, normalizzali in forma medico-scientifica corretta
    - evita tono da riassunto schematico: il risultato deve sembrare una sbobina pulita, accurata e pronta da studiare
    - usa elenchi solo dove migliorano davvero la leggibilità
    - restituisci solo Markdown valido, senza blocchi di codice
    """
).strip()


@dataclass(slots=True)
class PromptConfig:
    system_prompt: str
    chunk_prompt_template: str
    final_prompt_template: str
    ollama_max_input_chars: int
    ollama_num_predict: int
    ollama_final_max_input_chars: int

    @classmethod
    def defaults(cls) -> "PromptConfig":
        return cls(
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            chunk_prompt_template=DEFAULT_CHUNK_PROMPT_TEMPLATE,
            final_prompt_template=DEFAULT_FINAL_PROMPT_TEMPLATE,
            ollama_max_input_chars=6000,
            ollama_num_predict=2048,
            ollama_final_max_input_chars=24000,
        )

    @classmethod
    def from_file(cls, config_path: Path, profile_name: str | None = None) -> "PromptConfig":
        if not config_path.exists():
            logging.info("Config file %s not found, using built-in prompt defaults.", config_path)
            return cls.defaults()

        try:
            with config_path.open("r", encoding="utf-8") as handle:
                payload = yaml.safe_load(handle) or {}
        except yaml.YAMLError as exc:
            raise PipelineError(f"Invalid YAML config in {config_path}: {exc}") from exc

        profiles = payload.get("profiles") or {}
        prompts = payload.get("prompts") or {}

        if profiles and not isinstance(profiles, dict):
            raise PipelineError(f"Invalid config in {config_path}: 'profiles' must be a mapping.")
        if prompts and not isinstance(prompts, dict):
            raise PipelineError(f"Invalid config in {config_path}: 'prompts' must be a mapping.")

        selected_prompts = prompts
        if profiles:
            resolved_profile_name = profile_name or str(payload.get("default_profile") or "").strip()
            if not resolved_profile_name:
                resolved_profile_name = next(iter(profiles))

            profile_payload = profiles.get(resolved_profile_name)
            if not isinstance(profile_payload, dict):
                available_profiles = ", ".join(sorted(profiles))
                raise PipelineError(
                    f"Prompt profile '{resolved_profile_name}' not found in {config_path}. "
                    f"Available profiles: {available_profiles}"
                )

            selected_prompts = profile_payload.get("prompts") or {}
            if not isinstance(selected_prompts, dict):
                raise PipelineError(
                    f"Invalid config in {config_path}: profile '{resolved_profile_name}' must contain a 'prompts' mapping."
                )
            logging.info("Using prompt profile: %s", resolved_profile_name)
        elif profile_name:
            logging.warning("Prompt profile '%s' requested but config has no 'profiles' section; using root prompts.", profile_name)

        processing = payload.get("processing") or {}
        if profiles:
            profile_payload = profiles.get(resolved_profile_name) or {}
            processing = profile_payload.get("processing") or processing

        if processing and not isinstance(processing, dict):
            raise PipelineError(f"Invalid config in {config_path}: 'processing' must be a mapping.")

        return cls(
            system_prompt=str(selected_prompts.get("system") or DEFAULT_SYSTEM_PROMPT).strip(),
            chunk_prompt_template=str(selected_prompts.get("chunk_user") or DEFAULT_CHUNK_PROMPT_TEMPLATE).strip(),
            final_prompt_template=str(selected_prompts.get("final_user") or DEFAULT_FINAL_PROMPT_TEMPLATE).strip(),
            ollama_max_input_chars=max(1000, int(processing.get("ollama_max_input_chars") or 6000)),
            ollama_num_predict=max(256, int(processing.get("ollama_num_predict") or 2048)),
            ollama_final_max_input_chars=max(4000, int(processing.get("ollama_final_max_input_chars") or 24000)),
        )

    def build_chunk_prompt(
        self,
        lesson_title: str,
        chunk_index: int,
        total_chunks: int,
        transcript: str,
    ) -> str:
        return self._format_template(
            self.chunk_prompt_template,
            lesson_title=lesson_title,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            transcript=transcript,
        )

    def build_final_prompt(self, lesson_title: str, notes: str) -> str:
        return self._format_template(
            self.final_prompt_template,
            lesson_title=lesson_title,
            notes=notes,
        )

    def _format_template(self, template: str, **kwargs: object) -> str:
        try:
            return textwrap.dedent(template).strip().format(**kwargs)
        except KeyError as exc:
            missing_key = exc.args[0]
            raise PipelineError(f"Prompt template references unknown placeholder: {missing_key}") from exc


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
    config_path: Path
    prompt_profile: str | None
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
            config_path=resolve_path(os.getenv("CONFIG_PATH", "config.yaml"), "config.yaml"),
            prompt_profile=os.getenv("PROMPT_PROFILE") or None,
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


@dataclass(slots=True)
class TranscriptSegment:
    source_name: str
    note_stem: str
    text: str


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
        "prompt_profile": settings.prompt_profile,
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

    def chat(self, system_prompt: str, user_prompt: str, request_label: str, num_predict: int | None = None) -> str:
        endpoint = f"{self.settings.ollama_base_url}/api/chat"
        logging.info(
            "Ollama start | %s | model=%s | endpoint=%s",
            request_label,
            self.settings.ollama_model,
            endpoint,
        )
        started_at = time.perf_counter()
        progress = {"chars": 0, "thinking_chars": 0, "first_content_at": None}
        heartbeat_stop = threading.Event()

        def heartbeat() -> None:
            while not heartbeat_stop.wait(30):
                status = "waiting for first token" if progress["first_content_at"] is None else "receiving content"
                logging.info(
                    "Ollama waiting | %s | elapsed=%.1fs | status=%s | received_chars=%s | thinking_chars=%s",
                    request_label,
                    time.perf_counter() - started_at,
                    status,
                    progress["chars"],
                    progress["thinking_chars"],
                )

        heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
        heartbeat_thread.start()

        try:
            with requests.post(
                endpoint,
                json={
                    "model": self.settings.ollama_model,
                    "stream": True,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "options": {
                        "temperature": self.settings.ollama_temperature,
                        **(  {"num_predict": num_predict} if num_predict is not None else {}),
                    },
                },
                timeout=self.settings.ollama_timeout_seconds,
                stream=True,
            ) as response:
                elapsed = time.perf_counter() - started_at
                if not response.ok:
                    body_preview = response.text.strip().replace("\n", " ")[:300]
                    raise requests.HTTPError(
                        f"Ollama returned HTTP {response.status_code} for {request_label} "
                        f"at {endpoint} using model {self.settings.ollama_model}. "
                        f"Response: {body_preview or '<empty>'}",
                        response=response,
                    )

                fragments: list[str] = []
                thinking_fragments: list[str] = []
                last_progress_log = started_at
                for line in response.iter_lines(decode_unicode=True):
                    if not line:
                        continue

                    payload = json.loads(line)
                    piece = payload.get("message", {}).get("content", "")
                    thinking_piece = payload.get("message", {}).get("thinking", "")
                    if piece:
                        fragments.append(piece)
                        progress["chars"] += len(piece)
                        if progress["first_content_at"] is None:
                            progress["first_content_at"] = time.perf_counter()

                    if thinking_piece:
                        thinking_fragments.append(thinking_piece)
                        progress["thinking_chars"] += len(thinking_piece)

                    now = time.perf_counter()
                    if now - last_progress_log >= 10:
                        status = "waiting for first token" if progress["first_content_at"] is None else "receiving content"
                        logging.info(
                            "Ollama streaming | %s | elapsed=%.1fs | status=%s | received_chars=%s | thinking_chars=%s",
                            request_label,
                            now - started_at,
                            status,
                            progress["chars"],
                            progress["thinking_chars"],
                        )
                        last_progress_log = now

                    if payload.get("done"):
                        break

                content = "".join(fragments).strip()
        except requests.RequestException as exc:
            elapsed = time.perf_counter() - started_at
            logging.error(
                "Ollama request failed | %s | after %.1fs | model=%s | endpoint=%s | error=%s",
                request_label,
                elapsed,
                self.settings.ollama_model,
                endpoint,
                exc,
            )
            raise
        finally:
            heartbeat_stop.set()

        elapsed = time.perf_counter() - started_at
        first_token_elapsed = None
        if progress["first_content_at"] is not None:
            first_token_elapsed = progress["first_content_at"] - started_at
        logging.info(
            "Ollama done | %s | %.1fs | content_chars=%s | thinking_chars=%s | first_token=%s",
            request_label,
            elapsed,
            len(content),
            progress["thinking_chars"],
            f"{first_token_elapsed:.1f}s" if first_token_elapsed is not None else "never",
        )
        return content


def generate_markdown(
    lesson_paths: LessonPaths,
    transcripts: list[dict[str, str]],
    settings: Settings,
    prompt_config: PromptConfig,
) -> None:
    if not settings.ollama_enabled:
        logging.info("Ollama is disabled: skipping Markdown generation for %s.", lesson_paths.title)
        return

    client = OllamaClient(settings)
    transcript_segments = build_transcript_segments(transcripts, prompt_config.ollama_max_input_chars)
    oversized_count = sum(1 for item in transcripts if len(item["text"].strip()) > prompt_config.ollama_max_input_chars)
    logging.info(
        "Starting Ollama Markdown generation | lesson=%s | transcript_files=%s | prompt_segments=%s | split_files=%s | max_input_chars=%s | notes_dir=%s",
        lesson_paths.title,
        len(transcripts),
        len(transcript_segments),
        oversized_count,
        prompt_config.ollama_max_input_chars,
        lesson_paths.notes_dir,
    )

    chunk_notes: list[str] = []
    total_chunks = len(transcript_segments)
    for index, item in enumerate(transcript_segments, start=1):
        note_path = lesson_paths.notes_dir / f"{item.note_stem}.md"
        if note_path.exists() and not settings.overwrite:
            logging.info(
                "Reusing existing chunk note | %s/%s | source=%s | note=%s",
                index,
                total_chunks,
                item.source_name,
                note_path.name,
            )
            chunk_markdown = note_path.read_text(encoding="utf-8").strip()
        else:
            logging.info(
                "Generating chunk note | %s/%s | source=%s | transcript_chars=%s",
                index,
                total_chunks,
                item.source_name,
                len(item.text),
            )
            chunk_markdown = client.chat(
                system_prompt=prompt_config.system_prompt,
                user_prompt=prompt_config.build_chunk_prompt(
                    lesson_paths.title,
                    index,
                    total_chunks,
                    item.text,
                ),
                request_label=f"chunk {index}/{total_chunks} ({item.source_name})",
                num_predict=prompt_config.ollama_num_predict,
            )
            note_path.write_text(chunk_markdown + "\n", encoding="utf-8")
            logging.info("Saved chunk note | %s", note_path)
        chunk_notes.append(chunk_markdown)

    combined_notes = "\n\n".join(chunk_notes)
    logging.info(
        "Generating final lesson Markdown | lesson=%s | partial_notes=%s | combined_chars=%s | output=%s",
        lesson_paths.title,
        len(chunk_notes),
        len(combined_notes),
        lesson_paths.markdown_path,
    )

    if len(combined_notes) > prompt_config.ollama_final_max_input_chars:
        logging.warning(
            "Combined notes (%s chars) exceed ollama_final_max_input_chars (%s); "
            "concatenating chunk notes directly instead of a final LLM pass.",
            len(combined_notes),
            prompt_config.ollama_final_max_input_chars,
        )
        final_markdown = f"# {lesson_paths.title}\n\n{combined_notes}"
    else:
        final_markdown = client.chat(
            system_prompt=prompt_config.system_prompt,
            user_prompt=prompt_config.build_final_prompt(lesson_paths.title, combined_notes),
            request_label=f"final lesson markdown ({lesson_paths.title})",
        )
    lesson_paths.markdown_path.write_text(final_markdown + "\n", encoding="utf-8")
    logging.info("Saved final Markdown | %s", lesson_paths.markdown_path)


def cleanup_chunks(lesson_paths: LessonPaths, settings: Settings) -> None:
    if settings.keep_chunks:
        return
    clean_directory(lesson_paths.chunks_dir, "*.mp3")
    try:
        lesson_paths.chunks_dir.rmdir()
    except OSError:
        pass


def split_long_text(text: str, max_chars: int) -> list[str]:
    normalized_text = text.strip()
    if not normalized_text:
        return [""]
    if len(normalized_text) <= max_chars:
        return [normalized_text]

    sentence_like_parts = [
        part.strip()
        for part in re.split(r"(?<=[.!?])\s+|\n\s*\n+", normalized_text)
        if part.strip()
    ]
    if not sentence_like_parts:
        sentence_like_parts = [normalized_text]

    chunks: list[str] = []
    current_parts: list[str] = []
    current_length = 0

    def flush_current() -> None:
        nonlocal current_parts, current_length
        if current_parts:
            chunks.append(" ".join(current_parts).strip())
            current_parts = []
            current_length = 0

    for part in sentence_like_parts:
        if len(part) > max_chars:
            flush_current()
            words = part.split()
            word_buffer: list[str] = []
            word_length = 0
            for word in words:
                projected = word_length + len(word) + (1 if word_buffer else 0)
                if word_buffer and projected > max_chars:
                    chunks.append(" ".join(word_buffer).strip())
                    word_buffer = [word]
                    word_length = len(word)
                else:
                    word_buffer.append(word)
                    word_length = projected
            if word_buffer:
                chunks.append(" ".join(word_buffer).strip())
            continue

        projected_length = current_length + len(part) + (1 if current_parts else 0)
        if current_parts and projected_length > max_chars:
            flush_current()

        current_parts.append(part)
        current_length += len(part) + (1 if len(current_parts) > 1 else 0)

    flush_current()
    return chunks or [normalized_text]


def build_transcript_segments(
    transcripts: list[dict[str, str]],
    max_input_chars: int,
) -> list[TranscriptSegment]:
    segments: list[TranscriptSegment] = []

    for item in transcripts:
        source_name = item["chunk"]
        source_stem = Path(source_name).stem
        split_parts = split_long_text(item["text"], max_input_chars)

        if len(split_parts) == 1:
            segments.append(
                TranscriptSegment(
                    source_name=source_name,
                    note_stem=source_stem,
                    text=split_parts[0],
                )
            )
            continue

        for index, part in enumerate(split_parts, start=1):
            segments.append(
                TranscriptSegment(
                    source_name=f"{source_name} part {index}/{len(split_parts)}",
                    note_stem=f"{source_stem}_part_{index:02d}",
                    text=part,
                )
            )

    return segments


def load_transcripts_from_directory(transcripts_dir: Path) -> list[dict[str, str]]:
    if not transcripts_dir.exists() or not transcripts_dir.is_dir():
        raise PipelineError(f"Transcript directory does not exist: {transcripts_dir}")

    transcript_files = sorted(path for path in transcripts_dir.glob("*.txt") if path.is_file())
    if not transcript_files:
        raise PipelineError(f"No transcript .txt files found in {transcripts_dir}")

    transcripts: list[dict[str, str]] = []
    for transcript_file in transcript_files:
        text = transcript_file.read_text(encoding="utf-8").strip()
        transcripts.append({"chunk": transcript_file.name, "text": text})

    return transcripts

def resolve_lesson_paths_from_transcripts(transcripts_dir: Path, settings: Settings) -> LessonPaths:
    lesson_root = transcripts_dir.resolve().parent
    if lesson_root.parent.resolve() != settings.output_dir.resolve():
        raise PipelineError(
            "The transcript directory must be inside OUTPUT_DIR/<lesson>/transcripts to reuse the notes pipeline."
        )

    lesson_title = lesson_root.name
    lesson_paths = prepare_lesson_paths(lesson_root.with_suffix(".txt"), settings)
    if lesson_paths.root_dir != lesson_root:
        lesson_paths = LessonPaths(
            title=lesson_title,
            slug=lesson_root.name,
            root_dir=lesson_root,
            chunks_dir=lesson_root / "chunks",
            transcripts_dir=transcripts_dir,
            notes_dir=lesson_root / "notes",
            transcript_path=lesson_root / "transcript.txt",
            markdown_path=lesson_root / "lesson.md",
            metadata_path=lesson_root / "metadata.json",
        )
        lesson_paths.notes_dir.mkdir(parents=True, exist_ok=True)
    return lesson_paths
def process_transcript_directory(transcripts_dir: Path, settings: Settings, prompt_config: PromptConfig) -> None:
    lesson_paths = resolve_lesson_paths_from_transcripts(transcripts_dir, settings)
    transcripts = load_transcripts_from_directory(lesson_paths.transcripts_dir)
    logging.info(
        "Loaded transcript directory | lesson=%s | dir=%s | transcript_files=%s",
        lesson_paths.title,
        lesson_paths.transcripts_dir,
        len(transcripts),
    )
    write_combined_transcript(lesson_paths, transcripts)
    logging.info("Updated combined transcript | %s", lesson_paths.transcript_path)
    generate_markdown(lesson_paths, transcripts, settings, prompt_config)
    if settings.ollama_enabled:
        logging.info("Notes generated from transcripts: %s", lesson_paths.root_dir)
    else:
        logging.info("Transcripts processed without Ollama generation: %s", lesson_paths.root_dir)


def process_audio_file(
    audio_file: Path,
    model,
    settings: Settings,
    prompt_config: PromptConfig,
    skip_notes: bool,
) -> None:
    if not audio_file.exists():
        raise PipelineError(f"The audio file does not exist: {audio_file}")

    lesson_paths = prepare_lesson_paths(audio_file, settings)
    chunk_paths = split_audio(audio_file, lesson_paths, settings)
    transcripts = transcribe_chunks(model, chunk_paths, lesson_paths, settings)
    write_combined_transcript(lesson_paths, transcripts)
    write_metadata(audio_file, lesson_paths, transcripts, settings)

    if not skip_notes:
        generate_markdown(lesson_paths, transcripts, settings, prompt_config)

    cleanup_chunks(lesson_paths, settings)
    logging.info("Completed: %s", lesson_paths.root_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transcribe audio files with Whisper and generate Markdown notes with Ollama.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=("run", "notes"),
        default="run",
        help="Command to execute. Available: run, notes.",
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
    parser.add_argument(
        "--prompt-profile",
        dest="prompt_profile",
        help="Prompt profile name from config.yaml. Overrides PROMPT_PROFILE.",
    )
    parser.add_argument(
        "--transcripts-dir",
        help="Directory containing transcript .txt files to process with Ollama only (used with the 'notes' command).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    settings = Settings.from_env()
    if args.force:
        settings.overwrite = True
    if args.prompt_profile:
        settings.prompt_profile = args.prompt_profile

    configure_logging(settings.log_level)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    prompt_config = PromptConfig.from_file(settings.config_path, settings.prompt_profile)

    if args.command == "notes":
        if not args.transcripts_dir:
            raise PipelineError("The 'notes' command requires --transcripts-dir.")
        process_transcript_directory(Path(args.transcripts_dir).resolve(), settings, prompt_config)
        return

    ensure_ffmpeg()

    audio_files = get_audio_files(settings, args.audio_file)
    whisper_model = load_whisper_model(settings.whisper_model)

    for audio_file in audio_files:
        process_audio_file(
            audio_file=audio_file,
            model=whisper_model,
            settings=settings,
            prompt_config=prompt_config,
            skip_notes=args.skip_notes,
        )


if __name__ == "__main__":
    try:
        main()
    except PipelineError as exc:
        raise SystemExit(f"Pipeline error: {exc}") from exc
    except requests.RequestException as exc:
        raise SystemExit(
            "Ollama error: unable to complete the HTTP request. "
            f"Base URL: {os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')}. "
            f"Model: {os.getenv('OLLAMA_MODEL', 'llama3.1:8b')}. "
            "Make sure Ollama is running, the model is available, and the URL is correct."
        ) from exc
