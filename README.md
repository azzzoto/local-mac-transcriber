# Local Mac Transcriber

Pipeline locale per sbobinature universitarie e generazione di note Markdown da registrazioni audio.

1. divide la registrazione in segmenti audio con `ffmpeg`;
2. trascrive ciascun segmento con Whisper (locale, nessun cloud);
3. genera una sbobina Markdown fedele alla trascrizione con un modello Ollama locale.

Tutto è configurabile da `.env` e `config.yaml` senza toccare il codice.

## Requisiti

- Python 3.12+
- `ffmpeg` installato e disponibile nel PATH
- Ollama installato e avviato localmente per la generazione delle sbobine

Su macOS:

```bash
brew install ffmpeg
```

Installa Ollama da [https://ollama.com](https://ollama.com), poi scarica un modello **senza thinking mode**:

```bash
ollama pull llama3.1:latest   # consigliato: veloce, stabile, nessun thinking
```

> ⚠️ **Evita modelli "thinking"** come `qwen3.5`, `deepseek-r1` o simili: generano
> migliaia di token di ragionamento interno prima di rispondere, rendendo la pipeline
> inutilmente lenta per questo task.

## Setup

```bash
cp .env.example .env    # configura il tuo ambiente
uv sync                 # installa le dipendenze
```

Metti i file audio in `input/` oppure imposta `AUDIO_FILE` nel `.env`.

## `.env`

| variabile | default | descrizione |
|---|---|---|
| `INPUT_DIR` | `input` | cartella dei file audio |
| `OUTPUT_DIR` | `output` | cartella degli output |
| `CONFIG_PATH` | `config.yaml` | file con i profili prompt |
| `PROMPT_PROFILE` | _(dal config)_ | profilo prompt da usare |
| `AUDIO_FILE` | — | percorso di un singolo file audio |
| `CHUNK_MINUTES` | `30` | durata di ogni segmento audio |
| `WHISPER_MODEL` | `large-v3` | modello Whisper |
| `WHISPER_LANGUAGE` | `it` | lingua della trascrizione |
| `OLLAMA_ENABLED` | `true` | abilita/disabilita la generazione Markdown |
| `OLLAMA_MODEL` | `llama3.1:latest` | modello Ollama (usa uno senza thinking) |
| `OLLAMA_TEMPERATURE` | `0.1` | temperatura del modello |
| `OLLAMA_TIMEOUT_SECONDS` | `1800` | timeout per chiamata Ollama |
| `KEEP_CHUNKS` | `true` | conserva i segmenti audio intermedi |
| `OVERWRITE` | `false` | rigenera gli output già esistenti |

## `config.yaml`

I prompt vivono in `config.yaml` sotto profili nominati. Struttura:

```yaml
default_profile: gastro_it

profiles:
  gastro_it:
    processing:
      ollama_max_input_chars: 4500       # max caratteri per segmento inviato a Ollama
      ollama_num_predict: 2048           # max token in output per segmento
      ollama_final_max_input_chars: 24000 # sopra questa soglia il passo finale concatena direttamente
    prompts:
      system: |
        ...
      chunk_user: |
        ...  # placeholder: {lesson_title} {chunk_index} {total_chunks} {transcript}
      final_user: |
        ...  # placeholder: {lesson_title} {notes}
```

**Profili inclusi:**

| profilo | lingua | uso consigliato |
|---|---|---|
| `gastro_it` | italiano | lezioni di gastroenterologia/medicina, 4° anno |
| `general_it` | italiano | qualsiasi lezione universitaria in italiano |
| `medical_en` | inglese | lezioni medico-scientifiche in inglese |

### Parametri `processing`

| parametro | default | effetto |
|---|---|---|
| `ollama_max_input_chars` | `6000` | spezza i `.txt` in sottosegmenti prima di inviarli; abbassare se Ollama è lento |
| `ollama_num_predict` | `2048` | limita l'output per segmento; evita loop o output esplosivi |
| `ollama_final_max_input_chars` | `24000` | se le note totali superano questo valore, il passo finale concatena direttamente senza LLM |

## Utilizzo

Pipeline completa su tutti i file in `INPUT_DIR`:

```bash
uv run python main.py
```

Un solo file audio:

```bash
uv run python main.py --file "/path/to/lezione.m4a"
```

Solo trascrizione, senza generare Markdown:

```bash
uv run python main.py --skip-notes
```

Forza la rigenerazione di tutto:

```bash
uv run python main.py --force
```

### Comando `notes` — genera sbobine da trascrizioni già esistenti

Utile per testare i prompt senza rilanciare Whisper:

```bash
uv run python main.py notes --transcripts-dir output/lez-gastro-16-04/transcripts
```

Con profilo specifico:

```bash
uv run python main.py notes \
  --transcripts-dir output/lez-gastro-16-04/transcripts \
  --prompt-profile gastro_it
```

Con profilo diverso al volo:

```bash
uv run python main.py notes \
  --transcripts-dir output/lez-gastro-16-04/transcripts \
  --prompt-profile medical_en \
  --force
```

## Output generato

Per ogni lezione viene creata una cartella in `output/` con questa struttura:

```
output/<slug>/
├── chunks/           segmenti audio .mp3
├── transcripts/      trascrizioni per segmento (.txt)
├── transcript.txt    trascrizione completa aggregata
├── notes/            sbobine intermedie per sottosegmento (.md)
├── lesson.md         sbobina finale completa
└── metadata.json     metadati del processing
```

## Note operative

- Ollama è necessario solo per la fase di generazione Markdown; la trascrizione Whisper funziona senza.
- Le trascrizioni esistenti vengono riutilizzate se `OVERWRITE=false`.
- Se un segmento è ancora troppo pesante per Ollama, riduci `ollama_max_input_chars` nel profilo e rilancia con `--force`.
- Il passo finale di unificazione bypassa automaticamente Ollama se il materiale supera `ollama_final_max_input_chars`, concatenando le sbobine parziali direttamente.
- Prompt behavior can be changed by editing `config.yaml` instead of modifying `main.py`.
- The `notes` command is the fastest way to test only the Ollama prompt/output stage starting from existing transcript files.
