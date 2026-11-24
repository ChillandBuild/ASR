# Roman-script Tanglish ASR

Speech recognition for Tamil–English code-mixed speech (Tanglish), written in **Roman script**.
Base model: OpenAI Whisper-small, fine-tuned with LoRA.

> Fresh start (24 Nov 2026): new dataset, new training. The earlier pilot (Oct 2026) is archived outside this repo.

## Repo layout

```
src/tanglish_asr/      core pipeline: audit → prepare → transcribe → review → evaluate
experiments/           training + checkpoint evaluation scripts
tests/                 unit tests (python -m unittest discover -s tests)
configs/example.json   template: copy it and list your recordings
docs/                  annotation policy, recording guide, CPU run guide
references/            background papers (PDFs not committed)
Data/                  audio + transcripts      (NOT in git, shared privately)
private_outputs/       every run's output        (NOT in git)
```

## Setup (macOS / Linux, Python 3.11, ffmpeg installed)

```sh
git clone <repo-url> && cd ASR
uv venv --python 3.11 .venv
uv pip sync --python .venv/bin/python requirements-lock.txt
uv pip install --python .venv/bin/python --no-deps -e .
.venv/bin/python -m unittest discover -s tests -v
```

For training, also install `requirements-training.txt` (transformers, peft, accelerate):

```sh
uv pip install --python .venv/bin/python -r requirements-training.txt
```

No `uv`? Use `python3.11 -m venv .venv && .venv/bin/pip install -r requirements-lock.txt` instead.

## Pipeline commands

```sh
cp configs/example.json configs/run_01.json     # then edit the recordings list
.venv/bin/python -m tanglish_asr audit      --config configs/run_01.json --decode
.venv/bin/python -m tanglish_asr prepare    --config configs/run_01.json
.venv/bin/python -m tanglish_asr transcribe --config configs/run_01.json --smoke --limit 3
```

Commands: `audit`, `prepare`, `transcribe`, `review-import`, `evaluate`.
Rules the code enforces: output must go inside `private_outputs/`, a run never overwrites an earlier run,
and accuracy is scored only after a human has reviewed the reference transcripts
(`private_outputs/<run>/reference_review.csv`).

## Training

`experiments/train_whisper_small_aligned.py` reads JSONL splits, one clip per line:

```json
{"audio_path": "private_outputs/run_01/clips/speaker01_take01_00012_00041.wav", "text": "naan office ku late aa vandhen"}
```

Its default paths point at the old archived run, so **always pass the new ones**:

```sh
.venv/bin/python experiments/train_whisper_small_aligned.py \
  --base_model <whisper-small HF folder> --processor <processor folder> \
  --train_split private_outputs/run_01/train.jsonl --val_split private_outputs/run_01/val.jsonl \
  --test_split private_outputs/run_01/test.jsonl --output_dir private_outputs/train_01
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to get the data and how to work together.
