# Running the Tanglish tokenizer project locally (Mac, 16 GB, CPU-first)

**Machine:** Apple Silicon (arm64), 8 cores, 16 GB RAM, ~78 GB free disk. ffmpeg is not installed yet.
Apple Silicon has a built-in GPU, which PyTorch reaches through the "MPS" backend. It's not CUDA, but it's often 3–5× faster than CPU for small models. Every step below runs on plain CPU; MPS is an optional speed-up where noted.

## How the project is scaled down to fit a laptop

| Stage | GPU version (original plan) | Laptop version | Still valid? |
|---|---|---|---|
| 0–3 Data, resources, diagnosis, tokenizers | CPU already | Unchanged | Yes |
| 4a Text test | Continued pretraining of a 0.5–1B LLM | **Tiny GPT trained from scratch** (10–30M params), one per tokenizer, same data and steps. Bits-per-character + sentiment F1 | Yes. A from-scratch model is actually a *cleaner* tokenizer comparison, because no model has seen the vocabulary before |
| 4a Classifier | Fine-tune LLM | Logistic regression / linear SVM on each tokenizer's token n-grams (minutes) + optional tiny-GPT probe | Yes |
| 4b ASR | Whisper-large + LoRA | **Whisper-base (74M) or small (244M)** with old vs new tokenizer; baselines run zero-shot with large-v3-turbo via whisper.cpp | Yes, as a controlled comparison. State the model size as a limitation |

## Stage-by-stage

### Stage 0 — Environment and data (½ day)
```bash
brew install ffmpeg
conda create -n tanglish python=3.11 -y && conda activate tanglish
pip install torch torchaudio            # includes MPS support on Apple Silicon
pip install transformers tokenizers sentencepiece tiktoken datasets peft accelerate \
            jiwer rapidfuzz python-docx pandas scikit-learn matplotlib soundfile librosa
# for fast CPU inference of big Whisper models:
brew install whisper-cpp   # or: git clone https://github.com/ggerganov/whisper.cpp && make
```
Data (all text, small):
- **DravidianCodeMix / TamilMixSentiment:** Tanglish YouTube comments with sentiment labels, ~15k+ lines (HF datasets or the FIRE shared-task pages).
- **Dakshina (Google):** romanised Tamil sentences and word lexicons, ~few hundred MB.
- **Your 9 transcripts:** parse them into one-utterance-per-line text (the spoken test set).

Target: 100k–300k lines of training text. That is only tens of MB, so it's trivial for CPU.

### Stage 1 — Resources (CPU, seconds per run; the human checking takes the time)
- Spelling-variant lexicon: phonetic key rules + `rapidfuzz` edit distance + frequency → `variant_lexicon.csv`.
- Ending list: regex over the corpus for English root + Tamil ending (`-la`, `-ku/-kku`, `-nu`, `-dhaan`, `-um`, `-oda`, `-kitta`).
- Gold set: ~1,000 words sampled and checked by a Tamil speaker (this takes ~10 human-hours, not compute).

### Stage 2 — Baseline tokenizer diagnosis (CPU, minutes)
Load **tokenizers only, not model weights**, so each is a few MB:

| Tokenizer | How to load | Note |
|---|---|---|
| GPT-4o | `tiktoken.get_encoding("o200k_base")` | open |
| Whisper | `WhisperTokenizer.from_pretrained("openai/whisper-large-v3")` | open |
| Qwen 2.5/3 | `AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")` | open |
| Sarvam-1 | `AutoTokenizer.from_pretrained("sarvamai/sarvam-1")` | check licence |
| Llama 3 / Gemma 3 | `AutoTokenizer...` | **gated:** needs a free Hugging Face account + accepting the licence + `huggingface-cli login` |
| mT5 / XLM-R | `AutoTokenizer...` | open, classic multilingual baselines |

Metrics, each a few lines of Python: fertility, characters/token, normalised sequence length, parity, **variant consistency**, **boundary alignment**. Run on both typed (comments) and spoken (your transcripts) text.

### Stage 3 — Build tokenizers T1–T4 (CPU, seconds to minutes each)
- HF `tokenizers` trains BPE on 300k lines in under a minute.
- T1: extend the base vocab (e.g. Whisper / Qwen) via continued BPE training, +2k–8k tokens.
- T2: same, but on text augmented with spelling variants (swap words for lexicon variants at 20–40%).
- T3: a pre-tokenizer rule that splits known Tamil endings off English or Tamil roots before BPE.
- T4: T2 + T3. Sweep vocab sizes {2k, 4k, 8k} → 12 tokenizers, all built in under an hour total.

### Stage 4a — Text test (CPU/MPS, ~1–3 h per model)
- **Tiny GPT (nanoGPT-style):** 6 layers, d=384, context 256 → ~15–25M params. Train on ~20–50 MB text for a fixed character budget.
- One run per tokenizer (baseline + T1–T4 = 5) × 3 seeds = 15 runs. On MPS that's roughly 1–2 h each, so **2–3 overnight batches**. On pure CPU, allow ~2–3× longer, or halve model size.
- Report **bits per character** on held-out typed and spoken text. Use bits per character, not perplexity, because perplexity isn't comparable across tokenizers.
- Sentiment: TF-IDF of each tokenizer's tokens → logistic regression on TamilMixSentiment. Minutes per run, with 5-fold CV.

### Stage 4b — ASR test (CPU/MPS, overnight runs)
1. **Prepare audio:** `ffmpeg -i in.mp4 -ac 1 -ar 16000 out.wav`. Segment to ≤30 s using your transcript timestamps. **Precompute log-mel features once** and save them as `.npy`; this removes ~40% of training time.
2. **Zero-shot baselines:** whisper.cpp with `large-v3-turbo` quantised (q5/q8) runs at roughly real-time or faster on Apple Silicon. The 1.5 h eval set takes about an hour per system. Commercial APIs (Sarvam, Gemini) run on their servers, so they cost no local compute.
3. **Fine-tune:** Whisper-**base** (74M) on ~8 h train audio:
   - original tokenizer vs T4 tokenizer (resize decoder embeddings; initialise new tokens as the mean of their old sub-token embeddings)
   - full fine-tune fits easily in 16 GB; batch 16, ~3–5 epochs
   - expected: ~3–6 h per run on MPS, ~10–15 h on CPU → run overnight
   - optional: Whisper-small (244M) with LoRA for one confirmation run (~1 day)
4. **Score:** WER and spelling-tolerant WER (OIWER-style using your variant lexicon), plus tokens per utterance and decoding speed. Speed is where a better tokenizer shows clearly even on small models.

## Speed and memory tips
- `torch.set_num_threads(8)`; try `device="mps"` first and fall back to CPU if an op isn't supported (`PYTORCH_ENABLE_MPS_FALLBACK=1`).
- Keep fp32 on CPU; on MPS, fp16 autocast works for most Whisper ops.
- Cache everything: tokenised datasets (`datasets.save_to_disk`), mel features, and checkpoints every N steps so overnight runs can resume.
- Sort utterances by length and batch similar lengths together (less padding = faster).
- Don't run two training jobs at once on 16 GB. Queue them with a simple shell script overnight.
- Keep the Mac plugged in and stop it sleeping during runs: `caffeinate -i python train.py`.

## Realistic total local compute
| Stage | Wall-clock compute |
|---|---|
| 0–3 | < 1 day (mostly your reading/checking time) |
| 4a | ~3 nights |
| 4b | ~4–6 nights (baselines + 2–4 fine-tunes) |

**About two weeks of overnight runs.** Everything fits in 16 GB, with ~20–30 GB of disk for audio and features.

## When to leave the laptop (optional)
Only if reviewers ask for "does it hold at large scale?": one Whisper-large-v3-turbo LoRA run, using a free Kaggle/Colab GPU session or your ~$200 credit. The paper's core results don't depend on it.
