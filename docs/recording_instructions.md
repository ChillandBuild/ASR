# Recording a small Tanglish test set (~20–30 minutes of your time)

## What to record
**Part A — 19 read sentences** (`recording_prompts.csv`, IDs T01–T19). These are real lines from your transcripts, chosen to include English words, spelling-variant words (`ipo`, `apdi`, `maadhiri`) and mixed words (`schoolla`, `relationshipoda`, `generationku`).
- Read each one **the way you'd naturally say it**, not word-by-word.
- If you naturally say something slightly different (e.g. `ippo` instead of `ipo`, or drop a word), **edit the text in the CSV to match exactly what you said**. The text must match the audio.

**Part B — 5 spontaneous clips** (IDs S01–S05). Talk for 15–20 seconds each, in normal Tanglish, about anything (your day, a film, food, work). Then type what you said, in your usual spelling, as new rows in the CSV.

## How
- **Tool:** phone voice recorder, or on the Mac QuickTime Player → File → New Audio Recording.
- **Place:** quiet room, phone 15–30 cm from your mouth, no music or fan noise.
- **One file per clip**, named by ID: `T01.m4a` … `T19.m4a`, `S01.m4a` … `S05.m4a`. Any format is fine (m4a, wav, mp3); I'll convert.
- **Optional but valuable:** if 1–2 friends record the same 19 sentences, put each person's files in their own subfolder (`speaker1/`, `speaker2/`). More voices make the comparison much more reliable.

## Where to put the files
```
Data/  (inside your copy of the repo)
    recording_prompts.csv   (with any edits + S01–S05 rows)
    T01.m4a ... T19.m4a
    S01.m4a ... S05.m4a
```
Then tell me, and I'll run all the models.

## What the test will measure
For each model, on every clip:
- **Output script:** Roman Tanglish, Tamil script, or English translation.
- **Accuracy:** WER and character error rate (CER), plus a spelling-tolerant WER that doesn't penalise `ipo`/`ippo`-type differences.
- **English handling:** whether mixed words like `schoolla` and `generationku` come out intact.
- **Speed** on your Mac.

**Models to test:**
- indic-transcribe-flex (romanized and mixed modes; needs your Hugging Face token + licence acceptance)
- Dhanush66 and Badri0510 Tanglish Whisper models
- IndicConformer and SraVaani (Tamil script, plus a transliteration step)
- Whisper large-v3-turbo (reference)
