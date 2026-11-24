# Data/

Audio and transcripts live here. **Nothing in this folder except this README is committed to git.**
Get the dataset from the project owner (shared privately) and copy it in.

## Expected layout

```
Data/
├── speaker01_take01.mp3     # one audio stream per file
├── speaker01_take01.docx    # matching transcript (Roman-script Tanglish)
├── speaker02_take01.mp3
└── speaker02_take01.docx
```

- Each recording needs an entry in a config file (see `configs/example.json`).
- Recording IDs: letters, numbers, `_` or `-` only (e.g. `speaker01_take01`).
- Transcription rules: `docs/roman_tanglish_annotation_policy.md`.
- Recording guide: `docs/recording_instructions.md`.
