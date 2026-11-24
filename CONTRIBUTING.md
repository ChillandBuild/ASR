# Working on this repo

## 1. Get set up
Follow **Setup** in [README.md](README.md). The tests must pass before you change anything.

## 2. Get the data (never through git)
- Ask the project owner for the dataset link. Copy the files into `Data/` (layout: [Data/README.md](Data/README.md)).
- Audio (`*.mp3`, `*.wav`), transcripts (`*.docx`), PDFs, model weights and `.env*` files are blocked by `.gitignore`.
  If `git status` ever shows one of them, stop and tell the owner.
- These are recordings of real people. Don't upload them anywhere else, and don't paste transcript text into issues or PRs.

## 3. API keys (only if you use hosted models)
Copy `.env.example` and keep only the line you need, e.g. `.env.gemini` containing `GEMINI_API_KEY=...`.
Use your own key. Never commit it.

## 4. Make a change
```sh
git checkout -b yourname/short-topic     # never work on main directly
# ...edit...
.venv/bin/python -m unittest discover -s tests
git add <files> && git commit -m "Say what changed and why"
git push -u origin yourname/short-topic  # then open a Pull Request
```
- One topic per branch. Small PRs get reviewed faster.
- New code in `src/` needs a test in `tests/`.
- Run outputs stay in `private_outputs/`. To share a result, write a short `docs/<topic>_results.md` with the numbers.

## 5. Writing transcripts
Follow [docs/roman_tanglish_annotation_policy.md](docs/roman_tanglish_annotation_policy.md) exactly,
so that everyone's transcripts can be scored against each other.
