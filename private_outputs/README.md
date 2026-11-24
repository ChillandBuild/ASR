# private_outputs/

Every run writes here: audio clips, manifests, review sheets, training splits, checkpoints, results.
**Nothing in this folder except this README is committed to git.**

- The pipeline refuses to write anywhere else (`output_dir` in a config must be inside this folder).
- Use a new subfolder per run (e.g. `private_outputs/run_01`); runs never overwrite each other.
- Share results with the team by copying the summary (e.g. a `results.md`) into `docs/`, not the raw outputs.
