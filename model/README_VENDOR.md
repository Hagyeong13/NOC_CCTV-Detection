# vendor/

Inference code for the five-branch detector, written by the project's
co-authors and included here unmodified so results stay reproducible against
the version the paper reports.

Two consequences of keeping it unmodified:

- Several modules contain absolute paths from the machine they were developed
  on (`/home/tako/songhye/project4/...`). They are only read on the training
  and feature-extraction paths, not by `run_inference.py`, so the demo runs
  without them. Nothing here rewrites them, because diverging from the
  published version would defeat the point of vendoring it.
- The per-domain checkpoints these modules load are not in this repository.
  See the top-level README.
