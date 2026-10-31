# Project notes for Claude Code

Kaggle Solar Filament Segmentation Challenge 2026, scored with Panoptic Quality. Deadline Nov 15, 2026.

* **Start by reading `HANDOFF.md`**: status, data conventions, and specs for every remaining stage.
* Package code lives in `fseg/`, entry points in `scripts/`, cluster jobs in `slurm/`.
* Masks are COCO RLE at 2048 x 2048 (`fseg.rle`). Split by `observation_id` (image file stem),
  never by COCO record id. Evaluate with `fseg.metrics.evaluate(preds, references)` against
  `references.pkl` (all annotation sets) on fold-0 validation observations.
* Do not train on the JSON `spine` or chirality fields (competition rule); skeletonize masks instead.
* Every new script must run on the synthetic dataset (`tests/make_synthetic.py`) and on CPU, MPS and CUDA.
* Tests: `python tests/test_metrics.py` (add tests for new logic next to it).
* Docs and report style: crisp and precise, no em dashes.
