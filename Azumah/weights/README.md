# Model weights

All 28 checkpoints of the final pipeline (fp16, 1.6 GB) are in the public Kaggle dataset
[`richmondazumah/filament-seg-weights`](https://www.kaggle.com/datasets/richmondazumah/filament-seg-weights);
they are too large for git.

| Path in the dataset | Model |
|---|---|
| `fseg/runs/consensus_f{0-4,full}/weights/best.pt` | YOLO11l-seg proposers (5 folds + full data) |
| `fseg/unet/runs/unet_f{0-4}/best.pt` | ConvNeXt-tiny + ASPP U-Net, 1024 px |
| `fseg/unet/runs/unet2048_f{0-4}/best.pt` | same U-Net at native 2048 px (Charan) |
| `fseg/refiner/resnet60_{v0-4,full}/best.pt` | ResNet-34 refiners for YOLO proposals |
| `fseg/refiner/u2r_{v0-4,full}/best.pt` | ResNet-34 refiners for U-Net-2048 proposals (Charan) |
| `configs/*.json`, `manifest.csv` | tuned thresholds and folds |

Download and run the whole test pipeline:

```bash
kaggle datasets download richmondazumah/filament-seg-weights -p weights/ --unzip
WEIGHTS=weights FSEG_WORK_DIR=work_infer bash scripts/run_inference.sh submission.csv
```

Or run `notebooks/07_end_to_end.ipynb` on Kaggle with the dataset attached. Verified locally: the
script reproduces the submitted `union_c2r.csv` (public LB 0.40) mask for mask (median IoU 0.999).
