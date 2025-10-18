# MSSFN: Multi-stimulus Stereo Spatiotemporal Fusion Network

Reference implementation of **MSSFN** for eye-movement–based Alzheimer’s disease (AD) classification. The repository includes model code, training/evaluation scripts, and simple visualization utilities. 

## Quick Start

### 1) Environment

```bash
git clone https://github.com/zzwl77/MSSFN.git
cd MSSFN
pip install -r requirements.txt
```

### 2) Data

If you need access or have data-related questions, **please contact the paper’s corresponding author (as listed in the manuscript).**
To use your own data, follow the expected structure in `AD_Dataloader.py` and, if needed, precompute graph features with `generate_graph.py`. 

### 3) Training

`main_gcls.py` is the primary entry point (a demo launcher `main_gcls.sh` is also provided). Use `-h` to view options.

```bash
python main_gcls.py

# example
python main_gcls.py \
  --data_root ./data \
  --save_dir ./runs/mssfn_exp1 \
  --epochs 100 --batch_size 32 --lr 1e-3
```

### 4) Evaluation & Visualization

* **Classification test:** `test_cls.py`
* **ROC plotting:** `draw_roc_all.py`
* **Heatmaps/attention:** `heatmap.py`
  Run with your paths/checkpoints as needed. 

## Repository Layout (short)

```
MSSFN/
├─ models/               # network & modules
├─ utils/                # metrics, losses, logging, etc.
├─ AD_Dataloader.py      # data loading / preprocessing
├─ main_gcls.py          # training entry
├─ main_gcls.sh          # example launcher
├─ test_cls.py           # evaluation
├─ generate_graph.py     # (optional) graph/intermediate generation
├─ draw_roc_all.py       # ROC plotting
├─ heatmap.py            # visualization
└─ requirements.txt      # dependencies
```

Files as listed in the repository.

## Citation

If this code helps your research, please cite the associated manuscript (fill in your final details):

```bibtex
@article{MSSFN2025,
  title   = {MSSFN: Multi-stimulus Stereo Spatiotemporal Fusion Network with Pattern Disentanglement for Alzheimer's Disease Diagnosis},
  author  = {…},
  journal = {…},
  year    = {2025}
}
```

## License

No license file is currently provided in the repo. For redistribution or commercial use, please contact the authors or open an issue.

