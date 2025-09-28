<h1 align="center">PoseTriager</h1>

WARNING: This repository is rough and under development. Use at your own risk.

PoseTriager is a simple EGNN-based architecture for **pose classification** – filtering correct protein-ligand binding poses from incorrect ones in large docking datasets.

- **PoseTriager** – classifies poses as *near-native* vs *decoy*, enabling cleaner datasets for downstream training and evaluation.  
- Built around an **Equivariant Graph Neural Network (EGNN)** for learning directly from 3D protein-ligand graphs.  

---

## Installation

Clone this repository and install dependencies with `pip`:

```bash
pip install -r requirements.txt
```

## Usage
Training is launched via the CLI interface with configurable dataset paths, cache locations, and model hyperparameters.

e.g.

```
python -m src.cli \
    datamodule.dataset.train_df=$TRAIN_CSV \
    datamodule.dataset.val_df=$VAL_CSV \
    model.num_layers=4 \
    task.name=pbclassify \
    datamodule.data_cache=$CACHE_DIR \
    datamodule.dataset.data_path=$DATA_DIR
```

Input CSV files (`$TRAIN_CSV` and `$VAL_CSV`) require `protein`, `ligand` `label` and `pid`, for protein PDB, ligand SDF, pose label (0/1) and protein ID/unique identifier respectively.
`$CACHE_DIR` is a directory for caching processed data, and `$DATA_DIR` is the root directory containing protein and ligand files (which protein PDB and ligand SDF paths in the CSV are relative to).

Inference is also available via the CLI:


```
python -m src.inference \
    data_df=$DATA_CSV \
    out_df=$OUT_CSV \
    checkpoint=$CHECKPOINT \
    data_cache=$CACHE_DIR \
    data_path=$DATA_DIR
```

`$DATA_CSV` is a CSV file with `protein`, `ligand` and `pid` columns for protein PDB, ligand SDF and unique identifier respectively. Predictions are saved to `$OUT_CSV`.
`$CHECKPOINT` is a trained model checkpoint file (.pt).




