import argparse
import pandas as pd
import torch
from pathlib import Path
from src.datamodules.complex_features import (
    concat_structs,
    extract_coords,
    make_box,
    generate_edges,
    make_bit_vector,
)
from src.datamodules.smina_types import StructuralFileParser
from torch.nn.functional import one_hot
from torch_geometric.data import Data
import numpy as np
from tqdm import tqdm
import warnings

warnings.simplefilter("ignore", category=FutureWarning)

def process_row(row, base_path, n_features=11, edge_radius=10, save_dir=None):
    parser = StructuralFileParser()
    lig_path = Path(base_path) / row["ligand"]
    rec_path = Path(base_path) / row["protein"]
    lig_num = row.get("ligand_num", 0)
    label = row.get("label", 0)

    try:
        rec_df = parser.rdkitmol_to_df(rec_path, mol_type="receptor")
        lig_df = parser.rdkitmol_to_df(lig_path, mol_type="ligand", num=lig_num)
        struct = make_box(
            concat_structs(rec_df, lig_df, n_features),
            radius=6,
            relative_to_ligand=True,
        )
        struct = struct[struct["atomic_number"] > 1]
        pos = torch.tensor(np.vstack([struct["x"], struct["y"], struct["z"]]).T, dtype=torch.float)
        x = make_bit_vector(struct.types.to_numpy(), n_features, compact=True)

        struct, edge_indices, edge_attrs = generate_edges(
            struct, inter_radius=edge_radius, intra_radius=2.0
        )
        edge_index = torch.tensor(np.vstack(edge_indices), dtype=torch.long)
        edge_attr = one_hot(torch.tensor(edge_attrs, dtype=torch.long), 3)

        y = torch.tensor(label)
        data = Data(x=x.float(), edge_index=edge_index, edge_attr=edge_attr, pos=pos, y=y)

        ligand_stem = lig_path.stem
        ligand_dir = lig_path.parent
        save_path = Path(save_dir) / f"graph_{ligand_dir}" / f"{ligand_stem}_{lig_num}.pt"
        if not save_path.exists():
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(data, save_path)
    except Exception as e:
        print(f"[ERROR] {lig_path} → {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--save-path", required=True)
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.train_csv)
    total = len(df)
    chunk_size = total // args.num_shards
    start = args.shard_id * chunk_size
    end = total if args.shard_id == args.num_shards - 1 else (args.shard_id + 1) * chunk_size
    shard_df = df.iloc[start:end].reset_index(drop=True)
    from joblib import Parallel, delayed

    Parallel(n_jobs=8)(
        delayed(process_row)(row, base_path=args.data_path, save_dir=args.save_path)
        for _, row in tqdm(shard_df.iterrows(), total=len(shard_df))
    )
