from functools import partial
import pandas as pd
import numpy as np
import torch
import lmdb
import pickle
from pathlib import Path
from src.datamodules.base_module import BaseDataModule
from src.datamodules.complex_features import (
    concat_structs,
    make_box,
    generate_edges,
    make_bit_vector,
)
from src.datamodules.smina_types import StructuralFileParser
from torch.utils.data import Dataset
from torch.nn.functional import one_hot
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from joblib import Parallel, delayed
from tqdm import tqdm
from rdkit import RDLogger
from collections import defaultdict
from pathlib import Path
RDLogger.DisableLog("rdApp.*")

# hide warnings

import warnings

# hide future warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
import sys, pathlib
sys.modules['pathlib._local'] = pathlib



class PoseDataset(Dataset):
    def __init__(
        self,
        data_path,
        input_df,
        radius=6,
        edge_radius=10,
        save_name="test",
        save=True,
        **kwargs,
    ):
        super().__init__()
        self.radius = radius
        self.base_path = Path(data_path).expanduser()
        self.edge_radius = edge_radius
        # if not self.base_path.exists():
        #     raise FileNotFoundError(f"Dataset {self.base_path} does not exist.")
        self.compact = True

        labels = [0] * len(input_df)
        if "label" in input_df.columns:
            labels = input_df["label"].values
        if "receptor" in input_df.columns:
            self.receptor_fnames = input_df["receptor"].to_list()
        elif "protein" in input_df.columns:
            self.receptor_fnames = input_df["protein"].to_list()
        else:
            raise KeyError("No receptor or protein column found in input dataframe")
        if "ligand_num" not in input_df.columns:
            self.ligand_nums = [0] * len(input_df)
        else:
            self.ligand_nums = input_df["ligand_num"].to_list()
        self.ligand_fnames = input_df["ligand"].to_list()
        self.save_name = save_name
        self.save = save
        if self.save and self.save_name is None:
            raise ValueError("Save name must be provided if save is True")
        self.labels = labels
        self.n_features = 11  # + 8 * extended_atom_types
        self.feature_dim = self.n_features + 1
        if self.save:
            if not self.save_name:
                raise ValueError("Save name must be provided if save is True")
            Path(self.save_name).mkdir(parents=True, exist_ok=True)

            # for idx in tqdm(range(len(self.ligand_fnames)), desc="Saving graphs"):
            #     self.save_graph(idx)
                
            # do it in parallel
            # Parallel(n_jobs=10)(
            #     delayed(self.save_graph)(idx) for idx in tqdm(range(len(self.ligand_fnames)), desc="Saving graphs")
            # )
            # batch_size = 500
            # batches = [list(range(i, min(i + batch_size, len(self.ligand_fnames)))) 
            #         for i in range(0, len(self.ligand_fnames), batch_size)]
            # Parallel(n_jobs=-1)(
            #     delayed(self.save_graph_batch)(batch) for batch in tqdm(batches, desc="Saving graphs")
            # )
            grouped = defaultdict(list)
            for i, fname in enumerate(self.ligand_fnames):
                # key = str(Path(fname).parts[-2])[:2]  # -2 directory = index -3
                key = str(Path(fname).parents[0])[:2]  # -2 directory = index -3
                grouped[key].append(i)

            # Create batches based on these groups
            batches = list(grouped.values())


            Parallel(n_jobs=1)(
                delayed(self.save_graph_batch)(batch) for batch in tqdm(batches, desc="Saving graphs")
            )
            
            # self.data = Parallel(n_jobs=8)(
            #     delayed(self.load_graph)(idx) for idx in tqdm(range(len(self.ligand_fnames)), desc="Checking graphs")
            # )
            # self.data = Parallel(n_jobs=8)(
            #     delayed(self.load_graph)(idx, check=True) for idx in tqdm(range(len(self.ligand_fnames)), desc="Checking graphs")
            # )
            self.data = [
                idx for idx in tqdm(range(len(self.ligand_fnames)), desc="Checking graphs")
            ]
        else:
            # self.data = [
            #     self.process_graph(idx)
            #     for idx in tqdm(range(len(self.ligand_fnames)), desc="Processing graphs")
            # ]
            self.data = Parallel(n_jobs=8)(
                delayed(self.process_graph)(idx) for idx in tqdm(range(len(self.ligand_fnames)), desc="Processing graphs")
            )

        # # Filter valid entries
        # valid_data = [
        #     (self.ligand_fnames[i],
        #     self.receptor_fnames[i],
        #     self.ligand_nums[i],
        #     self.labels[i],
        #     i)
        #     for i in enumerate(self.data) if i is not None
        # ]

        # self.ligand_fnames, self.receptor_fnames, self.ligand_nums, self.labels, self.data = zip(*valid_data)
        # self.ligand_fnames = list(self.ligand_fnames)
        # self.receptor_fnames = list(self.receptor_fnames)
        # self.ligand_nums = list(self.ligand_nums)
        # self.labels = list(self.labels)
        # self.data = list(self.data)

    def __len__(self):
        """Return the total size of the dataset."""
        return len(self.data)

    def files_to_inputs(self, lig_fname, rec_fname, ligand_num, item=None):
        parser = StructuralFileParser()
        rec_fname = self.base_path / rec_fname
        lig_fname = self.base_path / lig_fname
        if not lig_fname.is_file():
            raise FileNotFoundError(lig_fname, "does not exist.")
        if not rec_fname.is_file():
            raise FileNotFoundError(rec_fname, "does not exist")
        rec_df = parser.rdkitmol_to_df(rec_fname, mol_type="receptor")
        lig_df = parser.rdkitmol_to_df(lig_fname, mol_type="ligand", num=ligand_num)

        # struct.types is +11 for receptor atoms, i.e. 0 -> 11, 1 -> 12, etc.
        struct = make_box(
            concat_structs(rec_df, lig_df, self.n_features),
            radius=self.radius,
            relative_to_ligand=True,
        )
        # Just in case any get through
        struct = struct[struct["atomic_number"] > 1]
        p = torch.from_numpy(np.vstack([struct["x"], struct["y"], struct["z"]]).T)
        v = make_bit_vector(struct.types.values, self.n_features, self.compact)
        return p.float(), v.float(), struct

    def process_graph(self, item):
        try:
            # item = self.ligand_fnames.index(item)
            lig_fname = Path(self.ligand_fnames[item])
            rec_fname = Path(self.receptor_fnames[item])
            lig_num = self.ligand_nums[item]
            label = self.labels[item]
            p, v, struct = self.files_to_inputs(lig_fname, rec_fname, lig_num, item=item)
            edge_radius = self.edge_radius if self.edge_radius > 0 else 4
            intra_radius = 2.0
            if self.edge_radius >= 0:
                struct, edge_indices, edge_attrs = generate_edges(
                    struct,
                    inter_radius=edge_radius,
                    intra_radius=intra_radius,
                )
                edge_indices = torch.from_numpy(np.vstack(edge_indices)).long()
                edge_attrs = one_hot(torch.from_numpy(edge_attrs).long(), 3)

            else:
                edge_indices, edge_attrs = torch.ones(1), torch.ones(1)

            try:
                y = torch.from_numpy(np.array(label))
                # y = y.long()
            except (TypeError, AttributeError):
                y = label
            return Data(
                x=v,
                edge_index=edge_indices,
                edge_attr=edge_attrs,
                pos=p,
                y=y,
                rec_fname=rec_fname,
                lig_fname=lig_fname,
                lig_num=lig_num,
            )
        except Exception as e:
            # print(self.ligand_fnames[item], self.ligand_nums[item], e)
            return None


    def save_graph(self, item):
        ligand_num = self.ligand_nums[item]
        ligand_path = Path(self.ligand_fnames[item])
        ligand_dir = ligand_path.parent
        ligand_stem = ligand_path.stem  # filename without suffix

        save_dir = Path(self.save_name) / f"graph_{ligand_dir}"
        save_path = save_dir / f"{ligand_stem}_{ligand_num}.pt"

        # if save_path.exists():
        #     return

        try:
            data_obj = self.process_graph(item)
            if not save_path.exists():
                save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(data_obj, save_path)
        except Exception as e:
            print(item, e)
    
    def save_graph_batch(self, items):
        # Use the -2 directory of the first item as the LMDB path name
        lmdb_dir = str(Path(self.ligand_fnames[items[0]]).parents[0])[:2]
        lmdb_path = Path(self.save_name) / f"{lmdb_dir}.lmdb"
        lmdb_path.parent.mkdir(parents=True, exist_ok=True)
        
        if lmdb_path.exists():
            return

        env = lmdb.open(str(lmdb_path), map_size=1 << 40)

        with env.begin(write=True) as txn:
            for idx in items:
                data_obj = self.process_graph(idx)
                key = self.ligand_fnames[idx].encode()
                txn.put(key, pickle.dumps(data_obj))

        env.close()
        print(f"Saved {len(items)} graphs to {lmdb_path}")


    # def load_graph(self, item, check=False):
    #     try:
    #         ligand_num = self.ligand_nums[item]
    #         ligand_path = Path(self.ligand_fnames[item])
    #         ligand_dir = ligand_path.parent
    #         ligand_stem = ligand_path.stem  # filename without suffix
    #         save_dir = Path(self.save_name) / f"graph_{ligand_dir}"
    #         save_path = save_dir / f"{ligand_stem}_{ligand_num}.pt"
    #         graph = torch.load(save_path, weights_only=False)
    #         if check:
    #             return item if graph is not None else None
    #         return graph
    #     except Exception as e:
    #         print(item, e)
    #         return None
    
    def load_graph(self, item, check=False):
        try:
            ligand_path = Path(self.ligand_fnames[item])
            key = self.ligand_fnames[item].encode()

            # Get LMDB directory based on your batch key logic
            lmdb_key = str(ligand_path.parents[0])[:2]  # Same key logic as in save_graph_batch
            lmdb_path = Path(self.save_name) / f"{lmdb_key}.lmdb"

            # Open LMDB environment and load entry
            env = lmdb.open(str(lmdb_path), readonly=True, lock=False)
            with env.begin() as txn:
                value = txn.get(key)
                if value is None:
                    raise KeyError(f"Key not found in LMDB: {key.decode()}")
                graph = pickle.loads(value)

            if check:
                return item if graph is not None else None
            return graph

        except Exception as e:
            print(f"Error loading item {item} from LMDB:", e)
            return None
    
    def __getitem__(self, item):
        if self.save:
            graph = self.load_graph(item)
            if graph is None:
                random_item = np.random.randint(0, len(self.data))
                return self.__getitem__(random_item)
            return graph
        else:
            graph = self.data[item]
            if graph is None:
                random_item = np.random.randint(0, len(self.data))
                return self.__getitem__(random_item)
            return graph


class TrainPoseDataModule(BaseDataModule):
    def __init__(
        self,
        task,
        data_path,
        train_df,
        val_df,
        datamodule_config,
        loader_config,
        task_config,
    ) -> None:

        super().__init__(datamodule_config, task)
        self.train_dataset = PoseDataset(
            data_path=data_path,
            input_df=train_df,
            save_name=datamodule_config.data_cache + "_train",
        )
        self.val_dataset = PoseDataset(
            data_path=data_path,
            input_df=val_df,
            save_name=datamodule_config.data_cache + "_val",
        )  # , **datamodule_config.dataset
        self.loader_config = loader_config

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            # collate_fn=self.safe_collate,
            **self.loader_config,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            # collate_fn=self.safe_collate,
            **self.loader_config,
        )


def module_factory(
    task_config,
    datamodule_config,
    loader_config,
):
    train_df = pd.read_csv(datamodule_config.dataset.pop("train_df"))
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    val_df = pd.read_csv(datamodule_config.dataset.pop("val_df"))
    val_df = val_df.sample(frac=1).reset_index(drop=True)

    return TrainPoseDataModule(
        data_path=datamodule_config.dataset.pop("data_path"),
        train_df=train_df,
        val_df=val_df,
        task=task_config,
        datamodule_config=datamodule_config,
        loader_config=loader_config,
        task_config=task_config,
    )


registry = {"base": partial(module_factory)}
