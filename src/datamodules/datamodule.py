from functools import partial
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from src.datamodules.base_module import BaseDataModule
from src.datamodules.complex_features import (
    concat_structs,
    extract_coords,
    make_box,
    generate_edges,
    make_bit_vector,
)
from src.datamodules.smina_types import StructuralFileParser
from torch.utils.data import Dataset
from torch.nn.functional import one_hot
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data


class PoseDataset(Dataset):
    def __init__(
        self,
        data_path,
        input_df,
        radius=6,
        edge_radius=10,
        **kwargs,
    ):
        super().__init__()
        self.radius = radius
        self.base_path = Path(data_path).expanduser()
        self.edge_radius = edge_radius
        self.parser = StructuralFileParser()
        if not self.base_path.exists():
            raise FileNotFoundError(f"Dataset {self.base_path} does not exist.")
        self.compact = True

        labels = []
        labels = input_df["label"].values
        self.receptor_fnames = input_df["protein"].to_list()
        self.ligand_fnames = input_df["ligand"].to_list()
        if labels[0] is not None:
            active_count = np.sum(labels)
            class_sample_count = np.array([len(labels) - active_count, active_count])
            weights = 1.0 / class_sample_count
            self.sample_weights = torch.from_numpy(
                np.array([weights[i] for i in labels])
            )
            self.sampler = torch.utils.data.WeightedRandomSampler(
                self.sample_weights, len(self.sample_weights)
            )
        self.labels = labels
        # print counts of active and inactive
        print(
            f"Active: {active_count}, Inactive: {len(labels) - active_count}, Total: {len(labels)}"
        )

        self.n_features = 11  # + 8 * extended_atom_types
        self.feature_dim = self.n_features + 1

    def __len__(self):
        """Return the total size of the dataset."""
        return len(self.ligand_fnames)

    def files_to_inputs(self, lig_fname, rec_fname, item=None):
        rec_fname = self.base_path / rec_fname
        lig_fname = self.base_path / lig_fname
        if not lig_fname.is_file():
            raise FileNotFoundError(lig_fname, "does not exist.")
        if not rec_fname.is_file():
            raise FileNotFoundError(rec_fname, "does not exist")
        rec_df = self.parser.rdkitmol_to_df(rec_fname, mol_type="receptor")
        lig_df = self.parser.rdkitmol_to_df(lig_fname, mol_type="ligand")

        # struct.types is +11 for receptor atoms, i.e. 0 -> 11, 1 -> 12, etc.
        struct = make_box(
            concat_structs(rec_df, lig_df, self.n_features),
            radius=self.radius,
            relative_to_ligand=True,
        )
        # Just in case any get through
        struct = struct[struct["atomic_number"] > 1]
        p = torch.from_numpy(np.vstack([struct["x"], struct["y"], struct["z"]]).T)
        v = make_bit_vector(struct.types.to_numpy(), self.n_features, self.compact)
        return p.float(), v.float(), struct

    def __getitem__(self, item):
        lig_fname = Path(self.ligand_fnames[item])
        rec_fname = Path(self.receptor_fnames[item])
        label = self.labels[item]
        p, v, struct = self.files_to_inputs(lig_fname, rec_fname, item=item)
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
            y = y.long()
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
        )


class PoseDataModule(BaseDataModule):
    def __init__(
        self,
        task,
        data_path,
        train_df,
        val_df,
        datamodule_config,
        loader_config,
    ) -> None:

        super().__init__(datamodule_config, task)
        self.train_dataset = PoseDataset(
            data_path=data_path, input_df=train_df  # , **datamodule_config.dataset
        )
        self.val_dataset = PoseDataset(
            data_path=data_path, input_df=val_df
        )  # , **datamodule_config.dataset
        self.loader_config = loader_config

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            **self.loader_config,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            **self.loader_config,
        )


def module_factory(
    task_config,
    datamodule_config,
    loader_config,
):

    train_df = pd.read_csv(datamodule_config.dataset.pop("train_df"))
    val_df = pd.read_csv(datamodule_config.dataset.pop("val_df"))

    return PoseDataModule(
        data_path=datamodule_config.dataset.pop("data_path"),
        train_df=train_df,
        val_df=val_df,
        task=task_config,
        datamodule_config=datamodule_config,
        loader_config=loader_config,
    )


registry = {"base": partial(module_factory)}
