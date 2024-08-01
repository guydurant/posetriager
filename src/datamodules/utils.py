import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import torch
import torch.nn.functional as F


def to_numpy(torch_tensor):
    """Switch from a torch tensor to a numpy array (on cpu)."""
    return torch_tensor.detach().cpu().numpy()


def concat_structs(
    rec_struct,
    lig_struct,
    n_features,
    # min_lig_rotation=0, parsers=None, extended=False
):
    """Concatenate the receptor and ligand parquet structures."""
    rec_struct.types += n_features  # + extended * 8

    concatted_structs = pd.concat([lig_struct, rec_struct], ignore_index=True)
    return concatted_structs


def extract_coords(struct, bp=None):
    """Get numpy coordinates from pd.DataFrame."""
    entity = struct[(struct.bp == bp)] if bp is not None else struct
    return np.vstack([entity.x.to_numpy(), entity.y.to_numpy(), entity.z.to_numpy()]).T


def make_box(struct, radius=4, relative_to_ligand=True):
    ligand_np = extract_coords(struct, 0)
    receptor_np = extract_coords(struct, 1)
    if relative_to_ligand:
        result = struct[struct.bp == 0].copy()
        rec_struct = struct[struct.bp == 1].copy()
        rec_struct.reset_index(inplace=True)
        distances = cdist(ligand_np, receptor_np, "euclidean")
        mask = distances < radius
        keep = np.where(np.sum(mask, axis=0))[0]
        result = pd.concat(
            [result, rec_struct[rec_struct.index.isin(keep)]], ignore_index=True
        )
        result.reset_index(drop=True, inplace=True)
        del result["index"]
        return result

    ligand_centre = np.mean(ligand_np, axis=0)

    struct["sq_dist"] = (
        (struct.x - ligand_centre[0]) ** 2
        + (struct.y - ligand_centre[1]) ** 2
        + (struct.z - ligand_centre[2]) ** 2
    )

    struct = struct[(struct.sq_dist < radius**2) | (struct.bp == 0)].copy()
    struct.reset_index(drop=True, inplace=True)
    del struct["sq_dist"]
    try:
        del struct["index"]
    except KeyError:
        pass
    return struct


def make_bit_vector(atom_types, n_atom_types, compact=True):
    if compact:
        # atom_types: 0 -> 0 0, 11 -> 0 1
        indices = torch.from_numpy(atom_types % n_atom_types).long()
        one_hot = F.one_hot(indices, num_classes=n_atom_types + 1)
        type_bit = torch.from_numpy((atom_types // n_atom_types)).int()
        one_hot[:, -1] = type_bit
    else:
        one_hot = F.one_hot(torch.from_numpy(atom_types), num_classes=n_atom_types * 2)
    return one_hot


def generate_edges(struct, inter_radius=4.0, intra_radius=2.0):
    struct.reset_index(inplace=True, drop=True)
    coords = extract_coords(struct)
    lig_or_rec = struct.bp.to_numpy()
    distances = cdist(coords, coords, "euclidean")
    adj_inter = (distances < inter_radius) & (distances > 1e-7)
    edge_indices_inter = np.where(adj_inter)
    inter_mask = abs(
        lig_or_rec[edge_indices_inter[0]] - lig_or_rec[edge_indices_inter[1]]
    )
    edge_indices_inter = (
        edge_indices_inter[0][np.where(inter_mask)],
        edge_indices_inter[1][np.where(inter_mask)],
    )
    n_edges_inter = sum(inter_mask)
    adj_intra = (distances < intra_radius) & (distances > 1e-7)
    n_edges_intra = np.sum(adj_intra)
    edge_indices_intra = np.where(adj_intra)
    bp_0_inter = lig_or_rec[edge_indices_inter[0]]
    bp_1_inter = lig_or_rec[edge_indices_inter[1]]
    bp_0_intra = lig_or_rec[edge_indices_intra[0]]
    bp_1_intra = lig_or_rec[edge_indices_intra[1]]
    edge_attrs_inter = np.zeros((n_edges_inter,), dtype="int32")
    edge_attrs_intra = np.zeros((n_edges_intra,), dtype="int32")
    edge_attrs_inter[np.where((bp_0_inter == 0) & (bp_1_inter == 1))] = 1
    edge_attrs_inter[np.where((bp_0_inter == 1) & (bp_1_inter == 0))] = 1
    edge_attrs_intra[np.where((bp_0_intra == 1) & (bp_1_intra == 1))] = 2
    edge_attrs = np.concatenate([edge_attrs_inter, edge_attrs_intra])
    edge_indices = (
        np.concatenate([edge_indices_inter[0], edge_indices_intra[0]]),
        np.concatenate([edge_indices_inter[1], edge_indices_intra[1]]),
    )
    return struct, edge_indices, edge_attrs
