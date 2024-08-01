import torch
import torch.nn as nn
from src.utils.utility import to_numpy


class PygLinearPass(nn.Module):
    """Helper class for neater forward passes.

    Gives a linear layer with the same semantic behaviour as the E_GCL and
    EGNN_Sparse layers.

    Arguments:
        module: nn.Module (usually a linear layer)
        feats_appended_to_coords: does the input include coordinates in the
            first three columns of the node feature vector
        return_coords_and_edges: return a tuple containing the node features,
            the coords and the edges rather than just the node features
    """

    def __init__(
        self, module, feats_appended_to_coords=False, return_coords_and_edges=False
    ):
        super().__init__()
        self.m = module
        self.feats_appended_to_coords = feats_appended_to_coords
        self.return_coords_and_edges = return_coords_and_edges
        self.intermediate_coords = None

    def forward(self, h, **kwargs):
        if self.feats_appended_to_coords:
            self.intermediate_coords = to_numpy(h[:, :3])
            feats = h[:, 3:]
            res = torch.hstack([h[:, :3], self.m(feats)])
        else:
            self.intermediate_coords = to_numpy(kwargs["coord"])
            res = self.m(h)
        if self.return_coords_and_edges:
            return (
                res,
                kwargs["coord"],
                kwargs["edge_attr"],
                kwargs.get("edge_messages", None),
            )
        return res