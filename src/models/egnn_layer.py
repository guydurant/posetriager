import torch
from torch import nn
from src.utils.utility import to_numpy


class EGNNLayer(nn.Module):
    """Modified from https://github.com/vgsatorras/egnn"""

    # pylint: disable = R, W, C
    def __init__(
        self,
        input_nf: int,
        output_nf: int,
        hidden_nf: int,
        edges_in_d: int = 0,
    ):
        super(EGNNLayer, self).__init__()
        input_edge = input_nf * 2
        self.epsilon = 1e-8
        self.att_val = None
        self.node_att_val = None
        self.intermediate_coords = None
        edge_coords_nf = 1
        act_fn = nn.SiLU()

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            nn.Identity(),
            act_fn,
            nn.Linear(hidden_nf, output_nf),
        )

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            layer,
            nn.Tanh(),
        )
        self.att_mlp = nn.Sequential(nn.Linear(hidden_nf, 1), nn.Sigmoid())

    @staticmethod
    def unsorted_segment_sum(data, segment_ids, num_segments):
        result_shape = (num_segments, data.size(1))
        result = data.new_full(result_shape, 0)  # Init empty result tensor.
        segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
        result.scatter_add_(0, segment_ids, data)
        return result

    @staticmethod
    def unsorted_segment_mean(data, segment_ids, num_segments):
        result_shape = (num_segments, data.size(1))
        segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
        result = data.new_full(result_shape, 0)  # Init empty result tensor.
        count = data.new_full(result_shape, 0)
        result.scatter_add_(0, segment_ids, data)
        count.scatter_add_(0, segment_ids, torch.ones_like(data))
        return result / count.clamp(min=1)

    def edge_model(self, source, target, radial, edge_attr):
        inp = [source, target, radial]
        if edge_attr is not None:
            inp.append(edge_attr)
        out = torch.cat(inp, dim=1)
        out = self.edge_mlp(out)
        return out

    def node_model(self, x, edge_index, m_ij):
        row, _ = edge_index
        # edge attention
        att_val = self.att_mlp(m_ij)
        self.att_val = to_numpy(att_val)
        agg = self.unsorted_segment_sum(att_val * m_ij, row, num_segments=x.size(0))

        agg = torch.cat([x, agg], dim=1)

        # Eq. 6: h_i = phi_h(h_i, m_i)
        out = self.node_mlp(agg)
        out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        agg = self.unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        coord += agg
        self.intermediate_coords = to_numpy(coord)
        return coord

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff**2, 1).unsqueeze(1)
        # normalise the coord_diff
        norm = torch.sqrt(radial).detach() + self.epsilon
        coord_diff = coord_diff / norm
        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_attr=None, edge_messages=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)
        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        edge_feat = (
            edge_feat + edge_messages if edge_messages is not None else edge_feat
        )
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        h, agg = self.node_model(h, edge_index, edge_feat)

        return h, coord, edge_attr, edge_feat
