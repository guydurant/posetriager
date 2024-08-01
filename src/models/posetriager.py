"""Equivariant graph neural network class.

EGNNLayer is modified from the code released with the original EGNN paper,
found at https://github.com/vgsatorras/egnn.
"""

import torch
import os
import math
from torch import nn
from torch_geometric.nn import global_mean_pool
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from src.utils.utility import to_numpy
from configs.schemas import ModelConfig
from src.models.egnn_layer import EGNNLayer
from src.models.linear_layer import PygLinearPass


from pathlib import Path
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PoseTriager(nn.Module):
    """Equivariant network based on EGNNLayer."""

    def __init__(
        self,
        config: ModelConfig,
    ):
        super().__init__()
        self.batch = 0
        self.epoch = 0
        self.save_path = Path(config.save_path).expanduser()
        self.only_save_best_models = config.only_save_best_models
        self.lr = config.learning_rate
        self.weight_decay = config.weight_decay
        self.n_layers = config.num_layers
        self.layers = self.build_net(
            dim_input=config.dim_input,
            k=config.k,
            num_layers=config.num_layers,
        )
        self.optimiser = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=config.weight_decay
        )

        # pc = self.param_count
        self.to(DEVICE)

    def forward(self, x):
        batch_size = torch.max(x.batch.int()).long() + 1
        feats, edges, coords, edge_attributes, batch = self.unpack_graph(x)
        feats, _ = self.get_embeddings(feats, edges, coords, edge_attributes, batch)
        feats = global_mean_pool(feats, batch, size=batch_size)  # (total_nodes, k)
        feats = self.feats_linear_layers(feats)  # (bs, k)
        return feats

    def unpack_input_data_and_predict(self, input_data):
        """See base class."""
        y_true = input_data.y
        try:
            y_true = y_true.float()
        except (AttributeError, TypeError):
            raise ValueError("y_true must be a float tensor.")
        y_pred = self(input_data).reshape(
            -1,
        )
        ligands = input_data.lig_fname
        receptors = input_data.rec_fname
        return y_pred, y_true, ligands, receptors

    def unpack_graph(self, graph):
        return (
            graph.x.float().to(DEVICE),
            graph.edge_index.to(DEVICE),
            graph.pos.float().to(DEVICE),
            graph.edge_attr.to(DEVICE),
            graph.batch.to(DEVICE),
        )

    # pylint: disable = R, W0201, W0613
    def build_net(
        self,
        dim_input: int,
        k: int,
        num_layers: int = 4,
    ):
        layers = [PygLinearPass(nn.Linear(dim_input, k), return_coords_and_edges=True)]
        self.n_layers = num_layers

        for _ in range(0, num_layers):
            layers.append(
                EGNNLayer(
                    k,
                    k,
                    k,
                    edges_in_d=3,
                )
            )
        self.feats_linear_layers = nn.Sequential(
            *[nn.Linear(k, 1), nn.Sigmoid()]
        )  # DO NOT LEAVE
        # self.feats_linear_layers = nn.Sequential(*[nn.Linear(k, 1)])
        return nn.Sequential(*layers)

    def get_embeddings(self, feats, edges, coords, edge_attributes, batch):
        edge_messages = None
        for i in self.layers:
            feats, coords, edge_attributes, edge_messages = i(
                h=feats,
                edge_index=edges,
                coord=coords,
                edge_attr=edge_attributes,
                edge_messages=edge_messages,
            )
        return feats, edge_messages

    def train_model(self, data_loader, epochs=1, epoch_end_validation_set=None):
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimiser, T_0=len(data_loader), T_mult=1, eta_min=0
        )
        epoch = 0
        for _ in tqdm(range(self.epoch, epochs)):
            self.train()
            epoch += 1
            for self.batch, graph in tqdm(enumerate(data_loader)):
                y_pred, y_true, _, _ = self.unpack_input_data_and_predict(graph)
                loss_ = self.backprop(y_true, y_pred)
                self.scheduler.step()
                # TODO Log to wandb
            self.eval()
            self.on_epoch_end(
                epoch_end_validation_set=epoch_end_validation_set,
                epochs=epochs,
            )

    def val(self, data_loader):
        self.eval()
        all_preds = []
        all_true = []
        with torch.no_grad():
            for self.batch, graph in tqdm(
                enumerate(data_loader), total=len(data_loader)
            ):
                y_pred, y_true, _, _ = self.unpack_input_data_and_predict(graph)
                y_true_np = to_numpy(y_true).reshape((-1,))
                all_true.extend(list(y_true_np))
                y_pred_np = to_numpy(y_pred).reshape((-1,))
                all_preds.extend(list(y_pred_np))
        return all_preds, all_true

    def backprop(self, y_true, y_pred):
        loss = nn.BCEWithLogitsLoss(y_pred, y_true.to(DEVICE))
        self.optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.parameters(), 1.0)
        self.optimiser.step()
        loss_ = float(to_numpy(loss))
        if math.isnan(loss_):
            raise ValueError("Loss is NaN, exiting.")
        return loss_

    def on_epoch_end(self, epoch_end_validation_set, epochs):
        self.epoch += 1
        epoch = self.epoch
        if not self.only_save_best_models:
            self.save()
        # end of epoch validation if requested
        if epoch_end_validation_set is not None and epoch < epochs:
            best = self.val(epoch_end_validation_set)
            if self.only_save_best_models and best:
                self.save()

    def save(self, save_data_dir):
        fname = f"classification_ckpt_epoch_{self.epoch}.pt"
        if not os.path.exists(save_data_dir):
            os.makedirs(save_data_dir)
        save_path = save_data_dir / "checkpoints" / fname
        if not os.path.exists(save_path.parent):
            os.makedirs(save_path.parent)
        torch.save(
            {
                "learning_rate": self.lr,
                "weight_decay": self.weight_decay,
                "epoch": self.epoch,
                "model_state_dict": self.state_dict(),
                "optimiser_state_dict": self.optimiser.state_dict(),
            },
            save_path,
        )

    def load_weights(self, checkpoint_file):
        checkpoint = torch.load(str(checkpoint_file), map_location=DEVICE)
        self.load_state_dict(checkpoint["model_state_dict"])
        self.optimiser.load_state_dict(checkpoint["optimiser_state_dict"])
        self.epoch = checkpoint.get("epoch", 0)

    @property
    def param_count(self):
        return sum([torch.numel(t) for t in self.parameters() if t.requires_grad])


registry = {
    "pose_triager": PoseTriager,
}
