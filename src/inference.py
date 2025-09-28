from src.models.posetriager import PoseTriager
from src.datamodules.datamodule import PoseDataset
from omegaconf import DictConfig
import torch
from torch import nn
import hydra
import wandb
from torch_geometric.loader import DataLoader
import pandas as pd
from tqdm import tqdm
import os
from sklearn.metrics import (
    matthews_corrcoef as matt_corr,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score as roc_auc,
    mean_absolute_error as mae,
    mean_squared_error as mse,
    r2_score,
    root_mean_squared_error,
)
from scipy.stats import pearsonr, spearmanr
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")


def print_classification_summary_statistics(predictions, true_values):
    # for just probabilities
    for metric in [roc_auc]:
        print(metric.__name__)
        print(metric(true_values, predictions))
    # for just binary
    binary_predictions = [1 if pred > 0.5 else 0 for pred in predictions]
    for metric in [matt_corr, f1_score, precision_score, recall_score]:
        print(metric.__name__)
        print(metric(true_values, binary_predictions))


def print_regression_summary_statistics(predictions, true_values):
    for metric in [mae, mse, r2_score, root_mean_squared_error, pearsonr, spearmanr]:
        print(metric.__name__)
        print(metric(true_values, predictions))


summary_statistics = {
    "pbvalidity": print_classification_summary_statistics,
    "pbclassify": print_classification_summary_statistics,
    "interaction_similarity": print_regression_summary_statistics,
}

def get_dataloader(data_path, data_df, batch_size=1, data_cache=None, save=True):
    dataset = PoseDataset(
        data_path=data_path,
        input_df=data_df,
        save_name=data_cache,
        save=save,
    )
    
    # def none_collate_fn(batch):
    #     batch = [graph for graph in batch if graph is not None]
    #     if len(batch) == 0:
    #         return Batch.from_data_list([Data(x=None, y=None)])
    #     return Batch.from_data_list(batch)
    
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        # collate_fn=none_collate_fn,
    )

def inference_from_config(config):
    if os.path.exists(config.out_df):
        print(f"Output file {config.out_df} already exists, exiting.")
        return
    checkpoint = torch.load(
        config.checkpoint, map_location=torch.device(config.accelerator), weights_only=False,
    )
    model_hparams = checkpoint["hyper_parameters"]
    model = PoseTriager(model_hparams["model"])
    state_dict = checkpoint["state_dict"]
    new_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(
        new_state_dict,
        strict=True,
    )
    model.eval()
    dataloader = get_dataloader(
        config.data_path,
        pd.read_csv(config.data_df),
        batch_size=config.batch_size,
        data_cache=config.data_cache,
    )
    predictions = []
    true_values = []
    ligand_names = []
    protein_names = []
    sigmoid = nn.Sigmoid()
    with torch.no_grad():
        for batch in tqdm(dataloader):
            preds = sigmoid(model(batch))
            predictions.extend([pred.item() for pred in preds])
            true_values.extend(batch.y.cpu().numpy())
            ligand_names.extend(batch.lig_fname)
            protein_names.extend(batch.rec_fname)

    # summary_statistics[config.task](predictions, true_values)
    out_df = pd.DataFrame(
        {
            "probs": predictions,
            "label": true_values,
            "ligand": ligand_names,
            "protein": protein_names,
        }
    )
    if config.out_df is None:
        return None
    if not os.path.exists("/".join(config.out_df.split("/")[:-1])):
        os.makedirs("/".join(config.out_df.split("/")[:-1]))
    out_df.to_csv(config.out_df, index=False)
    print(f"Output file {config.out_df} saved.")
    return None


@hydra.main(version_base=None, config_path="../configs", config_name="inference.yaml")
def main(config: DictConfig):
    inference_from_config(config)


def hydra_main():
    main()


if __name__ == "__main__":
    try:
        hydra_main()
    finally:
        wandb.finish()
