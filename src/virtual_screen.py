import os
from src.models.posetriager import PoseTriager
from omegaconf import DictConfig
import torch
from torch import nn
import hydra
import wandb
import pandas as pd
from tqdm import tqdm
import os
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")
from src.inference import get_dataloader
from rdkit import Chem
from io import StringIO
from joblib import Parallel, delayed

def process_ligand(lig_file, temp_dir):
    count = 0
    current_inchi = ""
    mols =  Chem.SDMolSupplier(lig_file)
    data = {}
    for i, m in tqdm(enumerate(mols), total=len(mols)):
        inchi = Chem.MolToInchiKey(m)
        if not os.path.exists(f"{temp_dir}/{inchi}"):
            os.makedirs(f"{temp_dir}/{inchi}")
        if inchi != current_inchi:
            data[current_inchi] = count
            count = 0
            current_inchi = inchi
        count += 1
        if not os.path.exists(f"{temp_dir}/{inchi}/{count}.mol"):
            Chem.MolToMolFile(m, f"{temp_dir}/{inchi}/{count}.mol")
    data.pop("")
    return data

def process_docks(cache_data_dir, ligand_file, protein_file, inchikeys_list=[]):
    cache_data_dir = cache_data_dir.replace("/scratch", "")
    if not os.path.exists(cache_data_dir):
        os.makedirs(cache_data_dir)
    if os.path.exists(cache_data_dir+".tar.gz"):
        print(f"Unzipping {cache_data_dir}.tar.gz")
        os.system(f"tar -xzf {cache_data_dir}.tar.gz -C {cache_data_dir}")
    lig_data = process_ligand(ligand_file, cache_data_dir)
    df = "num,protein,ligand\n"
    for inchi, count in lig_data.items():
        # print(f"Processing {inchi}")
        if len(inchikeys_list) == 0:
            for i in range(1, count+1):
                df += f"{i},{protein_file},{cache_data_dir}/{inchi}/{i}.mol\n"
        elif inchi.split('-')[0] in inchikeys_list:
            for i in range(1, count+1):
                df += f"{i},{protein_file},{cache_data_dir}/{inchi}/{i}.mol\n"
    return df


def single_virtual_screen_from_config(config, row, out_df, inchikeys_list=[]):
    if os.path.exists(out_df):
        print(f"Output file {out_df} already exists, exiting.")
        return
    individual_df = process_docks(row["cache_dir"], row["ligand"], row["protein"], inchikeys_list=inchikeys_list)
    checkpoint = torch.load(
        config.checkpoint, map_location=torch.device(config.accelerator), weights_only=False
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
        pd.read_csv(StringIO(individual_df)),
        batch_size=config.batch_size,
        data_cache=None,
        save=False,
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
    out_df_ = pd.DataFrame(
        {
            "probs": predictions,
            "label": true_values,
            "ligand": ligand_names,
            "protein": protein_names,
        }
    )
    if out_df is None:
        return None
    if not os.path.exists("/".join(out_df.split("/")[:-1])):
        os.makedirs("/".join(out_df.split("/")[:-1]))
    out_df_.to_csv(out_df, index=False)
    return None


@hydra.main(version_base=None, config_path="../configs", config_name="virtual_screen.yaml")
def main(config: DictConfig):
    model_name = "postriager"
    data_df = pd.read_csv(config.data_df)
    if config.inchikeys_list is not None:
        with open(config.inchikeys_list) as f:
            inchikeys_list = list(set([i.split('-')[0] for i in f.read().splitlines()]))
        inchikeys_list = inchikeys_list[:1000]
    else:
        inchikeys_list = set()
    Parallel(n_jobs=30)(delayed(single_virtual_screen_from_config)(config, row, "/".join(row["results"].split("/")[:-1])+f"/{model_name}_{row["results"].split("/")[-1]}", inchikeys_list=inchikeys_list) for i, row in tqdm(data_df.iterrows(), total=len(data_df)))


def hydra_main():
    main()


if __name__ == "__main__":
    try:
        hydra_main()
    finally:
        wandb.finish()