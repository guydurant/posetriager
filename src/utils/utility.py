import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import torch
import torch.nn.functional as F


def to_numpy(torch_tensor):
    """Switch from a torch tensor to a numpy array (on cpu)."""
    return torch_tensor.detach().cpu().numpy()
