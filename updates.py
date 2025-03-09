import torch
import torch_geometric
import os
from tqdm import tqdm


def reformat_graph_data(data):
    """
    ensure that the object structure is correct
    """
    # use `data.py` module within the `data` subpackage
    return torch_geometric.data.data.Data.from_dict(data.__dict__)


for file in tqdm(os.listdir("data/TCGA_GBMLGG/all_st_cpc_old")):
    old_data = torch.load(
        os.path.join("data/TCGA_GBMLGG/all_st_cpc_old", file),
    )
    torch.save(
        reformat_graph_data(old_data),
        os.path.join("data/TCGA_GBMLGG/all_st_cpc", file),
    )
