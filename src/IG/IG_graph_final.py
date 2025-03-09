"""!@file IG_graph_final.py
@breif This file is used to do local explanation for cell graphs, GCN, via Integrated Gradients.
@details In addition to adding the information about the centroid of the graph, we also let the cell graph to overlay on the pathology image.
@author Shizhe Xu
@date 28 June 2024
"""

import os
import sys
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(module_path)
from options import parse_args
from networks_IG import define_net, define_optimizer, define_scheduler, define_reg
from dataloader import PathgraphomicDatasetLoader, PathgraphomicFastDatasetLoader
from functions import (
    count_parameters,
    mixed_collate,
    unfreeze_unimodal,
    CoxLoss,
    CIndex_lifeline,
    cox_log_rank,
    accuracy_cox,
)
from torch.autograd import Variable, grad
from functions import getCleanAllDataset
import torch
from tqdm import tqdm
import pickle
import torch.nn.functional as F
from torch.autograd import grad
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import shap
import os
from torch_geometric.data import Batch, Data
import random
import torch_geometric
import networkx as nx
from PIL import Image

# random_seed = 2001
# random.seed(random_seed)
# np.random.seed(random_seed)
# torch.manual_seed(random_seed)

def predictions_and_gradients(inputs_grph, target_label_index, model):
    """
    Compute predictions and gradients for a batch of inputs with respect to the target label.
    
    Args:
    - inputs: A batch of input tensors.
    - target_label_index: Ignored for regression models like MaxNet.
    - model: The MaxNet model.
    
    Returns:
    - predictions: Predicted values for each input in the batch.
    - gradients: Gradients of the predictions with respect to the inputs.
    """
    # inputs_grph = torch.tensor(inputs_grph).to(device)
    inputs_grph.x = inputs_grph.x.to(device).requires_grad_()
    inputs_grph.edge_index = inputs_grph.edge_index.to(device)
    inputs_grph.edge_attr = inputs_grph.edge_attr.to(device)
    # features, predictions = model(x_path=x_path, x_grph=x_grph, x_omic=inputs)
    features, predictions = model(x_grph = inputs_grph)
    gradients = grad(
        outputs=predictions,
        inputs=inputs_grph.x,
        grad_outputs=torch.ones_like(predictions).to(device),  # Ensure grad_outputs are on the MPS device
        create_graph=True
    )[0]
    return predictions.cpu().detach().numpy(), gradients.cpu().detach().numpy()

def integrated_gradients(
    inp_grph,
    target_label_index,
    predictions_and_gradients,
    baseline_grph=None,
    steps=50):
    """Computes integrated gradients for a given network and prediction label."""

    # if baseline_grph is None:
    #     baseline_grph = torch.zeros_like(inp_grph)
    # assert baseline_grph.shape == inp_grph.shape

    if baseline_grph is None:
        baseline_grph = Data(
            x=torch.zeros_like(inp_grph.x),
            edge_index=inp_grph.edge_index,
            edge_attr=torch.zeros_like(inp_grph.edge_attr),
        )
    assert baseline_grph.x.shape == inp_grph.x.shape

    # DataBatch(x=[12040, 1036], edge_index=[2, 61494], edge_attr=[61494, 1], batch=[12040], ptr=[33])
    # the above reason is why we cannot do the following step
    # scaled_inputs_grph = [baseline_grph + (float(i)/steps)*(inp_grph-baseline_grph) for i in range(steps + 1)]
    # scaled_inputs_grph = torch.stack(scaled_inputs_grph).to(device)  # Convert to tensor and move to MPS device
    # print(scaled_inputs_grph.shape)


    scaled_inputs_grph_list = []
    for i in range(steps + 1):
        scale_factor = float(i) / steps
        scaled_x = baseline_grph.x + scale_factor * (inp_grph.x - baseline_grph.x)
        temp_grph = inp_grph.clone()
        temp_grph.x = scaled_x
        scaled_inputs_grph_list.append(temp_grph)
    print(len(scaled_inputs_grph_list))

    combined_grph = Batch.from_data_list(scaled_inputs_grph_list)

    predictions, grads = predictions_and_gradients(combined_grph, target_label_index)  # shapes: <steps+1>, <steps+1, inp.shape>

    # Use trapezoidal rule to approximate the integral.
    grads = torch.tensor(grads).to(device)  # Convert gradients to tensor and move to MPS device
    grads = (grads[:-1] + grads[1:]) / 2.0
    avg_grads = grads.mean(dim=0)
    integrated_gradients = (inp_grph.x - baseline_grph.x) * avg_grads  # shape: <inp.shape>
    
    return integrated_gradients.cpu().detach().numpy(), predictions


def plot_cell_graph_heatmap(graph_data, node_attributions, node_positions, background_image_path, save_folder="./visualisation_plots/", filename="shap_style_plot.png"):
    G = nx.Graph()
    for i, (attr, pos) in enumerate(zip(node_attributions, node_positions)):
        G.add_node(i, attr=attr, pos=pos)

    edge_index = graph_data.edge_index.cpu().numpy()
    edges = [(edge_index[0, i], edge_index[1, i]) for i in range(edge_index.shape[1])]
    G.add_edges_from(edges)

    pos = nx.get_node_attributes(G, 'pos')
    node_attr = nx.get_node_attributes(G, 'attr')
    node_colors = [node_attr[i] for i in G.nodes]

    # Load background image
    background_image = Image.open(background_image_path)
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(background_image)

    # Convert positions to numpy array
    pos_array = np.array([pos[i] for i in sorted(G.nodes)])

    # Overlay the graph with squares
    sc = ax.scatter(pos_array[:, 0], pos_array[:, 1], c=node_colors, cmap=plt.cm.viridis, marker='s', s=100)

    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=min(node_colors), vmax=max(node_colors)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)

    # remove the ticks on the colour bar
    cbar.set_ticks([])
    
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, filename)
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()


opt = parse_args()
if opt.gpu_ids and torch.cuda.is_available():
    device = torch.device(f"cuda:{opt.gpu_ids[0]}")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

model = define_net(opt, 1)
optimizer = define_optimizer(opt, model)
scheduler = define_scheduler(opt, optimizer)

pt_fname = "_15.pt"
best_grph_ckpt = torch.load(
    os.path.join(
        opt.checkpoints_dir, opt.exp_name, "graph", "graph" + pt_fname
    ),
    map_location=torch.device("cpu"),
)

# check if the pretrained graph net is loaded successfully
# print("Model parameters before loading checkpoint:")
# for name, param in model.named_parameters():
#     print(name, param.data)
model.load_state_dict(best_grph_ckpt["model_state_dict"])
# print("\nModel parameters after loading checkpoint:")
# for name, param in model.named_parameters():
#     print(name, param.data)
print(
    "Loading Models:\n",
    os.path.join(
        opt.checkpoints_dir, opt.exp_name, "graph", "graph" + pt_fname
    ))

model.to(device)
model.eval()

background_image_path = "TCGA-06-0174-01Z-00-DX3.23b6e12e-dfc1-4c6f-903e-170038a0e055_1.png"
graph_path = "TCGA-06-0174-01Z-00-DX3.23b6e12e-dfc1-4c6f-903e-170038a0e055_1.pt"
# graph_path = "data/TCGA_GBMLGG/all_st_patches_512_cpc/TCGA-02-0001-01Z-00-DX1.83fce43e-42ac-4dcd-b156-2908e75f2e47_1_0_0.pt"
graph_data = torch.load(graph_path)
# update the old data version
graph_data = torch_geometric.data.data.Data.from_dict(graph_data.__dict__)
print(graph_data)

graph_data = graph_data.to(device)
if hasattr(graph_data, 'centroid') and graph_data.centroid is not None:
    node_positions = graph_data.centroid.cpu().numpy()
else:
    edge_index = graph_data.edge_index.cpu().numpy()
    G = nx.Graph()
    edges = [(edge_index[0, i], edge_index[1, i]) for i in range(edge_index.shape[1])]
    G.add_edges_from(edges)
    pos = nx.spring_layout(G)
    node_positions = np.array([pos[i] for i in sorted(G.nodes)])

# Compute integrated gradients
target_label_index = 0
integrated_grads, prediction_trend = integrated_gradients(
    graph_data,
    target_label_index,
    lambda grph_inputs, label: predictions_and_gradients(grph_inputs, label, model),
    baseline_grph=None,
    steps=50
)
print(f"Integrated gradients shape: {integrated_grads.shape}")
print(f"Number of nodes in graph: {graph_data.num_nodes}")
if integrated_grads.shape[0] != graph_data.num_nodes:
    print(f"Error: The length of node_attributions ({integrated_grads.shape[0]} does not match the number of nodes ({graph_data.num_nodes}).")
else:
    node_attributions = integrated_grads.mean(axis=1)
    # node_attributions = integrated_grads[:graph_data.num_nodes]
    print(node_attributions.shape)
    plot_cell_graph_heatmap(graph_data, node_attributions, node_positions, background_image_path, save_folder="./visualisation_plots/", filename="IG_graph_final_9_june.png")