"""!@file IG_bilinear.py
@breif This file is used to create global explanations using the integrated gradients method.
@details The integrated gradients method is used to generate global explanations by using CNN+GCN+SNN models.
It takes longer time to run, and we use smaller batch size due to the memory constraints.
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
from torch_geometric.data import Batch
import random

random_seed = 2001
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)

def predictions_and_gradients(inputs_path, inputs_grph, inputs_omic, target_label_index, model):
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
    inputs_path = torch.tensor(inputs_path).to(device)
    # inputs_grph = torch.tensor(inputs_grph).to(device)
    inputs_grph.x = inputs_grph.x.to(device)
    # inputs_grph.edge_index = inputs_grph.edge_index.to(device)
    # inputs_grph.edge_attr = inputs_grph.edge_attr.to(device)
    inputs_omic = torch.tensor(inputs_omic, requires_grad=True).to(device)  # Ensure inputs are on the MPS device
    # features, predictions = model(x_path=x_path, x_grph=x_grph, x_omic=inputs)
    features, predictions = model(x_path=inputs_path, x_grph = inputs_grph, x_omic=inputs_omic)
    gradients = grad(
        outputs=predictions,
        inputs=inputs_omic,
        grad_outputs=torch.ones_like(predictions).to(device),  # Ensure grad_outputs are on the MPS device
        create_graph=True
    )[0]
    return predictions.cpu().detach().numpy(), gradients.cpu().detach().numpy()

def integrated_gradients(
    inp_path,
    inp_grph,
    inp_omic, 
    target_label_index,
    predictions_and_gradients,
    baseline_path=None,
    baseline_grph=None,
    baseline_omic=None,
    steps=50):
    """Computes integrated gradients for a given network and prediction label."""
    if baseline_path is None:
        baseline_path = torch.zeros_like(inp_path)
    assert baseline_path.shape == inp_path.shape

    # if baseline_grph is None:
    #     baseline_grph = torch.zeros_like(inp_grph)
    # assert baseline_grph.shape == inp_grph.shape

    if baseline_grph is None:
        baseline_grph = Batch(
            x=torch.zeros_like(inp_grph.x),
            edge_index=inp_grph.edge_index,
            edge_attr=torch.zeros_like(inp_grph.edge_attr),
            batch=inp_grph.batch,
            ptr=inp_grph.ptr
        )
    assert baseline_grph.x.shape == inp_grph.x.shape

    if baseline_omic is None:
        baseline_omic = torch.zeros_like(inp_omic)
    assert baseline_omic.shape == inp_omic.shape

    scaled_inputs_path = [baseline_path + (float(i)/steps)*(inp_path-baseline_path) for i in range(steps + 1)]
    scaled_inputs_path = torch.stack(scaled_inputs_path).to(device)  # Convert to tensor and move to MPS device
    print(scaled_inputs_path.shape)

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

    scaled_inputs_omic = [baseline_omic + (float(i)/steps)*(inp_omic-baseline_omic) for i in range(steps + 1)]
    scaled_inputs_omic = torch.stack(scaled_inputs_omic).to(device)  # Convert to tensor and move to MPS device
    print(scaled_inputs_omic.shape)

    combined_grph = Batch.from_data_list(scaled_inputs_grph_list)

    predictions, grads = predictions_and_gradients(scaled_inputs_path.cpu().numpy(), combined_grph, scaled_inputs_omic.cpu().numpy(), target_label_index)  # shapes: <steps+1>, <steps+1, inp.shape>

    # Use trapezoidal rule to approximate the integral.
    grads = torch.tensor(grads).to(device)  # Convert gradients to tensor and move to MPS device
    grads = (grads[:-1] + grads[1:]) / 2.0
    avg_grads = grads.mean(dim=0)
    integrated_gradients = (inp_omic - baseline_omic) * avg_grads  # shape: <inp.shape>
    
    return integrated_gradients.cpu().detach().numpy(), predictions


def pad_arrays(arrays):
    max_length = max(len(arr) for arr in arrays)
    return np.array([np.pad(arr, (0, max_length - len(arr)), 'constant', constant_values=np.nan) for arr in arrays])

# Function to visualize integrated gradients in a SHAP-styled plot
def plot_shap_style_integrated_gradients(attributions, features, top_n=20, xlim=(-2, 2), all_dataset=None, save_folder="./visualisation_plots/", filename="shap_style_plot.png",):
    """
    Plots the top N integrated gradients in a SHAP-styled plot.
    
    Args:
    - attributions: Array of integrated gradients with shape (64, 320).
    - features: Array of original input features with shape (64, 320).
    - top_n: Number of top features to plot.
    - feature_names: List of feature names for the x-axis labels.
    - xlim: Tuple specifying the x-axis limits.
    """
    # Aggregate attributions across samples (e.g., mean)
    # all_dataset = all_dataset[all_dataset['Histomolecular subtype'] == 'idhwt_ATC']
    # idhmut_ATC_dataset = all_dataset[all_dataset['Histomolecular subtype'] == 'idhmut_ATC']
    # ODG_dataset = all_dataset[all_dataset['Histomolecular subtype'] == 'ODG']
    drop_data = [
            "Histology",
            "Grade",
            "Molecular subtype",
            "censored",
            "TCGA ID",
            "Survival months",
            "Histomolecular subtype"
        ]
    all_dataset = all_dataset.drop(drop_data, axis=1)
    feature_names = all_dataset.columns.tolist()

    mean_attributions = np.mean(attributions, axis=0)
    
    # Get the top N features
    top_indices = np.argsort(np.abs(mean_attributions))[-top_n:][::-1]
    
    # if feature_names is False:
    #     feature_names = [f'Feature {i}' for i in range(mean_attributions.shape[0])]
    
    top_feature_names = [feature_names[i] for i in top_indices]
    
    # Prepare data for SHAP plot
    top_attributions = attributions[:, top_indices]
    top_features = features[:, top_indices]
    
    # Filter attributions to only include values within the specified range
    filtered_attributions = []
    filtered_features = []
    for i in range(top_n):
        mask = (top_attributions[:, i] >= xlim[0]) & (top_attributions[:, i] <= xlim[1])
        filtered_attributions.append(top_attributions[mask, i])
        filtered_features.append(top_features[mask, i])
    
    # Pad filtered arrays to the same length
    filtered_attributions = pad_arrays(filtered_attributions).T
    filtered_features = pad_arrays(filtered_features).T
    
    # Create SHAP summary plot
    # plt.figure(figsize=(10, 8))
    shap.summary_plot(filtered_attributions, filtered_features, feature_names=top_feature_names, plot_type="dot", max_display=top_n, show=False)
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, filename)
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()

# def plot_shap_style_integrated_gradients(attributions, features, top_n=20, feature_names=None):
#     """
#     Plots the top N integrated gradients in a SHAP-styled plot.
    
#     Args:
#     - attributions: Array of integrated gradients with shape (64, 320).
#     - features: Array of original input features with shape (64, 320).
#     - top_n: Number of top features to plot.
#     - feature_names: List of feature names for the x-axis labels.
#     - xlim: Tuple specifying the x-axis limits.
#     """
#     # Aggregate attributions across samples (e.g., mean)
#     mean_attributions = np.mean(attributions, axis=0)
    
#     # Get the top N features
#     top_indices = np.argsort(np.abs(mean_attributions))[-top_n:][::-1]
    
#     if feature_names is None:
#         feature_names = [f'Feature {i}' for i in range(mean_attributions.shape[0])]
    
#     top_feature_names = [feature_names[i] for i in top_indices]
    
#     # Prepare data for SHAP plot
#     top_attributions = attributions[:, top_indices]
#     top_features = features[:, top_indices]
    
#     # Create SHAP summary plot
#     plt.figure(figsize=(8, 8))
#     shap.summary_plot(top_attributions, top_features, feature_names=top_feature_names, plot_type="dot", max_display=top_n)


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

# model.eval()

# specify if the patches are used, and ROI directory
ignore_missing_histype = 1 if "grad" in opt.task else 0
ignore_missing_moltype = 1 if "omic" in opt.mode else 0
use_patch, roi_dir = (
    ("_patch_", "all_st_patches_512") if opt.use_vgg_features else ("_", "all_st")
)
use_rnaseq = "_rnaseq" if opt.use_rnaseq else ""

data_cv_path = "%s/splits_IG/gbmlgg15cv_%s_%d_%d_%d%s_%s.pkl" % (
    opt.dataroot,
    roi_dir,
    ignore_missing_moltype,
    ignore_missing_histype,
    opt.use_vgg_features,
    use_rnaseq,
    opt.histomolecular_type,
)
print("Loading %s" % data_cv_path)
data_cv = pickle.load(open(data_cv_path, "rb"))
data_cv_splits = data_cv["cv_splits"]

first_key = next(iter(data_cv_splits))
data = data_cv_splits[first_key]

custom_data_loader_train = (
    PathgraphomicFastDatasetLoader(opt, data, split="all", mode=opt.mode)
    if opt.use_vgg_features
    else PathgraphomicDatasetLoader(opt, data, split="all", mode=opt.mode)
)
custom_data_loader_test = (
    PathgraphomicFastDatasetLoader(opt, data, split="all", mode=opt.mode)
    if opt.use_vgg_features
    else PathgraphomicDatasetLoader(opt, data, split="all", mode=opt.mode)
)
train_loader = torch.utils.data.DataLoader(
    dataset=custom_data_loader_train,
    batch_size=opt.batch_size,
    shuffle=True,
    collate_fn=mixed_collate,
)

test_loader = torch.utils.data.DataLoader(
    dataset=custom_data_loader_test,
    batch_size=opt.batch_size,
    shuffle=False,
    collate_fn=mixed_collate,
)
for epoch in tqdm(range(opt.epoch_count, opt.niter + opt.niter_decay + 1)):
    # if finetuining is enabled/done, unfreeze the unimodal model
    if opt.finetune == 1:
        unfreeze_unimodal(opt, model, epoch)

    model.train()

    loss_epoch = 0
    for batch_idx, (x_path, x_grph, x_omic, censor, survtime, grade) in enumerate(
        train_loader
    ):
        # _, pred = model(
        #     x_path=x_path.to(device),
        #     x_grph=x_grph.to(device),
        #     x_omic=x_omic.to(device),
        # )
        _, pred = model(
            x_path=x_path.to(device),
            x_grph=x_grph.to(device),
            x_omic=x_omic.to(device),
        )
        censor = censor.to(device)
        loss_cox = (
            CoxLoss(survtime, censor, pred, device) if opt.task == "surv" else 0
            )
        loss_reg = define_reg(opt, model)
        loss_nll = F.nll_loss(pred, grade) if opt.task == "grad" else 0
        loss = (
            opt.lambda_cox * loss_cox
            + opt.lambda_nll * loss_nll
            + opt.lambda_reg * loss_reg
        )
        loss_epoch += loss.data.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()
print(loss_epoch)

model.eval()
for batch_idx, (x_path, x_grph, x_omic, censor, survtime, grade) in enumerate(
    test_loader
):
    input_tensor_path = x_path.to(device)
    input_tensor_grph = x_grph.to(device)
    input_tensor_omic = x_omic.to(device)
    # x_path = x_path.unsqueeze(0).expand(51, -1, -1)
    # x_grph = x_grph.unsqueeze(0)
    # x_path = x_path.to(device)
    # x_grph = x_grph.to(device)
    # print(x_grph.shape)
    print(f"input_tensor_path shape", input_tensor_path.shape)
    # print(f"input_tensor_grph shape", input_tensor_grph.shape)
    print(f"input_tensor_omic shape", input_tensor_omic.shape)
    # Define the target label index (not used in regression, but required by function signature)
    target_label_index = 0

    # Call the predictions_and_gradients function
    # predictions, gradients = predictions_and_gradients(input_tensor, target_label_index, model)

    # Check the shapes of the returned predictions and gradients
    # print("Predictions shape:", predictions.shape)  # Expected shape: (batch_size, 1)
    # print("Gradients shape:", gradients.shape)      # Expected shape: (batch_size, input_dim)
    # print(gradients)

    integrated_grads, prediction_trend = integrated_gradients(input_tensor_path, input_tensor_grph, input_tensor_omic,target_label_index, lambda path_inputs, grph_inputs, omic_inputs, label: predictions_and_gradients(path_inputs, grph_inputs, omic_inputs, label, model), baseline_path=None, baseline_grph=None, baseline_omic=None, steps=50)
    print("Integrated gradients shape:", integrated_grads.shape)  # Expected shape: (input_dim,)
    print("Prediction trend shape:", prediction_trend.shape)
    # print(integrated_grads)
    print(len(integrated_grads))
    sample_input_cpu = input_tensor_omic.cpu().detach().numpy()
    break
metadata, all_dataset = getCleanAllDataset(
    opt.dataroot,
    opt.ignore_missing_moltype,
    opt.ignore_missing_histype,
    opt.use_rnaseq,
)
plot_shap_style_integrated_gradients(
    integrated_grads,
    sample_input_cpu,
    top_n=20,
    xlim=(-5, 5),
    all_dataset=all_dataset,
    save_folder="./visualisation_plots/",
    filename="path+graph+omic_%s.png" % opt.histomolecular_type,
)