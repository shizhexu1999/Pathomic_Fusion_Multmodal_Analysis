"""!@file IG_updated_draft.py
@breif This draft file of using more refined Integrated Gradients for the model from networks.py.
@author Shizhe Xu
@date 28 June 2024
"""

import os
import sys
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(module_path)
from options import parse_args
from networks import define_net, define_optimizer, define_scheduler, define_reg
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

def predictions_and_gradients(inputs, target_label_index, model):
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
    inputs = torch.tensor(inputs, requires_grad=True).to("mps")  # Ensure inputs are on the MPS device
    features, predictions = model(x_omic=inputs)
    gradients = grad(
        outputs=predictions,
        inputs=inputs,
        grad_outputs=torch.ones_like(predictions).to("mps"),  # Ensure grad_outputs are on the MPS device
        create_graph=True
    )[0]
    return predictions.cpu().detach().numpy(), gradients.cpu().detach().numpy()

def integrated_gradients(
    inp, 
    target_label_index,
    predictions_and_gradients,
    baseline=None,
    steps=50):
    """Computes integrated gradients for a given network and prediction label."""
    if baseline is None:
        baseline = torch.zeros_like(inp)
    assert baseline.shape == inp.shape

    # Scale input and compute gradients.
    scaled_inputs = [baseline + (float(i)/steps)*(inp-baseline) for i in range(steps + 1)]
    scaled_inputs = torch.stack(scaled_inputs).to("mps")  # Convert to tensor and move to MPS device

    predictions, grads = predictions_and_gradients(scaled_inputs.cpu().numpy(), target_label_index)  # shapes: <steps+1>, <steps+1, inp.shape>

    # Use trapezoidal rule to approximate the integral.
    grads = torch.tensor(grads).to("mps")  # Convert gradients to tensor and move to MPS device
    grads = (grads[:-1] + grads[1:]) / 2.0
    avg_grads = grads.mean(dim=0)
    integrated_gradients = (inp - baseline) * avg_grads  # shape: <inp.shape>
    
    return integrated_gradients.cpu().detach().numpy(), predictions

# def plot_top_integrated_gradients(attributions, top_n=20, feature_names=None):
#     """
#     Plots the top N integrated gradients in a SHAP-styled bar plot.
    
#     Args:
#     - attributions: Array of integrated gradients with shape (64, 320).
#     - top_n: Number of top features to plot.
#     - feature_names: List of feature names for the x-axis labels.
#     """
#     # Aggregate attributions across samples (e.g., mean)
#     mean_attributions = np.mean(attributions, axis=0)
    
#     # Get the top N features
#     top_indices = np.argsort(np.abs(mean_attributions))[-top_n:][::-1]
#     top_attributions = mean_attributions[top_indices]
    
#     if feature_names is None:
#         feature_names = [f'Feature {i}' for i in range(len(mean_attributions))]
    
#     top_feature_names = [feature_names[i] for i in top_indices]
    
#     # Create bar plot
#     plt.figure(figsize=(12, 6))
#     plt.bar(range(top_n), top_attributions, color='skyblue')
#     plt.xlabel('Features')
#     plt.ylabel('Attribution')
#     plt.title(f'Top {top_n} Integrated Gradients Attributions')
#     plt.xticks(range(top_n), top_feature_names, rotation=90)
#     plt.show()

# def plot_shap_style_integrated_gradients(attributions, top_n=20, feature_names=None):
#     """
#     Plots the top N integrated gradients in a SHAP-styled plot.
    
#     Args:
#     - attributions: Array of integrated gradients with shape (64, 320).
#     - top_n: Number of top features to plot.
#     - feature_names: List of feature names for the x-axis labels.
#     """
#     # Aggregate attributions across samples (e.g., mean)
#     mean_attributions = np.mean(attributions, axis=0)
    
#     # Get the top N features
#     top_indices = np.argsort(np.abs(mean_attributions))[-top_n:][::-1]
    
#     if feature_names is None:
#         feature_names = [f'Feature {i}' for i in range(mean_attributions.shape[0])]
    
#     top_feature_names = [feature_names[i] for i in top_indices]
    
#     # Prepare data for seaborn plot
#     data = []
#     for i in top_indices:
#         for sample in attributions[:, i]:
#             data.append((top_feature_names[top_indices.tolist().index(i)], sample))
    
#     df = pd.DataFrame(data, columns=['Feature', 'Attribution'])
    
#     plt.figure(figsize=(12, 8))
#     sns.stripplot(x='Attribution', y='Feature', data=df, jitter=True, palette='coolwarm', orient='h')
#     plt.axvline(x=0, color='black', linestyle='--')
#     plt.xlabel('Attribution')
#     plt.ylabel('Features')
#     plt.title(f'Top {top_n} Integrated Gradients Attributions')
#     plt.xlim(-2, 2)
#     plt.show()

def pad_arrays(arrays):
    max_length = max(len(arr) for arr in arrays)
    return np.array([np.pad(arr, (0, max_length - len(arr)), 'constant', constant_values=np.nan) for arr in arrays])

# Function to visualize integrated gradients in a SHAP-styled plot
def plot_shap_style_integrated_gradients(attributions, features, top_n=20, xlim=(-2, 2), all_dataset=None):
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
    all_dataset = all_dataset[all_dataset['Histomolecular subtype'] == 'idhwt_ATC']
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
    plt.figure(figsize=(10, 8))
    shap.summary_plot(filtered_attributions, filtered_features, feature_names=top_feature_names, plot_type="dot", max_display=top_n)

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

model = define_net(opt, None)
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

data_cv_path = "%s/splits_original/gbmlgg15cv_%s_%d_%d_%d%s.pkl" % (
    opt.dataroot,
    roi_dir,
    ignore_missing_moltype,
    ignore_missing_histype,
    opt.use_vgg_features,
    use_rnaseq,
)
print("Loading %s" % data_cv_path)
data_cv = pickle.load(open(data_cv_path, "rb"))
data_cv_splits = data_cv["cv_splits"]

first_key = next(iter(data_cv_splits))
data = data_cv_splits[first_key]

custom_data_loader_train = (
    PathgraphomicFastDatasetLoader(opt, data, split="train", mode=opt.mode)
    if opt.use_vgg_features
    else PathgraphomicDatasetLoader(opt, data, split="train", mode=opt.mode)
)
custom_data_loader_test = (
    PathgraphomicFastDatasetLoader(opt, data, split="test", mode=opt.mode)
    if opt.use_vgg_features
    else PathgraphomicDatasetLoader(opt, data, split="test", mode=opt.mode)
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
        _, pred = model(
            x_path=x_path.to(device),
            x_grph=x_grph.to(device),
            x_omic=x_omic.to(device),
        )
        
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
    input_tensor = x_omic.to(device)
    print(input_tensor)
    # Define the target label index (not used in regression, but required by function signature)
    target_label_index = 0

    # Call the predictions_and_gradients function
    # predictions, gradients = predictions_and_gradients(input_tensor, target_label_index, model)

    # Check the shapes of the returned predictions and gradients
    # print("Predictions shape:", predictions.shape)  # Expected shape: (batch_size, 1)
    # print("Gradients shape:", gradients.shape)      # Expected shape: (batch_size, input_dim)
    # print(gradients)

    integrated_grads, prediction_trend = integrated_gradients(input_tensor, target_label_index, lambda inputs, label: predictions_and_gradients(inputs, label, model), baseline=None, steps=50)
    print("Integrated gradients shape:", integrated_grads.shape)  # Expected shape: (input_dim,)
    print("Prediction trend shape:", prediction_trend.shape)
    # print(integrated_grads)
    print(len(integrated_grads))
    sample_input_cpu = input_tensor.cpu().detach().numpy()
    break
metadata, all_dataset = getCleanAllDataset(
    opt.dataroot,
    opt.ignore_missing_moltype,
    opt.ignore_missing_histype,
    opt.use_rnaseq,
)
plot_shap_style_integrated_gradients(integrated_grads, sample_input_cpu, top_n=20, xlim=(-5, 5), all_dataset=all_dataset)