"""!@file IG_draft.py
@breif This file is used to do visulisation via Integrated Gradients.
@details This is a draft file for Integrated Gradients.
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
import torch
from tqdm import tqdm
import pickle
import torch.nn.functional as F

def integrated_gradients(model, input_tensor, baseline=None, num_steps=50):
    device = input_tensor.device
    # print(input_tensor.shape)
    # torch.Size([64, 320])
    if baseline is None:
        baseline = torch.zeros_like(input_tensor).to(device)
    
    assert input_tensor.shape == baseline.shape

    # do the scaling here
    scaled_inputs = [baseline + (float(i) / num_steps) * (input_tensor - baseline) for i in range(num_steps + 1)]
    # print(len(scaled_inputs))
    # length is 51
    scaled_inputs = torch.stack(scaled_inputs).to(device)
    # print(scaled_inputs.shape)
    # torch.Size([51, 64, 320])

    # scaled_inputs.requires_grad = True
    # scaled_inputs.requires_grad_(True)
    # print(scaled_inputs[1].shape)
    # torch.Size([64, 320])
    # forward
    # features, outputs = model(x_omic=scaled_inputs.to(device))
    
    # compute gradients
    grads = []
    for i in range(num_steps + 1):
        # scaled_inputs[i].requires_grad_(True)
        model.zero_grad()
        features, output = model(x_omic=scaled_inputs[i])
        # output = outputs[i]
        # print(output.shape)
        # torch.Size([64, 1])
        grad_outputs = torch.ones_like(output).to(device)
        grad_inputs = grad(outputs=output, inputs=scaled_inputs[i], grad_outputs=grad_outputs, create_graph=True)[0]
        grads.append(grad_inputs)
        
    grads = torch.stack(grads)
    
    # compute average gradient and approximate integral
    avg_grads = (grads[:-1] + grads[1:]) / 2.0
    integrated_grads = (input_tensor - baseline) * avg_grads.mean(dim=0)
    
    return integrated_grads

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
    # _, pred = model(
    #     x_path=x_path.to(device), x_grph=x_grph.to(device), x_omic=x_omic.to(device)
    # )
    input_tensor = x_omic.to(device)
    attributions = integrated_gradients(model, input_tensor)

        