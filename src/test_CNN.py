"""!@file test_CNN.py
@brief This file contains the test function for the CNN model.
@author Shizhe Xu
@date 28 June 2024
"""

import pickle
import os
import torch
import random
import numpy as np
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from dataloader import PathgraphomicDatasetLoader, PathgraphomicFastDatasetLoader, PathgraphomicFastDatasetLoader_CNN
from tqdm import tqdm

from networks import define_net, define_reg, define_optimizer, define_scheduler
from functions import (
    count_parameters,
    mixed_collate,
    unfreeze_unimodal,
    CoxLoss,
    CIndex_lifeline,
    cox_log_rank,
    accuracy_cox,
)

# data comes directly from the split previously made
def test(opt, model, data, split, device):
    model.eval()

    custom_data_loader = (
        PathgraphomicFastDatasetLoader_CNN(opt, data, split, mode=opt.mode)
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=custom_data_loader,
        batch_size=opt.batch_size,
        shuffle=False,
        collate_fn=mixed_collate,
    )

    risk_pred_all, censor_all, survtime_all = np.array([]), np.array([]), np.array([])
    probs_all, gt_all = None, np.array([])
    # loss_epoch, grad_acc_epoch = 0, 0 in the training function
    loss_test, grad_acc_test = 0, 0

    for batch_idx, (x_path, x_hazard, x_grph, x_omic, censor, survtime, grade) in enumerate(
        test_loader
    ):

        censor = censor.to(device) if "surv" in opt.task else censor
        grade = grade.to(device) if "grad" in opt.task else grade



        pred = x_hazard
        pred = pred.to(device)
        # print(pred.shape)
        # print(grade.shape)
        # _, pred = model(
        #     x_path=x_path.to(device), x_grph=x_grph.to(device), x_omic=x_omic.to(device)
        # )

        loss_cox = CoxLoss(survtime, censor, pred, device) if opt.task == "surv" else 0
        loss_reg = define_reg(opt, model)
        
        pred = pred.squeeze(1)

        loss_nll = F.nll_loss(pred, grade) if opt.task == "grad" else 0
        loss = (
            opt.lambda_cox * loss_cox
            + opt.lambda_nll * loss_nll
            + opt.lambda_reg * loss_reg
        )
        loss_test += loss.data.item()

        gt_all = np.concatenate((gt_all, grade.detach().cpu().numpy().reshape(-1)))

        if opt.task == "surv":
            risk_pred_all = np.concatenate(
                (risk_pred_all, pred.detach().cpu().numpy().reshape(-1))
            )
            censor_all = np.concatenate(
                (censor_all, censor.detach().cpu().numpy().reshape(-1))
            )
            survtime_all = np.concatenate(
                (survtime_all, survtime.detach().cpu().numpy().reshape(-1))
            )
        elif opt.task == "grad":
            grade_pred = pred.argmax(dim=1, keepdim=True)
            grad_acc_test += grade_pred.eq(grade.view_as(grade_pred)).sum().item()
            probs_np = pred.detach().cpu().numpy()
            probs_all = (
                probs_np
                if probs_all is None
                else np.concatenate((probs_all, probs_np), axis=0)
            )

    loss_test /= len(test_loader)
    # same as in the training function
    cindex_test = (
        CIndex_lifeline(risk_pred_all, censor_all, survtime_all)
        if opt.task == "surv"
        else None
    )
    pvalue_test = (
        cox_log_rank(risk_pred_all, censor_all, survtime_all)
        if opt.task == "surv"
        else None
    )
    surv_acc_test = (
        accuracy_cox(risk_pred_all, censor_all) if opt.task == "surv" else None
    )
    grad_acc_test = (
        grad_acc_test / len(test_loader.dataset) if opt.task == "grad" else None
    )

    pred_test = [risk_pred_all, survtime_all, censor_all, probs_all, gt_all]

    return loss_test, cindex_test, pvalue_test, surv_acc_test, grad_acc_test, pred_test