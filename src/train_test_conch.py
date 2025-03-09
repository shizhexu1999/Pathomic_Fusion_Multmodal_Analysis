"""!@file train_test_conch.py
@brief This file contains the training and testing functions for the CONCH model.
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
from dataloader import PathgraphomicDatasetLoader, PathgraphomicFastDatasetLoader
from tqdm import tqdm

from networks_conch import define_net_conch, define_reg, define_optimizer, define_scheduler
from functions import (
    count_parameters,
    mixed_collate,
    unfreeze_unimodal,
    CoxLoss,
    CIndex_lifeline,
    cox_log_rank,
    accuracy_cox,
)


def train(opt, data, device, k):
    # this code can be added to the train_cv.py file
    # device = torch.device(
    #     "cuda"
    #     if torch.cuda.is_available()
    #     else "mps" if torch.backends.mps.is_available() else "cpu"
    # )
    # print("Using device:", device)

    # set deterministic behavior based on the device type
    # if device.type == "cuda":
    #     cudnn.deterministic = True
    #     torch.cuda.manual_seed_all(2019)
    # elif device.type == "mps":
    #     pass

    # set seeds
    cudnn.deterministic = True
    torch.cuda.manual_seed_all(2019)
    torch.manual_seed(2019)
    random.seed(2019)

    # specify the model we use
    model = define_net_conch(opt, k)
    optimizer = define_optimizer(opt, model)
    scheduler = define_scheduler(opt, optimizer)
    print(model)
    print("Number of Trainable Parameters: %d" % count_parameters(model))
    print("Activation Type:", opt.act_type)
    print("Optimizer Type:", opt.optimizer_type)
    print("Regularization Type:", opt.reg_type)

    # specify if the patches are used, and ROI directory
    use_patch, roi_dir = (
        ("_patch_", "all_st_patches_512") if opt.use_vgg_features else ("_", "all_st")
    )
    custom_data_loader = (
        PathgraphomicFastDatasetLoader(opt, data, split="train", mode=opt.mode)
        if opt.use_vgg_features
        else PathgraphomicDatasetLoader(opt, data, split="train", mode=opt.mode)
    )

    # use our customised train loader
    train_loader = torch.utils.data.DataLoader(
        dataset=custom_data_loader,
        batch_size=opt.batch_size,
        shuffle=True,
        collate_fn=mixed_collate,
    )
    metric_logger = {
        "train": {
            "loss": [],
            "pvalue": [],
            "cindex": [],
            "surv_acc": [],
            "grad_acc": [],
        },
        "test": {
            "loss": [],
            "pvalue": [],
            "cindex": [],
            "surv_acc": [],
            "grad_acc": [],
        },
    }

    # default range is (1, 26)
    for epoch in tqdm(range(opt.epoch_count, opt.niter + opt.niter_decay + 1)):
        # if finetuining is enabled/done, unfreeze the unimodal model
        if opt.finetune == 1:
            unfreeze_unimodal(opt, model, epoch)

        model.train()

        # introduce metrics to calculate the C-index
        risk_pred_all, censor_all, survtime_all = (
            np.array([]),
            np.array([]),
            np.array([]),
        )
        loss_epoch, grad_acc_epoch = 0, 0

        for batch_idx, (x_path, x_grph, x_omic, censor, survtime, grade) in enumerate(
            train_loader
        ):
            censor = censor.to(device) if "surv" in opt.task else censor
            grade = grade.to(device) if "grad" in opt.task else grade
            # _ represents for features here
            _, pred = model(
                x_path=x_path.to(device),
                x_grph=x_grph.to(device),
                x_omic=x_omic.to(device),
            )

            # print(len(pred))

            loss_cox = (
                CoxLoss(survtime, censor, pred, device) if opt.task == "surv" else 0
            )
            loss_reg = define_reg(opt, model)
            # negative log-likelihood loss between the predicted outputs (pred) and the true labels (grade)
            # pred is hazard?
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
                pred = pred.argmax(dim=1, keepdim=True)
                grad_acc_epoch += pred.eq(grade.view_as(pred)).sum().item()

            if (
                opt.verbose > 0
                and opt.print_every > 0
                and (
                    batch_idx % opt.print_every == 0
                    or batch_idx + 1 == len(train_loader)
                )
            ):
                print(
                    "Epoch {:02d}/{:02d} Batch {:04d}/{:d}, Loss {:9.4f}".format(
                        epoch + 1,
                        opt.niter + opt.niter_decay,
                        batch_idx + 1,
                        len(train_loader),
                        loss.item(),
                    )
                )
        scheduler.step()

        # logging the results of both training and testing datasets
        # we can speed up our code by setting opt.measure to 0 (default is 1)
        # opt.niter+opt.niter_decay - 1 = 24, which is the last epoch


        if opt.measure or epoch == (opt.niter + opt.niter_decay - 1):

            loss_epoch /= len(train_loader)
            cindex_epoch = (
                CIndex_lifeline(risk_pred_all, censor_all, survtime_all)
                if opt.task == "surv"
                else None
            )
            pvalue_epoch = (
                cox_log_rank(risk_pred_all, censor_all, survtime_all)
                if opt.task == "surv"
                else None
            )
            surv_acc_epoch = (
                accuracy_cox(risk_pred_all, censor_all) if opt.task == "surv" else None
            )
            grad_acc_epoch = (
                grad_acc_epoch / len(train_loader.dataset)
                if opt.task == "grad"
                else None
            )
            # test function is defined below
            (
                loss_test,
                cindex_test,
                pvalue_test,
                surv_acc_test,
                grad_acc_test,
                pred_test,
            ) = test(opt, model, data, "test", device)

            metric_logger["train"]["loss"].append(loss_epoch)
            metric_logger["train"]["cindex"].append(cindex_epoch)
            metric_logger["train"]["pvalue"].append(pvalue_epoch)
            metric_logger["train"]["surv_acc"].append(surv_acc_epoch)
            metric_logger["train"]["grad_acc"].append(grad_acc_epoch)

            metric_logger["test"]["loss"].append(loss_test)
            metric_logger["test"]["cindex"].append(cindex_test)
            metric_logger["test"]["pvalue"].append(pvalue_test)
            metric_logger["test"]["surv_acc"].append(surv_acc_test)
            metric_logger["test"]["grad_acc"].append(grad_acc_test)

            pickle.dump(
                pred_test,
                open(
                    os.path.join(
                        opt.checkpoints_dir,
                        opt.exp_name,
                        opt.model_name,
                        "%s_%d%s%d_pred_test.pkl"
                        % (opt.model_name, k, use_patch, epoch),
                    ),
                    "wb",
                ),
            )

            if opt.verbose > 0:
                if opt.task == "surv":
                    print(
                        "[{:s}]\t\tLoss: {:.4f}, {:s}: {:.4f}".format(
                            "Train", loss_epoch, "C-Index", cindex_epoch
                        )
                    )
                    print(
                        "[{:s}]\t\tLoss: {:.4f}, {:s}: {:.4f}\n".format(
                            "Test", loss_test, "C-Index", cindex_test
                        )
                    )
                elif opt.task == "grad":
                    print(
                        "[{:s}]\t\tLoss: {:.4f}, {:s}: {:.4f}".format(
                            "Train", loss_epoch, "Accuracy", grad_acc_epoch
                        )
                    )
                    print(
                        "[{:s}]\t\tLoss: {:.4f}, {:s}: {:.4f}\n".format(
                            "Test", loss_test, "Accuracy", grad_acc_test
                        )
                    )

            if opt.task == "grad" and loss_epoch < opt.patience:
                print("Early stopping at Epoch %d" % epoch)
                break

    return model, optimizer, metric_logger


def test(opt, model, data, split, device):
    model.eval()

    custom_data_loader = (
        PathgraphomicFastDatasetLoader(opt, data, split, mode=opt.mode)
        if opt.use_vgg_features
        else PathgraphomicDatasetLoader(opt, data, split=split, mode=opt.mode)
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

    for batch_idx, (x_path, x_grph, x_omic, censor, survtime, grade) in enumerate(
        test_loader
    ):

        censor = censor.to(device) if "surv" in opt.task else censor
        grade = grade.to(device) if "grad" in opt.task else grade
        _, pred = model(
            x_path=x_path.to(device), x_grph=x_grph.to(device), x_omic=x_omic.to(device)
        )

        loss_cox = CoxLoss(survtime, censor, pred, device) if opt.task == "surv" else 0
        loss_reg = define_reg(opt, model)
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
