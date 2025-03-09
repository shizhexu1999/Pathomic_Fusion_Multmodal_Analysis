"""!@file main_1_fold.py
@breif This file is used to train and testthe model on the 1 fold cross validation.
@details The file uses the dataloader, options and train_test files. It is a simplified version of the main.py file.
@author Shizhe Xu
@date 28 June 2024
"""

import os
import sys
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(module_path)
import os
import logging
import numpy as np
import pickle

import torch

from dataloader import *
from options import parse_args
from train_test import train, test

# 1. Initializes parser and device
opt = parse_args()
if opt.gpu_ids and torch.cuda.is_available():
    device = torch.device(f"cuda:{opt.gpu_ids[0]}")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
# device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
print("Using device:", device)
if not os.path.exists(opt.checkpoints_dir):
    os.makedirs(opt.checkpoints_dir)
if not os.path.exists(os.path.join(opt.checkpoints_dir, opt.exp_name)):
    os.makedirs(os.path.join(opt.checkpoints_dir, opt.exp_name))
if not os.path.exists(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name)):
    os.makedirs(os.path.join(opt.checkpoints_dir, opt.exp_name, opt.model_name))

# 2. Initializes Data
ignore_missing_histype = 1 if "grad" in opt.task else 0
ignore_missing_moltype = 1 if "omic" in opt.mode else 0
use_patch, roi_dir = (
    ("_patch_", "all_st_patches_512") if opt.use_vgg_features else ("_", "all_st")
)
use_rnaseq = "_rnaseq" if opt.use_rnaseq else ""
# load the specific data split file
data_cv_path = "%s/splits/gbmlgg1cv_%s_%d_%d_%d%s.pkl" % (
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
# the results stores the final output we look for
results = []

# 3. Sets-Up Main Loop
first_key = next(iter(data_cv_splits))
data = data_cv_splits[first_key]

print("*******************************************")
print("************** SPLIT (1/1) **************")
print("*******************************************")
if os.path.exists(
    os.path.join(
        opt.checkpoints_dir,
        opt.exp_name,
        opt.model_name,
        "%s_%d_patch_pred_train.pkl" % (opt.model_name, first_key),
    )
):
    print("Train-Test Split already made.")
else:
    # 3.1 Train the model
    # model, optimizer, metric_logger = train(opt, data, device, first_key)

    # add try-except block to avoid NaNs during the training
    max_attempts = 5
    attempts = 0
    while attempts < max_attempts:
        try:
            model, optimizer, metric_logger = train(opt, data, device, first_key)
            # exit the loop on successful training
            break
        except ValueError as e:
            if "NaNs detected in inputs, please correct or drop" in str(e):
                print(f"Warning: NaNs detected in inputs on attempt {attempts + 1}, attempting to rerun the current split.")
            else:
                raise
        attempts += 1
    
    # 3.2 Use the trained model to predict on the training and testing set
    (
        loss_train,
        cindex_train,
        pvalue_train,
        surv_acc_train,
        grad_acc_train,
        pred_train,
    ) = test(opt, model, data, "train", device)
    loss_test, cindex_test, pvalue_test, surv_acc_test, grad_acc_test, pred_test = test(
        opt, model, data, "test", device
    )
    if opt.task == "surv":
        print(
            "[Final] Apply model to training set: C-Index: %.10f, P-Value: %.10e"
            % (cindex_train, pvalue_train)
        )
        logging.info(
            "[Final] Apply model to training set: C-Index: %.10f, P-Value: %.10e"
            % (cindex_train, pvalue_train)
        )
        print(
            "[Final] Apply model to testing set: C-Index: %.10f, P-Value: %.10e"
            % (cindex_test, pvalue_test)
        )
        logging.info(
            "[Final] Apply model to testing set: cC-Index: %.10f, P-Value: %.10e"
            % (cindex_test, pvalue_test)
        )
        results.append(cindex_test)

    elif opt.task == "grad":
        print(
            "[Final] Apply model to training set: Loss: %.10f, Acc: %.4f"
            % (loss_train, grad_acc_train)
        )
        logging.info(
            "[Final] Apply model to training set: Loss: %.10f, Acc: %.4f"
            % (loss_train, grad_acc_train)
        )
        print(
            "[Final] Apply model to testing set: Loss: %.10f, Acc: %.4f"
            % (loss_test, grad_acc_test)
        )
        logging.info(
            "[Final] Apply model to testing set: Loss: %.10f, Acc: %.4f"
            % (loss_test, grad_acc_test)
        )
        results.append(grad_acc_test)

    # 3.3 Saves Model
    if len(opt.gpu_ids) > 0 and torch.cuda.is_available():
        model_state_dict = model.module.cpu().state_dict()
    else:
        model_state_dict = model.cpu().state_dict()

    torch.save(
        {
            "split": 1,
            "opt": opt,
            "epoch": opt.niter + opt.niter_decay,
            "data": data,
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metric_logger,
        },
        os.path.join(
            opt.checkpoints_dir,
            opt.exp_name,
            opt.model_name,
            "%s_%d.pt" % (opt.model_name, 1),
        ),
    )
    print()

    pickle.dump(
        pred_train,
        open(
            os.path.join(
                opt.checkpoints_dir,
                opt.exp_name,
                opt.model_name,
                "%s_%d%spred_train.pkl" % (opt.model_name, 1, use_patch),
            ),
            "wb",
        ),
    )
    pickle.dump(
        pred_test,
        open(
            os.path.join(
                opt.checkpoints_dir,
                opt.exp_name,
                opt.model_name,
                "%s_%d%spred_test.pkl" % (opt.model_name, 1, use_patch),
            ),
            "wb",
        ),
    )

print("Split Results:", results)
print("Average:", np.array(results).mean())
pickle.dump(
    results,
    open(
        os.path.join(
            opt.checkpoints_dir,
            opt.exp_name,
            opt.model_name,
            "%s_results.pkl" % opt.model_name,
        ),
        "wb",
    ),
)
