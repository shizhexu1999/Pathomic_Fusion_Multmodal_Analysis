"""!@file test_15_fold.py
@brief This file is used to test CNN model.
@author Shizhe Xu
@date 28 June 2024
"""

import os
import logging
import numpy as np
import pickle

import torch

from dataloader import *
from options import parse_args
from test_CNN import test
from networks import define_net

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
data_cv_path = "%s/splits_original/gbmlgg15cv_CNN_%s_%d_%d_%d%s.pkl" % (
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

for k, data in data_cv_splits.items():
    print("*******************************************")
    print(
        "************** SPLIT (%d/%d) **************" % (k, len(data_cv_splits.items()))
    )
    print("*******************************************")
    load_path = os.path.join(
        opt.checkpoints_dir,
        opt.exp_name,
        opt.model_name,
        "%s_%d.pt" % (opt.model_name, k),
    )
    model_ckpt = torch.load(load_path, map_location=device)

    model_state_dict = model_ckpt["model_state_dict"]
    if hasattr(model_state_dict, "_metadata"):
        del model_state_dict._metadata

    model = define_net(opt, None)
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    print("Loading the model from %s" % load_path)
    model.load_state_dict(model_state_dict)

    loss_test, cindex_test, pvalue_test, surv_acc_test, grad_acc_test, pred_test = test(
        opt, model, data, "test", device
    )

    if opt.task == "surv":
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
            "[Final] Apply model to testing set: Loss: %.10f, Acc: %.4f"
            % (loss_test, grad_acc_test)
        )
        logging.info(
            "[Final] Apply model to testing set: Loss: %.10f, Acc: %.4f"
            % (loss_test, grad_acc_test)
        )
        results.append(grad_acc_test)

    pickle.dump(
        pred_test,
        open(
            os.path.join(
                opt.checkpoints_dir,
                opt.exp_name,
                opt.model_name,
                "%s_%d%spred_test.pkl" % (opt.model_name, k, use_patch),
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
