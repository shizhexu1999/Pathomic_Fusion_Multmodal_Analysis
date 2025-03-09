"""!@file make_splits_whole_data.py
@brief This file is used to create the data splits for the whole dataset.
@details It does not assign train and test groups, making all patients in the same group for IG global explanation.
@author Shizhe Xu
@date 28 June 2024
"""

import os
import sys
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(module_path)
import argparse
import os
import pickle
import torch
import numpy as np
import pandas as pd
from PIL import Image
from sklearn import preprocessing
from networks import define_net
from functions import getCleanAllDataset
from torchvision import transforms
import os
from options import parse_args

opt = parse_args()

# get pathology features (vgg features)
def get_vgg_features(model, device, img_path):
    if model is None:
        return img_path
    else:
        x_path = Image.open(img_path).convert("RGB")
        # do the normalization
        normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ],
        )
        # add a single dimension to the tensor at `dim=0`
        # it might be unnecessary to extract hazard score here
        x_path = torch.unsqueeze(normalize(x_path), dim=0)
        features, hazard = model(x_path=x_path.to(device))
        return features.cpu().detach().numpy()
    

def getAlignedMultimodalData(opt, model, device, all_dataset, pat_split, id2img):
    x_patname, x_path, x_grph, x_omic, e, t, g = [], [], [], [], [], [], []
    for pat_name in pat_split:
        if pat_name not in all_dataset.index:
            continue

        for img_fname in id2img[pat_name]:
            grph_fname = img_fname.rstrip(".png") + ".pt"
            # Check if the graph file name is correctly identified
            assert grph_fname in os.listdir(
                os.path.join(opt.dataroot, "%s_%s" % (opt.roi_dir, opt.graph_feat_type))
            )
            # Ensure exactly one row in all_dataset where the 'TCGA ID' matches the specified pat_name
            assert all_dataset[all_dataset["TCGA ID"] == pat_name].shape[0] == 1

            x_patname.append(pat_name)
            x_path.append(
                get_vgg_features(
                    model,
                    device,
                    os.path.join(opt.dataroot, opt.roi_dir, img_fname),
                )
            )
            x_grph.append(
                os.path.join(
                    opt.dataroot,
                    "%s_%s" % (opt.roi_dir, opt.graph_feat_type),
                    grph_fname,
                )
            )
            x_omic.append(
                np.array(
                    all_dataset[all_dataset["TCGA ID"] == pat_name].drop(
                        metadata, axis=1
                    ),
                )
            )
            e.append(int(all_dataset[all_dataset["TCGA ID"] == pat_name]["censored"].iloc[0]))
            t.append(
                int(all_dataset[all_dataset["TCGA ID"] == pat_name]["Survival months"].iloc[0])
            )
            g.append(int(all_dataset[all_dataset["TCGA ID"] == pat_name]["Grade"].iloc[0]))

    return x_patname, x_path, x_grph, x_omic, e, t, g

if opt.gpu_ids and torch.cuda.is_available():
    device = torch.device(f"cuda:{opt.gpu_ids[0]}")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

metadata, all_dataset = getCleanAllDataset(
    opt.dataroot,
    opt.ignore_missing_moltype,
    opt.ignore_missing_histype,
    opt.use_rnaseq,
)

if opt.histomolecular_type == "idhwt_ATC":
    all_dataset = all_dataset[all_dataset["Histomolecular subtype"] == "idhwt_ATC"]
elif opt.histomolecular_type == "idhmut_ATC":
    all_dataset = all_dataset[all_dataset["Histomolecular subtype"] == "idhmut_ATC"]
elif opt.histomolecular_type == "ODG":
    all_dataset = all_dataset[all_dataset["Histomolecular subtype"] == "ODG"]
elif opt.histomolecular_type == "None":
    all_dataset = all_dataset
else:
    raise ValueError("Unknown histomolecular subtype")
# print(all_dataset.shape)
# consturct a map from TCGA ID to ROI
img_fnames = os.listdir(os.path.join(opt.dataroot, opt.roi_dir))
id2img = {}
# we take the firts twelve characters of the file name (TCGA ID)
for id, img_fname in zip([img_fname[:12] for img_fname in img_fnames], img_fnames):
    if id not in id2img.keys():
        id2img[id] = []
        id2img[id].append(img_fname)

# create the dictionary file containing split information
data_dict = {}
data_dict["data_pd"] = all_dataset
cv_splits = {}

pnas_splits = pd.read_csv(opt.dataroot + "pnas_splits.csv")
pnas_splits.columns = ["TCGA ID"] + [str(k) for k in range(1, 16)]
pnas_splits.index = pnas_splits["TCGA ID"]
pnas_splits = pnas_splits.drop(["TCGA ID"], axis=1)

# combine all the data into a single set
all_patients = pnas_splits.index

model = None
if opt.use_vgg_features:
    # load the trained CNN model (pathology features)
    load_path = os.path.join(
        opt.checkpoints_dir,
        opt.exp_name,
        opt.model_name,
        # you can use any other trained pathology model here
        "%s_1.pt" % opt.model_name,
    )
    model_ckpt = torch.load(load_path, map_location=device)
    model_state_dict = model_ckpt["model_state_dict"]
    if hasattr(model_state_dict, "_metadata"):
        del model_state_dict._metadata
    # use the path model here (vgg)
    model = define_net(opt, None)
    # enable parallel processing on multiple GPUs
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model.load_state_dict(model_state_dict)
    model.eval()

(
    all_x_patname,
    all_x_path,
    all_x_grph,
    all_x_omic,
    all_e,
    all_t,
    all_g,
) = getAlignedMultimodalData(opt, model, device, all_dataset, all_patients, id2img)

all_x_omic, all_e, all_t = (
    np.array(all_x_omic).squeeze(axis=1),
    np.array(all_e, dtype=np.float64),
    np.array(all_t, dtype=np.float64),
)

# Using a standard scaler after removing the extra dimension for the omic data
scaler = preprocessing.StandardScaler().fit(all_x_omic)
all_x_omic = scaler.transform(all_x_omic)

all_data = {
    "x_patname": all_x_patname,
    "x_path": np.array(all_x_path),
    "x_grph": all_x_grph,
    "x_omic": all_x_omic,
    "e": np.array(all_e, dtype=np.float64),
    "t": np.array(all_t, dtype=np.float64),
    "g": np.array(all_g, dtype=np.float64),
}

dataset = {"all": all_data}
cv_splits[1] = dataset
# Store the combined data in the data dictionary
data_dict["cv_splits"] = cv_splits

pickle.dump(
    data_dict,
    open(
        "%s/splits_IG/gbmlgg15cv_%s_%d_%d_%d%s_%s.pkl"
        % (
            opt.dataroot,
            opt.roi_dir,
            opt.ignore_missing_moltype,
            opt.ignore_missing_histype,
            opt.use_vgg_features,
            "_rnaseq" if opt.use_rnaseq else "",
            opt.histomolecular_type,
        ),
        "wb",
    ),
)

# pickle.dump(
#     data_dict,
#     open(
#         "%s/splits_IG/gbmlgg15cv_%s_%d_%d_%d%s.pkl"
#         % (
#             opt.dataroot,
#             opt.roi_dir,
#             opt.ignore_missing_moltype,
#             opt.ignore_missing_histype,
#             opt.use_vgg_features,
#             "_rnaseq" if opt.use_rnaseq else "",
#         ),
#         "wb",
#     ),
# )