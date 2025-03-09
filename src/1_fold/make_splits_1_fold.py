"""!@file make_splits_1_fold.py
@breif This file is used to create the 1 fold cross validation splits.
@details The 1 fold cross validation splits are created using the data from the TCGA GBMLGG dataset.
The data is aligned and the splits are created using the pnas_splits.csv file.
The data is then saved in the splits folder in the data directory.
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

# from options import parse_gpuids


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataroot", type=str, default="./data/TCGA_GBMLGG/", help="datasets"
    )
    parser.add_argument("--roi_dir", type=str, default="all_st")
    parser.add_argument(
        "--graph_feat_type", type=str, default="cpc", help="graph features"
    )
    parser.add_argument(
        "--ignore_missing_moltype",
        type=int,
        default=0,
        help="Ignore data points with missing molecular subtype",
    )
    parser.add_argument(
        "--ignore_missing_histype",
        type=int,
        default=0,
        help="Ignore data points with missign histology subtype",
    )
    parser.add_argument("--use_vgg_features", type=int, default=0)
    parser.add_argument("--use_rnaseq", type=int, default=0)
    parser.add_argument(
        "--checkpoints_dir",
        type=str,
        default="./checkpoints/TCGA_GBMLGG/",
        help="trained models are saved here",
    )
    parser.add_argument("--model_name", type=str, default="path", help="model")
    # this mode enables to use the vgg features from the pathology (CNN) model
    parser.add_argument("--mode", type=str, default="path", help="mode")
    parser.add_argument(
        "--exp_name",
        type=str,
        default="surv_1_rnaseq",
        help="name of the project. It decides where to store samples and models",
    )
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default="0,1,2,3",
        help="gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU",
    )
    parser.add_argument(
        "--act_type", type=str, default="Sigmoid", help="activation function"
    )
    parser.add_argument(
        "--init_type",
        type=str,
        default="none",
        help="network initialization [normal | xavier | kaiming | orthogonal | max]. Max seems to work well",
    )
    parser.add_argument("--path_dim", type=int, default=32)
    parser.add_argument("--label_dim", type=int, default=1, help="size of output")
    parser.add_argument(
        "--init_gain",
        type=float,
        default=0.02,
        help="scaling factor for normal, xavier and orthogonal.",
    )

    # it will hold a namespace object containing all the command-line arguments
    # that the parser recognizes based on the defined argparse.ArgumentParser`.
    opt = parser.parse_known_args()[0]
    return opt


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
        # features, hazard = model(x_path)
        return features.cpu().detach().numpy()


# ensure the data is correctly aligned
def getAlignedMultimodalData(opt, model, device, all_dataset, pat_split, id2img):
    x_patname, x_path, x_grph, x_omic, e, t, g = [], [], [], [], [], [], []
    for pat_name in pat_split:
        if pat_name not in all_dataset.index:
            continue

        for img_fname in id2img[pat_name]:
            grph_fname = img_fname.rstrip(".png") + ".pt"
            # check if the graph file name is correctly identified
            assert grph_fname in os.listdir(
                os.path.join(opt.dataroot, "%s_%s" % (opt.roi_dir, opt.graph_feat_type))
            )
            # exactly one row in all_dataset where the 'TCGA ID'
            # matches the specified pat_name
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


opt = parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "mps")
# build metadata and all_dataset by using the function getCleanAllDataset
metadata, all_dataset = getCleanAllDataset(
    opt.dataroot,
    opt.ignore_missing_moltype,
    opt.ignore_missing_histype,
    opt.use_rnaseq,
)

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

# use the k-fold splis as the paper suggests
pnas_splits = pd.read_csv(opt.dataroot + "pnas_splits.csv")
pnas_splits.columns = ["TCGA ID"] + [str(k) for k in range(1, 16)]
pnas_splits.index = pnas_splits["TCGA ID"]
pnas_splits = pnas_splits.drop(["TCGA ID"], axis=1)

# print(all_dataset.shape)

k = pnas_splits.columns[0]
print("Creating Split %s" % k)
pat_train = pnas_splits.index[pnas_splits[k] == "Train"]
pat_test = pnas_splits.index[pnas_splits[k] == "Test"]
cv_splits = {}  # Initialize the dictionary to store the splits
cv_splits[int(k)] = {}

# The rest of the code remains unchanged, just ensure it's properly indented to run within the script without the loop
model = None
if opt.use_vgg_features:
    load_path = os.path.join(
        opt.checkpoints_dir,
        opt.exp_name,
        opt.model_name,
        "%s_%s.pt" % (opt.model_name, k),
    )
    model_ckpt = torch.load(load_path, map_location=device)
    model_state_dict = model_ckpt["model_state_dict"]
    if hasattr(model_state_dict, "_metadata"):
        del model_state_dict._metadata
    model = define_net(opt, None)
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model.load_state_dict(model_state_dict)
    model.eval()

# Obtain the data expected to use for training and testing
train_x_patname, train_x_path, train_x_grph, train_x_omic, train_e, train_t, train_g = (
    getAlignedMultimodalData(opt, model, device, all_dataset, pat_train, id2img)
)
test_x_patname, test_x_path, test_x_grph, test_x_omic, test_e, test_t, test_g = (
    getAlignedMultimodalData(opt, model, device, all_dataset, pat_test, id2img)
)

# Reformat the data
train_x_omic, train_e, train_t = (
    np.array(train_x_omic).squeeze(axis=1),
    np.array(train_e, dtype=np.float64),
    np.array(train_t, dtype=np.float64),
)
test_x_omic, test_e, test_t = (
    np.array(test_x_omic).squeeze(axis=1),
    np.array(test_e, dtype=np.float64),
    np.array(test_t, dtype=np.float64),
)

# Using a standard scaler after removing the extra dimension for the omic data
scaler = preprocessing.StandardScaler().fit(train_x_omic)
train_x_omic = scaler.transform(train_x_omic)
test_x_omic = scaler.transform(test_x_omic)

# List the train and test data we have
train_data = {
    "x_patname": train_x_patname,
    "x_path": np.array(train_x_path),
    "x_grph": train_x_grph,
    "x_omic": train_x_omic,
    "e": np.array(train_e, dtype=np.float64),
    "t": np.array(train_t, dtype=np.float64),
    "g": np.array(train_g, dtype=np.float64),
}

test_data = {
    "x_patname": test_x_patname,
    "x_path": np.array(test_x_path),
    "x_grph": test_x_grph,
    "x_omic": test_x_omic,
    "e": np.array(test_e, dtype=np.float64),
    "t": np.array(test_t, dtype=np.float64),
    "g": np.array(test_g, dtype=np.float64),
}

dataset = {"train": train_data, "test": test_data}
cv_splits[int(k)] = dataset


# if opt.make_all_train:
#     break

# use the data_dict mentioned above
data_dict["cv_splits"] = cv_splits

pickle.dump(
    data_dict,
    open(
        "%s/splits/gbmlgg1cv_%s_%d_%d_%d%s.pkl"
        % (
            opt.dataroot,
            opt.roi_dir,
            opt.ignore_missing_moltype,
            opt.ignore_missing_histype,
            opt.use_vgg_features,
            "_rnaseq" if opt.use_rnaseq else "",
        ),
        "wb",
    ),
)
