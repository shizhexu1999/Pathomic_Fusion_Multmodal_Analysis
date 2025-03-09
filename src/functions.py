"""!@file functions.py
@brief This file contains basic functions for data preparation and evaluation.
@author Shizhe Xu
@date 28 June 2024
"""

import pandas as pd
import math
import os
import numpy as np
import torch
from torch.nn import init, Parameter
import torch.nn as nn
from torch_geometric.data import Batch
import torch_geometric
from torch.utils.data.dataloader import default_collate

# lifelines
import lifelines
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test


######################################################################
# Prepare the dataset for the analysis
######################################################################
def getCleanAllDataset(
    dataroot="./data/TCGA_GBMLGG/",
    ignore_missing_moltype=False,
    ignore_missing_histype=False,
    use_rnaseq=False,
):
    ### 1. Joining all_datasets.csv with grade data. Looks at columns with misisng samples
    # first three comes from "grade_data.csv", the rest comes from "all_dataset.csv"
    # metadata can be used for our data preprocessing
    metadata = [
        "Histology",
        "Grade",
        "Molecular subtype",
        "TCGA ID",
        "censored",
        "Survival months",
    ]
    all_dataset = pd.read_csv(os.path.join(dataroot, "all_dataset.csv")).drop(
        "indexes", axis=1
    )
    all_dataset.index = all_dataset["TCGA ID"]

    all_grade = pd.read_csv(os.path.join(dataroot, "grade_data.csv"))
    all_grade["Histology"] = all_grade["Histology"].str.replace(
        "astrocytoma (glioblastoma)", "glioblastoma", regex=False
    )
    all_grade.index = all_grade["TCGA ID"]
    assert pd.Series(all_dataset.index).equals(pd.Series(sorted(all_grade.index)))

    all_dataset = all_dataset.join(
        all_grade[["Histology", "Grade", "Molecular subtype"]], how="inner"
    )
    cols = all_dataset.columns.tolist()
    cols = cols[-3:] + cols[:-3]
    all_dataset = all_dataset[cols]

    # include RNAseq data into the whole dataset
    if use_rnaseq:
        gbm = pd.read_csv(
            os.path.join(dataroot, "mRNA_Expression_z-Scores_RNA_Seq_RSEM.txt"),
            sep="\t",
            skiprows=1,
            index_col=0,
        )
        lgg = pd.read_csv(
            os.path.join(dataroot, "mRNA_Expression_Zscores_RSEM.txt"),
            sep="\t",
            skiprows=1,
            index_col=0,
        )
        # remove columns that contain only NaN (null) values from the dataframes
        gbm = gbm[gbm.columns[~gbm.isnull().all()]]
        lgg = lgg[lgg.columns[~lgg.isnull().all()]]
        glioma_RNAseq = gbm.join(lgg, how="inner").T
        glioma_RNAseq = glioma_RNAseq.dropna(axis=1)
        glioma_RNAseq.columns = [gene + "_rnaseq" for gene in glioma_RNAseq.columns]
        glioma_RNAseq.index = [patname[:12] for patname in glioma_RNAseq.index]
        glioma_RNAseq = glioma_RNAseq.iloc[~glioma_RNAseq.index.duplicated()]
        # this step is used to match all_dataset.index = all_dataset["TCGA ID"]
        glioma_RNAseq.index.name = "TCGA ID"
        all_dataset = all_dataset.join(glioma_RNAseq, how="inner")

    pat_missing_moltype = all_dataset[all_dataset["Molecular subtype"].isna()].index
    pat_missing_idh = all_dataset[all_dataset["idh mutation"].isna()].index
    pat_missing_1p19q = all_dataset[all_dataset["codeletion"].isna()].index
    print("# Missing Molecular Subtype:", len(pat_missing_moltype))
    print("# Missing IDH Mutation:", len(pat_missing_idh))
    print("# Missing 1p19q Codeletion:", len(pat_missing_1p19q))
    assert pat_missing_moltype.equals(pat_missing_idh)
    assert pat_missing_moltype.equals(pat_missing_1p19q)
    pat_missing_grade = all_dataset[all_dataset["Grade"].isna()].index
    pat_missing_histype = all_dataset[all_dataset["Histology"].isna()].index
    print("# Missing Histological Subtype:", len(pat_missing_histype))
    print("# Missing Grade:", len(pat_missing_grade))
    assert pat_missing_histype.equals(pat_missing_grade)

    ### 2. Impute Missing Genomic Data: Removes patients with missing molecular subtype 
    #/ idh mutation / 1p19q. Else imputes with median value of each column. 
    #Fills missing Molecular subtype with "Missing"
    if ignore_missing_moltype:
        all_dataset = all_dataset[all_dataset["Molecular subtype"].isna() == False]
    for col in all_dataset.drop(metadata, axis=1).columns:
        all_dataset["Molecular subtype"] = all_dataset["Molecular subtype"].fillna(
            "Missing"
        )
        all_dataset[col] = all_dataset[col].fillna(all_dataset[col].median())

    ### 3. Impute Missing Histological Data: Removes patients with missing histological subtype / grade. 
    # Else imputes with "missing" / grade -1
    if ignore_missing_histype:
    # filtered data only includes rows where the "Histology" column has non-null values
        all_dataset = all_dataset[all_dataset["Histology"].isna() == False]
    else:
        all_dataset["Grade"] = all_dataset["Grade"].fillna(1)
        all_dataset["Histology"] = all_dataset["Histology"].fillna("Missing")
    all_dataset["Grade"] = all_dataset["Grade"] - 2

    ### 4. Adds Histomolecular subtype
    ms2int = {"Missing": -1, "IDHwt": 0, "IDHmut-non-codel": 1, "IDHmut-codel": 2}
    all_dataset[["Molecular subtype"]] = all_dataset[["Molecular subtype"]].applymap(
        lambda s: ms2int.get(s) if s in ms2int else s
    )
    hs2int = {
        "Missing": -1,
        "astrocytoma": 0,
        "oligoastrocytoma": 1,
        "oligodendroglioma": 2,
        "glioblastoma": 3,
    }
    all_dataset[["Histology"]] = all_dataset[["Histology"]].applymap(
        lambda s: hs2int.get(s) if s in hs2int else s
    )
    all_dataset = addHistomolecularSubtype(all_dataset)
    metadata.extend(["Histomolecular subtype"])
    all_dataset["censored"] = 1 - all_dataset["censored"]
    return metadata, all_dataset

def addHistomolecularSubtype(data):
    subtyped_data = data.copy()
    subtyped_data.insert(
        loc=0, column="Histomolecular subtype", value=np.ones(len(data))
    )
    # convert the column to 'object' type if you are sure you want to store strings
    subtyped_data["Histomolecular subtype"] = subtyped_data[
        "Histomolecular subtype"
    ].astype("object")
    idhwt_ATC = np.logical_and(
        data["Molecular subtype"] == 0,
        np.logical_or(data["Histology"] == 0, data["Histology"] == 3),
    )
    subtyped_data.loc[idhwt_ATC, "Histomolecular subtype"] = "idhwt_ATC"

    idhmut_ATC = np.logical_and(
        data["Molecular subtype"] == 1,
        np.logical_or(data["Histology"] == 0, data["Histology"] == 3),
    )
    subtyped_data.loc[idhmut_ATC, "Histomolecular subtype"] = "idhmut_ATC"

    ODG = np.logical_and(data["Molecular subtype"] == 2, data["Histology"] == 2)
    subtyped_data.loc[ODG, "Histomolecular subtype"] = "ODG"
    return subtyped_data

def changeHistomolecularSubtype(data):
    """
    Molecular Subtype: IDHwt == 0, IDHmut-non-codel == 1, IDHmut-codel == 2
    Histology Subtype: astrocytoma == 0, oligoastrocytoma == 1, oligodendroglioma == 2, glioblastoma == 3
    """
    data = data.drop(['Histomolecular subtype'], axis=1)
    subtyped_data = data.copy()
    subtyped_data.insert(loc=0, column='Histomolecular subtype', value=np.ones(len(data)))
    idhwt_ATC = np.logical_and(data['Molecular subtype'] == 0, np.logical_or(data['Histology'] == 0, data['Histology'] == 3))
    subtyped_data.loc[idhwt_ATC, 'Histomolecular subtype'] = 'idhwt_ATC'
    
    idhmut_ATC = np.logical_and(data['Molecular subtype'] == 1, np.logical_or(data['Histology'] == 0, data['Histology'] == 3))
    subtyped_data.loc[idhmut_ATC, 'Histomolecular subtype'] = 'idhmut_ATC'
    
    ODG = np.logical_and(data['Molecular subtype'] == 2, data['Histology'] == 2)
    subtyped_data.loc[ODG, 'Histomolecular subtype'] = 'ODG'
    return subtyped_data

######################################################################
# Network Initilisation
# We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
# work better for some applications. We can try later on.
######################################################################
def init_func(m, init_type="normal", init_gain=0.02):
    """
    Function to initialize weights of a network layer based on the type of layer.
    Parameters:
        m (layer)       -- a layer of the network
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    """
    classname = m.__class__.__name__
    # check attribute first, then check the class name
    if hasattr(m, "weight") and (
        classname.find("Conv") != -1 or classname.find("Linear") != -1
    ):
        # use torch.nn.init
        if init_type == "normal":
            init.normal_(m.weight.data, 0.0, init_gain)
        elif init_type == "xavier":
            init.xavier_normal_(m.weight.data, gain=init_gain)
        elif init_type == "kaiming":
            init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
        elif init_type == "orthogonal":
            init.orthogonal_(m.weight.data, gain=init_gain)
        else:
            raise NotImplementedError(
                "initialization method [%s] is not implemented" % init_type
            )
        if hasattr(m, "bias") and m.bias is not None:
            init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm2d") != -1:
        init.normal_(m.weight.data, 1.0, init_gain)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type="normal", init_gain=0.02):
    """
    Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    """
    print("initialize network with %s" % init_type)
    net.apply(
        lambda m: init_func(m, init_type, init_gain)
    )  # apply the init function to every layer


# can use gpu_ids for parallel computing
def init_net(net, init_type="normal", init_gain=0.02, gpu_ids=[]):
    if torch.backends.mps.is_available():
        print("Using MPS")
        net.to("mps")
    elif len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to("cuda")
        # net = torch.nn.DataParallel(net, gpu_ids) 
    else:
        print("Neither MPS nor CUDA available, using CPU")
        net.to("cpu")

    if init_type != "max" and init_type != "none":
        print("Init Type:", init_type)
        init_weights(net, init_type, init_gain=init_gain)
    elif init_type == "none":
        print("Init Type: Not initializing networks.")
    # `max` only givs the self-normalizing weights
    elif init_type == "max":
        print("Init Type: Self-Normalizing Weights")

    return net


# "maximum effectiveness" technique for ensuring that
# the layer activations stay well-scaled during training
# it is used for GCN and SNN models, and the fusion process
def init_max_weights(module):
    for m in module.modules():
        # if type(m) == nn.Linear:
        # use isinstance() instead
        if isinstance(m, nn.Linear):
            # common technique for maximum effectiveness
            stdv = 1.0 / math.sqrt(m.weight.size(1))
            m.weight.data.normal_(0, stdv)
            m.bias.data.zero_()


######################################################################
# Regularization
######################################################################
def regularize_weights(model, reg_type=None):
    l1_reg = None
    # W represents for the weights of the model
    for W in model.parameters():
        if l1_reg is None:
            l1_reg = torch.abs(W).sum()
        else:
            # torch.abs(W).sum() is equivalent to W.norm(1)
            l1_reg = l1_reg + torch.abs(W).sum()
    return l1_reg


def regularize_path_weights(model, reg_type=None):
    l1_reg = None

    for W in model.module.classifier.parameters():
        if l1_reg is None:
            l1_reg = torch.abs(W).sum()
        else:
            l1_reg = l1_reg + torch.abs(W).sum()

    for W in model.module.linear.parameters():
        if l1_reg is None:
            l1_reg = torch.abs(W).sum()
        else:
            l1_reg = l1_reg + torch.abs(W).sum()

    return l1_reg


def regularize_MM_weights(model, reg_type=None):
    l1_reg = None

    # match with the definitions in the fusion process
    if model.module.__hasattr__("omic_net"):
        for W in model.module.omic_net.parameters():
            if l1_reg is None:
                l1_reg = torch.abs(W).sum()
            else:
                l1_reg = l1_reg + torch.abs(W).sum()

    if model.module.__hasattr__("linear_h_path"):
        for W in model.module.linear_h_path.parameters():
            if l1_reg is None:
                l1_reg = torch.abs(W).sum()
            else:
                l1_reg = l1_reg + torch.abs(W).sum()

    if model.module.__hasattr__("linear_h_omic"):
        for W in model.module.linear_h_omic.parameters():
            if l1_reg is None:
                l1_reg = torch.abs(W).sum()
            else:
                l1_reg = l1_reg + torch.abs(W).sum()

    if model.module.__hasattr__("linear_h_grph"):
        for W in model.module.linear_h_grph.parameters():
            if l1_reg is None:
                l1_reg = torch.abs(W).sum()
            else:
                l1_reg = l1_reg + torch.abs(W).sum()

    if model.module.__hasattr__("linear_z_path"):
        for W in model.module.linear_z_path.parameters():
            if l1_reg is None:
                l1_reg = torch.abs(W).sum()
            else:
                l1_reg = l1_reg + torch.abs(W).sum()

    if model.module.__hasattr__("linear_z_omic"):
        for W in model.module.linear_z_omic.parameters():
            if l1_reg is None:
                l1_reg = torch.abs(W).sum()
            else:
                l1_reg = l1_reg + torch.abs(W).sum()

    if model.module.__hasattr__("linear_z_grph"):
        for W in model.module.linear_z_grph.parameters():
            if l1_reg is None:
                l1_reg = torch.abs(W).sum()
            else:
                l1_reg = l1_reg + torch.abs(W).sum()

    if model.module.__hasattr__("linear_o_path"):
        for W in model.module.linear_o_path.parameters():
            if l1_reg is None:
                l1_reg = torch.abs(W).sum()
            else:
                l1_reg = l1_reg + torch.abs(W).sum()

    if model.module.__hasattr__("linear_o_omic"):
        for W in model.module.linear_o_omic.parameters():
            if l1_reg is None:
                l1_reg = torch.abs(W).sum()
            else:
                l1_reg = l1_reg + torch.abs(W).sum()

    if model.module.__hasattr__("linear_o_grph"):
        for W in model.module.linear_o_grph.parameters():
            if l1_reg is None:
                l1_reg = torch.abs(W).sum()
            else:
                l1_reg = l1_reg + torch.abs(W).sum()

    if model.module.__hasattr__("encoder1"):
        for W in model.module.encoder1.parameters():
            if l1_reg is None:
                l1_reg = torch.abs(W).sum()
            else:
                l1_reg = l1_reg + torch.abs(W).sum()

    if model.module.__hasattr__("encoder2"):
        for W in model.module.encoder2.parameters():
            if l1_reg is None:
                l1_reg = torch.abs(W).sum()
            else:
                l1_reg = l1_reg + torch.abs(W).sum()

    if model.module.__hasattr__("classifier"):
        for W in model.module.classifier.parameters():
            if l1_reg is None:
                l1_reg = torch.abs(W).sum()
            else:
                l1_reg = l1_reg + torch.abs(W).sum()

    return l1_reg


def regularize_MM_omic(model, reg_type=None):
    l1_reg = None

    # if model.module.__hasattr__("omic_net"):
    #     for W in model.module.omic_net.parameters():
    #         if l1_reg is None:
    #             l1_reg = torch.abs(W).sum()
    #         else:
    #             l1_reg = l1_reg + torch.abs(W).sum()

    if model.__hasattr__("omic_net"):
        for W in model.omic_net.parameters():
            if l1_reg is None:
                l1_reg = torch.abs(W).sum()
            else:
                l1_reg = l1_reg + torch.abs(W).sum()

    return l1_reg


######################################################################
# Freeze/unfreeze
######################################################################


# freeze all layers and parameters
def dfs_freeze(model):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = False
        dfs_freeze(child)


# unfreeze all layers and parameters
def dfs_unfreeze(model):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = True
        dfs_unfreeze(child)


# first train the  last  linear  layers  of  the  multimodal  network  with
# the  unimodal  network  modules  frozen
# At epoch 5, we unfroze the genomic and graph networks
def unfreeze_unimodal(opt, model, epoch):
    if opt.mode == "graphomic":
        if epoch == 5:
            dfs_unfreeze(model.omic_net)
            print("Unfreezing Omic")
        if epoch == 5:
            dfs_unfreeze(model.grph_net)
            print("Unfreezing Graph")
    elif opt.mode == "pathomic":
        if epoch == 5:
            dfs_unfreeze(model.omic_net)
            print("Unfreezing Omic")
    elif opt.mode == "pathgraph":
        if epoch == 5:
            dfs_unfreeze(model.module.grph_net)
            print("Unfreezing Graph")
    elif opt.mode == "pathgraphomic":
        if epoch == 5:
            dfs_unfreeze(model.omic_net)
            print("Unfreezing Omic")
        if epoch == 5:
            dfs_unfreeze(model.grph_net)
            print("Unfreezing Graph")
    elif opt.mode == "omicomic":
        if epoch == 5:
            dfs_unfreeze(model.omic_net)
            print("Unfreezing Omic")
    elif opt.mode == "graphgraph":
        if epoch == 5:
            dfs_unfreeze(model.grph_net)
            print("Unfreezing Graph")
    elif opt.mode == "multilinear":
        if epoch == 5:
            dfs_unfreeze(model.omic_net)
            print("Unfreezing Omic")
        if epoch == 5:
            dfs_unfreeze(model.grph_net)
            print("Unfreezing Graph")

######################################################################
# Batch Collation
######################################################################
def mixed_collate(batch):
    elem = batch[0]
    elem_type = type(elem)
    # transpose the batch
    transposed = zip(*batch)
    return [
        (
            Batch.from_data_list(samples, [])
            if type(samples[0]) is torch_geometric.data.data.Data
            else default_collate(samples)
        )
        for samples in transposed
    ]


######################################################################
# Analysis
######################################################################
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


######################################################################
# Survivial Functions
######################################################################
# def CoxLoss(survtime, censor, hazard_pred, device):
#     # This calculation credit to Travers Ching https://github.com/traversc/cox-nnet
#     # Cox-nnet: An artificial neural network method for prognosis prediction of high-throughput omics data
#     current_batch_len = len(survtime)
#     R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)
#     for i in range(current_batch_len):
#         for j in range(current_batch_len):
#             R_mat[i, j] = survtime[j] >= survtime[i]

#     R_mat = torch.FloatTensor(R_mat).to(device)
#     theta = hazard_pred.reshape(-1)
#     exp_theta = torch.exp(theta)
#     censor = torch.FloatTensor(censor).to(device)
#     loss_cox = -torch.mean(
#         (theta - torch.log(torch.sum(exp_theta * R_mat, dim=1))) * censor
#     )
#     return loss_cox

# we flag the issue why "censor = torch.FloatTensor(censor).to(device)" cause the TypeError
# we then reuse the coxloss from the original code
def CoxLoss(survtime, censor, hazard_pred, device):
    # This calculation credit to Travers Ching https://github.com/traversc/cox-nnet
    # Cox-nnet: An artificial neural network method for prognosis prediction of high-throughput omics data
    current_batch_len = len(survtime)
    R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)
    for i in range(current_batch_len):
        for j in range(current_batch_len):
            R_mat[i,j] = survtime[j] >= survtime[i]

    R_mat = torch.FloatTensor(R_mat).to(device)
    theta = hazard_pred.reshape(-1)
    exp_theta = torch.exp(theta)
    loss_cox = -torch.mean((theta - torch.log(torch.sum(exp_theta*R_mat, dim=1))) * censor)
    return loss_cox


def CIndex(hazards, labels, survtime_all):
    concord = 0.0
    total = 0.0
    N_test = labels.shape[0]
    for i in range(N_test):
        if labels[i] == 1:
            for j in range(N_test):
                if survtime_all[j] > survtime_all[i]:
                    total += 1
                    if hazards[j] < hazards[i]:
                        concord += 1
                    elif hazards[j] < hazards[i]:
                        concord += 0.5

    return concord / total


def CIndex_lifeline(hazards, labels, survtime_all):
    return concordance_index(survtime_all, -hazards, labels)


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def accuracy_cox(hazardsdata, labels):
    # the accuracy is based on estimated survival events against true survival events (censored and uncensored)
    # if the predicted hazard score is higher than its median, then it is censored
    median = np.median(hazardsdata)
    hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)
    hazards_dichotomize[hazardsdata > median] = 1
    correct = np.sum(hazards_dichotomize == labels)
    return correct / len(labels)


def cox_log_rank(hazardsdata, labels, survtime_all):
    median = np.median(hazardsdata)
    hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)
    hazards_dichotomize[hazardsdata > median] = 1
    idx = hazards_dichotomize == 0
    T1 = survtime_all[idx]
    T2 = survtime_all[~idx]
    E1 = labels[idx]
    E2 = labels[~idx]
    results = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)
    pvalue_pred = results.p_value
    return pvalue_pred
