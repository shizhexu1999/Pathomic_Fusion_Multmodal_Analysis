"""!@file fusion_IG.py
@brief This file contains the code for the fusion of the different modalities for IG.
@author Shizhe Xu
@date 28 June 2024
"""

import torch
import torch.nn as nn
from functions import init_max_weights


######################################################################
# Bilinear Fusion
######################################################################
class BilinearFusion(nn.Module):
    """
    Each modality was gated using three linear layers, with the second linear layer
    used to compute the attention scores.

    No feature dimension reduction was done in bimodal networks in any tasks.
    """

    def __init__(
        self,
        skip=1,
        use_bilinear=1,
        gate1=1,
        gate2=1,
        dim1=32,
        dim2=32,
        scale_dim1=1,
        scale_dim2=1,
        mmhid=64,
        dropout_rate=0.25,
    ):
        super(BilinearFusion, self).__init__()
        # incorporates skip connections between layers
        self.skip = skip
        self.use_bilinear = use_bilinear
        # gate mechanism
        self.gate1 = gate1
        self.gate2 = gate2

        dim1_og, dim2_og, dim1, dim2 = (
            dim1,
            dim2,
            dim1 // scale_dim1,
            dim2 // scale_dim2,
        )
        skip_dim = dim1 + dim2 + 2 if skip else 0

        # reduces the dimension of the input data (dim1_og) to dim1
        self.linear_h1 = nn.Sequential(nn.Linear(dim1_og, dim1), nn.ReLU())
        # bilinear layer takes two input vectors and computes a weighted sum of products of their elements
        # if false, it is a linear transformation to the concatenated vector to produce a vector of size dim1
        self.linear_z1 = (
            nn.Bilinear(dim1_og, dim2_og, dim1)
            if use_bilinear
            else nn.Sequential(nn.Linear(dim1_og + dim2_og, dim1))
        )
        # recombine these features into a new representation of the same dimensionality
        self.linear_o1 = nn.Sequential(
            nn.Linear(dim1, dim1), nn.ReLU(), nn.Dropout(p=dropout_rate)
        )

        self.linear_h2 = nn.Sequential(nn.Linear(dim2_og, dim2), nn.ReLU())
        self.linear_z2 = (
            nn.Bilinear(dim1_og, dim2_og, dim2)
            if use_bilinear
            else nn.Sequential(nn.Linear(dim1_og + dim2_og, dim2))
        )
        self.linear_o2 = nn.Sequential(
            nn.Linear(dim2, dim2), nn.ReLU(), nn.Dropout(p=dropout_rate)
        )

        self.post_fusion_dropout = nn.Dropout(p=dropout_rate)
        self.encoder1 = nn.Sequential(
            nn.Linear((dim1 + 1) * (dim2 + 1), mmhid),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
        )
        self.encoder2 = nn.Sequential(
            nn.Linear(mmhid + skip_dim, mmhid), nn.ReLU(), nn.Dropout(p=dropout_rate)
        )
        init_max_weights(self)

    def forward(self, vec1, vec2):
        if self.gate1:
            h1 = self.linear_h1(vec1)
            # compute the attention scores
            z1 = (
                self.linear_z1(vec1, vec2)
                if self.use_bilinear
                else self.linear_z1(torch.cat((vec1, vec2), dim=1))
            )
            # apply the attention scores to the hidden representation
            o1 = self.linear_o1(nn.Sigmoid()(z1) * h1)
        else:
            o1 = self.linear_o1(vec1)

        if self.gate2:
            h2 = self.linear_h2(vec2)
            z2 = (
                self.linear_z2(vec1, vec2)
                if self.use_bilinear
                else self.linear_z2(torch.cat((vec1, vec2), dim=1))
            )
            o2 = self.linear_o2(nn.Sigmoid()(z2) * h2)
        else:
            o2 = self.linear_o2(vec2)

        # bilinear fusion
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        
        o1 = o1.view(-1, 32)
        o2 = o2.view(-1, 32)

        o1 = torch.cat((o1, torch.ones(o1.shape[0], 1, device=device)), 1)
        o2 = torch.cat((o2, torch.ones(o2.shape[0], 1, device=device)), 1)
        # o1 = torch.cat((o1, torch.cuda.FloatTensor(o1.shape[0], 1).fill_(1)), 1)
        # o2 = torch.cat((o2, torch.cuda.FloatTensor(o2.shape[0], 1).fill_(1)), 1)
        # perform batch matrix multiplication
        o12 = torch.bmm(o1.unsqueeze(2), o2.unsqueeze(1)).flatten(start_dim=1)
        out = self.post_fusion_dropout(o12)
        out = self.encoder1(out)
        if self.skip:
            # To use theunperturbed unimodal features
            # we appended 1 to each feature vector before computing the Kronecker Product
            out = torch.cat((out, o1, o2), 1)
        out = self.encoder2(out)
        return out


######################################################################
# Trilinear Fusion
######################################################################
class TrilinearFusion_A(nn.Module):
    """
    The genomic modality was used to gate over the image and graph modalitie.

    For survival outcome prediction, the first and third linear layersfor the
    genomic modality have 32 hidden units to maintain the feature map dimension,
    with the linear layers in the image andgraph modalities having 16 hidden
    units in order to transform the feature representations into a lower dimension.
    """

    def __init__(
        self,
        skip=1,
        use_bilinear=1,
        gate1=1,
        gate2=1,
        gate3=1,
        dim1=32,
        dim2=32,
        dim3=32,
        scale_dim1=1,
        scale_dim2=1,
        scale_dim3=1,
        mmhid=96,
        dropout_rate=0.25,
    ):
        super(TrilinearFusion_A, self).__init__()
        self.skip = skip
        self.use_bilinear = use_bilinear
        self.gate1 = gate1
        self.gate2 = gate2
        self.gate3 = gate3

        dim1_og, dim2_og, dim3_og, dim1, dim2, dim3 = (
            dim1,
            dim2,
            dim3,
            dim1 // scale_dim1,
            dim2 // scale_dim2,
            dim3 // scale_dim3,
        )
        skip_dim = dim1 + dim2 + dim3 + 3 if skip else 0

        # path
        self.linear_h1 = nn.Sequential(nn.Linear(dim1_og, dim1), nn.ReLU())
        self.linear_z1 = (
            nn.Bilinear(dim1_og, dim3_og, dim1)
            if use_bilinear
            else nn.Sequential(nn.Linear(dim1_og + dim3_og, dim1))
        )
        self.linear_o1 = nn.Sequential(
            nn.Linear(dim1, dim1), nn.ReLU(), nn.Dropout(p=dropout_rate)
        )

        # graph
        self.linear_h2 = nn.Sequential(nn.Linear(dim2_og, dim2), nn.ReLU())
        self.linear_z2 = (
            nn.Bilinear(dim2_og, dim3_og, dim2)
            if use_bilinear
            else nn.Sequential(nn.Linear(dim2_og + dim3_og, dim2))
        )
        self.linear_o2 = nn.Sequential(
            nn.Linear(dim2, dim2), nn.ReLU(), nn.Dropout(p=dropout_rate)
        )

        # omic
        self.linear_h3 = nn.Sequential(nn.Linear(dim3_og, dim3), nn.ReLU())
        self.linear_z3 = (
            nn.Bilinear(dim1_og, dim3_og, dim3)
            if use_bilinear
            else nn.Sequential(nn.Linear(dim1_og + dim3_og, dim3))
        )
        self.linear_o3 = nn.Sequential(
            nn.Linear(dim3, dim3), nn.ReLU(), nn.Dropout(p=dropout_rate)
        )

        self.post_fusion_dropout = nn.Dropout(p=0.25)
        self.encoder1 = nn.Sequential(
            nn.Linear((dim1 + 1) * (dim2 + 1) * (dim3 + 1), mmhid),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
        )
        self.encoder2 = nn.Sequential(
            nn.Linear(mmhid + skip_dim, mmhid), nn.ReLU(), nn.Dropout(p=dropout_rate)
        )

        init_max_weights(self)

    def forward(self, vec1, vec2, vec3):
        if self.gate1:
            # print(f"vec1", vec1.shape)
            # print(f"vec3", vec3.shape)
            h1 = self.linear_h1(vec1)
            z1 = (
                self.linear_z1(vec1, vec3)
                if self.use_bilinear
                else self.linear_z1(torch.cat((vec1, vec3), dim=1))
            )
            # gate path with omic
            o1 = self.linear_o1(nn.Sigmoid()(z1) * h1)
        else:
            o1 = self.linear_o1(vec1)

        if self.gate2:
            vec3 = vec3.view(-1, 32)

            # print(f"vec2", vec2.shape)
            # print(f"vec3", vec3.shape)
            h2 = self.linear_h2(vec2)
            z2 = (
                self.linear_z2(vec2, vec3)
                if self.use_bilinear
                else self.linear_z2(torch.cat((vec2, vec3), dim=1))
            )
            # gate graph with omic
            o2 = self.linear_o2(nn.Sigmoid()(z2) * h2)
        else:
            o2 = self.linear_o2(vec2)

        # gate3=0 in the original paper setting
        if self.gate3:
            h3 = self.linear_h3(vec3)
            z3 = (
                self.linear_z3(vec1, vec3)
                if self.use_bilinear
                else self.linear_z3(torch.cat((vec1, vec3), dim=1))
            )
            # gate omic with path
            o3 = self.linear_o3(nn.Sigmoid()(z3) * h3)
        else:
            o3 = self.linear_o3(vec3)

        # trilinear fusion
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        # print(f"o1 shape", o1.shape)
        # print(f"o2 shape", o2.shape)
        # print(f"o3 shape", o3.shape)
        o1 = o1.view(-1, 32)
        o1 = torch.cat((o1, torch.ones(o1.shape[0], 1, device=device)), 1)
        o2 = torch.cat((o2, torch.ones(o2.shape[0], 1, device=device)), 1)
        o3 = torch.cat((o3, torch.ones(o3.shape[0], 1, device=device)), 1)
        # o1 = torch.cat((o1, torch.cuda.FloatTensor(o1.shape[0], 1).fill_(1)), 1)
        # o2 = torch.cat((o2, torch.cuda.FloatTensor(o2.shape[0], 1).fill_(1)), 1)
        # o3 = torch.cat((o3, torch.cuda.FloatTensor(o3.shape[0], 1).fill_(1)), 1)
        o12 = torch.bmm(o1.unsqueeze(2), o2.unsqueeze(1)).flatten(start_dim=1)
        o123 = torch.bmm(o12.unsqueeze(2), o3.unsqueeze(1)).flatten(start_dim=1)
        out = self.post_fusion_dropout(o123)
        out = self.encoder1(out)
        if self.skip:
            out = torch.cat((out, o1, o2, o3), 1)
        out = self.encoder2(out)
        return out


class TrilinearFusion_B(nn.Module):
    """
    The histology image modality was used to gate over the genomic and graph modalities

    For grade classification, we maintained the feature dimension of our histology image modality instead,
    and reduced the dimension ofthe graph and genomic modalities.
    """

    def __init__(
        self,
        skip=1,
        use_bilinear=1,
        gate1=1,
        gate2=1,
        gate3=1,
        dim1=32,
        dim2=32,
        dim3=32,
        scale_dim1=1,
        scale_dim2=1,
        scale_dim3=1,
        mmhid=96,
        dropout_rate=0.25,
    ):
        super(TrilinearFusion_B, self).__init__()
        self.skip = skip
        self.use_bilinear = use_bilinear
        self.gate1 = gate1
        self.gate2 = gate2
        self.gate3 = gate3

        dim1_og, dim2_og, dim3_og, dim1, dim2, dim3 = (
            dim1,
            dim2,
            dim3,
            dim1 // scale_dim1,
            dim2 // scale_dim2,
            dim3 // scale_dim3,
        )
        skip_dim = dim1 + dim2 + dim3 + 3 if skip else 0

        # path
        self.linear_h1 = nn.Sequential(nn.Linear(dim1_og, dim1), nn.ReLU())
        self.linear_z1 = (
            nn.Bilinear(dim1_og, dim3_og, dim1)
            if use_bilinear
            else nn.Sequential(nn.Linear(dim1_og + dim3_og, dim1))
        )
        self.linear_o1 = nn.Sequential(
            nn.Linear(dim1, dim1), nn.ReLU(), nn.Dropout(p=dropout_rate)
        )

        # graph
        self.linear_h2 = nn.Sequential(nn.Linear(dim2_og, dim2), nn.ReLU())
        # it might be clearer if swap the positions of dim2_og and dim1_og
        self.linear_z2 = (
            nn.Bilinear(dim2_og, dim1_og, dim2)
            if use_bilinear
            else nn.Sequential(nn.Linear(dim2_og + dim1_og, dim2))
        )
        self.linear_o2 = nn.Sequential(
            nn.Linear(dim2, dim2), nn.ReLU(), nn.Dropout(p=dropout_rate)
        )

        # omic
        self.linear_h3 = nn.Sequential(nn.Linear(dim3_og, dim3), nn.ReLU())
        self.linear_z3 = (
            nn.Bilinear(dim1_og, dim3_og, dim3)
            if use_bilinear
            else nn.Sequential(nn.Linear(dim1_og + dim3_og, dim3))
        )
        self.linear_o3 = nn.Sequential(
            nn.Linear(dim3, dim3), nn.ReLU(), nn.Dropout(p=dropout_rate)
        )

        self.post_fusion_dropout = nn.Dropout(p=0.25)
        self.encoder1 = nn.Sequential(
            nn.Linear((dim1 + 1) * (dim2 + 1) * (dim3 + 1), mmhid),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
        )
        self.encoder2 = nn.Sequential(
            nn.Linear(mmhid + skip_dim, mmhid), nn.ReLU(), nn.Dropout(p=dropout_rate)
        )

        init_max_weights(self)

    def forward(self, vec1, vec2, vec3):
        if self.gate1:
            h1 = self.linear_h1(vec1)
            z1 = (
                self.linear_z1(vec1, vec3)
                if self.use_bilinear
                else self.linear_z1(torch.cat((vec1, vec3), dim=1))
            )  # gate path with omic
            o1 = self.linear_o1(nn.Sigmoid()(z1) * h1)
        else:
            o1 = self.linear_o1(vec1)

        if self.gate2:
            h2 = self.linear_h2(vec2)
            z2 = (
                self.linear_z2(vec2, vec1)
                if self.use_bilinear
                else self.linear_z2(torch.cat((vec2, vec1), dim=1))
            )  # gate graph with path
            o2 = self.linear_o2(nn.Sigmoid()(z2) * h2)
        else:
            o2 = self.linear_o2(vec2)

        if self.gate3:
            h3 = self.linear_h3(vec3)
            z3 = (
                self.linear_z3(vec1, vec3)
                if self.use_bilinear
                else self.linear_z3(torch.cat((vec1, vec3), dim=1))
            )  # gate omic with path
            o3 = self.linear_o3(nn.Sigmoid()(z3) * h3)
        else:
            o3 = self.linear_o3(vec3)

        # trilinear fusion
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )

        o1 = torch.cat((o1, torch.ones(o1.shape[0], 1, device=device)), 1)
        o2 = torch.cat((o2, torch.ones(o2.shape[0], 1, device=device)), 1)
        o3 = torch.cat((o3, torch.ones(o3.shape[0], 1, device=device)), 1)
        # o1 = torch.cat((o1, torch.cuda.FloatTensor(o1.shape[0], 1).fill_(1)), 1)
        # o2 = torch.cat((o2, torch.cuda.FloatTensor(o2.shape[0], 1).fill_(1)), 1)
        # o3 = torch.cat((o3, torch.cuda.FloatTensor(o3.shape[0], 1).fill_(1)), 1)
        o12 = torch.bmm(o1.unsqueeze(2), o2.unsqueeze(1)).flatten(start_dim=1)
        o123 = torch.bmm(o12.unsqueeze(2), o3.unsqueeze(1)).flatten(start_dim=1)
        out = self.post_fusion_dropout(o123)
        out = self.encoder1(out)
        if self.skip:
            out = torch.cat((out, o1, o2, o3), 1)
        out = self.encoder2(out)
        return out
