"""!@file networks_conch.py
@brief This file modifies the PathNet by using CONCH image encoder.
@author Shizhe Xu
@date 28 June 2024
"""

import os

import torch
import torch.nn as nn
from torch.nn import init, Parameter
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.model_zoo import load_url as load_state_dict_from_url
import torch.nn.functional as F

import torch_geometric.nn


from fusion import *
from options import parse_args
from functions import *

# implement the models (feature encoder) directly
from torchvision.models import vgg19_bn
from conch.open_clip_custom import create_model_from_pretrained

######################################################################
# General Network Utility Functions
######################################################################
def define_act_layer(act_type="Tanh"):
    """
    it defines the activation layer for the network
    """
    if act_type == "Tanh":
        act_layer = nn.Tanh()
    elif act_type == "ReLU":
        act_layer = nn.ReLU()
    elif act_type == "Sigmoid":
        act_layer = nn.Sigmoid()
    elif act_type == "LSM":
        act_layer = nn.LogSoftmax(dim=1)
    elif act_type == "none":
        act_layer = None
    else:
        raise NotImplementedError(
            "activation layer [%s] is not found" % act_type,
        )
    return act_layer


def define_net_conch(opt, k):
    net = None
    # default activation is sigmoid
    act = define_act_layer(act_type=opt.act_type)
    # other initialisation such as HE or Xavier can also be considered here
    init_max = True if opt.init_type == "max" else False

    if opt.mode == "path":
        # label_dim is the size of the output
        # rather than using the `PathNet`
        net = conch(path_dim=opt.path_dim, act=act, label_dim=opt.label_dim)
    elif opt.mode == "graph":
        net = GraphNet(
            grph_dim=opt.grph_dim,
            dropout_rate=opt.dropout_rate,
            GNN=opt.GNN,
            use_edges=opt.use_edges,
            pooling_ratio=opt.pooling_ratio,
            act=act,
            label_dim=opt.label_dim,
            init_max=init_max,
        )
    elif opt.mode == "omic":
        net = MaxNet(
            input_dim=opt.input_size_omic,
            omic_dim=opt.omic_dim,
            dropout_rate=opt.dropout_rate,
            act=act,
            label_dim=opt.label_dim,
            init_max=init_max,
        )
    elif opt.mode == "graphomic":
        net = GraphomicNet(opt=opt, act=act, k=k)
    elif opt.mode == "pathomic":
        net = PathomicNet(opt=opt, act=act, k=k)
    elif opt.mode == "pathgraphomic":
        net = PathgraphomicNet(opt=opt, act=act, k=k)
    elif opt.mode == "pathpath":
        net = PathpathNet(opt=opt, act=act, k=k)
    elif opt.mode == "graphgraph":
        net = GraphgraphNet(opt=opt, act=act, k=k)
    elif opt.mode == "omicomic":
        net = OmicomicNet(opt=opt, act=act, k=k)
    else:
        raise NotImplementedError("model [%s] is not implemented" % opt.model)
    # init_net function is from functions.py
    return init_net(net, opt.init_type, opt.init_gain, opt.gpu_ids)


def define_optimizer(opt, model):
    """
    it defines the optimizer for the network
    """
    optimizer = None
    # the optimizer `ababound` is not implemented in the code
    # if opt.optimizer_type == 'adabound':
    #     optimizer = adabound.AdaBound(model.parameters(), lr=opt.lr, final_lr=opt.final_lr)
    if opt.optimizer_type == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=opt.lr,
            betas=(opt.beta1, opt.beta2),
            weight_decay=opt.weight_decay,
        )
    elif opt.optimizer_type == "adagrad":
        optimizer = torch.optim.Adagrad(
            model.parameters(),
            lr=opt.lr,
            weight_decay=opt.weight_decay,
            initial_accumulator_value=0.1,
        )
    else:
        raise NotImplementedError(
            "initialization optimizer [%s] is not implemented" % opt.optimizer
        )
    return optimizer


def define_reg(opt, model):
    """
    it defines the regularization for the network
    """
    loss_reg = None

    if opt.reg_type == "none":
        loss_reg = 0
    elif opt.reg_type == "path":
        loss_reg = regularize_path_weights(model=model)
    # 'mm' stands for the mixed models
    # it is designed specifically for the bifusion/trifusion models
    elif opt.reg_type == "mm":
        loss_reg = regularize_MM_weights(model=model)
    elif opt.reg_type == "all":
        loss_reg = regularize_weights(model=model)
    elif opt.reg_type == "omic":
        loss_reg = regularize_MM_omic(model=model)
    else:
        raise NotImplementedError("reg method [%s] is not implemented" % opt.reg_type)
    return loss_reg


def define_scheduler(opt, optimizer):
    """
    The learning rate scheduler adjusts the learning rate over time,
    and enhances the model's training by fine-tuning the optimization process.
    """
    if opt.lr_policy == "linear":

        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(
                opt.niter_decay + 1
            )
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == "exp":
        scheduler = lr_scheduler.ExponentialLR(optimizer, 0.1, last_epoch=-1)
    elif opt.lr_policy == "step":
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=opt.lr_decay_iters, gamma=0.1
        )
    elif opt.lr_policy == "plateau":
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.2, threshold=0.01, patience=5
        )
    elif opt.lr_policy == "cosine":
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=opt.niter, eta_min=0
        )
    else:
        return NotImplementedError(
            "learning rate policy [%s] is not implemented", opt.lr_policy
        )
    return scheduler


def define_bifusion(
    fusion_type,
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
    """
    it defines the bifusion process for the network,
    and more details can be found in the `fusion.py` module.
    """
    fusion = None
    if fusion_type == "pofusion":
        fusion = BilinearFusion(
            skip=skip,
            use_bilinear=use_bilinear,
            gate1=gate1,
            gate2=gate2,
            dim1=dim1,
            dim2=dim2,
            scale_dim1=scale_dim1,
            scale_dim2=scale_dim2,
            mmhid=mmhid,
            dropout_rate=dropout_rate,
        )
    else:
        raise NotImplementedError("fusion type [%s] is not found" % fusion_type)
    return fusion


def define_trifusion(
    fusion_type,
    skip=1,
    use_bilinear=1,
    gate1=1,
    gate2=1,
    gate3=3,
    dim1=32,
    dim2=32,
    dim3=32,
    scale_dim1=1,
    scale_dim2=1,
    scale_dim3=1,
    mmhid=96,
    dropout_rate=0.25,
):
    """
    it defines the trifusion process for the network,
    and more details can be found in the `fusion.py` module.
    """
    fusion = None
    if fusion_type == "pofusion_A":
        fusion = TrilinearFusion_A(
            skip=skip,
            use_bilinear=use_bilinear,
            gate1=gate1,
            gate2=gate2,
            gate3=gate3,
            dim1=dim1,
            dim2=dim2,
            dim3=dim3,
            scale_dim1=scale_dim1,
            scale_dim2=scale_dim2,
            scale_dim3=scale_dim3,
            mmhid=mmhid,
            dropout_rate=dropout_rate,
        )
    elif fusion_type == "pofusion_B":
        fusion = TrilinearFusion_B(
            skip=skip,
            use_bilinear=use_bilinear,
            gate1=gate1,
            gate2=gate2,
            gate3=gate3,
            dim1=dim1,
            dim2=dim2,
            dim3=dim3,
            scale_dim1=scale_dim1,
            scale_dim2=scale_dim2,
            scale_dim3=scale_dim3,
            mmhid=mmhid,
            dropout_rate=dropout_rate,
        )
    else:
        raise NotImplementedError("fusion type [%s] is not found" % fusion_type)
    return fusion


######################################################################
# Path Model
######################################################################
# download the pre-trained model from the PyTorch model zoo
model_urls = {
    "vgg11": "https://download.pytorch.org/models/vgg11-bbd30ac9.pth",
    "vgg13": "https://download.pytorch.org/models/vgg13-c768596a.pth",
    "vgg16": "https://download.pytorch.org/models/vgg16-397923af.pth",
    "vgg19": "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth",
    "vgg11_bn": "https://download.pytorch.org/models/vgg11_bn-6002323d.pth",
    "vgg13_bn": "https://download.pytorch.org/models/vgg13_bn-abd245e5.pth",
    "vgg16_bn": "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth",
    "vgg19_bn": "https://download.pytorch.org/models/vgg19_bn-c79401a0.pth",
}


class PathNet(nn.Module):

    def __init__(self, features, preprocess, path_dim=32, act=None, num_classes=1):
        super(PathNet, self).__init__()
        # encoder
        self.features = features
        self.preprocess = preprocess
        # adjusts the size of the output from the preceding layers to be exactly 7x7
        # average pooling is applied
        # we might need to modify it for the compatibility of mps
        # note that Adaptive pool MPS: input sizes must be divisible by output sizes
        # self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        # RuntimeError: linear(): input and weight.T shapes cannot be
        # multiplied (8x12800 and 25088x1024) if stride=3
        # this can be removed if the method can be improved
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2)

        # input is highly likely to be 512 * 1 * 1
        self.classifier = nn.Sequential(
            nn.Linear(512*1*1, 1024),
            nn.ReLU(True),
            nn.Dropout(0.25),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Dropout(0.25),
            nn.Linear(1024, path_dim),
            nn.ReLU(True),
            nn.Dropout(0.05),
        )

        # final layer to change from 32 to 1
        self.linear = nn.Linear(path_dim, num_classes)
        self.act = act

        # `shift` and `range` can be used to compute the hazard
        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

        # depth-first search (DFS) is used to freeze the features
        # freeze all layers
        dfs_freeze(self.features)

    def forward(self, **kwargs):
        x = kwargs["x_path"]
        # print(x)
        # print(x.size())
        # x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        # torch.Size([8, 3, 512, 512])
        # incorrect one is torch.Size([8, 32])
        # print to see if it is a feature
        
        # x is already be a tensor object
        # x = self.preprocess(x).unsqueeze(0)
        x = self.features.encode_image(x, proj_contrast=False, normalize=False)
        # print(x)
        # print(x.size())
        # torch.Size([8, 512, 16, 16])
        # note that Adaptive pool MPS: input sizes must be divisible by output sizes

        # x = self.avgpool(x)
        # print(x)
        # print(x.size())
        # torch.Size([8, 512, 7, 7])
        # example use: this will reshape x to [10, 10*28*28]

        # x = x.view(x.size(0), -1)
        # print(x.size())
        # torch.Size([8, 25088])

        features = self.classifier(x)
        hazard = self.linear(features)

        # for testing CNN purpose
        # features = x
        # #features = self.classifier(x)
        # hazard = self.linear(x)

        if self.act is not None:
            hazard = self.act(hazard)

            if isinstance(self.act, nn.Sigmoid):
                hazard = hazard * self.output_range + self.output_shift

        return features, hazard


# CNN configurations
# cfgs = {
#     "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
#     "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
#     "D": [
#         64,
#         64,
#         "M",
#         128,
#         128,
#         "M",
#         256,
#         256,
#         256,
#         "M",
#         512,
#         512,
#         512,
#         "M",
#         512,
#         512,
#         512,
#         "M",
#     ],
#     "E": [
#         64,
#         64,
#         "M",
#         128,
#         128,
#         "M",
#         256,
#         256,
#         256,
#         256,
#         "M",
#         512,
#         512,
#         512,
#         512,
#         "M",
#         512,
#         512,
#         512,
#         512,
#         "M",
#     ],
# }


# build networks based on the above architecture
def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


# build vgg for later training use
def conch(
    # arch="vgg19_bn",
    # cfg="E",
    # weights="IMAGENET1K_V1",
    act=None,
    # batch_norm=True,
    label_dim=1,
    # pretrained=True,
    # progress=True,
    **kwargs
):  
    features, preprocess = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path="checkpoints/conch/pytorch_model.bin")
    # make_layers here corresponds to `features` (encoder) at the beginning
    model = PathNet(
        features, preprocess,
        act=act,
        num_classes=label_dim,
        **kwargs
    )
    # model = PathNet(
    #     make_layers(cfgs[cfg], batch_norm=batch_norm),
    #     act=act,
    #     num_classes=label_dim,
    #     **kwargs
    # )

    # if pretrained:
    #     pretrained_dict = load_state_dict_from_url(model_urls[arch], progress=progress)

    #     for key in list(pretrained_dict.keys()):
    #         if "classifier" in key:
    #             pretrained_dict.pop(key)

    #     model.load_state_dict(pretrained_dict, strict=False)
    #     print("Initializing Path Weights")

    return model


######################################################################
# Omic Model
######################################################################
class MaxNet(nn.Module):
    def __init__(
        self,
        input_dim=80,
        omic_dim=32,
        dropout_rate=0.25,
        act=None,
        label_dim=1,
        init_max=True,
    ):
        super(MaxNet, self).__init__()
        hidden = [64, 48, 32, 32]
        self.act = act

        encoder1 = nn.Sequential(
            nn.Linear(input_dim, hidden[0]),
            nn.ELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False),
        )

        encoder2 = nn.Sequential(
            nn.Linear(hidden[0], hidden[1]),
            nn.ELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False),
        )

        encoder3 = nn.Sequential(
            nn.Linear(hidden[1], hidden[2]),
            nn.ELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False),
        )

        encoder4 = nn.Sequential(
            nn.Linear(hidden[2], omic_dim),
            nn.ELU(),
            nn.AlphaDropout(p=dropout_rate, inplace=False),
        )

        self.encoder = nn.Sequential(encoder1, encoder2, encoder3, encoder4)
        self.classifier = nn.Sequential(nn.Linear(omic_dim, label_dim))

        if init_max:
            init_max_weights(self)

        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

    def forward(self, **kwargs):
        x = kwargs["x_omic"]
        features = self.encoder(x)
        out = self.classifier(features)
        if self.act is not None:
            out = self.act(out)

            if isinstance(self.act, nn.Sigmoid):
                out = out * self.output_range + self.output_shift

        return features, out


######################################################################
# Graph Model
######################################################################

# The NormalizeFeaturesV2 class below is not used in our case
# class NormalizeFeaturesV2(object):

#     def __call__(self, data):
#         # normalize the feature matrix
#         data.x = data.x / data.x.max(0, keepdim=True)[0]
#         return data

#     def __repr__(self):
#         return "{}()".format(self.__class__.__name__)


class NormalizeFeaturesV2(object):

    def __init__(self, device=None):
        if device is None:
            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

    def __call__(self, data):
        # ensure the input tensor is on the correct device
        data.x = data.x.to(self.device, dtype=torch.float32)

        # normalize only the first 12 columns of x
        max_values = data.x[:, :12].max(0, keepdim=True)[0]
        data.x[:, :12] = data.x[:, :12] / max_values

        return data

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)


class NormalizeEdgesV2(object):

    def __init__(self, device=None):
        if device is None:
            # automatically select the device
            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

    def __call__(self, data):
        data.edge_attr = data.edge_attr.to(device=self.device, dtype=torch.float32)

        # normalize the edge attributes
        max_values = data.edge_attr.max(0, keepdim=True)[0]
        data.edge_attr = data.edge_attr / max_values

        return data

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)


class GraphNet(torch.nn.Module):
    def __init__(
        self,
        features=1036,
        nhid=128,
        grph_dim=32,
        nonlinearity=torch.tanh,
        dropout_rate=0.25,
        GNN="GCN",
        use_edges=0,
        pooling_ratio=0.20,
        act=None,
        label_dim=1,
        init_max=True,
    ):
        super(GraphNet, self).__init__()

        self.dropout_rate = dropout_rate
        self.use_edges = use_edges
        self.act = act

        self.conv1 = torch_geometric.nn.SAGEConv(features, nhid)
        # specify the graph neural network (GNN) for pooling
        # remove gnn=GNN from the original paper?
        self.pool1 = torch_geometric.nn.SAGPooling(
            nhid, ratio=pooling_ratio, GNN=torch_geometric.nn.GraphConv
        )
        self.conv2 = torch_geometric.nn.SAGEConv(nhid, nhid)
        self.pool2 = torch_geometric.nn.SAGPooling(
            nhid, ratio=pooling_ratio, GNN=torch_geometric.nn.GraphConv
        )
        self.conv3 = torch_geometric.nn.SAGEConv(nhid, nhid)
        self.pool3 = torch_geometric.nn.SAGPooling(
            nhid, ratio=pooling_ratio, GNN=torch_geometric.nn.GraphConv
        )

        # self.pool1 = torch_geometric.nn.SAGPooling(nhid, ratio=pooling_ratio)
        # self.conv2 = torch_geometric.nn.SAGEConv(nhid, nhid)
        # self.pool2 = torch_geometric.nn.SAGPooling(nhid, ratio=pooling_ratio)
        # self.conv3 = torch_geometric.nn.SAGEConv(nhid, nhid)
        # self.pool3 = torch_geometric.nn.SAGPooling(nhid, ratio=pooling_ratio)

        self.lin1 = torch.nn.Linear(nhid * 2, nhid)
        self.lin2 = torch.nn.Linear(nhid, grph_dim)
        self.lin3 = torch.nn.Linear(grph_dim, label_dim)

        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

        if init_max:
            init_max_weights(self)
            print("Initialzing with Max")

    def forward(self, **kwargs):
        # extract graph data from keyword arguments
        data = kwargs["x_grph"]
        data = NormalizeFeaturesV2()(data)
        data = NormalizeEdgesV2()(data)
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        x = F.relu(self.conv1(x, edge_index))
        # we add an additional _ here
        # return (x, connect_out.edge_index, connect_out.edge_attr,connect_out.batch, perm, score)
        # attn is set to `None` by default
        x, edge_index, edge_attr, batch, _, _ = self.pool1(
            x, edge_index, edge_attr, batch
        )
        # use global mean pool and global maximum pool

        x1 = torch.cat(
            [
                torch_geometric.nn.global_max_pool(x, batch),
                torch_geometric.nn.global_mean_pool(x, batch),
            ],
            dim=1,
        )

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, edge_attr, batch, _, _ = self.pool2(
            x, edge_index, edge_attr, batch
        )
        x2 = torch.cat(
            [
                torch_geometric.nn.global_max_pool(x, batch),
                torch_geometric.nn.global_mean_pool(x, batch),
            ],
            dim=1,
        )

        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, edge_attr, batch, _, _ = self.pool3(
            x, edge_index, edge_attr, batch
        )
        x3 = torch.cat(
            [
                torch_geometric.nn.global_max_pool(x, batch),
                torch_geometric.nn.global_mean_pool(x, batch),
            ],
            dim=1,
        )

        x = x1 + x2 + x3

        x = F.relu(self.lin1(x))
        # self.training here
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        # apply ReLU and dropout
        features = F.relu(self.lin2(x))
        out = self.lin3(features)
        if self.act is not None:
            out = self.act(out)

            if isinstance(self.act, nn.Sigmoid):
                out = out * self.output_range + self.output_shift

        return features, out


######################################################################
# Graph + Omic
######################################################################
class GraphomicNet(nn.Module):
    def __init__(self, opt, act, k):
        super(GraphomicNet, self).__init__()
        # define the graph model
        self.grph_net = GraphNet(
            grph_dim=opt.grph_dim,
            dropout_rate=opt.dropout_rate,
            use_edges=1,
            pooling_ratio=0.20,
            label_dim=opt.label_dim,
            init_max=False,
        )
        # define the omic model
        self.omic_net = MaxNet(
            input_dim=opt.input_size_omic,
            omic_dim=opt.omic_dim,
            dropout_rate=opt.dropout_rate,
            act=act,
            label_dim=opt.label_dim,
            init_max=False,
        )

        # load the saved checkpoint pt files
        if k is not None:
            pt_fname = "_%d.pt" % k
            best_grph_ckpt = torch.load(
                os.path.join(
                    opt.checkpoints_dir, opt.exp_name, "graph", "graph" + pt_fname
                ),
                map_location=torch.device("cpu"),
            )
            best_omic_ckpt = torch.load(
                os.path.join(
                    opt.checkpoints_dir, opt.exp_name, "omic", "omic" + pt_fname
                ),
                map_location=torch.device("cpu"),
            )
            self.grph_net.load_state_dict(best_grph_ckpt["model_state_dict"])
            self.omic_net.load_state_dict(best_omic_ckpt["model_state_dict"])
            print(
                "Loading Models:\n",
                os.path.join(
                    opt.checkpoints_dir, opt.exp_name, "graph", "graph" + pt_fname
                ),
                "\n",
                os.path.join(
                    opt.checkpoints_dir, opt.exp_name, "omic", "omic" + pt_fname
                ),
            )
        # use the bilinear fusion here
        self.fusion = define_bifusion(
            fusion_type=opt.fusion_type,
            skip=opt.skip,
            use_bilinear=opt.use_bilinear,
            gate1=opt.grph_gate,
            gate2=opt.omic_gate,
            dim1=opt.grph_dim,
            dim2=opt.omic_dim,
            scale_dim1=opt.grph_scale,
            scale_dim2=opt.omic_scale,
            mmhid=opt.mmhid,
            dropout_rate=opt.dropout_rate,
        )
        self.classifier = nn.Sequential(nn.Linear(opt.mmhid, opt.label_dim))
        self.act = act

        # freeze all the layers
        dfs_freeze(self.grph_net)
        dfs_freeze(self.omic_net)
        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

    def forward(self, **kwargs):
        # extract the corresponding features
        grph_vec, _ = self.grph_net(x_grph=kwargs["x_grph"])
        omic_vec, _ = self.omic_net(x_omic=kwargs["x_omic"])
        features = self.fusion(grph_vec, omic_vec)
        hazard = self.classifier(features)
        if self.act is not None:
            hazard = self.act(hazard)

            if isinstance(self.act, nn.Sigmoid):
                hazard = hazard * self.output_range + self.output_shift

        return features, hazard

    # check for the existence of an attribute within the private dictionaries
    # which should include _parameters, _buffers, and _modules
    def __hasattr__(self, name):
        if "_parameters" in self.__dict__:
            _parameters = self.__dict__["_parameters"]
            if name in _parameters:
                return True
        if "_buffers" in self.__dict__:
            _buffers = self.__dict__["_buffers"]
            if name in _buffers:
                return True
        if "_modules" in self.__dict__:
            modules = self.__dict__["_modules"]
            if name in modules:
                return True
        return False


######################################################################
# Path + Omic
######################################################################
class PathomicNet(nn.Module):
    def __init__(self, opt, act, k):
        super(PathomicNet, self).__init__()
        self.omic_net = MaxNet(
            input_dim=opt.input_size_omic,
            omic_dim=opt.omic_dim,
            dropout_rate=opt.dropout_rate,
            act=act,
            label_dim=opt.label_dim,
            init_max=False,
        )

        if k is not None:
            pt_fname = "_%d.pt" % k
            best_omic_ckpt = torch.load(
                os.path.join(
                    opt.checkpoints_dir, opt.exp_name, "omic", "omic" + pt_fname
                ),
                map_location=torch.device("cpu"),
            )
            self.omic_net.load_state_dict(best_omic_ckpt["model_state_dict"])
            print(
                "Loading Models:\n",
                os.path.join(
                    opt.checkpoints_dir, opt.exp_name, "omic", "omic" + pt_fname
                ),
            )

        self.fusion = define_bifusion(
            fusion_type=opt.fusion_type,
            skip=opt.skip,
            use_bilinear=opt.use_bilinear,
            gate1=opt.path_gate,
            gate2=opt.omic_gate,
            dim1=opt.path_dim,
            dim2=opt.omic_dim,
            scale_dim1=opt.path_scale,
            scale_dim2=opt.omic_scale,
            mmhid=opt.mmhid,
            dropout_rate=opt.dropout_rate,
        )
        self.classifier = nn.Sequential(nn.Linear(opt.mmhid, opt.label_dim))
        self.act = act

        dfs_freeze(self.omic_net)
        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

    def forward(self, **kwargs):
        path_vec = kwargs["x_path"]
        omic_vec, _ = self.omic_net(x_omic=kwargs["x_omic"])
        features = self.fusion(path_vec, omic_vec)
        hazard = self.classifier(features)
        if self.act is not None:
            hazard = self.act(hazard)

            if isinstance(self.act, nn.Sigmoid):
                hazard = hazard * self.output_range + self.output_shift

        return features, hazard

    def __hasattr__(self, name):
        if "_parameters" in self.__dict__:
            _parameters = self.__dict__["_parameters"]
            if name in _parameters:
                return True
        if "_buffers" in self.__dict__:
            _buffers = self.__dict__["_buffers"]
            if name in _buffers:
                return True
        if "_modules" in self.__dict__:
            modules = self.__dict__["_modules"]
            if name in modules:
                return True
        return False


######################################################################
# Path + Graph
######################################################################
class PathgraphNet(nn.Module):
    def __init__(self, opt, act, k):
        super(PathgraphNet, self).__init__()
        self.grph_net = GraphNet(
            grph_dim=opt.grph_dim,
            dropout_rate=opt.dropout_rate,
            use_edges=1,
            pooling_ratio=0.20,
            label_dim=opt.label_dim,
            init_max=False,
        )

        if k is not None:
            pt_fname = "_%d.pt" % k
            best_grph_ckpt = torch.load(
                os.path.join(
                    opt.checkpoints_dir, opt.exp_name, "graph", "graph" + pt_fname
                ),
                map_location=torch.device("cpu"),
            )
            self.grph_net.load_state_dict(best_grph_ckpt["model_state_dict"])
            print(
                "Loading Models:\n",
                os.path.join(
                    opt.checkpoints_dir, opt.exp_name, "graph", "graph" + pt_fname
                ),
            )

        self.fusion = define_bifusion(
            fusion_type=opt.fusion_type,
            skip=opt.skip,
            use_bilinear=opt.use_bilinear,
            gate1=opt.path_gate,
            gate2=opt.grph_gate,
            dim1=opt.path_dim,
            dim2=opt.grph_dim,
            scale_dim1=opt.path_scale,
            scale_dim2=opt.grph_scale,
            mmhid=opt.mmhid,
            dropout_rate=opt.dropout_rate,
        )
        self.classifier = nn.Sequential(nn.Linear(opt.mmhid, opt.label_dim))
        self.act = act

        dfs_freeze(self.grph_net)
        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

    def forward(self, **kwargs):
        path_vec = kwargs["x_path"]
        grph_vec, _ = self.grph_net(x_grph=kwargs["x_grph"])
        features = self.fusion(path_vec, grph_vec)
        hazard = self.classifier(features)
        if self.act is not None:
            hazard = self.act(hazard)

            if isinstance(self.act, nn.Sigmoid):
                hazard = hazard * self.output_range + self.output_shift

        return features, hazard

    def __hasattr__(self, name):
        if "_parameters" in self.__dict__:
            _parameters = self.__dict__["_parameters"]
            if name in _parameters:
                return True
        if "_buffers" in self.__dict__:
            _buffers = self.__dict__["_buffers"]
            if name in _buffers:
                return True
        if "_modules" in self.__dict__:
            modules = self.__dict__["_modules"]
            if name in modules:
                return True
        return False


######################################################################
# Path + Graph + Omic
######################################################################
class PathgraphomicNet(nn.Module):
    def __init__(self, opt, act, k):
        super(PathgraphomicNet, self).__init__()
        self.grph_net = GraphNet(
            grph_dim=opt.grph_dim,
            dropout_rate=opt.dropout_rate,
            use_edges=1,
            pooling_ratio=0.20,
            label_dim=opt.label_dim,
            init_max=False,
        )
        self.omic_net = MaxNet(
            input_dim=opt.input_size_omic,
            omic_dim=opt.omic_dim,
            dropout_rate=opt.dropout_rate,
            act=act,
            label_dim=opt.label_dim,
            init_max=False,
        )

        # load the checkpoint .pt files
        if k is not None:
            pt_fname = "_%d.pt" % k
            best_grph_ckpt = torch.load(
                os.path.join(
                    opt.checkpoints_dir, opt.exp_name, "graph", "graph" + pt_fname
                ),
                map_location=torch.device("cpu"),
            )
            best_omic_ckpt = torch.load(
                os.path.join(
                    opt.checkpoints_dir, opt.exp_name, "omic", "omic" + pt_fname
                ),
                map_location=torch.device("cpu"),
            )
            self.grph_net.load_state_dict(best_grph_ckpt["model_state_dict"])
            self.omic_net.load_state_dict(best_omic_ckpt["model_state_dict"])
            print(
                "Loading Models:\n",
                os.path.join(
                    opt.checkpoints_dir, opt.exp_name, "graph", "graph" + pt_fname
                ),
                "\n",
                os.path.join(
                    opt.checkpoints_dir, opt.exp_name, "omic", "omic" + pt_fname
                ),
            )

            self.fusion = define_trifusion(
                fusion_type=opt.fusion_type,
                skip=opt.skip,
                use_bilinear=opt.use_bilinear,
                gate1=opt.path_gate,
                gate2=opt.grph_gate,
                gate3=opt.omic_gate,
                dim1=opt.path_dim,
                dim2=opt.grph_dim,
                dim3=opt.omic_dim,
                scale_dim1=opt.path_scale,
                scale_dim2=opt.grph_scale,
                scale_dim3=opt.omic_scale,
                mmhid=opt.mmhid,
                dropout_rate=opt.dropout_rate,
            )
            self.classifier = nn.Sequential(nn.Linear(opt.mmhid, opt.label_dim))
            self.act = act

            dfs_freeze(self.grph_net)
            dfs_freeze(self.omic_net)
            self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
            self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

    def forward(self, **kwargs):
        path_vec = kwargs["x_path"]
        grph_vec, _ = self.grph_net(x_grph=kwargs["x_grph"])
        omic_vec, _ = self.omic_net(x_omic=kwargs["x_omic"])
        features = self.fusion(path_vec, grph_vec, omic_vec)
        hazard = self.classifier(features)
        if self.act is not None:
            hazard = self.act(hazard)

            if isinstance(self.act, nn.Sigmoid):
                hazard = hazard * self.output_range + self.output_shift

        return features, hazard

    def __hasattr__(self, name):
        if "_parameters" in self.__dict__:
            _parameters = self.__dict__["_parameters"]
            if name in _parameters:
                return True
        if "_buffers" in self.__dict__:
            _buffers = self.__dict__["_buffers"]
            if name in _buffers:
                return True
        if "_modules" in self.__dict__:
            modules = self.__dict__["_modules"]
            if name in modules:
                return True
        return False


######################################################################
# Path + Path
######################################################################


class PathpathNet(nn.Module):
    def __init__(self, opt, act, k):
        super(PathpathNet, self).__init__()
        self.fusion = define_bifusion(
            fusion_type=opt.fusion_type,
            skip=opt.skip,
            use_bilinear=opt.use_bilinear,
            gate1=opt.path_gate,
            gate2=1 - opt.path_gate if opt.path_gate else 0,
            dim1=opt.path_dim,
            dim2=opt.path_dim,
            scale_dim1=opt.path_scale,
            scale_dim2=opt.path_scale,
            mmhid=opt.mmhid,
            dropout_rate=opt.dropout_rate,
        )
        self.classifier = nn.Sequential(nn.Linear(opt.mmhid, opt.label_dim))
        self.act = act
        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

    def forward(self, **kwargs):
        path_vec = kwargs["x_path"]
        features = self.fusion(path_vec, path_vec)
        hazard = self.classifier(features)
        if self.act is not None:
            hazard = self.act(hazard)
            if isinstance(self.act, nn.Sigmoid):
                hazard = hazard * self.output_range + self.output_shift
        return features, hazard

    def __hasattr__(self, name):
        if "_parameters" in self.__dict__:
            _parameters = self.__dict__["_parameters"]
            if name in _parameters:
                return True
        if "_buffers" in self.__dict__:
            _buffers = self.__dict__["_buffers"]
            if name in _buffers:
                return True
        if "_modules" in self.__dict__:
            modules = self.__dict__["_modules"]
            if name in modules:
                return True
        return False


######################################################################
# Graph + Graph
######################################################################


class GraphgraphNet(nn.Module):
    def __init__(self, opt, act, k):
        super(GraphgraphNet, self).__init__()
        self.grph_net = GraphNet(
            grph_dim=opt.grph_dim,
            dropout_rate=opt.dropout_rate,
            use_edges=1,
            pooling_ratio=0.20,
            label_dim=opt.label_dim,
            init_max=False,
        )
        if k is not None:
            pt_fname = "_%d.pt" % k
            best_grph_ckpt = torch.load(
                os.path.join(
                    opt.checkpoints_dir, opt.exp_name, "graph", "graph" + pt_fname
                ),
                map_location=torch.device("cpu"),
            )
            self.grph_net.load_state_dict(best_grph_ckpt["model_state_dict"])
            print(
                "Loading Models:\n",
                os.path.join(
                    opt.checkpoints_dir, opt.exp_name, "graph", "graph" + pt_fname
                ),
            )
        self.fusion = define_bifusion(
            fusion_type=opt.fusion_type,
            skip=opt.skip,
            use_bilinear=opt.use_bilinear,
            gate1=opt.grph_gate,
            gate2=1 - opt.grph_gate if opt.grph_gate else 0,
            dim1=opt.grph_dim,
            dim2=opt.grph_dim,
            scale_dim1=opt.grph_scale,
            scale_dim2=opt.grph_scale,
            mmhid=opt.mmhid,
            dropout_rate=opt.dropout_rate,
        )
        self.classifier = nn.Sequential(nn.Linear(opt.mmhid, opt.label_dim))
        self.act = act
        dfs_freeze(self.grph_net)
        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

    def forward(self, **kwargs):
        grph_vec, _ = self.grph_net(x_grph=kwargs["x_grph"])
        features = self.fusion(grph_vec, grph_vec)
        hazard = self.classifier(features)
        if self.act is not None:
            hazard = self.act(hazard)
            if isinstance(self.act, nn.Sigmoid):
                hazard = hazard * self.output_range + self.output_shift
        return features, hazard

    def __hasattr__(self, name):
        if "_parameters" in self.__dict__:
            _parameters = self.__dict__["_parameters"]
            if name in _parameters:
                return True
        if "_buffers" in self.__dict__:
            _buffers = self.__dict__["_buffers"]
            if name in _buffers:
                return True
        if "_modules" in self.__dict__:
            modules = self.__dict__["_modules"]
            if name in modules:
                return True
        return False


######################################################################
# Omic + Omic
######################################################################


class OmicomicNet(nn.Module):
    def __init__(self, opt, act, k):
        super(OmicomicNet, self).__init__()
        self.omic_net = MaxNet(
            input_dim=opt.input_size_omic,
            omic_dim=opt.omic_dim,
            dropout_rate=opt.dropout_rate,
            act=act,
            label_dim=opt.label_dim,
            init_max=False,
        )
        if k is not None:
            pt_fname = "_%d.pt" % k
            best_omic_ckpt = torch.load(
                os.path.join(
                    opt.checkpoints_dir, opt.exp_name, "omic", "omic" + pt_fname
                ),
                map_location=torch.device("cpu"),
            )
            self.omic_net.load_state_dict(best_omic_ckpt["model_state_dict"])
            print(
                "Loading Models:\n",
                os.path.join(
                    opt.checkpoints_dir, opt.exp_name, "omic", "omic" + pt_fname
                ),
            )
        self.fusion = define_bifusion(
            fusion_type=opt.fusion_type,
            skip=opt.skip,
            use_bilinear=opt.use_bilinear,
            gate1=opt.omic_gate,
            gate2=1 - opt.omic_gate if opt.omic_gate else 0,
            dim1=opt.omic_dim,
            dim2=opt.omic_dim,
            scale_dim1=opt.omic_scale,
            scale_dim2=opt.omic_scale,
            mmhid=opt.mmhid,
            dropout_rate=opt.dropout_rate,
        )
        self.classifier = nn.Sequential(nn.Linear(opt.mmhid, opt.label_dim))
        self.act = act
        dfs_freeze(self.omic_net)
        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)

    def forward(self, **kwargs):
        omic_vec, _ = self.omic_net(x_omic=kwargs["x_omic"])
        features = self.fusion(omic_vec, omic_vec)
        hazard = self.classifier(features)
        if self.act is not None:
            hazard = self.act(hazard)
            if isinstance(self.act, nn.Sigmoid):
                hazard = hazard * self.output_range + self.output_shift
        return features, hazard

    def __hasattr__(self, name):
        if "_parameters" in self.__dict__:
            _parameters = self.__dict__["_parameters"]
            if name in _parameters:
                return True
        if "_buffers" in self.__dict__:
            _buffers = self.__dict__["_buffers"]
            if name in _buffers:
                return True
        if "_modules" in self.__dict__:
            modules = self.__dict__["_modules"]
            if name in modules:
                return True
        return False
