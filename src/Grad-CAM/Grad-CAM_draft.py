"""!@file Grad-CAM_draft.py
@breif This file si our first draft of the Grad-CAM implementation.
@author Shizhe Xu
@date 28 June 2024
"""

import os
import sys
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(module_path)
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt

from options import parse_args
from networks import define_net, define_reg, define_optimizer, define_scheduler
from functions import count_parameters
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, ClassifierOutputSoftmaxTarget
from torchvision.models import vgg19

class RegressionOutputTarget:
    def __init__(self):
        pass

    def __call__(self, model_output):
        # Target the single output neuron
        return model_output
    
class ModelWrapper(nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x_path=x)[1]

opt = parse_args()
# model = define_net(opt, None)
# optimizer = define_optimizer(opt, model)
# scheduler = define_scheduler(opt, optimizer)
model = vgg19(pretrained=True)
print(model)
# print("Number of Trainable Parameters: %d" % count_parameters(model))
# print("Activation Type:", opt.act_type)
# print("Optimizer Type:", opt.optimizer_type)
# print("Regularization Type:", opt.reg_type)
model.eval()

wrapped_model = ModelWrapper(model)
img_path = 'cell_graph_reconstruction/example_data/imgs/TCGA-06-0174-01Z-00-DX3.23b6e12e-dfc1-4c6f-903e-170038a0e055_1.png'
single_X_path = Image.open(img_path).convert("RGB")
preprocess = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomVerticalFlip(0.5),
                transforms.RandomCrop(opt.input_size_path),
                transforms.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.05, hue=0.01
                ),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
input_tensor = preprocess(single_X_path)
single_X_path = single_X_path.resize((512, 512))
input_tensor = input_tensor.unsqueeze(0)
# print(input_tensor.shape)
# cam = GradCAM(model=wrapped_model, target_layers=[model.features[-1]])
cam = GradCAM(model=model, target_layers=[model.features[-1]])
# targets = [RegressionOutputTarget()]
# targets = [ClassifierOutputTarget(281)]
targets = None
# generate the CAM with smoothing options
grayscale_cam = cam(input_tensor=input_tensor, targets=targets, aug_smooth=True, eigen_smooth=True)

grayscale_cam = grayscale_cam[0, :]
# convert the PIL image to a NumPy array and normalize it
single_X_path = np.array(single_X_path) / 255.0
single_X_path = single_X_path.astype(np.float32)
# visualize the CAM
visualization = show_cam_on_image(single_X_path, grayscale_cam, use_rgb=True)

plt.imshow(visualization)
plt.show()
