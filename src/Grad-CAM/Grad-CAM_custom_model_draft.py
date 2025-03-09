"""!@file Grad-CAM_custom_model_draft.py
@breif This file is used to create the Grad-CAM for the model from networks.py rather than importing from torchvision.models.
@details The file uses the GradCAM class to generate the CAM for the model from networks.py.
@author Shizhe Xu
@date 28 June 2024
"""

import os
import sys
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(module_path)
# from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.models as models

from options import parse_args
from networks import define_net, define_reg, define_optimizer, define_scheduler, get_vgg
from functions import count_parameters
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, ClassifierOutputSoftmaxTarget
from torchvision.models import vgg19
from torch.autograd import Function

class GradCAM:
    def __init__(self, model, target_layer_name):
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
            print(f"Forward hook activated. Activations shape: {output.shape}")

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
            print(f"Backward hook activated. Gradients shape: {grad_out[0].shape}")

        # Get the actual target layer using the target_layer_name
        target_layer = dict([*self.model.named_modules()])[self.target_layer_name]

        # Register hooks
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_image, target_class=None):
        features = self.model.features(input_image.unsqueeze(0))
        flattened_features = features.view(features.size(0), -1)
        print(f"Flattened features shape: {flattened_features.shape}")
        model_output = self.model.classifier(flattened_features)
        jitter = torch.randn_like(model_output) * 1e-5  # Small jitter
        model_output = model_output + jitter
        print(f"model output shape is", model_output)
        # model_output = self.model.linear(model_output)
        # model_output = self.model.classifier(flattened_features)
        print(f"model output shape is", model_output.shape)
        if target_class is None:
            target_class = model_output.argmax(dim=1).item()
        print(f"Target class: {target_class}")
        self.model.zero_grad()

        one_hot_output = torch.zeros(model_output.shape, device=input_image.device)
        one_hot_output[0][target_class] = 1
        # print(one_hot_output)
        model_output.backward(gradient=one_hot_output)

        if self.gradients is None:
            raise RuntimeError("Gradients not captured. Make sure the backward hook is working correctly.")
        print(f"Gradients captured. Shape: {self.gradients.shape}")

        # weights = torch.mean(self.gradients, dim=[0, 2, 3])
        # cam = torch.sum(weights * self.activations, dim=0)

        # Calculate the weights and the CAM
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)

        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / cam.max()
        return cam.squeeze().cpu().numpy()
    
        # if self.activations is not None:
        #     self.activations.requires_grad = True
        
        # target = output[:, class_idx]
        # print(f"Target for backward pass: {target.item()}")
        # target.backward()

        # gradients = torch.autograd.grad(outputs=target, inputs=self.activations,
        #                                 grad_outputs=torch.ones_like(target),
        #                                 create_graph=True, retain_graph=True, allow_unused=True)[0]

        # if self.activations is None:
        #     raise RuntimeError("Activations not captured. Make sure the forward hook is working correctly.")
        
        # # if gradients is None:
        # #     raise RuntimeError("Gradients not captured. Make sure the backward hook is working correctly.")
        # if self.gradients is None:
        #     raise RuntimeError("Gradients not captured. Make sure the backward hook is working correctly.")

        # print(f"Gradients shape: {self.gradients.shape}")
        # print(f"Activations shape: {self.activations.shape}")

        # gradients = self.gradients
        # activations = self.activations

        # b, k, u, v = gradients.size()
        # alpha = gradients.view(b, k, -1).mean(2)
        # weights = alpha.view(b, k, 1, 1)
        # heatmap = (weights * activations).sum(1, keepdim=True)

        # heatmap = F.relu(heatmap)
        # heatmap = F.interpolate(heatmap, size=(input_image.size(2), input_image.size(3)), mode='bilinear', align_corners=False)
        # heatmap = heatmap.view(input_image.size(2), input_image.size(3)).cpu().data.numpy()
        # heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        
        # return heatmap

    # def visualize_heatmap(self, heatmap, original_image):
    #     plt.imshow(original_image.permute(1, 2, 0).cpu().data.numpy(), alpha=0.5)
    #     plt.imshow(heatmap, cmap='jet', alpha=0.5)
    #     plt.colorbar()
    #     plt.show()

def show_cam_on_image(img, mask):
    if mask.ndim > 2:
        mask = mask.squeeze()
    # Resize the mask to match the input image dimensions
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

opt = parse_args()
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model = define_net(opt, None)
# model = get_vgg(cfg="E", batch_norm=True, pretrained=True).to(device)
# model = vgg19(pretrained=True).to(device)
# model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).to(device)
optimizer = define_optimizer(opt, model)
scheduler = define_scheduler(opt, optimizer)
for name, module in model.named_modules():
    print(name)
model.eval()

img_path = 'cell_graph_reconstruction/example_data/imgs/TCGA-06-0174-01Z-00-DX3.23b6e12e-dfc1-4c6f-903e-170038a0e055_1.png'
single_X_path = Image.open(img_path).convert("RGB")
# preprocess = transforms.Compose(
#             [
#                 transforms.RandomHorizontalFlip(0.5),
#                 transforms.RandomVerticalFlip(0.5),
#                 transforms.RandomCrop(opt.input_size_path),
#                 transforms.ColorJitter(
#                     brightness=0.1, contrast=0.1, saturation=0.05, hue=0.01
#                 ),
#                 transforms.ToTensor(),
#                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#             ]
#         )
preprocess = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
input_image = preprocess(single_X_path).to(device)
# input_image = input_tensor.unsqueeze(0)

# target_layer = model.features[-1]
target_layer_name = "features.32"

grad_cam = GradCAM(model, target_layer_name)
cam_mask = grad_cam.generate_cam(input_image)
# gradcam = GradCAM(model, target_layer)

# heatmap = gradcam.generate_heatmap(input_image)

original_image = cv2.imread(img_path, 1)
original_image = cv2.resize(original_image, (224, 224))
original_image = np.float32(original_image) / 255

# generate the CAM overlay on the image
cam_image = show_cam_on_image(original_image, cam_mask)

plt.imshow(cam_image)
plt.axis('off')
plt.show()