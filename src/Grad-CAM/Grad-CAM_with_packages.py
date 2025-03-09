"""!@file Grad-CAM_with_packages.py
@breif This file is used to create the Grad-CAM using the code from the pytorch-grad-cam package.
@details The package is sourced from (https://github.com/jacobgil/pytorch-grad-cam).
@author Shizhe Xu
@date 28 June 2024
"""

import os
import sys
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(module_path)
import os
import argparse
from pytorch_grad_cam import (
    GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus,
    AblationCAM, XGradCAM, EigenCAM, EigenGradCAM,
    LayerCAM, FullGrad, GradCAMElementWise
)
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
from pytorch_grad_cam.utils.image import (
    show_cam_on_image, deprocess_image, preprocess_image
)
from torchvision.models import vgg19, resnet50
from pytorch_grad_cam import GuidedBackpropReLUModel

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image-path',
        type=str,
        default='./examples/both.png',
        help='Input image path')
    parser.add_argument('--method', type=str, default='gradcam',
                        choices=[
                            'gradcam', 'hirescam', 'gradcam++',
                            'scorecam', 'xgradcam', 'ablationcam',
                            'eigencam', 'eigengradcam', 'layercam',
                            'fullgrad', 'gradcamelementwise'
                        ],
                        help='CAM method')

    parser.add_argument('--output_dir', type=str, default='Grad_CAM_output_with_packages',
                        help='Output directory to save the images')
    args = parser.parse_args()
    return args

methods = {
        "gradcam": GradCAM,
        "hirescam": HiResCAM,
        "scorecam": ScoreCAM,
        "gradcam++": GradCAMPlusPlus,
        "ablationcam": AblationCAM,
        "xgradcam": XGradCAM,
        "eigencam": EigenCAM,
        "eigengradcam": EigenGradCAM,
        "layercam": LayerCAM,
        "fullgrad": FullGrad,
        "gradcamelementwise": GradCAMElementWise
    }

device = torch.device('cpu')
args = get_args()
opt = parse_args()
# model = define_net(opt, None)
# optimizer = define_optimizer(opt, model)
# scheduler = define_scheduler(opt, optimizer)
# model = vgg19(pretrained=True).to(device)
model = resnet50(pretrained=True).to(device)
print(model)
# print("Number of Trainable Parameters: %d" % count_parameters(model))
# print("Activation Type:", opt.act_type)
# print("Optimizer Type:", opt.optimizer_type)
# print("Regularization Type:", opt.reg_type)
model.eval()

target_layers = [model.layer4]
# target_layers = [model.features[36]]

img_path = 'cell_graph_reconstruction/example_data/imgs/TCGA-06-0174-01Z-00-DX3.23b6e12e-dfc1-4c6f-903e-170038a0e055_1.png'
rgb_img = cv2.imread(img_path, 1)[:, :, ::-1]
# ensure the input image should np.float32 in the range [0, 1]
rgb_img = np.float32(rgb_img) / 255
input_tensor = preprocess_image(rgb_img,
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]).to(device)

targets = None
cam_algorithm = methods[args.method]
with cam_algorithm(model=model,
                    target_layers=target_layers) as cam:

    # it is possible to override the internal batch size for faster computation
    cam.batch_size = 32
    grayscale_cam = cam(input_tensor=input_tensor,
                        targets=targets,
                        aug_smooth=False,
                        eigen_smooth=False)

    grayscale_cam = grayscale_cam[0, :]

    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

gb_model = GuidedBackpropReLUModel(model=model, device=opt.device)
gb = gb_model(input_tensor, target_category=None)

cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
cam_gb = deprocess_image(cam_mask * gb)
gb = deprocess_image(gb)

os.makedirs(args.output_dir, exist_ok=True)

plt.imshow(cam_image)
plt.show()
cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
cam_output_path = os.path.join(args.output_dir, f'{args.method}_cam.jpg')
gb_output_path = os.path.join(args.output_dir, f'{args.method}_gb.jpg')
cam_gb_output_path = os.path.join(args.output_dir, f'{args.method}_cam_gb.jpg')

cv2.imwrite(cam_output_path, cam_image)
cv2.imwrite(gb_output_path, gb)
cv2.imwrite(cam_gb_output_path, cam_gb)
