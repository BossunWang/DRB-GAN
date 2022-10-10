"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import torch
import torch.nn as nn
from torchvision import models


def normalize(x, eps=1e-10):
    return x * torch.rsqrt(torch.sum(x ** 2, dim=1, keepdim=True) + eps)


class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = models.alexnet(pretrained=True).features
        self.channels = []
        for layer in self.layers:
            if isinstance(layer, nn.Conv2d):
                self.channels.append(layer.out_channels)

    def forward(self, x):
        fmaps = []
        for layer in self.layers:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                fmaps.append(x)
        return fmaps


class Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False))

    def forward(self, x):
        return self.main(x)


class LPIPS(nn.Module):
    def __init__(self):
        super().__init__()
        self.alexnet = AlexNet()
        self.lpips_weights = nn.ModuleList()
        for channels in self.alexnet.channels:
            self.lpips_weights.append(Conv1x1(channels, 1))
        self._load_lpips_weights()
        # imagenet normalization for range [-1, 1]
        self.mu = torch.tensor([-0.03, -0.088, -0.188]).view(1, 3, 1, 1).cuda()
        self.sigma = torch.tensor([0.458, 0.448, 0.450]).view(1, 3, 1, 1).cuda()

    def _load_lpips_weights(self):
        own_state_dict = self.state_dict()
        if torch.cuda.is_available():
            state_dict = torch.load('lpips_weights.ckpt')
        else:
            state_dict = torch.load('lpips_weights.ckpt',
                                    map_location=torch.device('cpu'))
        for name, param in state_dict.items():
            if name in own_state_dict:
                own_state_dict[name].copy_(param)

    def forward(self, x, y):
        x = (x - self.mu) / self.sigma
        y = (y - self.mu) / self.sigma
        x_fmaps = self.alexnet(x)
        y_fmaps = self.alexnet(y)
        lpips_value = 0
        for x_fmap, y_fmap, conv1x1 in zip(x_fmaps, y_fmaps, self.lpips_weights):
            x_fmap = normalize(x_fmap)
            y_fmap = normalize(y_fmap)
            lpips_value += torch.mean(conv1x1((x_fmap - y_fmap) ** 2))
        return lpips_value


@torch.no_grad()
def calculate_lpips_given_images(group_of_images, lpips):
    lpips_values = []

    for i in range(1, len(group_of_images)):
        lpips_values.append(lpips(group_of_images[0], group_of_images[i]))
    lpips_value = torch.mean(torch.stack(lpips_values, dim=0))
    return lpips_value.item()


if __name__ == '__main__':
    import argparse
    import os
    from PIL import Image
    from torchvision import transforms as T
    import numpy as np
    from tqdm import tqdm

    parser = argparse.ArgumentParser(description='LPIPS metric')
    parser.add_argument('--src_dir', type=str, default='', help='target dataset path')
    parser.add_argument('--tgt_dir', type=str, default='', help='test dataset path')

    args = parser.parse_args()

    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    test_transform = [T.ToTensor(), T.Normalize(mean=mean, std=std)]
    test_transform = T.Compose(test_transform)

    label_dict = os.listdir(args.tgt_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lpips = LPIPS().eval().to(device)

    # Reference-guided
    lpips_values = []
    for dirPath, dirNames, fileNames in os.walk(args.src_dir):
        for f in tqdm(fileNames):
            path = os.path.join(dirPath, f)
            image = Image.open(path).convert('RGB')
            image = test_transform(image).to(device)

            # compared with all style
            group_of_images = [image]
            for style_label in label_dict:
                stylized_img_path = os.path.join(args.tgt_dir, style_label, f)
                stylized_image = Image.open(stylized_img_path).convert('RGB')
                stylized_image = test_transform(stylized_image).to(device)
                group_of_images.append(stylized_image)

            lpips_value = calculate_lpips_given_images(group_of_images, lpips)
            lpips_values.append(lpips_value)

    # calculate LPIPS for all style
    lpips_mean = np.array(lpips_values).mean()
    print("Reference-guided LPIPS values for all style:", lpips_mean)
