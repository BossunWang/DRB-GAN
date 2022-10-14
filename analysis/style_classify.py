import argparse
import sys
import os
import numpy as np
from tqdm import tqdm
import cv2
import shutil
import torch
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from torch.nn import functional as F

abs_path = os.getcwd().split('DRB-GAN')[0]
sys.path.append(os.path.join(abs_path, 'DRB-GAN', "model"))
sys.path.append(os.path.join(abs_path, 'DRB-GAN'))

from data.dataset import ImageDataset, ImageClassDataset
from model.StyleEncodNet import StyleEncodingNetwork
from model.StyleTransferNet import StyleTransferNetwork
from model.DiscriminativeNet import Discriminator
from model.vgg_nets import Vgg19


def main(conf):
    conf.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data setting
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)

    # dataset setting
    train_transform = [T.Resize(conf.tgt_img_size, InterpolationMode.BICUBIC)
                       , T.CenterCrop(conf.tgt_crop_size)
                       , T.ToTensor()
                       , T.Normalize(mean=mean, std=std)]
    train_transform = T.Compose(train_transform)

    train_data_tgt = ImageClassDataset(conf.tgt_dataset, train_transform, sample_size=1)
    test_data_tgt = ImageDataset(conf.tgt_dataset, train_transform, get_path=True)
    label_dict = train_data_tgt.label_dict
    num_classes = len(train_data_tgt)
    print('num_classes:', num_classes)

    # create folders
    if not os.path.exists(conf.incorrect_dir):
        os.makedirs(conf.incorrect_dir)

    for label_str in label_dict:
        dir_path = os.path.join(conf.incorrect_dir, label_str)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    test_loader_tgt = torch.utils.data.DataLoader(test_data_tgt
                                                  , batch_size=1
                                                  , shuffle=False
                                                  , num_workers=conf.workers
                                                  , pin_memory=False)

    # model setting
    VGG = Vgg19().to(conf.device)
    VGG.eval()
    style_encoding_net = StyleEncodingNetwork(conf.feature_dim
                                              , num_classes
                                              , VGG
                                              , conf.gamma_dim
                                              , conf.beta_dim
                                              , conf.omega_dim).to(conf.device)

    checkpoint = torch.load(conf.pretrain_model, map_location=conf.device)
    if "iter" in checkpoint \
            and "style_encoder" in checkpoint \
            and "generator" in checkpoint \
            and "discriminator" in checkpoint \
            and "G_optimizer" in checkpoint \
            and "D_optimizer" in checkpoint:
        print('load models')
        style_encoding_net.load_state_dict(checkpoint['style_encoder'])

    # evaluation
    style_encoding_net.eval()

    # evaluation
    data_size = len(test_data_tgt)
    correct_count = 0

    for it, (style_images, style_img_path) in enumerate(tqdm(test_loader_tgt)):
        style_img_path = style_img_path[0]
        style_label = None
        for si, label_str in enumerate(label_dict):
            if style_img_path.find(label_str) > 0:
                style_label = si
                break
        if style_label is None:
            continue

        with torch.no_grad():
            style_label = torch.Tensor([style_label]).long().to(conf.device)
            style_images = style_images.to(conf.device)
            style_prob, style_gamma, style_beta, style_omega = style_encoding_net(style_images, style_label)
            # saved incorrect images
            if style_prob.argmax() != style_label:
                incorrect_img_path = style_img_path.replace(conf.tgt_dataset, conf.incorrect_dir)
                incorrect_extend_name \
                    = "_" + label_dict[style_label] + "_to_" + label_dict[style_prob.argmax()] + ".jpg"
                incorrect_img_path = incorrect_img_path.replace(".jpg", incorrect_extend_name)
                shutil.copyfile(style_img_path, incorrect_img_path)
            else:
                correct_count += 1

    print("class accuracy:", correct_count / data_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DRB-GAN test')
    parser.add_argument('--tgt_dataset', type=str, default='', help='target dataset path')

    parser.add_argument('--tgt_img_size', type=int, default=256, help='The size of image: H and W')
    parser.add_argument('--tgt_crop_size', type=int, default=256, help='The size of cropped image: H and W')
    parser.add_argument('--workers', type=int, default=1, help='The number of workers for dataloader')

    parser.add_argument('--feature_dim', type=int, default=156160, help='The dimension of style encoder feature')
    parser.add_argument('--gamma_dim', type=int, default=256, help='The dimension of style code for ADIN mean')
    parser.add_argument('--beta_dim', type=int, default=256, help='The dimension of style code for ADIN std')
    parser.add_argument('--omega_dim', type=int, default=4, help='The dimension of style code for DConv')
    parser.add_argument('--encoder_out_ch', type=int, default=128, help='The dimension of content encoder output')
    parser.add_argument('--db_number', type=int, default=4, help='The number of Dynamic ResBlock')
    parser.add_argument('--ws', type=int, default=64, help='The window size of SW-LIN Decoder')
    parser.add_argument('--M', type=int, default=2, help='The number of style reference image for Discriminator')

    parser.add_argument('--vgg_model', type=str, default='vgg19-dcbb9e9d.pth',
                        help='file name to load the vgg model for feature extraction')
    parser.add_argument('--pretrain_model', type=str, default='checkpoint/',
                        help='file name to load the model for training')
    parser.add_argument('--incorrect_dir', type=str, default='incorrect',
                        help='Directory name to save the incorrect style prediction images')
    args = parser.parse_args()

    main(args)