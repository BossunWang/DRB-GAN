import argparse
import sys
import os
import numpy as np
from tqdm import tqdm
import cv2
import torch
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

abs_path = os.getcwd().split('DRB-GAN')[0]
sys.path.append(os.path.join(abs_path, 'DRB-GAN', "model"))

from data.dataset import ImageDataset, ImageClassDataset
from model.StyleEncodNet import StyleEncodingNetwork
from model.StyleTransferNet import StyleTransferNetwork
from model.DiscriminativeNet import Discriminator
from model.vgg_nets import Vgg19


def test(source, targets, style_encoder, generator, cur_it, label_dict, mean, std, conf):
    style_mix_gamma = torch.zeros((1, conf.gamma_dim)).to(conf.device)
    style_mix_beta = torch.zeros((1, conf.beta_dim)).to(conf.device)
    style_mix_omega = torch.zeros((1, conf.omega_dim)).to(conf.device)

    for ti, target in enumerate(targets):
        with torch.no_grad():
            style_label = torch.Tensor([ti]).long()
            style_prob, style_gamma, style_beta, style_omega = style_encoder(target.unsqueeze(0), style_label)
            fake_img = generator(source, style_gamma, style_beta, style_omega)

        result = torch.cat((source[0], fake_img[0]), 2).detach().cpu().numpy().transpose(1, 2, 0)
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        result = (result * std) + mean
        result = np.clip(result * 255.0, 0, 255)
        path = os.path.join(conf.sample_dir,
                            str(cur_it) + '_test_style_' + label_dict[ti] + '.png')
        cv2.imwrite(path, result)

        # collect style mixture code
        if label_dict[ti] in conf.mixture_list:
            li = conf.mixture_list.index(label_dict[ti])
            style_mix_gamma += conf.mixture_weights[li] * style_gamma
            style_mix_beta += conf.mixture_weights[li] * style_beta
            style_mix_omega += conf.mixture_weights[li] * style_omega

    # style mixed
    mixture_img = generator(source, style_mix_gamma, style_mix_beta, style_mix_omega)
    mixture_result = torch.cat((source[0], mixture_img[0]), 2).detach().cpu().numpy().transpose(1, 2, 0)
    mixture_result = cv2.cvtColor(mixture_result, cv2.COLOR_BGR2RGB)
    mixture_result = (mixture_result * std) + mean
    mixture_result = np.clip(mixture_result * 255.0, 0, 255)
    path = os.path.join(conf.sample_dir,
                        str(cur_it) + '_mix_style_' + '_'.join(conf.mixture_list) + '.png')
    cv2.imwrite(path, mixture_result)


def main(conf):
    conf.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data setting
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)

    mean_array = np.array(mean).reshape(1, 1, -1)
    std_array = np.array(std).reshape(1, 1, -1)

    # dataset setting
    train_transform = [T.Resize(conf.tgt_img_size, InterpolationMode.BICUBIC)
                       , T.CenterCrop(conf.tgt_crop_size)
                       , T.ToTensor()
                       , T.Normalize(mean=mean, std=std)]
    train_transform = T.Compose(train_transform)

    test_transform = [T.ToTensor()
                      , T.Normalize(mean=mean, std=std)]
    test_transform = T.Compose(test_transform)

    train_data_tgt = ImageClassDataset(conf.tgt_dataset, train_transform, sample_size=1)
    test_data_src = ImageDataset(conf.test_dataset, test_transform)
    label_dict = train_data_tgt.label_dict

    test_loader_src = torch.utils.data.DataLoader(test_data_src
                                                  , batch_size=1
                                                  , num_workers=conf.workers
                                                  , pin_memory=False
                                                  , shuffle=False)
    test_loader_tgt = torch.utils.data.DataLoader(train_data_tgt
                                                  , batch_size=1
                                                  , shuffle=False
                                                  , num_workers=conf.workers
                                                  , pin_memory=False)

    target_test_batch_iterator = iter(test_loader_tgt)

    num_classes = len(train_data_tgt)

    print('num_classes:', num_classes)
    print("mixture list:", conf.mixture_list)
    print("mixture weight:", conf.mixture_weights)

    # model setting
    VGG = Vgg19().to(conf.device)
    VGG.eval()
    style_encoding_net = StyleEncodingNetwork(conf.feature_dim
                                              , num_classes
                                              , VGG
                                              , conf.gamma_dim
                                              , conf.beta_dim
                                              , conf.omega_dim).to(conf.device)
    style_transfer_net = StyleTransferNetwork(conf.encoder_out_ch
                                              , conf.gamma_dim
                                              , conf.beta_dim
                                              , conf.omega_dim
                                              , conf.db_number, conf.ws).to(conf.device)

    checkpoint = torch.load(conf.pretrain_model, map_location=conf.device)
    if "iter" in checkpoint \
            and "style_encoder" in checkpoint \
            and "generator" in checkpoint \
            and "discriminator" in checkpoint \
            and "G_optimizer" in checkpoint \
            and "D_optimizer" in checkpoint:
        print('load models')
        style_encoding_net.load_state_dict(checkpoint['style_encoder'])
        style_transfer_net.load_state_dict(checkpoint['generator'])

    # evaluation
    style_encoding_net.eval()
    style_transfer_net.eval()

    for it, content_image in enumerate(tqdm(test_loader_src)):
        content_image = content_image.to(conf.device)

        style_all_cls_images = torch.tensor(
            [], dtype=torch.float, device=conf.device
        )
        for ni in range(num_classes):
            try:
                style_images, style_labels = next(target_test_batch_iterator)
            except StopIteration:
                target_test_batch_iterator = iter(test_loader_tgt)
                style_images, style_labels = next(target_test_batch_iterator)

            style_images = style_images.to(conf.device)
            style_image = style_images[:, :content_image.size(1), :, :]

            style_all_cls_images = torch.cat([style_all_cls_images, style_image])

        test(content_image, style_all_cls_images
             , style_encoding_net, style_transfer_net
             , it, label_dict, mean_array, std_array, conf)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DRB-GAN tes')
    parser.add_argument('--tgt_dataset', type=str, default='', help='target dataset path')
    parser.add_argument('--test_dataset', type=str, default='', help='test dataset path')

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

    parser.add_argument('--sample_dir', type=str, default='samples',
                        help='Directory name to save the samples on training')
    parser.add_argument('--vgg_model', type=str, default='vgg19-dcbb9e9d.pth',
                        help='file name to load the vgg model for feature extraction')
    parser.add_argument('--pretrain_model', type=str, default='checkpoint/',
                        help='file name to load the model for training')
    parser.add_argument('--mixture_list', type=str, nargs='+')
    parser.add_argument('--mixture_weights', type=float, nargs='+')
    args = parser.parse_args()

    assert len(args.mixture_list) == len(args.mixture_weights), "mixture_list and mixture_weights should be same size"
    assert sum(args.mixture_weights) == 1.0, "sum of mixture weights should be 1."

    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)

    main(args)
