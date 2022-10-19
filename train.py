"""
@author: Jun Wang
@date: 20201019
@contact: jun21wangustc@gmail.com
"""
import os
import sys
import shutil
import argparse
import numpy as np
import logging as logger
import cv2
from functools import partial

abs_path = os.getcwd().split('DRB-GAN')[0]
sys.path.append(os.path.join(abs_path, 'DRB-GAN', "model"))

import torch
from torch import optim
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from tensorboardX import SummaryWriter

from data.dataset import ImageDataset, ImageClassDataset
from model.StyleEncodNet import StyleEncodingNetwork
from model.StyleTransferNet import StyleTransferNetwork
from model.DiscriminativeNet import Discriminator
from model.vgg_nets import Vgg19
from loss.loss import DRBGANLoss
from utils.AverageMeter import AverageMeter
from utils.utils import init_seeds, worker_init_fn, get_lr


def set_logging(rank=-1):
    logger.basicConfig(
        format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M',
        level=logger.INFO if rank in [-1, 0] else logger.WARN)


def train_one_iter(source, targets, target_labels
                   , style_encoder, generator, discriminator
                   , criterion, G_optimizer, D_optimizer, scaler
                   , discriminator_loss_meter, generator_loss_meter
                   , content_loss_meter, style_loss_meter, per_loss_meter, style_cls_loss_meter
                   , it, conf, update_G_only=False):
    batch_size, C, H, W = source.size()
    source = source.to(conf.device)
    targets = targets.to(conf.device)
    target_labels = target_labels.to(conf.device)

    target = targets[:, :C, :, :]
    collection_targets = targets[:, C:, :, :]
    target_label = target_labels[0][0]

    # generate fake image
    with torch.cuda.amp.autocast(enabled=conf.mixed_precision):
        style_prob, style_gamma, style_beta, style_omega = style_encoder(target, target_label.unsqueeze(0))
        fake_img = generator(source, style_gamma, style_beta, style_omega)

    # training rate: G : D = self.training_rate : 1
    if not update_G_only:
        # Update D
        with torch.cuda.amp.autocast(enabled=conf.mixed_precision):
            # from true distribution
            real_prep = discriminator(target, collection_targets, False)
            # from fake distribution
            fake_prep = discriminator(fake_img.detach(), collection_targets, False)
            adv_loss_d = criterion.compute_loss_D(fake_prep, real_prep)
            d_loss = adv_loss_d
        # backward
        D_optimizer.zero_grad()
        if conf.mixed_precision:
            scaler.scale(d_loss).backward()
            scaler.step(D_optimizer)
            scaler.update()
        else:
            d_loss.backward()
            D_optimizer.step()

        discriminator_loss_meter.update(adv_loss_d.item(), batch_size)

    # Update G
    with torch.cuda.amp.autocast(enabled=conf.mixed_precision):
        fake_prep = discriminator(fake_img, collection_targets, False)
        adv_loss_g, per_loss, style_cls_loss, content_loss, style_loss \
            = criterion.compute_loss_G(fake_img
                                       , source
                                       , target
                                       , fake_prep
                                       , style_prob
                                       , target_label.unsqueeze(0)
                                       , mixed_precision=conf.mixed_precision)
        g_loss = adv_loss_g + per_loss + style_cls_loss

    # backward
    G_optimizer.zero_grad()
    if conf.mixed_precision:
        scaler.scale(g_loss).backward()
        scaler.step(G_optimizer)
        scaler.update()
    else:
        g_loss.backward()
        G_optimizer.step()

    generator_loss_meter.update(adv_loss_g.item(), batch_size)
    content_loss_meter.update(content_loss.item(), batch_size)
    style_loss_meter.update(style_loss.item(), batch_size)
    per_loss_meter.update(per_loss.item(), batch_size)
    style_cls_loss_meter.update(style_cls_loss.item(), batch_size)

    if it % conf.print_freq == 0:
        discriminator_loss_val = discriminator_loss_meter.avg
        generator_loss_val = generator_loss_meter.avg
        content_loss_val = content_loss_meter.avg
        style_loss_val = style_loss_meter.avg
        per_loss_val = per_loss_meter.avg
        style_cls_loss_val = style_cls_loss_meter.avg

        g_lr = get_lr(G_optimizer)
        d_lr = get_lr(D_optimizer)
        logger.info('iter %d, g lr %.6f, d lr %.6f'
                    ', discriminator loss %.6f'
                    ', generator loss %.6f'
                    ', content loss %.6f'
                    ', style loss %.6f'
                    ', per loss %.6f'
                    ', style cls loss %.6f'
                    % (it, g_lr, d_lr
                       , discriminator_loss_val
                       , generator_loss_val
                       , content_loss_val
                       , style_loss_val
                       , per_loss_val
                       , style_cls_loss_val))

        conf.writer.add_scalar('discriminator_loss', discriminator_loss_val, it)
        conf.writer.add_scalar('generator_loss', generator_loss_val, it)
        conf.writer.add_scalar('content_loss', content_loss_val, it)
        conf.writer.add_scalar('style_loss', style_loss_val, it)
        conf.writer.add_scalar('per_loss', per_loss_val, it)
        conf.writer.add_scalar('style_cls_loss', style_cls_loss_val, it)
        conf.writer.add_scalar('g_lr', g_lr, it)
        conf.writer.add_scalar('d_lr', d_lr, it)

        discriminator_loss_meter.reset()
        generator_loss_meter.reset()
        content_loss_meter.reset()
        style_loss_meter.reset()
        per_loss_meter.reset()
        style_cls_loss_meter.reset()

    if it % conf.save_freq == 0 or it == conf.iter - 1:
        saved_name = 'DRBGAN_it_%d.pt' % it
        state = {
            'iter': it
            , 'style_encoder': style_encoder.module.state_dict()
            , 'generator': generator.module.state_dict()
            , 'discriminator': discriminator.module.state_dict()
            , 'G_optimizer': G_optimizer.state_dict()
            , 'D_optimizer': D_optimizer.state_dict()
        }

        torch.save(state, os.path.join(conf.checkpoint_dir, saved_name))
        logger.info('save checkpoint %s to disk...' % saved_name)
        del state


def test(source, targets, style_encoder, generator, cur_it, label_dict, mean, std, conf):
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
                            str(cur_it) + '_iter_' + 'test_style_' + label_dict[ti] + '.png')
        cv2.imwrite(path, result)


def train(conf):
    """Total training procedure.
    """
    init_seeds(3)
    conf.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data setting
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)

    mean_array = np.array(mean).reshape(1, 1, -1)
    std_array = np.array(std).reshape(1, 1, -1)

    # dataset setting
    train_transform = [T.RandomHorizontalFlip()
                       , T.Resize(conf.img_size, InterpolationMode.BICUBIC)
                       , T.RandomCrop(conf.crop_size)
                       , T.ToTensor()
                       , T.Normalize(mean=mean, std=std)]
    train_transform = T.Compose(train_transform)
    test_transform = [T.Resize(conf.img_size, InterpolationMode.BICUBIC)
                      , T.ToTensor()
                      , T.Normalize(mean=mean, std=std)]
    test_transform = T.Compose(test_transform)

    train_data_src = ImageDataset(conf.src_dataset, train_transform)
    train_data_tgt = ImageClassDataset(conf.tgt_dataset, train_transform, sample_size=conf.M + 1)
    test_data_src = ImageDataset(conf.test_dataset, test_transform)
    label_dict = train_data_tgt.label_dict

    init_fn = partial(worker_init_fn, num_workers=conf.workers, rank=0, seed=2)
    train_loader_src = torch.utils.data.DataLoader(train_data_src
                                                   , batch_size=conf.batch_size
                                                   , shuffle=True
                                                   , num_workers=conf.workers
                                                   , pin_memory=False
                                                   , drop_last=True
                                                   , worker_init_fn=init_fn)
    train_loader_tgt = torch.utils.data.DataLoader(train_data_tgt
                                                   , batch_size=conf.batch_size
                                                   , shuffle=True
                                                   , num_workers=conf.workers
                                                   , pin_memory=False
                                                   , drop_last=True
                                                   , worker_init_fn=init_fn)
    test_loader_src = torch.utils.data.DataLoader(test_data_src
                                                  , batch_size=1
                                                  , num_workers=conf.workers
                                                  , pin_memory=False
                                                  , shuffle=True)
    test_loader_tgt = torch.utils.data.DataLoader(train_data_tgt
                                                  , batch_size=1
                                                  , shuffle=False
                                                  , num_workers=conf.workers
                                                  , pin_memory=False)

    source_train_batch_iterator = iter(train_loader_src)
    target_train_batch_iterator = iter(train_loader_tgt)

    source_test_batch_iterator = iter(test_loader_src)
    target_test_batch_iterator = iter(test_loader_tgt)

    num_classes = len(train_data_tgt)

    logger.info('num_classes: %d', num_classes)

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
    discriminator = Discriminator(conf.M + 1).to(conf.device)

    # optimizer setting
    generator_group_dict = [
        {"params": style_encoding_net.parameters(), "weight_decay": 5e-4},
        {"params": style_transfer_net.parameters(), "weight_decay": 5e-4},
    ]
    G_optimizer = optim.Adam(generator_group_dict, lr=conf.g_lr, betas=(0.5, 0.999))
    D_optimizer = optim.Adam(discriminator.parameters(), lr=conf.d_lr, betas=(0.5, 0.999))

    # loss setting
    criterion = DRBGANLoss(conf, VGG)
    scaler = torch.cuda.amp.GradScaler(enabled=conf.mixed_precision)

    ori_iter = 0

    if conf.pretrained:
        checkpoint = torch.load(conf.pretrain_model, map_location=conf.device)
        if "iter" in checkpoint \
                and "style_encoder" in checkpoint \
                and "generator" in checkpoint \
                and "discriminator" in checkpoint \
                and "G_optimizer" in checkpoint \
                and "D_optimizer" in checkpoint:
            logger.info('load model')
            ori_iter = checkpoint['iter'] + 1
            style_encoding_net.load_state_dict(checkpoint['style_encoder'])
            style_transfer_net.load_state_dict(checkpoint['generator'])
            discriminator.load_state_dict(checkpoint['discriminator'])
            G_optimizer.load_state_dict(checkpoint['G_optimizer'])
            D_optimizer.load_state_dict(checkpoint['D_optimizer'])

    style_encoder = torch.nn.DataParallel(style_encoding_net)
    generator = torch.nn.DataParallel(style_transfer_net)
    discriminator = torch.nn.DataParallel(discriminator)

    discriminator_loss_meter = AverageMeter()
    generator_loss_meter = AverageMeter()
    content_loss_meter = AverageMeter()
    style_loss_meter = AverageMeter()
    per_loss_meter = AverageMeter()
    style_cls_loss_meter = AverageMeter()

    logger.info('start iter: %d', ori_iter)
    j = conf.training_rate
    for it in range(ori_iter, conf.iter):
        # training procedure
        style_encoder.train()
        generator.train()
        discriminator.train()
        try:
            content_image = next(source_train_batch_iterator)
        except StopIteration:
            source_train_batch_iterator = iter(train_loader_src)
            content_image = next(source_train_batch_iterator)

        try:
            style_images, style_labels = next(target_train_batch_iterator)
        except StopIteration:
            target_train_batch_iterator = iter(train_loader_tgt)
            style_images, style_labels = next(target_train_batch_iterator)

        update_G_only = True
        if j == conf.training_rate:
            update_G_only = False

        train_one_iter(content_image, style_images, style_labels
                       , style_encoder, generator, discriminator
                       , criterion, G_optimizer, D_optimizer, scaler
                       , discriminator_loss_meter, generator_loss_meter
                       , content_loss_meter, style_loss_meter, per_loss_meter, style_cls_loss_meter
                       , it, conf, update_G_only)

        j = j - 1
        if j < 1:
            j = conf.training_rate

        # testing procedure
        if it % conf.print_freq == 0:
            style_encoder.eval()
            generator.eval()
            try:
                content_image = next(source_test_batch_iterator)
            except StopIteration:
                source_test_batch_iterator = iter(test_loader_src)
                content_image = next(source_test_batch_iterator)

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
                 , style_encoder, generator
                 , it, label_dict, mean_array, std_array, conf)

    logger.info("training done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DRB-GAN')
    parser.add_argument('--src_dataset', type=str, default='', help='source dataset path')
    parser.add_argument('--tgt_dataset', type=str, default='', help='target dataset path')
    parser.add_argument('--test_dataset', type=str, default='', help='test dataset path')

    parser.add_argument('--iter', type=int, default=101, help='The number of iteration to run')
    parser.add_argument('--batch_size', type=int, default=1, help='The size of batch size')
    parser.add_argument('--training_rate', type=int, default=1, help='training rate about G & D')
    parser.add_argument('--img_size', type=int, default=256, help='The size of image: H and W')
    parser.add_argument('--crop_size', type=int, default=256, help='The size of cropped image: H and W')
    parser.add_argument('--workers', type=int, default=1, help='The number of workers for dataloader')
    parser.add_argument('--print_freq', type=int, default=100, help='The number of loss print freq')
    parser.add_argument('--save_freq', type=int, default=1, help='The number of ckpt_save_freq')

    parser.add_argument('--g_lr', type=float, default=2e-5, help='The learning rate')
    parser.add_argument('--d_lr', type=float, default=2e-5, help='The learning rate')
    parser.add_argument('--feature_dim', type=int, default=156160, help='The dimension of style encoder feature')
    parser.add_argument('--gamma_dim', type=int, default=256, help='The dimension of style code for ADIN mean')
    parser.add_argument('--beta_dim', type=int, default=256, help='The dimension of style code for ADIN std')
    parser.add_argument('--omega_dim', type=int, default=4, help='The dimension of style code for DConv')
    parser.add_argument('--encoder_out_ch', type=int, default=128, help='The dimension of content encoder output')
    parser.add_argument('--db_number', type=int, default=4, help='The number of Dynamic ResBlock')
    parser.add_argument('--ws', type=int, default=64, help='The window size of SW-LIN Decoder')
    parser.add_argument('--M', type=int, default=2, help='The number of style reference image for Discriminator')

    parser.add_argument('--g_adv_weight', type=float, default=1.0, help='Weight about Generator loss')
    parser.add_argument('--d_adv_weight', type=float, default=1.0, help='Weight about Discriminator loss')
    parser.add_argument('--con_weight', type=float, default=1.0, help='Weight about VGG19')
    parser.add_argument('--sty_weight', type=float, default=0.02, help='Weight about style')
    parser.add_argument('--perceptual_weight', type=float, default=1.0, help='Weight about perceptual loss')
    parser.add_argument('--class_weight', type=float, default=1.0, help='Weight about classification loss')
    parser.add_argument('--tv_weight', type=float, default=1.0, help='Weight about tv')
    parser.add_argument('--ld', type=float, default=10.0, help='The gradient penalty lambda')
    parser.add_argument('--gan_loss', type=str, default='lsgan', help='type of Generator loss')

    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')
    parser.add_argument('--sample_dir', type=str, default='samples',
                        help='Directory name to save the samples on training')
    parser.add_argument('--vgg_model', type=str, default='vgg19-dcbb9e9d.pth',
                        help='file name to load the vgg model for feature extraction')
    parser.add_argument('--pretrained', type=bool, default=False,
                        help='whether to pretrained')
    parser.add_argument('--pretrain_model', type=str, default='checkpoint/',
                        help='file name to load the model for training')
    parser.add_argument('--mixed_precision', action='store_true', default=False)
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    tensorboardx_logdir = os.path.join(args.log_dir)
    if os.path.exists(tensorboardx_logdir):
        shutil.rmtree(tensorboardx_logdir)
    writer = SummaryWriter(log_dir=tensorboardx_logdir)
    args.writer = writer

    set_logging(-1)
    train(args)
