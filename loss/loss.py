import torch
from torch.autograd import Variable
from torch.autograd import grad as torch_grad
import torch.nn.functional as F
import torch.nn as nn
import os
import sys

abs_path = os.getcwd().split('DRB-GAN')[0]
sys.path.append(os.path.join(abs_path, 'DRB-GAN', "model"))

from vgg_nets import Vgg19


def gram(input):
    """
    Calculate Gram Matrix
    https://pytorch.org/tutorials/advanced/neural_style_tutorial.html#style-loss
    """
    b, c, w, h = input.size()
    x = input.view(b * c, w * h)
    G = torch.mm(x, x.T)
    # normalize by total elements
    return G.div(b * c * w * h)


def calc_gradient_penalty(netD, real_data, fake_data, gp_weight):
    batch_size = real_data.size()[0]

    # Calculate interpolation
    alpha = torch.rand(batch_size, 1, 1, 1)
    alpha = alpha.expand_as(real_data)
    alpha = alpha.to(real_data.device)
    interpolated = alpha * real_data.data + (1 - alpha) * fake_data.data
    interpolated = Variable(interpolated, requires_grad=True)
    interpolated = interpolated.to(real_data.device)

    # Calculate probability of interpolated examples
    prob_interpolated = netD(interpolated)

    # Calculate gradients of probabilities with respect to examples
    gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                           grad_outputs=torch.ones(prob_interpolated.size()).to(real_data.device),
                           create_graph=True, retain_graph=True)[0]

    # Gradients have shape (batch_size, num_channels, img_width, img_height),
    # so flatten to easily take norm per example in batch
    gradients = gradients.view(batch_size, -1)
    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    # Return gradient penalty
    return gp_weight * ((gradients_norm - 1) ** 2).mean()


def total_variation_loss(image):
    tv_h = torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))
    tv_w = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:]))
    tv_loss = (tv_h + tv_w)
    return tv_loss


class DRBGANLoss:
    def __init__(self, args, vgg19):
        self.content_loss = nn.MSELoss().to(args.device)
        self.style_mean_loss = nn.L1Loss().to(args.device)
        self.gram_loss = nn.L1Loss().to(args.device)
        self.style_cls_loss = torch.nn.CrossEntropyLoss().to(args.device)
        self.bce_loss = nn.BCELoss()
        self.wadvg = args.g_adv_weight
        self.wadvd = args.d_adv_weight
        self.wcon = args.con_weight
        self.wsty = args.sty_weight
        self.wcls = args.class_weight
        self.wper = args.perceptual_weight
        self.vgg19 = vgg19
        self.adv_type = args.gan_loss

    def compute_loss_G(self, fake_img, img, style_img, fake_logit, style_logit, style_label, mixed_precision=False):
        '''
        Compute loss for Generator
        @Arugments:
            - fake_img: generated image
            - img: image
            - style_img: target style image
            - fake_logit: output of Discriminator given fake image
            - style_logit: output of AuxiliaryClassifier given style image
            - style_label: ground truth of class labels given style image
        @Returns:
            loss
        '''
        fake_feat = self.vgg19(fake_img)
        img_feat = self.vgg19(img)
        style_feat = self.vgg19(style_img)
        fake_feat_list \
            = [fake_feat.relu1_2, fake_feat.relu2_2, fake_feat.relu3_3, fake_feat.relu4_3, fake_feat.relu5_1]
        style_feat_list \
            = [style_feat.relu1_2, style_feat.relu2_2, style_feat.relu3_3, style_feat.relu4_3, style_feat.relu5_1]

        content_loss = self.content_loss(img_feat.relu4_1.detach(), fake_feat.relu4_1)

        with torch.cuda.amp.autocast(enabled=False):
            style_loss = torch.Tensor([self.style_mean_loss(ff.mean(), sf.mean().detach()) ** 2
                                       + self.gram_loss(gram(ff.float()), gram(sf.float().detach())) ** 2
                                       for ff, sf in zip(fake_feat_list, style_feat_list)]).mean()
        with torch.cuda.amp.autocast(enabled=mixed_precision):
            per_loss = self.wcon * content_loss + self.wsty * style_loss
            style_cls_loss = self.style_cls_loss(style_logit, style_label)

            return [
                self.wadvg * self.adv_loss_g(fake_logit),
                self.wper * per_loss,
                self.wcls * style_cls_loss,
                content_loss,
                style_loss
            ]

    def compute_loss_D(self, fake_img_logits, real_img_logits):
        loss = self.wadvd * self.adv_loss_d(fake_img_logits, real_img_logits)
        return loss

    def content_loss_vgg(self, image, recontruction):
        feat = self.vgg19(image)
        re_feat = self.vgg19(recontruction)

        return self.content_loss(feat, re_feat)

    def adv_loss_d(self, fake_img_logits, real_img_logits):
        loss = 0
        for it, (fake_img_logit, real_img_logit) in enumerate(zip(fake_img_logits, real_img_logits)):
            if self.adv_type == 'lsgan':
                loss += torch.mean((fake_img_logit - 0) ** 2) + torch.mean((real_img_logit - 1) ** 2)
            elif self.adv_type == 'nsgan':
                all0 = Variable(torch.zeros_like(fake_img_logit.data).cuda(), requires_grad=False)
                all1 = Variable(torch.ones_like(real_img_logit.data).cuda(), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy_with_logits(fake_img_logit, all0) +
                                   F.binary_cross_entropy_with_logits(real_img_logit, all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.adv_type)

        return loss

    def adv_loss_g(self, fake_img_logits):
        loss = 0
        for it, (out0) in enumerate(fake_img_logits):
            if self.adv_type == 'lsgan':
                loss += torch.mean((out0 - 1) ** 2)  # LSGAN
            elif self.adv_type == 'nsgan':
                all1 = Variable(torch.ones_like(out0.data).cuda(), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy_with_logits(out0, all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.adv_type)

        return loss


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='DRB-GAN')
    parser.add_argument('--g_adv_weight', type=float, default=1.0, help='Weight about Generator loss')
    parser.add_argument('--d_adv_weight', type=float, default=1.0, help='Weight about Discriminator loss')
    parser.add_argument('--con_weight', type=float, default=1.0,
                        help='Weight about VGG19')
    parser.add_argument('--sty_weight', type=float, default=0.02,
                        help='Weight about style')
    parser.add_argument('--perceptual_weight', type=float, default=1.0,
                        help='Weight about perceptual loss')
    parser.add_argument('--class_weight', type=float, default=1.0,
                        help='Weight about classification loss')
    parser.add_argument('--tv_weight', type=float, default=1.0,
                        help='Weight about tv')
    parser.add_argument('--gan_loss', type=str, default='lsgan', help='type of Generator loss')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    num_styles = 4
    num_scales = 3
    fake_image = torch.rand((1, 3, 256, 256)).to(device)
    content_image = torch.rand((1, 3, 256, 256)).to(device)
    style_image = torch.rand((1, 3, 256, 256)).to(device)
    style_label = torch.randint(0, num_styles, (1,)).to(device)
    fake_logits = [torch.rand((1, 1, 7, 7), requires_grad=True).to(device) for _ in range(num_scales)]
    real_logits = [torch.rand((1, 1, 7, 7)).to(device) for _ in range(num_scales)]
    style_logit = torch.rand((1, num_styles)).to(device)

    VGG = Vgg19().to(device)
    VGG.eval()
    loss = DRBGANLoss(args, VGG)

    adv_loss_d = loss.compute_loss_D(fake_logits, real_logits)
    d_loss = adv_loss_d
    print("d_loss:", d_loss)
    d_loss.backward(retain_graph=True)

    adv_loss_g, per_loss, style_cls_loss, content_loss, style_loss \
        = loss.compute_loss_G(fake_image, content_image, style_image, fake_logits, style_logit, style_label)

    g_loss = adv_loss_g + per_loss + style_cls_loss
    print("g_loss:", g_loss)
    g_loss.backward()

