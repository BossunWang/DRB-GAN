import torch
import torch.nn.functional as F
import math
from tools import edge_extracter
from tools.guided_filter import GuidedFilter


def extract_image_patches(x, kernel, stride=1, dilation=1):
    # Do TF 'SAME' Padding
    # b, c, h, w = x.shape
    # h2 = math.ceil(h / stride)
    # w2 = math.ceil(w / stride)
    # pad_row = (h2 - 1) * stride + (kernel - 1) * dilation + 1 - h
    # pad_col = (w2 - 1) * stride + (kernel - 1) * dilation + 1 - w
    # x = F.pad(x, (pad_row // 2, pad_row - pad_row // 2, pad_col // 2, pad_col - pad_col // 2))

    # Extract patches
    patches = x.unfold(2, kernel, stride).unfold(3, kernel, stride)
    # print(patches.size())
    patches = patches.permute(1, 0, 2, 3, 4, 5).contiguous()

    return patches.view(patches.shape[0], -1, patches.shape[-2], patches.shape[-1])


def extract_top_k_img_patches_by_sum(img, patch_size, stride, k, guided_filter):
    '''
    :param img: input image of shape b, h, w, 3  -1 ~ 1 float32
    :param patch_size: the size of each extracted patch
    :param stride: the stride to slide window on original image
    :param k: the number of patches to extract from img
    :return: image patches with shape  k, patch_size, patch_size, 3  -1 ~ 1
    '''
    img_blur = guided_filter(img, img)
    edge_map = edge_extracter.edge_map(img_blur, enhance=True)   # 0 ~ 1    b, 1, h, w
    img_edge = torch.cat([img, edge_map], dim=1)               # b, 4, h, w
    img_edge_patches = extract_image_patches(img_edge, patch_size, stride)
    img_patches = img_edge_patches[0:3, ...]
    edge_patches = img_edge_patches[-1, ...]
    edge_intensity_list = torch.sum(edge_patches, dim=(-2, -1))
    top_k = torch.topk(edge_intensity_list, k=k)[1]
    img_patches = img_patches.permute(1, 0, 2, 3).contiguous()
    top_k_img_patches = img_patches[top_k, ...]  # 3, k, patch_size, patch_size
    return top_k_img_patches


if __name__ == '__main__':
    import cv2
    import torch
    import numpy as np
    import sys
    sys.path.append("../model")
    from model.DiscriminativeNet import Discriminator

    patch_size = 96
    stride = 48
    k = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gf = GuidedFilter(r=5, eps=0.2).to(device)

    image = cv2.imread(
        "/media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/Style_Transfer/art_dataset_v2/kaka/136964374_447443723304110_7976762053155757037_n.jpg")
    image = cv2.resize(image, (512, 512))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
    image = (image - 0.5) / 0.5
    image = torch.from_numpy(image.transpose((2, 0, 1))[None]).float().to(device)
    image.requires_grad_()

    top_k_img_patches = extract_top_k_img_patches_by_sum(image, patch_size, stride, k, gf)
    top_k_img_patches_gray = torch.sum(top_k_img_patches, dim=1, keepdims=True) / 3

    # show img
    for ik in range(k):
        image_patch = top_k_img_patches_gray[ik].data.cpu().numpy().transpose(1, 2, 0)
        image_patch = np.asarray((image_patch * 0.5 + 0.5).clip(0, 1) * 255, dtype=np.uint8)
        cv2.imwrite("image_patch{}.jpg".format(ik), image_patch)

    M = 0
    style_collection_inputs = torch.rand(1, 3 * M, patch_size, patch_size).to(device)
    discriminator = Discriminator(M + 1, input_dim=1).to(device)
    output_logits = discriminator(top_k_img_patches_gray, style_collection_inputs.detach(), shuffle=False)

    loss = 0
    for output_logit in output_logits:
        print("output_logit:", output_logit.size())
        loss += output_logit.mean()

    loss.backward()