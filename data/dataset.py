"""
@author: Hang Du, Jun Wang
@date: 20201101
@contact: jun21wangustc@gmail.com
"""

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from pathlib import Path


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for dirPath, dirNames, fileNames in os.walk(dir):
        for f in sorted(fileNames):
            path = os.path.join(dirPath, f)
            if "jpg" in Path(path).suffix \
                    or "png" in Path(path).suffix \
                    or "jpeg" in Path(path).suffix \
                    or "jfif" in Path(path).suffix:
                images.append(path)

    return images


class ImageDataset(Dataset):
    def __init__(self, data_file, transform, get_path=False):
        if os.path.isdir(data_file):
            self.files = make_dataset(data_file)
        else:
            self.files = []
            f = open(data_file, 'r')
            self.files = [line.strip() for line in f.readlines()]
            f.close()

        self.transform = transform
        self.get_path = get_path

    def __len__(self):
        return len(self.files)

    def to_Tensor(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image

    def __getitem__(self, index):
        image_path = self.files[index]
        image = self.to_Tensor(image_path)

        if self.get_path:
            return image, image_path
        else:
            return image


class ImageClassDataset(Dataset):
    def __init__(self, data_root, transform, sample_size=1, assigned_labels=[], assigned_transform=[]):
        self.data_root = data_root
        self.label_dict = os.listdir(data_root)
        self.files = make_dataset(data_root)
        self.transform = transform
        self.sample_size = sample_size
        self.assigned_labels = assigned_labels
        self.assigned_transform = assigned_transform
        print("label dict:", self.label_dict)

    def __len__(self):
        return len(self.label_dict)

    def to_Tensor(self, image_path, transform):
        image = Image.open(image_path).convert('RGB')
        image = transform(image)
        return image

    def __getitem__(self, index):
        image_dir = self.label_dict[index]
        path, dirs, files = next(os.walk(os.path.join(self.data_root, image_dir)))
        file_index_list = np.random.permutation(len(files))[:self.sample_size]

        image_list = []
        label_list = []
        for fi in file_index_list:
            image_path = os.path.join(self.data_root, image_dir, files[fi])
            if len(self.assigned_labels) > 0 and index in self.assigned_labels:
                image = self.to_Tensor(image_path, self.assigned_transform[self.assigned_labels.index(index)])
                # style_image = image.cpu().numpy().transpose(1, 2, 0)
                # style_image = (style_image * std_array) + mean_array
                # style_image = np.clip(style_image * 255.0, 0, 255)
                # cv2.imwrite('sample_assigned_style.png', style_image)
            else:
                image = self.to_Tensor(image_path, self.transform)
            image_list.append(image)
            label_list.append(torch.Tensor([index]))
        return torch.cat(image_list), torch.cat(label_list).long()


if __name__ == '__main__':
    from torchvision import transforms as T
    from torchvision.transforms import InterpolationMode
    import matplotlib.pyplot as plt
    import numpy as np
    import cv2
    from tqdm import tqdm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = 512
    crop_size = 512
    sample_isze = 3
    train_transform = [T.RandomHorizontalFlip()
                       , T.Resize(input_size, InterpolationMode.BICUBIC)
                       , T.RandomCrop(crop_size)
                       , T.ToTensor()
                       , T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
    train_transform = T.Compose(train_transform)
    train_assigned_transform = [T.RandomHorizontalFlip()
                                , T.CenterCrop(crop_size)
                                , T.ToTensor()
                                , T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
    train_assigned_transform = T.Compose(train_assigned_transform)
    test_transform = [T.Resize(input_size, InterpolationMode.BICUBIC)
                      , T.ToTensor()
                      , T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
    test_transform = T.Compose(test_transform)

    train_data_src = ImageDataset('train.txt'
                                  , train_transform)
    train_data_tgt = ImageClassDataset('../../art_dataset_v2'
                                       , train_transform
                                       , sample_isze
                                       , assigned_labels=[6]
                                       , assigned_transform=[train_assigned_transform])
    test_data_tgt = ImageDataset('test.txt'
                                 , test_transform)

    batch_size = 1
    train_loader_src = torch.utils.data.DataLoader(train_data_src
                                                   , batch_size=batch_size
                                                   , shuffle=True
                                                   , drop_last=True)
    train_loader_tgt = torch.utils.data.DataLoader(train_data_tgt
                                                   , batch_size=batch_size
                                                   , shuffle=True
                                                   , drop_last=True)
    test_loader_src = torch.utils.data.DataLoader(test_data_tgt
                                                  , batch_size=1
                                                  , shuffle=False
                                                  , drop_last=False)

    style_batch_iterator = iter(train_loader_tgt)

    mean_array = np.array((0.5, 0.5, 0.5)).reshape(1, 1, -1)
    std_array = np.array((0.5, 0.5, 0.5)).reshape(1, 1, -1)

    for content_image in tqdm(train_loader_src):
        content_image = content_image.to(device)
        content_image = content_image[0].cpu().numpy().transpose(1, 2, 0)

        content_image = cv2.cvtColor(content_image, cv2.COLOR_BGR2RGB)
        content_image = (content_image * std_array) + mean_array
        content_image = np.clip(content_image * 255.0, 0, 255)
        cv2.imwrite('sample_train_content.png', content_image)

        try:
            style_image_list, label_list = next(style_batch_iterator)
        except StopIteration:
            style_batch_iterator = iter(train_loader_tgt)
            style_image_list, label_list = next(style_batch_iterator)

        # print("style_image_list:", style_image_list.size())
        # print("label_list:", label_list.size())

        style_image = style_image_list[0, :3, :, :].cpu().numpy().transpose(1, 2, 0)
        style_image = (style_image * std_array) + mean_array
        style_image = np.clip(style_image * 255.0, 0, 255)
        cv2.imwrite('sample_train_style.png', style_image)

    for content_image in tqdm(test_loader_src):
        content_image = content_image.to(device)
        content_image = content_image[0].cpu().numpy().transpose(1, 2, 0)

        content_image = cv2.cvtColor(content_image, cv2.COLOR_BGR2RGB)
        content_image = (content_image * std_array) + mean_array
        content_image = np.clip(content_image * 255.0, 0, 255)
        cv2.imwrite('sample_test_content.png', content_image)

