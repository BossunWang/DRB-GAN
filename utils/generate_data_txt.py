import os
from pathlib import Path


def make_dataset(dir_list):
    images = []
    for dir in dir_list:
        assert os.path.isdir(dir), '%s is not a valid directory' % dir

        for dirPath, dirNames, fileNames in os.walk(dir):
            for f in sorted(fileNames):
                path = os.path.join(dirPath, f)
                if "jpg" in Path(path).suffix or "png" in Path(path).suffix or "jpeg" in Path(path).suffix:
                    images.append(path)

    return images


if __name__ == '__main__':
    train_dir_list = ["/media/glory/Transcend/Dataset/Scene/DIV2K_HR/origin"
                      , "/media/glory/Transcend/Dataset/Scene/Flickr2K-001/Flickr2K"
                      , "/media/glory/Transcend/Dataset/Scene/Flickr1024/Train"
                      , "/media/glory/Transcend/Dataset/Scene/landscape_set"
                      , "/media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/Animal_dataset/train"]
    test_dir_list = ["/media/glory/Transcend/Dataset/Scene/DIV2K_valid_HR"
                     , "/media/glory/Transcend/Dataset/Scene/Flickr1024/Validation"
                     , "/media/glory/Transcend/Dataset/Scene/Flickr1024/Test"
                     , "/media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/Animal_dataset/val"]

    train_img_list = make_dataset(train_dir_list)
    test_img_list = make_dataset(test_dir_list)

    with open('../data/train.txt', 'w') as f:
        for line in train_img_list:
            f.write(f"{line}\n")

    with open('../data/test.txt', 'w') as f:
        for line in test_img_list:
            f.write(f"{line}\n")
