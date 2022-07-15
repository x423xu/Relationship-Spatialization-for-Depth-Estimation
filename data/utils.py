import torch
import random
from torchvision import transforms
import numpy as np
from PIL import Image


def augment_image(image, dataset):
    # gamma augmentation
    gamma = random.uniform(0.8, 1.2)
    image_aug = image ** gamma

    # brightness augmentation
    if dataset == "nyu":
        brightness = random.uniform(0.75, 1.25)
    else:
        brightness = random.uniform(0.9, 1.1)
    image_aug = image_aug * brightness

    # color augmentation
    colors = np.random.uniform(0.9, 1.1, size=3)
    white = np.ones((image.shape[0], image.shape[1]))
    color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
    image_aug *= color_image
    image_aug = np.clip(image_aug, 0, 1)

    return image_aug


def train_preprocess(image, dataset):
    do_augment = random.random()
    if do_augment > 0.5:
        image = augment_image(image, dataset)

    return image


def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def preprocessing_transforms(mode):
    return transforms.Compose([ToTensor(mode=mode)])


def remove_leading_slash(s):
    if s[0] == "/" or s[0] == "\\":
        return s[1:]
    return s


class ToTensor(object):
    def __init__(self, mode):
        self.mode = mode
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def __call__(self, sample):
        image, focal = sample["image"], sample["focal"]
        image = self.to_tensor(image)
        image = self.normalize(image)

        if self.mode == "test":
            return {"image": image, "focal": focal}

        depth = sample["depth"]
        if self.mode == "train":
            depth = self.to_tensor(depth)
            return {"image": image, "depth": depth, "focal": focal}
        else:
            has_valid_depth = sample["has_valid_depth"]
            return {
                "image": image,
                "depth": depth,
                "focal": focal,
                "has_valid_depth": has_valid_depth,
            }

    def to_tensor(self, pic):
        if not (_is_pil_image(pic) or _is_numpy_image(pic)):
            raise TypeError(
                "pic should be PIL Image or ndarray. Got {}".format(type(pic))
            )

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img

        # handle PIL Image
        if pic.mode == "I":
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == "I;16":
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == "YCbCr":
            nchannel = 3
        elif pic.mode == "I;16":
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float()
        else:
            return img
