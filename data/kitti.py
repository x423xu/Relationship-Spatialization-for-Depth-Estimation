import os, random

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import h5py, torch

from .utils import ToTensor, remove_leading_slash, train_preprocess
from lib.data.transforms import build_transforms
from lib.config import cfg


class kitti(Dataset):
    def __init__(self, args, mode, transform=None, is_for_online_eval=False, ext="png"):
        self.args = args
        self.ext = ext
        if mode == "online_eval":
            with open(args.filenames_file_eval, "r") as f:
                self.filenames = f.readlines()
            self.filenames = self.filenames[: args.max_eval_num]
        else:
            with open(args.filenames_file, "r") as f:
                self.filenames = f.readlines()
        # print('length:{}'.format(len(self.filenames)), self.ext)
        self.mode = mode
        self.transform = transform
        self.to_tensor = ToTensor
        self.is_for_online_eval = is_for_online_eval
        self.rel_transforms = build_transforms(cfg, is_train=True, nyuv=True)

    def __getitem__(self, idx):
        sample_path = self.filenames[idx]
        focal = float(sample_path.split()[2])
        if self.mode == "train":
            image_path = os.path.join(
                self.args.data_path, remove_leading_slash(sample_path.split()[0])
            )
            depth_path = os.path.join(
                self.args.gt_path, remove_leading_slash(sample_path.split()[1])
            )
            relation_path = os.path.join(
                self.args.data_path,
                remove_leading_slash(sample_path.split()[0]).replace(".png", ".h5"),
            )
            image = Image.open(image_path)
            depth_gt = Image.open(depth_path)
            if self.args.do_kb_crop is True:
                height = image.height
                width = image.width
                top_margin = int(height - 352)
                left_margin = int((width - 1216) / 2)
                depth_gt = depth_gt.crop(
                    (left_margin, top_margin, left_margin + 1216, top_margin + 352)
                )
                image = image.crop(
                    (left_margin, top_margin, left_margin + 1216, top_margin + 352)
                )
            image = np.asarray(image, dtype=np.float32) / 255.0
            depth_gt = np.asarray(depth_gt, dtype=np.float32)
            depth_gt = np.expand_dims(depth_gt, axis=2)
            depth_gt = depth_gt / 256.0
            image = train_preprocess(image, "kitti")
            sample = {"image": image, "depth": depth_gt, "focal": focal}
        else:
            if self.mode == "online_eval":
                data_path = self.args.data_path_eval
            else:
                data_path = self.args.data_path

            image_path = os.path.join(
                data_path, remove_leading_slash(sample_path.split()[0])
            )
            img_pil = Image.open(image_path)
            image = np.asarray(img_pil, dtype=np.float32) / 255.0
            if self.mode == "online_eval":
                gt_path = self.args.gt_path_eval
                depth_path = os.path.join(
                    gt_path, remove_leading_slash(sample_path.split()[1])
                )
                has_valid_depth = False
                try:
                    depth_gt = Image.open(depth_path)
                    has_valid_depth = True
                except IOError:
                    depth_gt = False
                    print("Missing gt for {}".format(image_path))

                if has_valid_depth:
                    depth_gt = np.asarray(depth_gt, dtype=np.float32)
                    depth_gt = np.expand_dims(depth_gt, axis=2)
                    if self.args.dataset == "kitti":
                        depth_gt = depth_gt / 256.0
                    else:
                        raise "depth scaling only works for kitti"

            relation_path = os.path.join(
                data_path,
                remove_leading_slash(sample_path.split()[0]).replace(".png", ".h5"),
            )
            if self.mode == "online_eval":
                sample = {
                    "image": image,
                    "depth": depth_gt,
                    "focal": focal,
                    "has_valid_depth": has_valid_depth,
                    "image_path": sample_path.split()[0],
                    "depth_path": sample_path.split()[1],
                }
            else:
                sample = {"image": image, "focal": focal}

        if self.transform:
            sample = self.transform(sample)
        rel_h5 = h5py.File(relation_path, "r")
        rel_features = torch.from_numpy(rel_h5["rel_features"][:])
        bbox_pairs = torch.from_numpy(rel_h5["bbox"][:])
        bbox = bbox_pairs
        if bbox.shape[0] < 256:
            new_zeros_b = torch.zeros(
                [256 - bbox.shape[0], bbox.shape[1]],
                dtype=bbox.dtype,
                device=bbox.device,
            )
            bbox = torch.cat([bbox, new_zeros_b], dim=0)
        bbox_pairs = bbox
        sample.update({"rel_features": rel_features, "bbox_pairs": bbox_pairs})
        return sample

    def __len__(self):
        return len(self.filenames)

