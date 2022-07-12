import os, random, glob

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import h5py, torch, cv2

from utils import ToTensor
from lib.data.transforms import build_transforms
from lib.config import cfg


class nyu(Dataset):
    def __init__(self, args, mode, transform=None, is_for_online_eval=False, ext="png"):
        self.args = args
        self.ext = ext
        if mode == "online_eval":
            cat_dir = os.listdir(args.data_path_eval)
            self.filenames = []
            for cd in cat_dir:
                f = glob.glob(os.path.join(args.data_path_eval, cd, "*.h5"))
                self.filenames.extend(f)
            self.filenames = self.filenames[: args.max_eval_num]
        if mode == "train":
            cat_dir = os.listdir(args.data_path)
            self.filenames = []
            for cd in cat_dir:
                f = glob.glob(os.path.join(args.data_path, cd, "*.h5"))
                self.filenames.extend(f)
        self.mode = mode
        self.transform = transform
        self.to_tensor = ToTensor
        self.is_for_online_eval = is_for_online_eval
        self.rel_transforms = build_transforms(cfg, is_train=True, nyuv=True)

    def __getitem__(self, idx):
        sample_path = self.filenames[idx]
        focal = torch.tensor([518.8597])
        if self.mode == "train":
            im_h5 = h5py.File(sample_path, "r")
            image = im_h5["rgb"][:].transpose(1, 2, 0)
            do_hist_equil = random.random()
            if do_hist_equil > 0.5:
                img = np.array(image, dtype=np.uint8)
                img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
                img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
                img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
                image = Image.fromarray(img_output)
            depth_gt = im_h5["depth"][:]
            relation_path = sample_path.replace("train", "train_rel")
            image = np.asarray(image, dtype=np.float32) / 255.0
            depth_gt = np.asarray(depth_gt, dtype=np.float32)
            depth_gt = np.expand_dims(depth_gt, axis=2)
            image, depth_gt = self.train_preprocess(image, depth_gt)
            sample = {"image": image, "depth": depth_gt, "focal": focal}
        else:
            image_path = self.filenames[idx]
            img_h5 = h5py.File(image_path, "r")
            image = img_h5["rgb"][:].transpose(1, 2, 0) / 255.0
            image = image.astype(np.float32)
            depth_gt = img_h5["depth"][:]

            depth_gt = np.asarray(depth_gt, dtype=np.float32)
            depth_gt = np.expand_dims(depth_gt, axis=2)
            has_valid_depth = True
            if self.args.dataset == "nyu":
                depth_gt = depth_gt / 1.0
            else:
                depth_gt = depth_gt / 256.0
            relation_path = sample_path.replace("val", "val_rel")

            sample = {
                "image": image,
                "depth": depth_gt,
                "focal": focal,
                "has_valid_depth": has_valid_depth,
            }

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

