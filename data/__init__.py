import torch
import torch.utils.data.distributed
from torch.utils.data import DataLoader

from .kitti import kitti
from .nyu import nyu

from .utils import preprocessing_transforms

"""
DepthDataloader works for kitti and nyu depth v2. It constructs datasets from kitti or nyu, under train, online_eval or test mode.
"""


class DepthDataLoader(object):
    def __init__(self, args, mode):
        if args.dataset == "nyu":
            ext = "jpg"
            dataset = nyu
        if args.dataset == "kitti":
            ext = "png"
            dataset = kitti
        if mode == "train":
            self.training_samples = dataset(
                args, mode, transform=preprocessing_transforms(mode), ext=ext
            )
            if args.distributed:
                self.train_sampler = torch.utils.data.distributed.DistributedSampler(
                    self.training_samples
                )
            else:
                self.train_sampler = None

            print(args.num_threads, args.world_size)
            self.data = DataLoader(
                self.training_samples,
                args.batch_size,
                shuffle=(self.train_sampler is None),
                num_workers=args.num_threads,
                pin_memory=True,
                sampler=self.train_sampler,
            )

        elif mode == "online_eval":
            self.testing_samples = dataset(
                args, mode, transform=preprocessing_transforms(mode), ext=ext
            )
            if (
                args.distributed
            ):  # redundant. here only for readability and to be more explicit
                # Give whole test set to all processes (and perform/report evaluation only on one) regardless
                self.eval_sampler = None
            else:
                self.eval_sampler = None
            self.data = DataLoader(
                self.testing_samples,
                1,
                shuffle=False,
                num_workers=1,
                pin_memory=False,
                sampler=self.eval_sampler,
            )

        elif mode == "test":
            self.testing_samples = dataset(
                args, mode, transform=preprocessing_transforms(mode)
            )
            self.data = DataLoader(
                self.testing_samples, 1, shuffle=False, num_workers=1
            )

        else:
            print(
                "mode should be one of 'train, test, online_eval'. Got {}".format(mode)
            )

