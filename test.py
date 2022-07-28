import train
import torch
import sys
import numpy as np
from yacs.config import CfgNode as CN
from tqdm import tqdm


class Test:
    def __init__(self, args):
        self.args = args
        pretrained_path = args.pretrained
        # model init
        model = train.models.RSMDE.build(
            n_bins=args.n_bins,
            min_val=args.min_depth,
            max_val=args.max_depth,
            norm=args.norm,
            orthogonal_disable=args.orthogonal_disable,
            attention_disable=args.attention_disable,
            no_relation=args.no_relation,
            pretrained=pretrained_path,
            algo=args.algo,
            do_kb_crop=args.do_kb_crop,
        )
        if args.gpu is not None:  # If a gpu is set by user: NO PARALLELISM!!
            torch.cuda.set_device(args.gpu)
            model = model.cuda(args.gpu)
        self.model = model
        self.model.eval()

        # dataloader
        self.test_loader = train.DepthDataLoader(args, "online_eval").data

    def __call__(self):
        device = self.args.gpu
        with torch.no_grad():
            metrics = train.utils.RunningAverageDict()
            for batch in tqdm(self.test_loader, desc="Evaluaion loop"):
                img = batch["image"].to(device)
                depth = batch["depth"].to(device)
                rel_features = batch["rel_features"].to(device)
                bbox = batch["bbox_pairs"].to(device)
                if "has_valid_depth" in batch:
                    if not batch["has_valid_depth"]:
                        continue
                depth = depth.squeeze().unsqueeze(0).unsqueeze(0)
                if args.do_kb_crop:
                    top_margin = batch["top_margin"]
                    left_margin = batch["left_margin"]
                if "has_valid_depth" in batch:
                    if not batch["has_valid_depth"]:
                        continue
                if args.do_kb_crop:
                    _, pred = self.model(
                        img,
                        bbox,
                        rel_features,
                        top_margin=top_margin,
                        left_margin=left_margin,
                    )
                else:
                    _, pred = self.model(img, bbox, rel_features)
                pred = torch.nn.functional.interpolate(
                    pred, depth.shape[-2:], mode="bilinear", align_corners=True
                )
                pred = pred.squeeze().cpu().numpy()
                pred[pred < self.args.min_depth_eval] = self.args.min_depth_eval
                pred[pred > self.args.max_depth_eval] = self.args.max_depth_eval
                pred[np.isinf(pred)] = self.args.max_depth_eval
                pred[np.isnan(pred)] = self.args.min_depth_eval

                gt_depth = depth.squeeze().cpu().numpy()
                valid_mask = np.logical_and(
                    gt_depth > self.args.min_depth_eval,
                    gt_depth < self.args.max_depth_eval,
                )
                if self.args.garg_crop or self.args.eigen_crop:
                    gt_height, gt_width = gt_depth.shape
                    eval_mask = np.zeros(valid_mask.shape)

                    if self.args.garg_crop:
                        eval_mask[
                            int(0.40810811 * gt_height) : int(0.99189189 * gt_height),
                            int(0.03594771 * gt_width) : int(0.96405229 * gt_width),
                        ] = 1

                    elif self.args.eigen_crop:
                        if self.args.dataset == "kitti":
                            eval_mask[
                                int(0.3324324 * gt_height) : int(
                                    0.91351351 * gt_height
                                ),
                                int(0.0359477 * gt_width) : int(0.96405229 * gt_width),
                            ] = 1
                        else:
                            eval_mask[45:471, 41:601] = 1
                    valid_mask = np.logical_and(valid_mask, eval_mask)
                metric = train.utils.compute_errors(gt_depth[valid_mask], pred[valid_mask])
                # print(metric)
                metrics.update(metric)
        avg_metrics = metrics.get_value()
        print("average: ",avg_metrics)


if __name__ == "__main__":
    if not len(sys.argv) > 1:
        raise " Please spcify which dataset to test (nyu/kitti)."
    if sys.argv[1] == "nyu":
        # --------------nyu-----------------#
        args = CN()
        args.dataset = "nyu"
        args.n_bins = 256
        args.min_depth = 1e-3
        args.max_depth = 10.0
        args.norm = "linear"
        args.orthogonal_disable = False
        args.attention_disable = False
        args.no_relation = False
        args.algo = "baseline"
        args.do_kb_crop = False
        args.min_depth_eval = 1e-3
        args.max_depth_eval = 10
        args.gpu = 0
        args.garg_crop = False
        args.eigen_crop = True
        args.num_threads = 1
        args.rank = 0
        args.world_size = 1
        args.batch_size = 1
        args.distributed = False
        args.data_path = "/scratch/xiaoyu/dataset/nyuv2_test/"
        args.data_path_eval = "/scratch/xiaoyu/dataset/nyuv2_test/"
        args.max_eval_num = -1
        args.pretrained = '/scratch/xiaoyu/Relationship-Spatialization-for-Depth-Estimation/checkpoints/RSMDE_27-Jul_12-01-nyu-baseline-nodebs20-tep25-lr0.0001-wd0.1-be2fdb2e-a266-4641-b5e6-991cc241e211-Ortho-Atten-Relat/model_best.pt'
        # -------------------------------#
    elif sys.argv[1] == "kitti":
        # --------------kitti-----------------#
        args = CN()
        args.dataset = "kitti"
        args.n_bins = 256
        args.min_depth = 1e-3
        args.max_depth = 80.0
        args.norm = "linear"
        args.orthogonal_disable = False
        args.attention_disable = False
        args.no_relation = False
        args.algo = "baseline"
        args.do_kb_crop = True
        args.min_depth_eval = 1e-3
        args.max_depth_eval = 80
        args.gpu = 0
        args.garg_crop = False
        args.eigen_crop = True
        args.num_threads = 1
        args.rank = 0
        args.world_size = 1
        args.batch_size = 1
        args.distributed = False
        args.data_path = "/scratch/xiaoyu/dataset/kitti_data"
        args.filenames_file_eval = "./train_test_inputs/kitti_eigen_test_files_with_gt.txt"
        args.gt_path_eval = "/scratch/xiaoyu/dataset/kitti_depth/"
        args.data_path_eval = "/scratch/xiaoyu/dataset/kitti_data"
        args.max_eval_num = -1
        args.pretrained = "/scratch/xiaoyu/Relationship-Spatialization-for-Depth-Estimation/checkpoints/RSMDE_27-Jul_12-07-kitti-baseline-nodebs20-tep25-lr0.0001-wd0.1-a5644bb3-ae85-43bd-8412-a50e98b477c6-Ortho-Atten-Relat/model_latest.pt"
        # -------------------------------#
    else:
        raise "Wrong dataset"
    t = Test(args)
    t()
