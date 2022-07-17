import argparse
import os
import sys
import uuid
from datetime import datetime as dt

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.distributed
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt

from utils import model_io
import models
import utils
from data import DepthDataLoader
from loss import SILogLoss, BinsChamferLoss
from utils import RunningAverage, colorize

# ---------------------------------------------------------#
# config logging
import logging

logging.basicConfig(
    filename="log",
    filemode="w",
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s", datefmt="%H:%M:%S"
)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
PROJECT = "RaMDE"

# Some tool functions
def is_rank_zero(args):
    return args.rank == 0


def colorize(value, vmin=10, vmax=1000, cmap="plasma"):
    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.0
    # squeeze last dim if it exists
    # value = value.squeeze(axis=0)

    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value, bytes=True)  # (nxmx4)

    img = value[:, :, :3]

    return img


def log_images(img, depth, pred, args, step):
    depth = colorize(depth, vmin=args.min_depth, vmax=args.max_depth)
    pred = colorize(pred, vmin=args.min_depth, vmax=args.max_depth)


# --------------------------------------------#
# main worker
def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    # Using pretrained models
    # if args.dataset == "nyu":
    #     pretrained_path = (
    #         "./pretrained/RaMDE-nyu-baseline-Relation-Orthogonal-Attention.pt"
    #     )
    # elif args.dataset == "kitti":
    #     pretrained_path = (
    #         "./pretrained/RaMDE-kitti-baseline-Relation-Orthogonal-Attention.pt"
    #     )
    # else:
    pretrained_path = None
    model = models.RSMDE.build(
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
    args.multigpu = False
    if args.distributed:
        # Use DDP
        args.multigpu = True
        args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.workers = int((args.num_workers + ngpus_per_node - 1) / ngpus_per_node)
        print(args.gpu, args.rank, args.batch_size, args.workers)
        torch.cuda.set_device(args.gpu)
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu],
            output_device=args.gpu,
            find_unused_parameters=True,
        )

    elif args.gpu is None:
        # Use DP
        args.multigpu = True
        model = model.cuda()
        model = torch.nn.DataParallel(model)

    args.epoch = 0
    args.last_epoch = -1
    torch.autograd.set_detect_anomaly(True)
    train(
        model,
        args,
        epochs=args.epochs,
        lr=args.lr,
        device=args.gpu,
        root=args.root,
        experiment_name="RSMDE",
        optimizer_state_dict=None,
    )


def train(
    model,
    args,
    epochs=10,
    experiment_name="RSMDE",
    lr=0.0001,
    root=".",
    device=None,
    optimizer_state_dict=None,
):
    global PROJECT
    print(f"Training {experiment_name}")
    run_id = f"{dt.now().strftime('%d-%h_%H-%M')}-{args.dataset}-{args.algo}-nodebs{args.bs}-tep{epochs}-lr{lr}-wd{args.wd}-{uuid.uuid4()}"
    if not args.orthogonal_disable:
        run_id = f"{run_id}-Ortho"
    if not args.attention_disable:
        run_id = f"{run_id}-Atten"
    if not args.no_relation:
        run_id = f"{run_id}-Relat"
    name = f"{experiment_name}_{run_id}"
    log_path = os.path.join("./checkpoints", name)
    print(log_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    logger.info(f"Training {experiment_name}")
    logger.info(log_path)
    should_write = (not args.distributed) or args.rank == 0
    train_loader = DepthDataLoader(args, "train").data
    test_loader = DepthDataLoader(args, "online_eval").data
    criterion_ueff = SILogLoss()
    criterion_bins = BinsChamferLoss() if args.chamfer else None
    model.train()
    params = model.parameters()
    optimizer = optim.AdamW(params, weight_decay=args.wd, lr=args.lr)
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)

    iters = len(train_loader)
    step = args.epoch * iters
    best_loss = np.inf
    train_loss = []
    eval_loss = []

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        lr,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        cycle_momentum=True,
        base_momentum=0.85,
        max_momentum=0.95,
        last_epoch=args.last_epoch,
        div_factor=args.div_factor,
        final_div_factor=args.final_div_factor,
    )
    # resume
    resume_model_name = f"{experiment_name}-{args.dataset}-{args.algo}"
    if not args.no_relation:
        resume_model_name = f"{resume_model_name}-Relation"
    if not args.orthogonal_disable:
        resume_model_name = f"{resume_model_name}-Orthogonal"
    if not args.attention_disable:
        resume_model_name = f"{resume_model_name}-Attention"
    resume_model_name = f"{resume_model_name}.pt"
    if args.resume != "" and scheduler is not None:
        print("resume training")
        model, optimizer, _ = model_io.load_checkpoint(
            os.path.join("./pretrained", resume_model_name), model, optimizer
        )

    print("start training")
    for epoch in range(args.epoch, epochs):
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            img = batch["image"].to(device)
            depth = batch["depth"].to(device)
            rel_features = batch["rel_features"].to(device)
            bbox = batch["bbox_pairs"].to(device)
            if args.do_kb_crop:
                top_margin = batch["top_margin"]
                left_margin = batch["left_margin"]
            if "has_valid_depth" in batch:
                if not batch["has_valid_depth"]:
                    continue
            if args.do_kb_crop:
                bin_edges, pred = model(
                    img,
                    bbox,
                    rel_features,
                    top_margin=top_margin,
                    left_margin=left_margin,
                )
            else:
                bin_edges, pred = model(img, bbox, rel_features)
            mask = depth > args.min_depth

            print(
                pred.max().detach().cpu().numpy(),
                pred.min().detach().cpu().numpy(),
                depth[mask].max().detach().cpu().numpy(),
                depth[mask].min().detach().cpu().numpy(),
            )

            l_dense = criterion_ueff(
                pred, depth, mask=mask.to(torch.bool), interpolate=True
            )
            if args.w_chamfer > 0:
                l_chamfer = criterion_bins(bin_edges, depth)
            else:
                l_chamfer = torch.Tensor([0]).to(img.device)
            pred_interpolate = nn.functional.interpolate(
                pred, depth.shape[-2:], mode="bilinear", align_corners=True
            )
            l_L1 = (pred_interpolate[mask] - depth[mask]).abs().mean()
            loss = (
                l_dense
                + args.w_chamfer * l_chamfer
                + 0.01 * l_L1
                + 0.01 * torch.abs(pred.max() - depth[mask].max())
                + 0.01 * torch.abs(pred.min() - depth[mask].min())
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.1)  # optional
            optimizer.step()
            step += 1
            scheduler.step()

            if i % args.print_every == 0:
                log = "[e:{}-{}/{}], loss {:.4f}".format(
                    epoch, i, len(train_loader), loss.detach().cpu().numpy()
                )
                print(log)
                logger.info(log)
            if should_write and step % args.validate_every == 0:
                train_loss.append(loss.detach().cpu().numpy())
                model.eval()
                metrics, val_si = validate(
                    args, model, test_loader, criterion_ueff, epoch, epochs, device
                )
                eval_loss.append(metrics["abs_rel"])
                log = "e: {}, step: {}, abs_rel: {:.4f}, rmse:{:.4f}".format(
                    epoch, step, metrics["abs_rel"], metrics["rmse"]
                )
                print(log)
                logger.info(log)
                if metrics["abs_rel"] < best_loss and should_write:
                    model_io.save_checkpoint(
                        model, optimizer, epoch, f"model_best.pt", root=log_path
                    )
                    best_loss = metrics["abs_rel"]
                model_io.save_checkpoint(
                    model, optimizer, epoch, f"model_latest.pt", root=log_path
                )
                model_io.save_checkpoint(
                    model, optimizer, epoch, resume_model_name, root="./pretrained"
                )
                fig, ax = plt.subplots(1, 1)
                ax.plot(train_loss, label="train_loss")
                ax.legend()
                fig.savefig(
                    os.path.join(log_path, "./{}_loss.png".format(experiment_name))
                )
                fig, ax = plt.subplots(1, 1)
                ax.plot(eval_loss, label="abs_rel")
                ax.legend()
                fig.savefig(
                    os.path.join(log_path, "./{}_absrel.png".format(experiment_name))
                )
                plt.close("all")
                np.save(
                    os.path.join(log_path, "./{}_loss.npy".format(experiment_name)),
                    np.array(train_loss),
                )
                np.save(
                    os.path.join(log_path, "./{}_absrel.npy".format(experiment_name)),
                    np.array(eval_loss),
                )
                model.train()
        logger.info("train_loss: {:.4f}".format(loss.item()))
        print("train_loss: {:.4f}".format(loss.item()))


def validate(args, model, test_loader, criterion_ueff, epoch, epochs, device="cpu"):
    with torch.no_grad():
        val_si = RunningAverage()
        metrics = utils.RunningAverageDict()
        for batch in (
            tqdm(test_loader, desc=f"Epoch: {epoch + 1}/{epochs}. Loop: Validation")
            if is_rank_zero(args)
            else test_loader
        ):
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
                _, pred = model(
                    img,
                    bbox,
                    rel_features,
                    top_margin=top_margin,
                    left_margin=left_margin,
                )
            else:
                _, pred = model(img, bbox, rel_features)
            mask = depth > args.min_depth
            l_dense = criterion_ueff(
                pred, depth, mask=mask.to(torch.bool), interpolate=True
            )
            val_si.append(l_dense.item())
            pred = nn.functional.interpolate(
                pred, depth.shape[-2:], mode="bilinear", align_corners=True
            )
            pred = pred.squeeze().cpu().numpy()
            pred[pred < args.min_depth_eval] = args.min_depth_eval
            pred[pred > args.max_depth_eval] = args.max_depth_eval
            pred[np.isinf(pred)] = args.max_depth_eval
            pred[np.isnan(pred)] = args.min_depth_eval

            gt_depth = depth.squeeze().cpu().numpy()
            valid_mask = np.logical_and(
                gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval
            )
            if args.garg_crop or args.eigen_crop:
                gt_height, gt_width = gt_depth.shape
                eval_mask = np.zeros(valid_mask.shape)

                if args.garg_crop:
                    eval_mask[
                        int(0.40810811 * gt_height) : int(0.99189189 * gt_height),
                        int(0.03594771 * gt_width) : int(0.96405229 * gt_width),
                    ] = 1

                elif args.eigen_crop:
                    if args.dataset == "kitti":
                        eval_mask[
                            int(0.3324324 * gt_height) : int(0.91351351 * gt_height),
                            int(0.0359477 * gt_width) : int(0.96405229 * gt_width),
                        ] = 1
                    else:
                        eval_mask[45:471, 41:601] = 1
                valid_mask = np.logical_and(valid_mask, eval_mask)
            metrics.update(utils.compute_errors(gt_depth[valid_mask], pred[valid_mask]))

        return metrics.get_value(), val_si


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield str(arg)


if __name__ == "__main__":
    __spec__ = None
    # Arguments
    parser = argparse.ArgumentParser(
        description="Training script. Default values of all arguments are recommended for reproducibility",
        fromfile_prefix_chars="@",
        conflict_handler="resolve",
    )
    parser.convert_arg_line_to_args = convert_arg_line_to_args
    parser.add_argument(
        "--epochs", default=25, type=int, help="number of total epochs to run"
    )
    parser.add_argument(
        "--n-bins",
        "--n_bins",
        default=256,
        type=int,
        help="number of bins/buckets to divide depth range into",
    )
    # parser.add_argument('--lr', '--learning-rate', default=0.000357, type=float, help='max learning rate')
    parser.add_argument(
        "--lr", "--learning-rate", default=5e-3, type=float, help="max learning rate"
    )
    parser.add_argument(
        "--wd", "--weight-decay", default=0.1, type=float, help="weight decay"
    )
    parser.add_argument(
        "--w_chamfer", default=0.05, type=float, help="weight value for chamfer loss",
    )
    parser.add_argument(
        "--div-factor", default=25, type=float, help="Initial div factor for lr",
    )
    parser.add_argument(
        "--final-div-factor", default=100, type=float, help="final div factor for lr",
    )

    parser.add_argument("--bs", default=2, type=int, help="batch size")
    parser.add_argument("--print_every", default=100, type=int, help="print period")
    parser.add_argument(
        "--validate_every", default=1000, type=int, help="validation period"
    )
    parser.add_argument("--gpu", default=0, type=int, help="Which gpu to use")

    parser.add_argument(
        "--norm",
        default="linear",
        type=str,
        help="Type of norm/competition for bin-widths",
        choices=["linear", "softmax", "sigmoid"],
    )
    parser.add_argument(
        "--same-lr",
        default=False,
        action="store_true",
        help="Use same LR for all param groups",
    )
    parser.add_argument(
        "--distributed", default=False, action="store_true", help="Use DDP if set"
    )
    parser.add_argument(
        "--root", default=".", type=str, help="Root folder to save data in"
    )
    parser.add_argument("--resume", default="", type=str, help="Resume from checkpoint")

    parser.add_argument("--notes", default="", type=str, help="Wandb notes")
    parser.add_argument("--tags", default="sweep", type=str, help="Wandb tags")

    parser.add_argument(
        "--workers", default=8, type=int, help="Number of workers for data loading"
    )
    parser.add_argument(
        "--dataset", default="nyu", type=str, help="Dataset to train on"
    )

    parser.add_argument(
        "--data_path",
        default="../../HDD/dataset/NYUv2Whole/nyuv2/sync",
        type=str,
        help="path to dataset",
    )
    parser.add_argument(
        "--gt_path",
        default="../../HDD/dataset/NYUv2Whole/nyuv2/sync",
        type=str,
        help="path to dataset",
    )
    parser.add_argument(
        "--filenames_file",
        default="./train_test_inputs/nyuv2_train_gt_new.txt",
        type=str,
        help="path to the filenames text file",
    )

    parser.add_argument("--input_height", type=int, help="input height", default=416)
    parser.add_argument("--input_width", type=int, help="input width", default=544)
    parser.add_argument(
        "--max_depth", type=float, help="maximum depth in estimation", default=10
    )
    parser.add_argument(
        "--min_depth", type=float, help="minimum depth in estimation", default=1e-3
    )

    parser.add_argument(
        "--do_random_rotate",
        default=False,
        help="if set, will perform random rotation for augmentation",
        action="store_true",
    )
    parser.add_argument(
        "--degree", type=float, help="random rotation maximum degree", default=2.5
    )
    parser.add_argument(
        "--do_kb_crop",
        help="if set, crop input images as kitti benchmark images",
        action="store_true",
    )
    parser.add_argument(
        "--use_right",
        help="if set, will randomly use right images when train on KITTI",
        action="store_true",
    )

    parser.add_argument(
        "--data_path_eval",
        default="../../HDD/dataset/NYUv2Whole/nyuv2_test/",
        type=str,
        help="path to the data for online evaluation",
    )
    parser.add_argument(
        "--gt_path_eval",
        default="../../data/kitti_depth/val",
        type=str,
        help="path to the groundtruth data for online evaluation",
    )
    parser.add_argument(
        "--filenames_file_eval",
        default="./train_test_inputs/nyuv2_test_gt_new.txt",
        type=str,
        help="path to the filenames text file for online evaluation",
    )

    parser.add_argument(
        "--min_depth_eval",
        type=float,
        help="minimum depth for evaluation",
        default=1e-3,
    )
    parser.add_argument(
        "--max_depth_eval", type=float, help="maximum depth for evaluation", default=10
    )
    parser.add_argument(
        "--eigen_crop",
        default=True,
        help="if set, crops according to Eigen NIPS14",
        action="store_true",
    )
    parser.add_argument(
        "--garg_crop",
        help="if set, crops according to Garg  ECCV16",
        action="store_true",
    )

    parser.add_argument(
        "--max_eval_num", type=int, default=-1, help="maximal evaluation number"
    )
    parser.add_argument(
        "--attention_disable",
        default=False,
        help="attention disable",
        action="store_true",
    )
    parser.add_argument(
        "--orthogonal_disable",
        default=False,
        help="orthogonal disable",
        action="store_true",
    )

    parser.add_argument("--algo", type=str, default="tri_graph")

    parser.add_argument(
        "--no_relation", default=False, help="using relationship", action="store_true"
    )

    if sys.argv.__len__() == 2:
        arg_filename_with_prefix = "@" + sys.argv[1]
        args = parser.parse_args([arg_filename_with_prefix])
    else:
        args = parser.parse_args()

    args.batch_size = args.bs
    args.num_threads = args.workers
    args.mode = "train"
    args.chamfer = args.w_chamfer > 0
    if args.root != "." and not os.path.isdir(args.root):
        os.makedirs(args.root)

    print(args)

    try:
        node_str = os.environ["SLURM_JOB_NODELIST"].replace("[", "").replace("]", "")
        nodes = node_str.split(",")

        args.world_size = len(nodes)
        args.rank = int(os.environ["SLURM_PROCID"])

    except KeyError as e:
        # We are NOT using SLURM
        args.world_size = 1
        args.rank = 0
        nodes = ["127.0.0.1"]

    if args.distributed:
        mp.set_start_method("forkserver")

        print(args.rank)
        port = np.random.randint(15000, 15025)
        args.dist_url = "tcp://{}:{}".format(nodes[0], port)
        print(args.dist_url)
        args.dist_backend = "nccl"
        args.gpu = None

    ngpus_per_node = torch.cuda.device_count()
    args.num_workers = args.workers
    args.ngpus_per_node = ngpus_per_node

    if args.distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        if ngpus_per_node == 1:
            args.gpu = 0
        main_worker(args.gpu, ngpus_per_node, args)
