import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import h5py
import torch
from glob import glob
from PIL import Image
from tqdm import tqdm
from PIL import Image
import argparse

import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lib.config import cfg
from lib.scene_parser.parser import build_scene_parser
from lib.data.transforms import build_transforms

# import pdb

parser = argparse.ArgumentParser(description="get relationship features")
parser.add_argument(
    "--dataset",
    default="kitti",
    type=str,
    choices=["nyu", "kitti", "test", "single", "icl_nuim"],
    help="what dataset used",
)
# parser.add_argument("--data_path", default="/home/xxy/HDD/dataset/nyuv2/sync", type=str)
parser.add_argument("--data_path", default="/home/xxy/HDD/dataset/kitti_data", type=str)
args = parser.parse_args()

cfg.merge_from_file("train_test_inputs/sgg_res101_step.yaml")
cfg.inference = True
cfg.resume = 1
cfg.MODEL.ALGORITHM = "sg_grcnn"
cfg.DATASET.NAME = args.dataset
rel_model = build_scene_parser(cfg)
ckpt = torch.load(cfg.MODEL.WEIGHT_DET, map_location=torch.device("cpu")).pop("model")
load_dict = {}
for k, v in ckpt.items():
    if k.startswith("module."):
        k_ = k.replace("module.", "")
        load_dict[k_] = v
    else:
        load_dict[k] = v

rel_model.load_state_dict(load_dict)
if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOXES:
    rel_model.rel_heads.box_feature_extractor.load_state_dict(
        rel_model.roi_heads.box.feature_extractor.state_dict()
    )
    rel_model.rel_heads.box_predictor.load_state_dict(
        rel_model.roi_heads.box.predictor.state_dict()
    )
rel_model.eval()
rel_model.to("cuda:0")
rel_transforms = build_transforms(cfg, is_train=True, nyuv=True)
print(args.dataset)
if args.dataset == "nyu":
    train_data_path = args.data_path
    train_rel_path = args.data_path
    cat_dir = os.listdir(train_data_path)
    t1 = tqdm(cat_dir, total=len(cat_dir))
    for cd in t1:
        t1.set_description(cd)
        if not os.path.exists(os.path.join(train_rel_path, cd)):
            os.makedirs(os.path.join(train_rel_path, cd))
        filenames = glob(os.path.join(train_data_path, cd, "*.jpg"))

        t2 = tqdm(filenames, total=len(filenames))
        for f in t2:
            t2.set_description(os.path.basename(f))
            if os.path.exists(
                os.path.join(
                    train_rel_path, cd, os.path.basename(f).replace("jpg", "h5")
                )
            ):
                continue
            img = Image.open(f)
            img, _ = rel_transforms(img, None)
            img = img.cuda()
            # pdb.set_trace()
            rel_out = rel_model(img)
            rel_features = rel_out[0][1]
            bbox = torch.cat([b.bbox for b in rel_out[1][1]], dim=0)
            if rel_features.shape[0] < 256:
                new_zeros_t = torch.zeros(
                    [256 - rel_features.shape[0], rel_features.shape[1]],
                    dtype=rel_features.dtype,
                    device=rel_features.device,
                )
                new_zeros_b = torch.zeros(
                    [256 - bbox.shape[0], bbox.shape[1]],
                    dtype=bbox.dtype,
                    device=bbox.device,
                )
                rel_features = torch.cat([rel_features, new_zeros_t], dim=0)
                bbox = torch.cat([bbox, new_zeros_b], dim=0)
            rel_features = rel_features.cpu().detach().numpy()
            bbox = bbox.cpu().detach().numpy()

            im_rel_h5 = h5py.File(
                os.path.join(
                    train_rel_path, cd, os.path.basename(f).replace("jpg", "h5")
                ),
                "w",
            )
            im_rel_h5.create_dataset("rel_features", data=rel_features)
            im_rel_h5.create_dataset("bbox", data=bbox)
            im_rel_h5.close()

if args.dataset == "test":
    ext = ".jpg"
    # test_data_path = '/scratch/xiaoyu/depth-estimation/data/NYUv2/image/test'
    test_data_path = "/scratch/xiaoyu/depth-estimation/source/RaMDE/test_imgs"
    # filenames = glob(os.path.join(test_data_path,'*.png'))
    filenames = glob(os.path.join(test_data_path, "*{}".format(ext)))
    for f in filenames:
        print(f)
        img = Image.open(f)
        img, _ = rel_transforms(img, None)
        img = img.cuda()
        rel_out = rel_model(img)
        rel_features = rel_out[0][1]
        bbox = torch.cat([b.bbox for b in rel_out[1][1]], dim=0)
        if rel_features.shape[0] < 256:
            new_zeros_t = torch.zeros(
                [256 - rel_features.shape[0], rel_features.shape[1]],
                dtype=rel_features.dtype,
                device=rel_features.device,
            )
            new_zeros_b = torch.zeros(
                [256 - bbox.shape[0], bbox.shape[1]],
                dtype=bbox.dtype,
                device=bbox.device,
            )
            rel_features = torch.cat([rel_features, new_zeros_t], dim=0)
            bbox = torch.cat([bbox, new_zeros_b], dim=0)
        rel_features = rel_features.cpu().detach().numpy()
        bbox = bbox.cpu().detach().numpy()
        im_rel_h5 = h5py.File(f.replace(ext, ".h5"), "w")
        im_rel_h5.create_dataset("rel_features", data=rel_features)
        im_rel_h5.create_dataset("bbox", data=bbox)
        im_rel_h5.close()

if args.dataset == "kitti":
    # data_path = "/scratch/xiaoyu/depth-estimation/data/kitti_data"
    data_path = args.data_path
    files_path = [
        "train_test_inputs/kitti_eigen_train_files_with_gt.txt",
        "train_test_inputs/kitti_eigen_test_files_with_gt.txt",
    ]
    for fp in files_path:
        file = open(fp, "r")
        lines = file.readlines()
        t1 = tqdm(lines, total=len(lines))
        for l in t1:
            [img_path, depth, focal] = l.split(" ")
            t1.set_description(img_path)
            v = img_path.strip()
            f = os.path.join(data_path, img_path)
            if os.path.exists(f.replace(".png", ".h5")):
                print(f.replace(".png", ".h5"))
                continue
            with torch.no_grad():
                img = Image.open(f)
                img, _ = rel_transforms(img, None)
                img = img.cuda()
                rel_out = rel_model(img)
                rel_features = rel_out[0][1]
                bbox = torch.cat([b.bbox for b in rel_out[1][1]], dim=0)
                if rel_features.shape[0] < 256:
                    new_zeros_t = torch.zeros(
                        [256 - rel_features.shape[0], rel_features.shape[1]],
                        dtype=rel_features.dtype,
                        device=rel_features.device,
                    )
                    new_zeros_b = torch.zeros(
                        [256 - bbox.shape[0], bbox.shape[1]],
                        dtype=bbox.dtype,
                        device=bbox.device,
                    )
                    rel_features = torch.cat([rel_features, new_zeros_t], dim=0)
                    bbox = torch.cat([bbox, new_zeros_b], dim=0)
                rel_features = rel_features.cpu().detach().numpy()
                bbox = bbox.cpu().detach().numpy()

                im_rel_h5 = h5py.File(f.replace(".png", ".h5"), "w")
                im_rel_h5.create_dataset("rel_features", data=rel_features)
                im_rel_h5.create_dataset("bbox", data=bbox)
                im_rel_h5.close()

if args.dataset == "single":
    f = "./test_imgs/150272.jpg"
    img = Image.open(f)
    print(img.height, img.width)
    img, _ = rel_transforms(img, None)
    img = img.cuda()
    rel_out = rel_model(img)
    rel_features = rel_out[0][1]
    bbox = torch.cat([b.bbox for b in rel_out[1][1]], dim=0)
    if rel_features.shape[0] < 256:
        new_zeros_t = torch.zeros(
            [256 - rel_features.shape[0], rel_features.shape[1]],
            dtype=rel_features.dtype,
            device=rel_features.device,
        )
        new_zeros_b = torch.zeros(
            [256 - bbox.shape[0], bbox.shape[1]], dtype=bbox.dtype, device=bbox.device
        )
        rel_features = torch.cat([rel_features, new_zeros_t], dim=0)
        bbox = torch.cat([bbox, new_zeros_b], dim=0)
    rel_features = rel_features.cpu().detach().numpy()
    bbox = bbox.cpu().detach().numpy()
    ext = os.path.splitext(f)[-1]
    print(ext)
    im_rel_h5 = h5py.File(f.replace(ext, ".h5"), "w")
    im_rel_h5.create_dataset("rel_features", data=rel_features)
    im_rel_h5.create_dataset("bbox", data=bbox)
    im_rel_h5.close()

if args.dataset == "icl_nuim":
    data_path = "/scratch/xiaoyu/depth-estimation/data/icl_nuim"
    image_list_file = "./train_test_inputs/icl_nuim_test_files_with_gt.txt"
    image_list = open(image_list_file, "r")
    image_list = image_list.readlines()
    image_counts = len(image_list)
    for il in image_list:
        [index, depth_path, _, rgb_path] = il.split()
        print("{}/{}".format(index, image_counts))
        rgb_path = rgb_path.strip()
        f = os.path.join(data_path, rgb_path)
        if os.path.exists(f.replace(".png", ".h5")):
            print(f.replace(".png", ".h5"))
            continue
        with torch.no_grad():
            img = Image.open(f)
            img, _ = rel_transforms(img, None)
            img = img.cuda()
            rel_out = rel_model(img)
            rel_features = rel_out[0][1]
            bbox = torch.cat([b.bbox for b in rel_out[1][1]], dim=0)
            if rel_features.shape[0] < 256:
                new_zeros_t = torch.zeros(
                    [256 - rel_features.shape[0], rel_features.shape[1]],
                    dtype=rel_features.dtype,
                    device=rel_features.device,
                )
                new_zeros_b = torch.zeros(
                    [256 - bbox.shape[0], bbox.shape[1]],
                    dtype=bbox.dtype,
                    device=bbox.device,
                )
                rel_features = torch.cat([rel_features, new_zeros_t], dim=0)
                bbox = torch.cat([bbox, new_zeros_b], dim=0)
            rel_features = rel_features.cpu().detach().numpy()
            bbox = bbox.cpu().detach().numpy()

            im_rel_h5 = h5py.File(f.replace(".png", ".h5"), "w")
            im_rel_h5.create_dataset("rel_features", data=rel_features)
            im_rel_h5.create_dataset("bbox", data=bbox)
            im_rel_h5.close()
