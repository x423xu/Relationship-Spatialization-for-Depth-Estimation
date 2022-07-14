from typing import ForwardRef
import torch
import torch.nn as nn
from torch.nn.modules import conv  #
from torchvision.transforms import functional as F
import sys

from collections import OrderedDict

import pickle
from PIL import Image
import numpy as np
from torchvision import transforms
import cv2
import os

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

"""
message collecting
message updating
orthogonal

step:
1. input image roi pooling: image, proposals --> output
proposal size is (256, 8), in which subject proposal is (256, 4), object size is (256, 4),
image --> output, subject images (256, 2048, 7, 7), object images (256, 2048, 7, 7)
(resnet backbone to scale image to 1/16)

2. embedding: subject images --> subject embeddings (256, C, 7, 7), object embeddings (256, C, 7, 7)

3. depth feature collection: direction 1, image embeddings -(edge features)-> depth feature (sub & obj); direction 2, 
relation features -(edge features & attention)-> depth feature (sub & obj)

4. message updating: conv

5. ** feature re-projection (hard) ** idea: collect same id subj and obj, apply channel avgpool to same channel size and 
then interpolate to original size. 
"""


class Updater(nn.Module):
    def __init__(self):
        super(Updater, self).__init__()

    def forward(self, x, y):
        return x + y


class TriGraph(nn.Module):
    def __init__(self, ch):
        super(TriGraph, self).__init__()
        self.max_iters = 2
        self.depth_message_collector = nn.ModuleList(
            [
                nn.Conv2d(ch, ch * 2, 3, 1, 1),  # image->depth
                nn.Conv2d(ch, ch * 2, 3, 1, 1),
                nn.Conv2d(ch, ch * 2, 3, 1, 1),  # relation->depth
                nn.Conv2d(ch, ch * 2, 3, 1, 1),
            ]
        )
        self.image_message_collector = nn.ModuleList(
            [
                nn.Conv2d(ch * 2, ch, 3, 1, 1),  # depth->image
                nn.Conv2d(ch * 2, ch, 3, 1, 1),
                nn.Conv2d(ch, ch, 3, 1, 1),  # relation->image
                nn.Conv2d(ch, ch, 3, 1, 1),
            ]
        )
        self.relation_message_collector = nn.ModuleList(
            [
                nn.Conv2d(ch * 2, ch, 3, 1, 1),  # depth->relation
                nn.Conv2d(ch * 2, ch, 3, 1, 1),
                nn.Conv2d(ch, ch, 3, 1, 1),  # image->relation
                nn.Conv2d(ch, ch, 3, 1, 1),
            ]
        )
        # self.updater1 = Updater()
        # self.updater2 = Updater()
        # self.updater3 = Updater()
        self.activation_fn = nn.ReLU()

    def forward(self, image_features, relation_features, depth_features):
        depth_message_from_image = [image_features]
        depth_message_from_relation = [relation_features]
        image_message_from_depth = [depth_features]
        image_message_from_relation = [relation_features]
        relation_message_from_depth = [depth_features]
        relation_message_from_image = [image_features]
        for i in range(self.max_iters):
            # collect message
            depth_message1 = self.depth_message_collector[i](
                depth_message_from_image[i].clone()
            )
            depth_message2 = self.depth_message_collector[2 + i](
                depth_message_from_relation[i].clone()
            )

            image_message1 = self.image_message_collector[i](
                image_message_from_depth[i].clone()
            )
            image_message2 = self.image_message_collector[2 + i](
                image_message_from_relation[i].clone()
            )

            relation_message1 = self.relation_message_collector[i](
                relation_message_from_depth[i].clone()
            )
            relation_message2 = self.relation_message_collector[2 + i](
                relation_message_from_image[i].clone()
            )

            # update message
            depth_features += (depth_message1 + depth_message2) / 2
            image_features += (image_message1 + image_message2) / 2
            relation_features += (relation_message1 + relation_message2) / 2

            # activation
            depth_features = self.activation_fn(depth_features)
            image_features = self.activation_fn(image_features)
            relation_features = self.activation_fn(relation_features)

            depth_message_from_image.append(image_features)
            depth_message_from_relation.append(relation_features)
            image_message_from_depth.append(depth_features)
            image_message_from_relation.append(relation_features)
            relation_message_from_depth.append(depth_features)
            relation_message_from_image.append(image_features)

        return depth_features


class RaMDE(nn.Module):
    def __init__(
        self,
        cfg,
        input_channel,
        ch=128,
        orthogonal_disable=True,
        attention_disable=True,
        algo="tri_graph",
    ):
        super(RaMDE, self).__init__()
        assert algo in ["baseline", "tri_graph"]
        self.orthogonal_disable = orthogonal_disable
        self.attention_disable = attention_disable
        self.algo = algo

        self.channel_attention = nn.AdaptiveAvgPool2d(1)
        self.rel_embedding = nn.Sequential(
            nn.Conv2d(input_channel, ch, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(ch, ch, 1, 1, 0),
        )
        self.embedding = nn.Sequential(
            nn.Conv2d(ch * 2, ch, 3, 1, 1), nn.ReLU(), nn.Conv2d(ch, ch, 3, 1, 1)
        )
        if self.algo == "tri_graph":
            self.tri_graph = TriGraph(ch=ch)

    """
    range_attention_maps: (4, 32, 240, 320)
    image: (b, 3, 480, 320)
    bbox: (b*256, 8)
    rel_feature: (b*256, 1024)
    """

    def get_random(self, shape):
        if shape > 1:
            a = torch.rand(shape)
            while a[:-1].sum() >= 1:
                a = torch.rand(shape - 1)
            a[-1] = 1 - a[:-1].sum()
        else:
            a = torch.ones([1], dtype=torch.float32)
        return a

    def interpolate(
        self, rel_features, bbox, size
    ):  # (b,128,1024,1)->(b, 128, 240, 320),bbox(4,256,8)
        out = torch.zeros(
            [rel_features.shape[0], rel_features.shape[1], size[0], size[1]],
            dtype=rel_features.dtype,
            device=rel_features.device,
        )
        # print(rel_features.shape, out.shape, bbox.shape)
        interpolate_ = nn.functional.interpolate
        # crop = F.crop
        sub_box = bbox[:, :, :4].int() // 2
        obj_box = bbox[:, :, 4:].int() // 2
        for i in range(bbox.shape[0]):
            for j in range(bbox.shape[1]):
                sx1, sy1, sx2, sy2 = sub_box[i, j, :]
                sh, sw = sy2 - sy1, sx2 - sx1
                if sh < 5 or sw < 5:
                    continue
                sf = interpolate_(rel_features[i, :, :].unsqueeze(0), size=[sh, sw])

                ox1, oy1, ox2, oy2 = obj_box[i, j, :]
                oh, ow = oy2 - oy1, ox2 - ox1
                if oh < 5 or ow < 5:
                    continue
                of = interpolate_(rel_features[i, :, :].unsqueeze(0), size=[oh, ow])
                # print(sy1, sy2)
                out[i, :, sy1:sy2, sx1:sx2] += sf.squeeze()
                out[i, :, oy1:oy2, ox1:ox2] += of.squeeze()
        return out

    def forward(self, range_attention_maps, bbox, tuple_features):
        tuple_features = tuple_features.unsqueeze(-1)
        rel_features = self.rel_embedding(tuple_features)
        # print(rel_features.shape)
        rel_features = self.interpolate(
            rel_features,
            bbox,
            size=[range_attention_maps.shape[2], range_attention_maps.shape[3]],
        )
        # rel_features = nn.functional.interpolate(rel_features, size = [range_attention_maps.shape[2],\
        #                 range_attention_maps.shape[3]])
        rel_features = rel_features.contiguous()
        image_features = range_attention_maps.contiguous()

        # print(rel_features.shape, range_attention_maps.shape)
        if self.algo == "baseline":
            depth_features = torch.cat([image_features, rel_features], dim=1)

        if self.algo == "tri_graph":
            if not self.orthogonal_disable:
                rel_features = (
                    rel_features
                    - (
                        (rel_features * image_features).sum()
                        / (image_features ** 2).sum()
                    )
                    * image_features
                )
            if not self.attention_disable:
                attention = self.channel_attention(rel_features)
                rel_features *= attention
            depth_features = torch.cat([image_features, rel_features], dim=1)
            depth_features = self.tri_graph(
                image_features.clone(), rel_features.clone(), depth_features.clone()
            )

        out = self.embedding(depth_features)

        return out

