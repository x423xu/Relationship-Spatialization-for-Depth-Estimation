import torch
import torch.nn as nn
import torch.nn.functional as F
from .relation_spatialization import RaMDE

from .miniViT import mViT
from models.config import _C as cfg
from utils import model_io


class UpSampleBN(nn.Module):
    def __init__(self, skip_input, output_features):
        super(UpSampleBN, self).__init__()

        self._net = nn.Sequential(
            nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_features),
            nn.LeakyReLU(),
            nn.Conv2d(
                output_features, output_features, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(output_features),
            nn.LeakyReLU(),
        )

    def forward(self, x, concat_with):
        up_x = F.interpolate(
            x,
            size=[concat_with.size(2), concat_with.size(3)],
            mode="bilinear",
            align_corners=True,
        )
        f = torch.cat([up_x, concat_with], dim=1)
        return self._net(f)


class DecoderBN(nn.Module):
    def __init__(self, num_features=2048, num_classes=1, bottleneck_features=2048):
        super(DecoderBN, self).__init__()
        features = int(num_features)

        self.conv2 = nn.Conv2d(
            bottleneck_features, features, kernel_size=1, stride=1, padding=1
        )

        self.up1 = UpSampleBN(
            skip_input=features // 1 + 112 + 64, output_features=features // 2
        )
        self.up2 = UpSampleBN(
            skip_input=features // 2 + 40 + 24, output_features=features // 4
        )
        self.up3 = UpSampleBN(
            skip_input=features // 4 + 24 + 16, output_features=features // 8
        )
        self.up4 = UpSampleBN(
            skip_input=features // 8 + 16 + 8, output_features=features // 16
        )

        #         self.up5 = UpSample(skip_input=features // 16 + 3, output_features=features//16)
        self.conv3 = nn.Conv2d(
            features // 16, num_classes, kernel_size=3, stride=1, padding=1
        )
        # self.act_out = nn.Softmax(dim=1) if output_activation == 'softmax' else nn.Identity()

    def forward(self, features):
        x_block0, x_block1, x_block2, x_block3, x_block4 = (
            features[4],
            features[5],
            features[6],
            features[8],
            features[11],
        )

        x_d0 = self.conv2(x_block4)

        x_d1 = self.up1(x_d0, x_block3)
        x_d2 = self.up2(x_d1, x_block2)
        x_d3 = self.up3(x_d2, x_block1)
        x_d4 = self.up4(x_d3, x_block0)
        #         x_d5 = self.up5(x_d4, features[0])
        out = self.conv3(x_d4)
        # out = self.act_out(out)
        # if with_features:
        #     return out, features[-1]
        # elif with_intermediate:
        #     return out, [x_block0, x_block1, x_block2, x_block3, x_block4, x_d1, x_d2, x_d3, x_d4]
        return out


class Encoder(nn.Module):
    def __init__(self, backend):
        super(Encoder, self).__init__()
        self.original_model = backend

    def forward(self, x):
        features = [x]
        for k, v in self.original_model._modules.items():
            if k == "blocks":
                for ki, vi in v._modules.items():
                    features.append(vi(features[-1]))
            else:
                features.append(v(features[-1]))
        return features


class Backbone(nn.Module):
    def __init__(
        self, backend, n_bins=100, min_val=0.1, max_val=10, norm="linear", **kwargs
    ):
        super(Backbone, self).__init__()
        self.num_classes = n_bins
        self.min_val = min_val
        self.max_val = max_val
        self.encoder = Encoder(backend)
        self.adaptive_bins_layer = mViT(
            128,
            n_query_channels=128,
            patch_size=16,
            dim_out=n_bins,
            embedding_dim=128,
            norm=norm,
        )

        self.decoder = DecoderBN(num_classes=128)
        self.conv_out = nn.Sequential(
            nn.Conv2d(128, n_bins, kernel_size=1, stride=1, padding=0),
            nn.Softmax(dim=1),
        )
        orthogonal_disable = kwargs["orthogonal_disable"]
        attention_disable = kwargs["attention_disable"]
        no_relation = kwargs["no_relation"]
        self.no_relation = no_relation
        algo = kwargs["algo"]
        do_kb_crop = kwargs["do_kb_crop"]
        print(
            "attention disable {} -- orthogonal disable {}, -- no relation {}".format(
                attention_disable, orthogonal_disable, no_relation
            )
        )
        if not no_relation:
            self.ramde = RaMDE(
                input_channel=256,
                orthogonal_disable=orthogonal_disable,
                attention_disable=attention_disable,
                algo=algo,
                do_kb_crop=do_kb_crop,
            )

    def forward(self, x, bbox, rel_features, **kwargs):
        unet_out = self.decoder(self.encoder(x))
        bin_widths_normed, range_attention_maps = self.adaptive_bins_layer(unet_out)
        if not self.no_relation:
            range_attention_maps = self.ramde(
                range_attention_maps, bbox, rel_features, **kwargs
            )
        out = self.conv_out(range_attention_maps)

        bin_widths = (
            self.max_val - self.min_val
        ) * bin_widths_normed  # .shape = N, dim_out
        bin_widths = nn.functional.pad(
            bin_widths, (1, 0), mode="constant", value=self.min_val
        )
        bin_edges = torch.cumsum(bin_widths, dim=1)

        centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
        n, dout = centers.size()
        centers = centers.view(n, dout, 1, 1)

        pred = torch.sum(out * centers, dim=1, keepdim=True)

        return bin_edges, pred

    def _freeze(self):
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.decoder.parameters():
            p.requires_grad = False
        for p in self.adaptive_bins_layer.parameters():
            p.requires_grad = False

    def get_1x_lr_params(self):  # lr/10 learning rate
        return self.encoder.parameters()

    def get_10x_lr_params(self):  # lr learning rate
        modules = [self.decoder, self.adaptive_bins_layer, self.conv_out]
        for m in modules:
            yield from m.parameters()

    @classmethod
    def build(cls, n_bins, **kwargs):
        basemodel_name = "tf_efficientnet_b5_ap"

        print("Loading base model ()...".format(basemodel_name), end="")
        basemodel = torch.hub.load(
            "./pretrained/gen-efficientnet-pytorch",
            basemodel_name,
            pretrained=True,
            source="local",
        )
        print("Done.")

        # Remove last layer
        print("Removing last two layers (global_pool & classifier).")
        basemodel.global_pool = nn.Identity()
        basemodel.classifier = nn.Identity()

        # Building Encoder-Decoder model
        print("Building Encoder-Decoder model..", end="")
        m = cls(basemodel, n_bins=n_bins, **kwargs)

        # Loading pretrained model
        path = kwargs["pretrained"]
        if path is not None:
            print("\nLoading RSMDE model from " + path)
            m, _, _ = model_io.load_checkpoint(path, m)
        print("build {} done.".format(cls.__name__))
        return m


if __name__ == "__main__":
    model = Backbone.build(100)
    x = torch.rand(2, 3, 480, 640)
    bins, pred = model(x)
    print(bins.shape, pred.shape)
