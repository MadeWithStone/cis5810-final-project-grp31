# Copyright 2021 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PoolFormer implementation
"""
import copy
import os

import numpy as np
import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_
from timm.models.layers.helpers import to_2tuple
from timm.models.registry import register_model


class PatchEmbed(nn.Module):
    """
    Patch Embedding that is implemented by a layer of conv.
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H/stride, W/stride]
    """

    def __init__(
        self,
        patch_size=16,
        stride=16,
        padding=0,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
    ):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class LayerNormChannel(nn.Module):
    """
    LayerNorm only for Channel Dimension.
    Input: tensor in shape [B, C, H, W]
    """

    def __init__(self, num_channels, eps=1e-05):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight.unsqueeze(-1).unsqueeze(-1) * x + self.bias.unsqueeze(
            -1
        ).unsqueeze(-1)
        return x


class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """

    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


class Pooling(nn.Module):
    """
    Implementation of pooling for PoolFormer
    --pool_size: pooling size
    """

    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size // 2, count_include_pad=False
        )

    def forward(self, x):
        return self.pool(x) - x


class PoolAttn(nn.Module):
    """
    Implementation of pooling for PoolFormer
    --pool_size: pooling size
    """

    def __init__(
        self,
        dim=256,
        norm_layer=GroupNorm,
    ):
        super().__init__()
        # TODO: Initialize adaptive average pooling in path-wise pooling attention
        self.patch_pool1 = nn.AdaptiveAvgPool2d((None, 4))
        self.patch_pool2 = nn.AdaptiveAvgPool2d((4, None))
        # TODO: Initialize adaptive average pooling in embed-wise pooling attention
        self.embdim_pool1 = nn.AdaptiveAvgPool2d((None, 4))
        self.embdim_pool2 = nn.AdaptiveAvgPool2d((4, None))
        # Norm before last projection
        self.norm = norm_layer(dim)
        # Projection (conv layer)
        self.proj0 = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)


    def forward(self, x):
        # Shape of X_0
        B, D, H, W = x.shape

        ## Patch-wise pooling attention ##
        # TODO: 1. Squeeze along h-axis; get X_ph ~ (B,D,H,4)
        squeeze_h = self.patch_pool1(x)

        # TODO: 2. Squeeze along w-axis; get X_pw ~ (B,D,4,W)
        squeeze_w = self.patch_pool2(x)

        # TODO: 3. Matrix multiplication; get X_1 = X_ph @ X_pw
        x_1 = squeeze_h @ squeeze_w

        # TODO: 4. Projection of X_1 by proj0
        x_1 = self.proj0(x_1)

        ## Embed-wise pooling attention ##
        # 1. Reshape X_0 to get X_0_ ~ (B,N,Dh,Dw), where N=H x W, Dh=32, Dw=D/Dh (2 for D=64)
        x_0_ = x.view(B, D, H * W).transpose(1, 2).view(B, H * W, 32, -1)

        # TODO: 2. Squeeze along Dh-axis; get X_PDh ~ (B,N,D_h,4)
        squeeze_dh = self.embdim_pool1(x_0_)

        # TODO: 2. Squeeze along Dw-axis; get X_PDw ~ (B,N,4,D_w)
        squeeze_dw = self.embdim_pool2(x_0_)

        # TODO: 3. Matrix multiplication; get X2 ~ (B,N,Dh,Dw)
        x_2 = squeeze_dh @ squeeze_dw

        # TODO: 4. Reshape X2; get X3 ~ (B,D,H,W)
        x_3 = x_2.view(B, H * W, D).transpose(1, 2).view(B, D, H, W)

        # TODO: Projection of x_3 by proj1
        x_3 = self.proj1(x_3)


        # TODO: Add X_1 and X_3 with norm and then projected by proj2
        x_out = self.norm(x_1 + x_3)
        x_out = self.proj2(x_out)

        return x_out


class Mlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PoolFormerBlock(nn.Module):
    """
    Implementation of one PoolFormer block.
    --dim: embedding dim
    --pool_size: pooling size
    --mlp_ratio: mlp expansion ratio
    --act_layer: activation
    --norm_layer: normalization
    --drop: dropout rate
    --drop path: Stochastic Depth,
        refer to https://arxiv.org/abs/1603.09382
    --use_layer_scale, --layer_scale_init_value: LayerScale,
        refer to https://arxiv.org/abs/2103.17239
    """

    def __init__(
        self,
        dim,
        pool_size=3,
        mlp_ratio=4.0,
        act_layer=nn.GELU,
        norm_layer=GroupNorm,
        drop=0.0,
        drop_path=0.0,
        use_layer_scale=True,
        layer_scale_init_value=1e-5,
        block=None,
        sub_idx=None,
    ):

        super().__init__()
        self.block = block
        self.sub_idx = sub_idx
        self.norm1 = norm_layer(dim)
        # Pooling Attention
        self.token_mixer = PoolAttn(
            dim=dim,
            norm_layer=norm_layer,
        )
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        # MLP
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        # The following two techniques are useful to train deep PoolFormers.
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True
            )
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True
            )

    def forward(self, x):
        if self.use_layer_scale:
            # Token mixing
            x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.token_mixer(self.norm1(x)))
            # MLP
            x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        else:
            # Token mixing
            x = x + self.drop_path(self.token_mixer(self.norm1(x)))
            # MLP
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def basic_blocks(
    dim,
    index,
    layers,
    pool_size=3,
    mlp_ratio=4.0,
    act_layer=nn.GELU,
    norm_layer=GroupNorm,
    drop_rate=0.0,
    drop_path_rate=0.0,
    use_layer_scale=True,
    layer_scale_init_value=1e-5,
):
    """
    generate PoolFormer blocks for a stage
    return: PoolFormer blocks
    """
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = (
            drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)
        )
        blocks.append(
            PoolFormerBlock(
                dim,
                pool_size=pool_size,
                mlp_ratio=mlp_ratio,
                act_layer=act_layer,
                norm_layer=norm_layer,
                drop=drop_rate,
                drop_path=block_dpr,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
            )
        )
    blocks = nn.Sequential(*blocks)

    return blocks


class PoolAttnFormer(nn.Module):
    def __init__(
        self,
        layers,
        embed_dims=None,
        mlp_ratios=None,
        downsamples=None,
        pool_size=3,
        norm_layer=GroupNorm,
        act_layer=nn.GELU,
        num_classes=1000,
        in_patch_size=7,
        in_stride=4,
        in_pad=2,
        down_patch_size=3,
        down_stride=2,
        down_pad=1,
        drop_rate=0.0,
        drop_path_rate=0.0,
        use_layer_scale=True,
        layer_scale_init_value=1e-5,
        fork_feat=False,
        init_cfg=None,
        pretrained=None,
        **kwargs,
    ):

        super().__init__()

        if not fork_feat:
            self.num_classes = num_classes
        self.fork_feat = fork_feat

        self.patch_embed = PatchEmbed(
            patch_size=in_patch_size,
            stride=in_stride,
            padding=in_pad,
            in_chans=3,
            embed_dim=embed_dims[0],
        )

        # set the main block in network
        network = []
        for i in range(len(layers)):
            stage = basic_blocks(
                embed_dims[i],
                i,
                layers,
                pool_size=pool_size,
                mlp_ratio=mlp_ratios[i],
                act_layer=act_layer,
                norm_layer=norm_layer,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
            )
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if downsamples[i] or embed_dims[i] != embed_dims[i + 1]:
                # downsampling between two stages
                network.append(
                    PatchEmbed(
                        patch_size=down_patch_size,
                        stride=down_stride,
                        padding=down_pad,
                        in_chans=embed_dims[i],
                        embed_dim=embed_dims[i + 1],
                    )
                )
        self.network = nn.ModuleList(network)

        if self.fork_feat:
            # add a norm layer for each output
            self.out_indices = [0, 2, 4, 6]
            for i_emb, i_layer in enumerate(self.out_indices):
                if i_emb == 0 and os.environ.get("FORK_LAST3", None):
                    # TODO: more elegant way
                    """For RetinaNet, `start_level=1`. The first norm layer will not used.
                    cmd: `FORK_LAST3=1 python -m torch.distributed.launch ...`
                    """
                    layer = nn.Identity()
                else:
                    layer = norm_layer(embed_dims[i_emb])
                layer_name = f"norm{i_layer}"
                self.add_module(layer_name, layer)
        else:
            # Classifier head
            self.norm = norm_layer(embed_dims[-1])
            self.head = (
                nn.Linear(embed_dims[-1], num_classes)
                if num_classes > 0
                else nn.Identity()
            )
        self.apply(self.cls_init_weights)

    # init for classification
    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    def forward_embeddings(self, x):
        x = self.patch_embed(x)
        return x

    def forward_tokens(self, x):
        outs = []
        for idx, block in enumerate(self.network):
            x = block(x)
            if self.fork_feat and idx in self.out_indices:
                norm_layer = getattr(self, f"norm{idx}")
                x_out = norm_layer(x)
                outs.append(x_out)
        if self.fork_feat:
            # output the features of four stages for dense prediction
            return outs
        # output only the features of last layer for image classification
        return x

    def forward(self, x):
        # input embedding
        x = self.forward_embeddings(x)
        # through backbone
        x = self.forward_tokens(x)
        if self.fork_feat:
            # otuput features of four stages for dense prediction
            return x
        x = self.norm(x)
        cls_out = self.head(x.mean([-2, -1]))
        # for image classification
        return cls_out


class PatchSplit(nn.Module):
    def __init__(self, stride=2, in_chans=256, embed_dim=128, norm_layer=None):
        super().__init__()
        self.stride = stride
        self.upsample = nn.Upsample(scale_factor=stride, mode="nearest")
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=1, padding=1)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.upsample(x)
        x = self.proj(x)
        x = self.norm(x)

        return x


class HR_stream(nn.Module):
    def __init__(
        self,
        dim,
        layers,
        mlp_ratio,
        act_layer=nn.GELU,
        norm_layer=GroupNorm,
        drop_rate=0.0,
        drop_path_rate=0.0,
        use_layer_scale=True,
        layer_scale_init_value=1e-5,
    ):
        super().__init__()
        # Layer parameter initialization
        layers = [2, 2, 2, 4]
        mlp_ratio = [2, 2, 2, 4]

        ## Stage 2 in HR-steam, Block 1
        depth1 = layers[1]
        mlp_ratio1 = mlp_ratio[1]
        dpr1 = [x.item() for x in torch.linspace(0, drop_path_rate, depth1)]
        self.patch_emb1 = PatchSplit(stride=2, in_chans=dim[1], embed_dim=dim[0])
        self.Block1 = nn.Sequential(
            *[
                PoolFormerBlock(
                    dim[0],
                    mlp_ratio=mlp_ratio1,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    drop=drop_rate,
                    drop_path=dpr1[i],
                    use_layer_scale=use_layer_scale,
                    layer_scale_init_value=layer_scale_init_value,
                    block=1,
                    sub_idx=i,
                )
                for i in range(depth1)
            ]
        )

        # Stage 3 in HR-stream, Block 2
        depth2 = layers[2]
        mlp_ratio2 = mlp_ratio[2]
        dpr2 = [x.item() for x in torch.linspace(0, drop_path_rate, depth2)]
        self.patch_emb2 = nn.Sequential(
            PatchSplit(
                stride=2,
                in_chans=dim[2],
                embed_dim=dim[1],
                norm_layer=norm_layer,
            ),
            PatchSplit(stride=2, in_chans=dim[1], embed_dim=dim[0]),
        )
        self.Block2 = nn.Sequential(
            *[
                PoolFormerBlock(
                    dim[0],
                    mlp_ratio=mlp_ratio2,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    drop=drop_rate,
                    drop_path=dpr2[i],
                    use_layer_scale=use_layer_scale,
                    layer_scale_init_value=layer_scale_init_value,
                    block=2,
                    sub_idx=i,
                )
                for i in range(depth2)
            ]
        )

        # Stage 4 in HR-stream, Block 3
        depth3 = layers[3]
        mlp_ratio3 = mlp_ratio[3]
        dpr3 = [x.item() for x in torch.linspace(0, drop_path_rate, depth3)]
        self.patch_emb3 = nn.Sequential(
            PatchSplit(
                stride=2,
                in_chans=dim[3],
                embed_dim=dim[2],
                norm_layer=norm_layer,
            ),
            PatchSplit(
                stride=2,
                in_chans=dim[2],
                embed_dim=dim[1],
                norm_layer=norm_layer,
            ),
            PatchSplit(stride=2, in_chans=dim[1], embed_dim=dim[0]),
        )
        self.Block3 = nn.Sequential(
            *[
                PoolFormerBlock(
                    dim[0],
                    mlp_ratio=mlp_ratio3,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    drop=drop_rate,
                    drop_path=dpr3[i],
                    use_layer_scale=use_layer_scale,
                    layer_scale_init_value=layer_scale_init_value,
                    block=3,
                    sub_idx=i,
                )
                for i in range(depth3)
            ]
        )

    def forward(self, x):
        # Feature map from each stage in Basic stream
        x0 = x[0]
        x1 = x[1]
        x2 = x[2]
        x3 = x[3]
        # Stage 2 in HR stream
        x1_new = self.patch_emb1(x1) + x0
        x1_new = self.Block1(x1_new)
        # Stage 3 in HR stream
        x2_new = self.patch_emb2(x2) + x1_new
        x2_new = self.Block2(x2_new)
        # Stage 4 in HR stream
        x3_new = self.patch_emb3(x3) + x2_new
        x3_new = self.Block3(x3_new)
        return x3_new, x3


class PoolAttnFormer_hr(nn.Module):
    def __init__(
        self,
        img_size=224,
        layers=None,
        embed_dims=None,
        mlp_ratios=None,
        num_classes=1000,
        norm_layer=GroupNorm,
        act_layer=nn.GELU,
        drop_rate=0.1,
        drop_path_rate=0.1,
        use_layer_scale=True,
        layer_scale_init_value=1e-5,
        pretrained=None,
        **kwargs,
    ):

        super().__init__()

        self.num_classes = num_classes

        # Basic Stream
        self.poolattn_cls = PoolAttnFormer(
            layers,
            embed_dims=embed_dims,
            downsamples=[True, True, True, True],
            mlp_ratios=mlp_ratios,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            fork_feat=True,
        )
        # HR Stream
        self.stage1 = HR_stream(
            embed_dims,
            layers,
            mlp_ratio=mlp_ratios,
            act_layer=act_layer,
            norm_layer=norm_layer,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            use_layer_scale=use_layer_scale,
            layer_scale_init_value=layer_scale_init_value,
        )

        self.norm0 = norm_layer(embed_dims[0])
        self.norm3 = norm_layer(embed_dims[3])

        img_size = [img_size[1], img_size[0]]

        self.apply(self.init_weights)

        if pretrained is not None:
            pt_checkpoint = torch.load(
                pretrained, map_location=lambda storage, loc: storage
            )
            self.poolattn_cls = load_pretrained_weights(
                self.poolattn_cls, pt_checkpoint
            )
            # self.poolattn_cls.load_state_dict(pt_checkpoint, False)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.ConvTranspose2d):
            trunc_normal_(m.weight, std=0.02)

    def forward(self, x):
        # through backbone
        x = self.poolattn_cls(x)
        x3_new, x3 = self.stage1(x)
        x3_new = self.norm0(x3_new)
        x3 = self.norm3(x3)

        return x3_new, x3


def load_pretrained_weights(model, checkpoint):
    import collections

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    model_dict = model.state_dict()
    new_state_dict = collections.OrderedDict()
    matched_layers, discarded_layers = [], []
    for k, v in state_dict.items():
        # If the pretrained state_dict was saved as nn.DataParallel,
        # keys would contain "module.", which should be ignored.
        if k.startswith("module."):
            k = k[7:]
        if k.startswith("backbone."):
            k = k[9:]
        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)
    # new_state_dict.requires_grad = False
    model_dict.update(new_state_dict)

    model.load_state_dict(model_dict)
    print("load_weight", len(matched_layers))
    return model

