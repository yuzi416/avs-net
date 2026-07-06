# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, Scale
from mmcv.ops import DeformConv2d
from torch import Tensor

from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures.bbox import bbox_overlaps
from mmdet.utils import (ConfigType, InstanceList, MultiConfig,
                         OptInstanceList, RangeType, reduce_mean)
from ..task_modules.prior_generators import MlvlPointGenerator
from ..task_modules.samplers import PseudoSampler
from ..utils import multi_apply
from .atss_head import ATSSHead
from .fcos_head import FCOSHead

INF = 1e8


@MODELS.register_module()
class VFNetHead(ATSSHead, FCOSHead):
    """Head of `VarifocalNet (VFNet): An IoU-aware Dense Object Detector`."""

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 regress_ranges: RangeType = ((-1, 64), (64, 128), (128, 256),
                                              (256, 512), (512, INF)),
                 center_sampling: bool = False,
                 center_sample_radius: float = 1.5,
                 sync_num_pos: bool = True,
                 gradient_mul: float = 0.1,
                 bbox_norm_type: str = 'reg_denom',
                 loss_cls_fl: ConfigType = dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 use_vfl: bool = True,
                 loss_cls: ConfigType = dict(
                     type='VarifocalLoss',
                     use_sigmoid=True,
                     alpha=0.75,
                     gamma=2.0,
                     iou_weighted=True,
                     loss_weight=1.0),
                 loss_bbox: ConfigType = dict(
                     type='GIoULoss', loss_weight=1.5),
                 loss_bbox_refine: ConfigType = dict(
                     type='GIoULoss', loss_weight=2.0),
                 norm_cfg: ConfigType = dict(
                     type='GN', num_groups=32, requires_grad=True),
                 use_atss: bool = True,
                 reg_decoded_bbox: bool = True,
                 anchor_generator: ConfigType = dict(
                     type='AnchorGenerator',
                     ratios=[1.0],
                     octave_base_scale=8,
                     scales_per_octave=1,
                     center_offset=0.0,
                     strides=[8, 16, 32, 64, 128]),
                 init_cfg: MultiConfig = dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='vfnet_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 extra_cls_convs: int = 0,
                 cls_stem_convs: int = 0,
                 reg_stem_convs: int = 0,
                 use_color_gate: bool = False,
                 color_gate_reduction: int = 16,
                 use_color_spatial_gate: bool = True,
                 **kwargs) -> None:

        # ---- must be set BEFORE super().__init__() ----
        self.extra_cls_convs = extra_cls_convs
        self.cls_stem_convs = cls_stem_convs
        self.reg_stem_convs = reg_stem_convs
        self.use_color_gate = use_color_gate
        self.color_gate_reduction = color_gate_reduction
        self.use_color_spatial_gate = use_color_spatial_gate

        # dcn base offsets, adapted from reppoints_head.py
        self.num_dconv_points = 9
        self.dcn_kernel = int(np.sqrt(self.num_dconv_points))
        self.dcn_pad = int((self.dcn_kernel - 1) / 2)
        dcn_base = np.arange(-self.dcn_pad,
                             self.dcn_pad + 1).astype(np.float64)
        dcn_base_y = np.repeat(dcn_base, self.dcn_kernel)
        dcn_base_x = np.tile(dcn_base, self.dcn_kernel)
        dcn_base_offset = np.stack([dcn_base_y, dcn_base_x], axis=1).reshape((-1))
        self.dcn_base_offset = torch.tensor(dcn_base_offset).view(1, -1, 1, 1)

        super(FCOSHead, self).__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            norm_cfg=norm_cfg,
            init_cfg=init_cfg,
            **kwargs)

        self.regress_ranges = regress_ranges
        self.reg_denoms = [regress_range[-1] for regress_range in regress_ranges]
        self.reg_denoms[-1] = self.reg_denoms[-2] * 2
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        self.sync_num_pos = sync_num_pos
        self.bbox_norm_type = bbox_norm_type
        self.gradient_mul = gradient_mul
        self.use_vfl = use_vfl

        if self.use_vfl:
            self.loss_cls = MODELS.build(loss_cls)
        else:
            self.loss_cls = MODELS.build(loss_cls_fl)

        self.loss_bbox = MODELS.build(loss_bbox)
        self.loss_bbox_refine = MODELS.build(loss_bbox_refine)

        # for getting ATSS targets
        self.use_atss = use_atss
        self.reg_decoded_bbox = reg_decoded_bbox
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)

        self.anchor_center_offset = anchor_generator['center_offset']
        self.num_base_priors = self.prior_generator.num_base_priors[0]

        if self.train_cfg:
            self.assigner = TASK_UTILS.build(self.train_cfg['assigner'])
            if self.train_cfg.get('sampler', None) is not None:
                self.sampler = TASK_UTILS.build(
                    self.train_cfg['sampler'],
                    default_args=dict(context=self))
            else:
                self.sampler = PseudoSampler()

        # only be used in `get_atss_targets` when `use_atss` is True
        self.atss_prior_generator = TASK_UTILS.build(anchor_generator)

        self.fcos_prior_generator = MlvlPointGenerator(
            anchor_generator['strides'],
            self.anchor_center_offset if self.use_atss else 0.5)

        # In order to reuse the `get_bboxes` in `BaseDenseHead`.
        # Only be used in testing phase.
        self.prior_generator = self.fcos_prior_generator

    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        super(FCOSHead, self)._init_cls_convs()
        super(FCOSHead, self)._init_reg_convs()

        # ---- task-specific stems (decouple early feature) ----
        self.cls_stem = nn.ModuleList()
        self.reg_stem = nn.ModuleList()

        in_ch = self.in_channels
        for i in range(self.cls_stem_convs):
            self.cls_stem.append(
                ConvModule(
                    in_ch if i == 0 else self.feat_channels,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias))

        for i in range(self.reg_stem_convs):
            self.reg_stem.append(
                ConvModule(
                    in_ch if i == 0 else self.feat_channels,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias))

        # ---- extra cls convs ----
        if self.extra_cls_convs > 0:
            for _ in range(self.extra_cls_convs):
                self.cls_convs.append(
                    ConvModule(
                        self.feat_channels,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        bias=self.conv_bias))

        # ---- color-aware gate for classification branch only ----
        if self.use_color_gate:
            hidden_channels = max(self.feat_channels // self.color_gate_reduction, 16)

            self.color_avgpool = nn.AdaptiveAvgPool2d(1)
            self.color_maxpool = nn.AdaptiveMaxPool2d(1)
            self.color_mlp = nn.Sequential(
                nn.Conv2d(self.feat_channels, hidden_channels, kernel_size=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_channels, self.feat_channels, kernel_size=1, bias=True)
            )

            if self.use_color_spatial_gate:
                self.color_spatial = nn.Conv2d(
                    2, 1, kernel_size=7, stride=1, padding=3, bias=False)

        self.relu = nn.ReLU(inplace=True)

        self.vfnet_reg_conv = ConvModule(
            self.feat_channels,
            self.feat_channels,
            3,
            stride=1,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            bias=self.conv_bias)
        self.vfnet_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

        self.vfnet_reg_refine_dconv = DeformConv2d(
            self.feat_channels,
            self.feat_channels,
            self.dcn_kernel,
            1,
            padding=self.dcn_pad)
        self.vfnet_reg_refine = nn.Conv2d(self.feat_channels, 4, 3, padding=1)
        self.scales_refine = nn.ModuleList([Scale(1.0) for _ in self.strides])

        self.vfnet_cls_dconv = DeformConv2d(
            self.feat_channels,
            self.feat_channels,
            self.dcn_kernel,
            1,
            padding=self.dcn_pad)
        self.vfnet_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)

    def _apply_color_gate(self, x: Tensor) -> Tensor:
        """Color-aware gate for classification branch only."""
        if not self.use_color_gate:
            return x

        # channel attention
        avg_attn = self.color_mlp(self.color_avgpool(x))
        max_attn = self.color_mlp(self.color_maxpool(x))
        ch_attn = torch.sigmoid(avg_attn + max_attn)

        # residual style: safer than pure multiplication
        out = x * (1.0 + ch_attn)

        # spatial attention
        if self.use_color_spatial_gate:
            avg_map = torch.mean(out, dim=1, keepdim=True)
            max_map, _ = torch.max(out, dim=1, keepdim=True)
            sp_attn = torch.sigmoid(
                self.color_spatial(torch.cat([avg_map, max_map], dim=1)))
            out = out * (1.0 + sp_attn)

        return out

    def forward(self, x: Tuple[Tensor]) -> Tuple[List[Tensor]]:
        """Forward features from the upstream network."""
        return multi_apply(self.forward_single, x, self.scales,
                           self.scales_refine, self.strides, self.reg_denoms)

    def forward_single(self, x: Tensor, scale: Scale, scale_refine: Scale,
                       stride: int, reg_denom: int) -> tuple:
        """Forward features of a single scale level."""
        cls_feat = x
        reg_feat = x

        for layer in self.cls_stem:
            cls_feat = layer(cls_feat)

        for layer in self.reg_stem:
            reg_feat = layer(reg_feat)

        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)

        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)

        # classification-only color-aware gate
        cls_feat = self._apply_color_gate(cls_feat)

        # predict bbox_pred
        reg_feat_init = self.vfnet_reg_conv(reg_feat)
        if self.bbox_norm_type == 'reg_denom':
            bbox_pred = scale(self.vfnet_reg(reg_feat_init)).float().exp() * reg_denom
        elif self.bbox_norm_type == 'stride':
            bbox_pred = scale(self.vfnet_reg(reg_feat_init)).float().exp() * stride
        else:
            raise NotImplementedError

        # decoupled offsets
        dcn_offset_reg = self.star_dcn_offset(
            bbox_pred, self.gradient_mul, stride).to(reg_feat.dtype)

        dcn_offset_cls = self.star_dcn_offset(
            bbox_pred.detach(), 0.0, stride).to(cls_feat.dtype)

        # refine bbox
        reg_feat = self.relu(self.vfnet_reg_refine_dconv(reg_feat, dcn_offset_reg))
        bbox_pred_refine = scale_refine(
            self.vfnet_reg_refine(reg_feat)).float().exp()
        bbox_pred_refine = bbox_pred_refine * bbox_pred.detach()

        # cls score
        cls_feat = self.relu(self.vfnet_cls_dconv(cls_feat, dcn_offset_cls))
        cls_score = self.vfnet_cls(cls_feat)

        if self.training:
            return cls_score, bbox_pred, bbox_pred_refine
        else:
            return cls_score, bbox_pred_refine

    def star_dcn_offset(self, bbox_pred: Tensor, gradient_mul: float,
                        stride: int) -> Tensor:
        """Compute the star deformable conv offsets."""
        dcn_base_offset = self.dcn_base_offset.type_as(bbox_pred)
        bbox_pred_grad_mul = (1 - gradient_mul) * bbox_pred.detach() + \
            gradient_mul * bbox_pred
        bbox_pred_grad_mul = bbox_pred_grad_mul / stride
        N, C, H, W = bbox_pred.size()

        x1 = bbox_pred_grad_mul[:, 0, :, :]
        y1 = bbox_pred_grad_mul[:, 1, :, :]
        x2 = bbox_pred_grad_mul[:, 2, :, :]
        y2 = bbox_pred_grad_mul[:, 3, :, :]
        bbox_pred_grad_mul_offset = bbox_pred.new_zeros(
            N, 2 * self.num_dconv_points, H, W)
        bbox_pred_grad_mul_offset[:, 0, :, :] = -1.0 * y1
        bbox_pred_grad_mul_offset[:, 1, :, :] = -1.0 * x1
        bbox_pred_grad_mul_offset[:, 2, :, :] = -1.0 * y1
        bbox_pred_grad_mul_offset[:, 4, :, :] = -1.0 * y1
        bbox_pred_grad_mul_offset[:, 5, :, :] = x2
        bbox_pred_grad_mul_offset[:, 7, :, :] = -1.0 * x1
        bbox_pred_grad_mul_offset[:, 11, :, :] = x2
        bbox_pred_grad_mul_offset[:, 12, :, :] = y2
        bbox_pred_grad_mul_offset[:, 13, :, :] = -1.0 * x1
        bbox_pred_grad_mul_offset[:, 14, :, :] = y2
        bbox_pred_grad_mul_offset[:, 16, :, :] = y2
        bbox_pred_grad_mul_offset[:, 17, :, :] = x2
        dcn_offset = bbox_pred_grad_mul_offset - dcn_base_offset

        return dcn_offset

    def loss_by_feat(
            self,
            cls_scores: List[Tensor],
            bbox_preds: List[Tensor],
            bbox_preds_refine: List[Tensor],
            batch_gt_instances: InstanceList,
            batch_img_metas: List[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """Compute loss of the head."""
        assert len(cls_scores) == len(bbox_preds) == len(bbox_preds_refine)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.fcos_prior_generator.grid_priors(
            featmap_sizes, bbox_preds[0].dtype, bbox_preds[0].device)
        labels, label_weights, bbox_targets, bbox_weights = self.get_targets(
            cls_scores,
            all_level_points,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore)

        num_imgs = cls_scores[0].size(0)

        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels).contiguous()
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4).contiguous()
            for bbox_pred in bbox_preds
        ]
        flatten_bbox_preds_refine = [
            bbox_pred_refine.permute(0, 2, 3, 1).reshape(-1, 4).contiguous()
            for bbox_pred_refine in bbox_preds_refine
        ]

        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_bbox_preds_refine = torch.cat(flatten_bbox_preds_refine)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)

        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        bg_class_ind = self.num_classes
        pos_inds = torch.where(
            ((flatten_labels >= 0) & (flatten_labels < bg_class_ind)) > 0)[0]
        num_pos = len(pos_inds)

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_bbox_preds_refine = flatten_bbox_preds_refine[pos_inds]
        pos_labels = flatten_labels[pos_inds]

        if self.sync_num_pos:
            num_pos_avg_per_gpu = reduce_mean(
                pos_inds.new_tensor(num_pos).float()).item()
            num_pos_avg_per_gpu = max(num_pos_avg_per_gpu, 1.0)
        else:
            num_pos_avg_per_gpu = num_pos

        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_points = flatten_points[pos_inds]

        pos_decoded_bbox_preds = self.bbox_coder.decode(
            pos_points, pos_bbox_preds)
        pos_decoded_target_preds = self.bbox_coder.decode(
            pos_points, pos_bbox_targets)
        iou_targets_ini = bbox_overlaps(
            pos_decoded_bbox_preds,
            pos_decoded_target_preds.detach(),
            is_aligned=True).clamp(min=1e-6)
        bbox_weights_ini = iou_targets_ini.clone().detach()
        bbox_avg_factor_ini = reduce_mean(
            bbox_weights_ini.sum()).clamp_(min=1).item()

        pos_decoded_bbox_preds_refine = self.bbox_coder.decode(
            pos_points, pos_bbox_preds_refine)
        iou_targets_rf = bbox_overlaps(
            pos_decoded_bbox_preds_refine,
            pos_decoded_target_preds.detach(),
            is_aligned=True).clamp(min=1e-6)
        bbox_weights_rf = iou_targets_rf.clone().detach()
        bbox_avg_factor_rf = reduce_mean(
            bbox_weights_rf.sum()).clamp_(min=1).item()

        if num_pos > 0:
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds.detach(),
                weight=bbox_weights_ini,
                avg_factor=bbox_avg_factor_ini)

            loss_bbox_refine = self.loss_bbox_refine(
                pos_decoded_bbox_preds_refine,
                pos_decoded_target_preds.detach(),
                weight=bbox_weights_rf,
                avg_factor=bbox_avg_factor_rf)

            if self.use_vfl:
                pos_ious = iou_targets_rf.clone().detach()
                cls_iou_targets = torch.zeros_like(flatten_cls_scores)
                cls_iou_targets[pos_inds, pos_labels] = pos_ious
        else:
            loss_bbox = pos_bbox_preds.sum() * 0
            loss_bbox_refine = pos_bbox_preds_refine.sum() * 0
            if self.use_vfl:
                cls_iou_targets = torch.zeros_like(flatten_cls_scores)

        if self.use_vfl:
            loss_cls = self.loss_cls(
                flatten_cls_scores,
                cls_iou_targets,
                avg_factor=num_pos_avg_per_gpu)
        else:
            loss_cls = self.loss_cls(
                flatten_cls_scores,
                flatten_labels,
                weight=label_weights,
                avg_factor=num_pos_avg_per_gpu)

        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_bbox_rf=loss_bbox_refine)

    def get_targets(
            self,
            cls_scores: List[Tensor],
            mlvl_points: List[Tensor],
            batch_gt_instances: InstanceList,
            batch_img_metas: List[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> tuple:
        """A wrapper for computing ATSS and FCOS targets."""
        if self.use_atss:
            return self.get_atss_targets(
                cls_scores,
                mlvl_points,
                batch_gt_instances,
                batch_img_metas,
                batch_gt_instances_ignore)
        else:
            self.norm_on_bbox = False
            return self.get_fcos_targets(mlvl_points, batch_gt_instances)

    def _get_targets_single(self, *args, **kwargs):
        """Avoid ambiguity in multiple inheritance."""
        if self.use_atss:
            return ATSSHead._get_targets_single(self, *args, **kwargs)
        else:
            return FCOSHead._get_targets_single(self, *args, **kwargs)

    def get_fcos_targets(self, points: List[Tensor],
                         batch_gt_instances: InstanceList) -> tuple:
        """Compute FCOS regression and classification targets."""
        labels, bbox_targets = FCOSHead.get_targets(self, points, batch_gt_instances)
        label_weights = None
        bbox_weights = None
        return labels, label_weights, bbox_targets, bbox_weights

    def get_anchors(self,
                    featmap_sizes: List[Tuple],
                    batch_img_metas: List[dict],
                    device: str = 'cuda') -> tuple:
        """Get anchors according to feature map sizes."""
        num_imgs = len(batch_img_metas)

        multi_level_anchors = self.atss_prior_generator.grid_priors(
            featmap_sizes, device=device)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]

        valid_flag_list = []
        for img_id, img_meta in enumerate(batch_img_metas):
            multi_level_flags = self.atss_prior_generator.valid_flags(
                featmap_sizes, img_meta['pad_shape'], device=device)
            valid_flag_list.append(multi_level_flags)

        return anchor_list, valid_flag_list

    def get_atss_targets(
            self,
            cls_scores: List[Tensor],
            mlvl_points: List[Tensor],
            batch_gt_instances: InstanceList,
            batch_img_metas: List[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> tuple:
        """A wrapper for computing ATSS targets for points in multiple images."""
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.atss_prior_generator.num_levels == \
            self.fcos_prior_generator.num_levels

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, batch_img_metas, device=device)

        cls_reg_targets = ATSSHead.get_targets(
            self,
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore,
            unmap_outputs=True)

        (anchor_list, labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, avg_factor) = cls_reg_targets

        bbox_targets_list = [
            bbox_targets.reshape(-1, 4) for bbox_targets in bbox_targets_list
        ]

        num_imgs = len(batch_img_metas)
        bbox_targets_list = self.transform_bbox_targets(
            bbox_targets_list, mlvl_points, num_imgs)

        labels_list = [labels.reshape(-1) for labels in labels_list]
        label_weights_list = [
            label_weights.reshape(-1) for label_weights in label_weights_list
        ]
        bbox_weights_list = [
            bbox_weights.reshape(-1) for bbox_weights in bbox_weights_list
        ]
        label_weights = torch.cat(label_weights_list)
        bbox_weights = torch.cat(bbox_weights_list)
        return labels_list, label_weights, bbox_targets_list, bbox_weights

    def transform_bbox_targets(self, decoded_bboxes: List[Tensor],
                               mlvl_points: List[Tensor],
                               num_imgs: int) -> List[Tensor]:
        """Transform bbox_targets (x1, y1, x2, y2) into (l, t, r, b) format."""
        assert len(decoded_bboxes) == len(mlvl_points)
        num_levels = len(decoded_bboxes)
        mlvl_points = [points.repeat(num_imgs, 1) for points in mlvl_points]
        bbox_targets = []
        for i in range(num_levels):
            bbox_target = self.bbox_coder.encode(mlvl_points[i], decoded_bboxes[i])
            bbox_targets.append(bbox_target)

        return bbox_targets

    def _load_from_state_dict(self, state_dict: dict, prefix: str,
                              local_metadata: dict, strict: bool,
                              missing_keys: Union[List[str], str],
                              unexpected_keys: Union[List[str], str],
                              error_msgs: Union[List[str], str]) -> None:
        """Override the method in the parent class to avoid changing para's name."""
        pass