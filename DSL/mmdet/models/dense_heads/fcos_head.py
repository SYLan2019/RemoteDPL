import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Scale
from mmcv.runner import force_fp32

from mmdet.core import distance2bbox, multi_apply, multiclass_nms, reduce_mean, build_assigner
from ..builder import HEADS, build_loss
from .anchor_free_head import AnchorFreeHead

INF = 1e8


@HEADS.register_module()
class FCOSHead(AnchorFreeHead):
    """Anchor-free head used in `FCOS <https://arxiv.org/abs/1904.01355>`_.

    The FCOS head does not use anchor boxes. Instead bounding boxes are
    predicted at each pixel and a centerness measure is used to suppress
    low-quality predictions.
    Here norm_on_bbox, centerness_on_reg, dcn_on_last_conv are training
    tricks used in official repo, which will bring remarkable mAP gains
    of up to 4.9. Please see https://github.com/tianzhi0549/FCOS for
    more detail.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        strides (list[int] | list[tuple[int, int]]): Strides of points
            in multiple feature levels. Default: (4, 8, 16, 32, 64).
        regress_ranges (tuple[tuple[int, int]]): Regress range of multiple
            level points.
        center_sampling (bool): If true, use center sampling. Default: False.
        center_sample_radius (float): Radius of center sampling. Default: 1.5.
        norm_on_bbox (bool): If true, normalize the regression targets
            with FPN strides. Default: False.
        centerness_on_reg (bool): If true, position centerness on the
            regress branch. Please refer to https://github.com/tianzhi0549/FCOS/issues/89#issuecomment-516877042.
            Default: False.
        conv_bias (bool | str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias of conv will be set as True if `norm_cfg` is None, otherwise
            False. Default: "auto".
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        loss_centerness (dict): Config of centerness loss.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: norm_cfg=dict(type='GN', num_groups=32, requires_grad=True).
        init_cfg (dict or list[dict], optional): Initialization config dict.

    Example:
        >>> self = FCOSHead(11, 7)
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> cls_score, bbox_pred, centerness = self.forward(feats)
        >>> assert len(cls_score) == len(self.scales)
    """  # noqa: E501

    def __init__(self,
                 num_classes,
                 in_channels,
                 # myj 增加richness
                 #richness=False,
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF)),
                 center_sampling=False,
                 center_sample_radius=1.5,
                 norm_on_bbox=False,
                 centerness_on_reg=False,
                 # fpn_layer = 'p2',
                 # fraction = 1/2,
                 # for unlabel loss weight
                 loss_weight = 1.0,
                 soft_weight = 0.0,
                 soft_warm_up = 0,
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(type='IoULoss', loss_weight=1.0),
                 loss_centerness=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='conv_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 # assigner = None,
                 **kwargs):
        # myj 增加richness
        #self.richness_toggle = richness
        # self.fpn_layer = fpn_layer
        # self.fraction = fraction
        self.regress_ranges = regress_ranges
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        self.norm_on_bbox = norm_on_bbox
        self.centerness_on_reg = centerness_on_reg
        self.loss_weight = loss_weight
        self.soft_weight = soft_weight
        self.soft_warm_up = soft_warm_up
        self.cur_iter = 0
        super().__init__(
            num_classes,
            in_channels,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            norm_cfg=norm_cfg,
            init_cfg=init_cfg,
            **kwargs)
        #myj进行修改
        #self.down_strides = [stride * 2 for stride in self.strides]
        self.loss_centerness = build_loss(loss_centerness)
        # self.down_assigner = build_assigner(assigner)
        # #assign不应该为空
        # self.lp2 = nn.Parameter(torch.tensor(0.6, dtype=torch.float))
        # self.lp3 = nn.Parameter(torch.tensor(0.6, dtype=torch.float))
        # self.lp4 = nn.Parameter(torch.tensor(0.6, dtype=torch.float))
        # self.lp5 = nn.Parameter(torch.tensor(0.6, dtype=torch.float))
        # self.lp6 = nn.Parameter(torch.tensor(0.6, dtype=torch.float))
        # self.lp7 = nn.Parameter(torch.tensor(0.6, dtype=torch.float))

    def _init_layers(self):
        """Initialize layers of the head."""
        super()._init_layers()
        self.conv_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])
        #myj的修改，增加一个头，这个头专门计算richness的值，类比centerness
        #if self.richness_toggle:
        # self.conv_richness = nn.Sequential(
        #     nn.Conv2d(self.in_channels, self.feat_channels, 3, padding=1),
        #     nn.Conv2d(self.feat_channels, self.feat_channels, 3, padding=1),
        #     nn.Conv2d(self.feat_channels, self.feat_channels, 3, padding=1),
        #     nn.Conv2d(self.feat_channels, self.feat_channels, 3, padding=1),
        #     #这个是预测头，预测的是richness的值
        #     nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        # )

    # 用于可视化
    def vision(self, feats):
        return multi_apply(self.vision_single, feats, self.scales,
                           self.strides)[0]

    def vision_single(self, feat, scale, stride):
        cls_score, bbox_pred, cls_feat, reg_feat = super().forward_single(feat)

        return reg_feat


    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple:
                cls_scores (list[Tensor]): Box scores for each scale level, \
                    each is a 4D-tensor, the channel number is \
                    num_points * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for each \
                    scale level, each is a 4D-tensor, the channel number is \
                    num_points * 4.
                centernesses (list[Tensor]): centerness for each scale level, \
                    each is a 4D-tensor, the channel number is num_points * 1.
        """
        # if down_train:
        #     #print("OOOOOOOOOOOOOOOOOOOOKKKKKKKKKKKKKKKKKKKK\n")
        #     return multi_apply(self.forward_single, feats, self.scales,
        #                        self.down_strides)
        return multi_apply(self.forward_single, feats, self.scales,
                           self.strides)

    def forward_single(self, x, scale, stride):
        """Forward features of a single scale level.

        Args:
            x (Tensor): FPN feature maps of the specified stride.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.
            stride (int): The corresponding stride for feature maps, only
                used to normalize the bbox prediction when self.norm_on_bbox
                is True.

        Returns:
            tuple: scores for each class, bbox predictions and centerness \
                predictions of input feature maps.
        """
        #需要注意下采样的stride
        cls_score, bbox_pred, cls_feat, reg_feat = super().forward_single(x)
        #myj的修改，增加一个头，这个头专门计算richness的值，类比centerness
        #if self.richness_toggle:
        # richness_score = self.conv_richness(x)#tenorsize为（2,1,h,w）
        #丰富度得分为正值
        # richness_score = F.relu(richness_score)
        if self.centerness_on_reg:
            centerness = self.conv_centerness(reg_feat)
        else:
            centerness = self.conv_centerness(cls_feat)
        # scale the bbox_pred of different level
        # float to avoid overflow when enabling FP16
        bbox_pred = scale(bbox_pred).float()
        if self.norm_on_bbox:
            bbox_pred = F.relu(bbox_pred)
            if not self.training:
                bbox_pred *= stride
        else:
            bbox_pred = bbox_pred.exp()

        #if self.richness_toggle:
        return cls_score, bbox_pred, centerness#, richness_score

    #针对小目标进行损失计算
    def forward_TO(self,
                  x,
                  img_metas,
                  gt_bboxes,
                  gt_labels=None,
                  gt_bboxes_ignore=None,
                  proposal_cfg=None,
                  **kwargs):
        #进行修改，需要设计小目标损失计算。
        outs = self(x)
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)

        losses = self.loss_to(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(*outs, img_metas, cfg=proposal_cfg)
            return losses, proposal_list
    #产生理论感受野
    def gen_trf(self):
        '''
        Calculate the theoretical receptive field from P2-p7 of a standard ResNet-50-FPN.
        # ref: https://distill.pub/2019/computing-receptive-fields/
        '''

        j_i = [1]
        for i in range(7):
            j = j_i[i] * 2
            j_i.append(j)

        # j_i = [1,2,4,8,16,32,64,128]
        r0 = 1
        r1 = r0 + (7 - 1) * j_i[0]
        #对下面的所有3-1改成可学习的参数
        r2 = r1 + (3 - 1) * j_i[1]
        trf_p2 = r2 + 2 * j_i[2] * 3

        r3 = trf_p2 + (3 - 1) * j_i[2]
        trf_p3 = r3 + 2 * j_i[3] * 3

        r4 = trf_p3 + (3 - 1) * j_i[3]
        trf_p4 = r4 + 2 * j_i[4] * 5

        r5 = trf_p4 + (3 - 1) * j_i[4]
        trf_p5 = r5 + 2 * j_i[5] * 2

        trf_p6 = trf_p5 + 2 * j_i[6]

        trf_p7 = trf_p6 + 2 * j_i[7]
        # trfs = [35, 91, 267, 427, 555, 811]
        trfs = [trf_p2, trf_p3, trf_p4, trf_p5, trf_p6, trf_p7]

        return trfs

    def get_targets_to(self, points, gt_bboxes_list, gt_labels_list):
        """Compute regression, classification and centerness targets for points
        in multiple images.

        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points, 2).
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels_list (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).

        Returns:
            tuple:
                concat_lvl_labels (list[Tensor]): Labels of each level. \
                concat_lvl_bbox_targets (list[Tensor]): BBox targets of each \
                    level.
        """
        assert len(points) == len(self.regress_ranges)

        # convert points to rf according to its layer
        rfields = []
        trfs = self.gen_trf()  # trfs = [35, 91, 267, 427, 555, 811] p2~p7
        for num in range(len(points)):
            rfield = []
            if self.fpn_layer == 'p3':
                rfnum = num + 1
            else:
                rfnum = num
            # fraction = 1/2,
            if rfnum == 0:
                # rf = trfs[0] * self.fraction
                rf = trfs[0] * self.lp2
            elif rfnum == 1:
                # rf = trfs[1] * self.fraction
                rf = trfs[1] * self.lp3
            elif rfnum == 2:
                # rf = trfs[2] * self.fraction
                rf = trfs[2] * self.lp4
            elif rfnum == 3:
                # rf = trfs[3] * self.fraction
                rf = trfs[3] * self.lp5
            elif rfnum == 4:
                # rf = trfs[4] * self.fraction
                rf = trfs[4] * self.lp6
            else:
                # rf = trfs[5] * self.fraction
                rf = trfs[5] * self.lp7

            point = points[num]
            px1 = point[..., 0] - rf / 2
            py1 = point[..., 1] - rf / 2
            px2 = point[..., 0] + rf / 2
            py2 = point[..., 1] + rf / 2
            rfield = torch.cat((px1[..., None], py1[..., None]), dim=1)
            rfield = torch.cat((rfield, px2[..., None]), dim=1)
            rfield = torch.cat((rfield, py2[..., None]), dim=1)
            # rfield  =  tensor(点的个数,4)
            rfields.append(rfield)
        rfields = torch.cat(rfields, dim=0)  # tensor(所有特征层点的个数, 4)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(  # [[-1,64],[-1,64]] num*2*2
                points[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)  # tenor(总特征点个数，2)
        concat_points = torch.cat(points, dim=0)  # tensor(特征点个数,2)

        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]

        # get labels and bbox_targets of each image
        labels_list, bbox_targets_list, gt_centers_list = multi_apply(
            self._get_target_single_to,
            gt_bboxes_list,
            gt_labels_list,
            points=concat_points,
            rfields=rfields,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points)

        # split to per img, per level
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]
        #增加对于gt_box的中心点的处理
        gt_centers_list = [gt_centers.split(num_points, 0) for gt_centers in gt_centers_list]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        concat_lvl_gt_centers = []
        for i in range(num_levels):
            #增加
            concat_lvl_gt_centers.append(torch.cat([gt_centers[i] for gt_centers in gt_centers_list]))
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            bbox_targets = torch.cat(
                [bbox_targets[i] for bbox_targets in bbox_targets_list])
            if self.norm_on_bbox:
                bbox_targets = bbox_targets / self.strides[i]
            concat_lvl_bbox_targets.append(bbox_targets)
        return concat_lvl_labels, concat_lvl_bbox_targets, concat_lvl_gt_centers

    def _get_target_single_to(self, gt_bboxes, gt_labels, points, rfields, regress_ranges,
                           num_points_per_lvl):
        """Compute regression and classification targets for a single image."""
        num_points = points.size(0)
        num_gts = gt_labels.size(0)  # 一张图片真实框的个数=====
        gt_ori = gt_bboxes
        if num_gts == 0:
            #myj 这里要进行修改
            return gt_labels.new_full((num_points,), self.num_classes), \
                gt_bboxes.new_zeros((num_points, 4)), gt_bboxes.new_zeros((num_points, 2))

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
                gt_bboxes[:, 3] - gt_bboxes[:, 1])
        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1)  # points, gts
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        left = xs - gt_bboxes[..., 0]  # numpoints, num_gt
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)  # numpoints, num_gt, 4

        assign_result = self.down_assigner.assign(rfields, gt_ori,
                                             gt_bboxes_ignore=None)  # assign_result.gt_inds = tensor(总特征点个数，)
        inds = assign_result.gt_inds  # num_points
        inds = inds - 1

        inds_mask = inds[..., None].repeat(1, num_gts)  # num_points, num_gts
        point_mask = torch.arange(num_gts).repeat(num_points, 1).cuda()  # num_points, num_gts
        assigned_mask = (inds_mask == point_mask)
        areas[assigned_mask == False] = INF
        min_area, min_area_inds = areas.min(dim=1)

        #增加真实框的中心点坐标
        dot_gt_box = gt_ori[min_area_inds]
        #返回的类型为tensor(总特征点个数，2)
        gt_center = torch.stack(((dot_gt_box[:, 0] + dot_gt_box[:, 2]) / 2, (dot_gt_box[:, 1] + dot_gt_box[:, 3]) / 2), dim=-1)

        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = self.num_classes  # set as BG
        bbox_targets = bbox_targets[range(num_points), min_area_inds]
        #myj进行修改，增加了高斯分布的目标回归值
        return labels, bbox_targets, gt_center


    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def loss_to(self,
             cls_scores,
             bbox_preds,
             centernesses,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute loss of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * 4.
            centernesses (list[Tensor]): centerness for each scale level, each
                is a 4D-tensor, the channel number is num_points * 1.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert len(cls_scores) == len(bbox_preds) == len(centernesses)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                           bbox_preds[0].device)
        labels, bbox_targets, gt_centers = self.get_targets_to(all_level_points, gt_bboxes,
                                                gt_labels)

        ig_labels = None
        if gt_bboxes_ignore is not None:
            gt_labels_ignore = []
            for i in range(len(gt_bboxes_ignore)):
                _size = gt_bboxes_ignore[i].size(0)
                gt_labels_ignore.append(
                    torch.zeros([_size], device=gt_bboxes_ignore[i].device, dtype=torch.int64) + self.num_classes - 1)
            ig_labels, _, _ = self.get_targets_to(all_level_points, gt_bboxes_ignore, gt_labels_ignore)
        #添加未标记标签的计算
        batch_sizes = len(img_metas)
        flatten_As_labels = None
        if self.loss_weight != 1.0:
            flatten_As_labels = []
            label_weights = 1
            unlabel_weights = self.loss_weight
            for i in range(len(ig_labels)):
                flatten_As_labels.append(torch.ones_like(ig_labels[i], device=ig_labels[i].device, dtype=torch.float32))
                # support 1:1 batch
                if batch_sizes % 2 == 0:
                    flatten_As_labels[i][:int(len(flatten_As_labels[i]) / 2)] *= label_weights
                    flatten_As_labels[i][int(len(flatten_As_labels[i]) / 2):] *= unlabel_weights
                # support 1:1 with extra one scale invariant input
                else:
                    flatten_As_labels[i][
                    :int(len(flatten_As_labels[i]) / batch_sizes * (batch_sizes - 1) / 2)] *= label_weights
                    flatten_As_labels[i][
                    int(len(flatten_As_labels[i]) / batch_sizes * (batch_sizes - 1) / 2):] *= unlabel_weights
            flatten_As_labels = torch.cat(flatten_As_labels)

        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds and centerness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        #myj
        flatten_gt_centers = torch.cat(gt_centers)
        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = torch.tensor(
            len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)
        num_pos = max(reduce_mean(num_pos), 1.0)


        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]
        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        #myj这里修改为高斯距离
        #得到真实框中心点坐标
        pos_centerness_targets = self.centerness_target_to(flatten_points[pos_inds], flatten_gt_centers[pos_inds])
        # centerness weighted iou loss
        centerness_denorm = max(
            reduce_mean(pos_centerness_targets.sum().detach()), 1e-6)
        #修改为正样本点的数量
        #centerness_denorm = num_pos

        if len(pos_inds) > 0:
            pos_points = flatten_points[pos_inds]
            pos_decoded_bbox_preds = distance2bbox(pos_points, pos_bbox_preds)
            pos_decoded_target_preds = distance2bbox(pos_points,
                                                     pos_bbox_targets)

            # for unlabel weights
            flatten_weights = torch.ones_like(pos_centerness_targets, device=pos_centerness_targets.device)
            if flatten_As_labels is not None:
                        pos_As_labels = flatten_As_labels[pos_inds]
                        flatten_weights = flatten_weights * pos_As_labels
            #将权重改为不需要pos_centerness_targets，在小样本检测过程中，不需要中心度损失
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                #weight=flatten_weights,
                weight=pos_centerness_targets * flatten_weights,
                avg_factor=centerness_denorm)
            loss_centerness = self.loss_centerness(
                pos_centerness, pos_centerness_targets, weight=flatten_weights, avg_factor=num_pos)
            # myj进行修改，看denorm的值
            # print('loss_bbox种centerness_denorm的值为：', centerness_denorm)
            # print('现在看centerness中的变量值：')
            # print('pos_centerness的值为:', pos_centerness)
            # print('pos_centerness_targets的值为：', pos_centerness_targets)
            # print('num_pos的值为：', num_pos)
            # import pdb;
            # pdb.set_trace()
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_centerness = pos_centerness.sum()

        weight = torch.ones_like(flatten_labels, dtype=torch.float32, device=flatten_cls_scores.device)
        if ig_labels is not None:
                flatten_ig_labels = torch.cat(ig_labels)
                # find the inter set between ig_labels and labels, and do not ignore these points
                inter_inds = ((flatten_ig_labels-self.num_classes)*(flatten_labels-self.num_classes)).nonzero().reshape(-1)
                if inter_inds.size(0) >0:
                        flatten_ig_labels[inter_inds] = self.num_classes
                weight = flatten_ig_labels.float() - self.num_classes + 1

        if flatten_As_labels is not None:
                weight = weight * flatten_As_labels
        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_labels, weight=weight, avg_factor=num_pos)

        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_centerness=loss_centerness)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def loss(self,
             cls_scores,
             bbox_preds,
             centernesses,
             #richness_scores,#进行修改，增加丰富度的预测值（list[Tensor]）
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute loss of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * 4.
            centernesses (list[Tensor]): centerness for each scale level, each
                is a 4D-tensor, the channel number is num_points * 1.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert len(cls_scores) == len(bbox_preds) == len(centernesses)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        #all_level_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                          # bbox_preds[0].device)
        # if not down_train:
        all_level_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                           bbox_preds[0].device)
        #需要修改target使得能够返回一个richnes参数，这个参数表示每个特征点所在gtboxes的数目
        labels, bbox_targets = self.get_targets(all_level_points, gt_bboxes,
                                            gt_labels)
        # else:
        #     all_level_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
        #                                         bbox_preds[0].device, down_train = True)
        #     labels, bbox_targets = self.get_targets(all_level_points, gt_bboxes,
        #                                         gt_labels, down_train = True)
        # for ignore labels
        ig_labels = None
        if gt_bboxes_ignore is not None:
                gt_labels_ignore = []
                for i in range(len(gt_bboxes_ignore)):
                        _size = gt_bboxes_ignore[i].size(0)
                        gt_labels_ignore.append(torch.zeros([_size], device=gt_bboxes_ignore[i].device, dtype=torch.int64) + self.num_classes - 1)
                ig_labels, _ = self.get_targets(all_level_points, gt_bboxes_ignore, gt_labels_ignore)

        # bhchen 08/09/2021 add loss weights for unlabel training via label; loss weight is for unlabeled data. labeled data is set default to 1.
        batch_sizes = len(img_metas)
        flatten_As_labels = None
        if self.loss_weight != 1.0:
                flatten_As_labels = []
                label_weights = 1
                unlabel_weights = self.loss_weight
                for i in range(len(ig_labels)):
                        flatten_As_labels.append(torch.ones_like(ig_labels[i], device=ig_labels[i].device, dtype=torch.float32))
                        # support 1:1 batch
                        if batch_sizes%2 == 0:
                                flatten_As_labels[i][:int(len(flatten_As_labels[i])/2)] *=label_weights 
                                flatten_As_labels[i][int(len(flatten_As_labels[i])/2):] *=unlabel_weights
                        # support 1:1 with extra one scale invariant input 
                        else:
                                flatten_As_labels[i][:int(len(flatten_As_labels[i])/batch_sizes*(batch_sizes-1)/2)] *=label_weights 
                                flatten_As_labels[i][int(len(flatten_As_labels[i])/batch_sizes*(batch_sizes-1)/2):] *=unlabel_weights
                flatten_As_labels = torch.cat(flatten_As_labels)
        # Done

        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds and centerness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]
        #对每一个richness_score进行flatten，类似于centerness
        # flatten_richness = [
        #     richness_score.permute(0, 2, 3, 1).reshape(-1)
        #     for richness_score in richness_scores
        # ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        #依旧类似centerness,依旧是预测值部分
        #flatten_richness = torch.cat(flatten_richness)

        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        #flatten_richness_targets = torch.cat(richness_targets)

        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        #得到前景的特征点的索引，nonzero返回是的是二维张量
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = torch.tensor(
            len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)
        num_pos = max(reduce_mean(num_pos), 1.0)

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]
        #myj 正样本的丰富度预测得分
        #pos_richness = flatten_richness[pos_inds]

        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_centerness_targets = self.centerness_target(pos_bbox_targets)
        #pos_richness_targets = flatten_richness_targets[pos_inds]

        # centerness weighted iou loss
        centerness_denorm = max(
            reduce_mean(pos_centerness_targets.sum().detach()), 1e-6)

        if len(pos_inds) > 0:
            pos_points = flatten_points[pos_inds]
            pos_decoded_bbox_preds = distance2bbox(pos_points, pos_bbox_preds)
            pos_decoded_target_preds = distance2bbox(pos_points,
                                                     pos_bbox_targets)
            # for unlabel weights
            flatten_weights = torch.ones_like(pos_centerness_targets, device=pos_centerness_targets.device)
            if flatten_As_labels is not None:
                        pos_As_labels = flatten_As_labels[pos_inds]
                        flatten_weights = flatten_weights * pos_As_labels
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                weight=pos_centerness_targets * flatten_weights,
                avg_factor=centerness_denorm)
            loss_centerness = self.loss_centerness(
                pos_centerness, pos_centerness_targets, weight=flatten_weights, avg_factor=num_pos)
            #myj 丰富损失计算，参照centerness
            # loss_richness = self.loss_centerness(
            #     pos_richness, pos_richness_targets, weight=flatten_weights, avg_factor=num_pos)
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_centerness = pos_centerness.sum()
            #loss_richness = pos_richness.sum()

        weight = torch.ones_like(flatten_labels, dtype=torch.float32, device=flatten_cls_scores.device)
        if ig_labels is not None:
                flatten_ig_labels = torch.cat(ig_labels)
                # find the inter set between ig_labels and labels, and do not ignore these points
                inter_inds = ((flatten_ig_labels-self.num_classes)*(flatten_labels-self.num_classes)).nonzero().reshape(-1)
                if inter_inds.size(0) >0:
                        flatten_ig_labels[inter_inds] = self.num_classes
                weight = flatten_ig_labels.float() - self.num_classes + 1

        # bhchen 08/10/2021 add for label&unlabel loss weights
        if flatten_As_labels is not None:
                weight = weight * flatten_As_labels

        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_labels, weight = weight, avg_factor=num_pos)

        # bhchen add for scale invariant with soft label
        # 下面的只执行else里面的程序，注意bach_size为偶数
        loss_sisoft = 0.0
        if batch_sizes%2 !=0 and self.soft_weight!=0.0:
            for i in range(1, len(cls_scores)):
                n_pre, c_pre, h_pre, w_pre = cls_scores[i-1].shape
                n, c, h, w = cls_scores[i].shape
                # L2 reg
                loss_sisoft = loss_sisoft + ((cls_scores[i][int(batch_sizes-1-1),:,:,:] - cls_scores[i-1][int(batch_sizes-1),:,:h,:w]) * (cls_scores[i][int(batch_sizes-1-1),:,:,:] - cls_scores[i-1][int(batch_sizes-1),:,:h,:w])).mean()
                # KL reg
                #loss_sisoft = loss_sisoft + F.kl_div(cls_scores[i][int(batch_sizes-1-1),:,:,:].permute(1,2,0).softmax(dim=-1).log(), cls_scores[i-1][int(batch_sizes-1),:,:h,:w].permute(1,2,0).softmax(dim=-1), reduction='sum')
                #loss_sisoft = loss_sisoft + F.kl_div(cls_scores[i-1][int(batch_sizes-1),:,:h,:w].permute(1,2,0).softmax(dim=-1).log(), cls_scores[i][int(batch_sizes-1-1),:,:,:].permute(1,2,0).softmax(dim=-1), reduction='sum')
                #loss_sisoft = loss_sisoft/(h*w)
            soft_weight = self.soft_weight * 1.0
            if self.soft_warm_up >= self.cur_iter:
                    self.cur_iter +=1
                    soft_weight = self.soft_weight/1000.0
            loss_sisoft*=soft_weight

            return dict(
                loss_cls=loss_cls,
                loss_bbox=loss_bbox,
                loss_centerness=loss_centerness,
                loss_sisoft=loss_sisoft)
        else: 
            return dict(
                loss_cls=loss_cls,
                loss_bbox=loss_bbox,
                loss_centerness=loss_centerness
                #loss_richness=loss_richness
            )

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   centernesses,
                   #richness_scores,#进行修改，需要对丰富度得分进行处理
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        """Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                with shape (N, num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_points * 4, H, W).
            centernesses (list[Tensor]): Centerness for each scale level with
                shape (N, num_points * 1, H, W).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used. Default: None.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where 5 represent
                (tl_x, tl_y, br_x, br_y, score) and the score between 0 and 1.
                The shape of the second tensor in the tuple is (n,), and
                each element represents the class label of the corresponding
                box.
        """
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                      bbox_preds[0].device)

        cls_score_list = [cls_scores[i].detach() for i in range(num_levels)]
        bbox_pred_list = [bbox_preds[i].detach() for i in range(num_levels)]
        centerness_pred_list = [
            centernesses[i].detach() for i in range(num_levels)
        ]
        # richness_pred_list = [
        #     richness_scores[i].detach() for i in range(num_levels)
        # ]
        if torch.onnx.is_in_onnx_export():
            assert len(
                img_metas
            ) == 1, 'Only support one input image while in exporting to ONNX'
            img_shapes = img_metas[0]['img_shape_for_onnx']
        else:
            img_shapes = [
                img_metas[i]['img_shape']
                for i in range(cls_scores[0].shape[0])
            ]
        scale_factors = [
            img_metas[i]['scale_factor'] for i in range(cls_scores[0].shape[0])
        ]
        result_list = self._get_bboxes(cls_score_list, bbox_pred_list,
                                       centerness_pred_list, mlvl_points,
                                       img_shapes, scale_factors, cfg, rescale,
                                       with_nms)
        return result_list

    def _get_bboxes(self,
                    cls_scores,
                    bbox_preds,
                    centernesses,
                    #richness_scores,
                    mlvl_points,
                    img_shapes,
                    scale_factors,
                    cfg,#None
                    rescale=False,#True
                    with_nms=True):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for a single scale level
                with shape (N, num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for a single scale
                level with shape (N, num_points * 4, H, W).
            centernesses (list[Tensor]): Centerness for a single scale level
                with shape (N, num_points, H, W).
            mlvl_points (list[Tensor]): Box reference for a single scale level
                with shape (num_total_points, 4).
            img_shapes (list[tuple[int]]): Shape of the input image,
                list[(height, width, 3)].
            scale_factors (list[ndarray]): Scale factor of the image arrange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            tuple(Tensor):
                det_bboxes (Tensor): BBox predictions in shape (n, 5), where
                    the first 4 columns are bounding box positions
                    (tl_x, tl_y, br_x, br_y) and the 5-th column is a score
                    between 0 and 1.
                det_labels (Tensor): A (n,) tensor where each item is the
                    predicted class label of the corresponding box.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        device = cls_scores[0].device
        batch_size = cls_scores[0].shape[0]
        #myj 增加类别个数
        num_classes = cls_scores[0].shape[1]
        # convert to tensor to keep tracing
        nms_pre_tensor = torch.tensor(#1000
            cfg.get('nms_pre', -1), device=device, dtype=torch.long)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_centerness = []
        #和centerness一样
        #mlvl_richness = []
        for cls_score, bbox_pred, centerness, points in zip(
                cls_scores, bbox_preds, centernesses, mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(0, 2, 3, 1).reshape(
                batch_size, -1, self.cls_out_channels).sigmoid()
            centerness = centerness.permute(0, 2, 3,
                                            1).reshape(batch_size,
                                                       -1).sigmoid()
            #richness = richness.permute(0, 2, 3, 1).reshape(batch_size, -1).sigmoid()
            bbox_pred = bbox_pred.permute(0, 2, 3,
                                          1).reshape(batch_size, -1, 4)
            points = points.expand(batch_size, -1, 2)
            # Get top-k prediction
            from mmdet.core.export import get_k_for_topk
            nms_pre = get_k_for_topk(nms_pre_tensor, bbox_pred.shape[1])
            if nms_pre > 0:
                max_scores, _ = (scores * centerness[..., None]).max(-1)
                _, topk_inds = max_scores.topk(nms_pre)
                batch_inds = torch.arange(batch_size).view(
                    -1, 1).expand_as(topk_inds).long()
                # Avoid onnx2tensorrt issue in https://github.com/NVIDIA/TensorRT/issues/1134 # noqa: E501
                if torch.onnx.is_in_onnx_export():
                    transformed_inds = bbox_pred.shape[
                        1] * batch_inds + topk_inds
                    points = points.reshape(-1,
                                            2)[transformed_inds, :].reshape(
                                                batch_size, -1, 2)
                    bbox_pred = bbox_pred.reshape(
                        -1, 4)[transformed_inds, :].reshape(batch_size, -1, 4)
                    scores = scores.reshape(
                        -1, self.num_classes)[transformed_inds, :].reshape(
                            batch_size, -1, self.num_classes)
                    centerness = centerness.reshape(
                        -1, 1)[transformed_inds].reshape(batch_size, -1)
                    # richness = richness.reshape(
                    #     -1, 1)[transformed_inds].reshape(batch_size, -1)
                else:
                    points = points[batch_inds, topk_inds, :]
                    bbox_pred = bbox_pred[batch_inds, topk_inds, :]
                    scores = scores[batch_inds, topk_inds, :]
                    centerness = centerness[batch_inds, topk_inds]
                    #richness = richness[batch_inds, topk_inds]
            #根据score和centerness乘积的大小进行筛选特征点
            bboxes = distance2bbox(points, bbox_pred, max_shape=img_shapes)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_centerness.append(centerness)
            #mlvl_richness.append(richness)
        #（1，五个尺寸的总的特征点数，4）
        batch_mlvl_bboxes = torch.cat(mlvl_bboxes, dim=1)
        # 返回原来的图片尺寸，因为原图再进行训练的时候会进行resize
        if rescale:
            batch_mlvl_bboxes /= batch_mlvl_bboxes.new_tensor(
                np.array(scale_factors)).unsqueeze(1)
        batch_mlvl_scores = torch.cat(mlvl_scores, dim=1)
        batch_mlvl_centerness = torch.cat(mlvl_centerness, dim=1)
        #batch_mlvl_richness = torch.cat(mlvl_richness, dim=1)

        # Replace multiclass_nms with ONNX::NonMaxSuppression in deployment
        if torch.onnx.is_in_onnx_export() and with_nms:
            from mmdet.core.export import add_dummy_nms_for_onnx
            batch_mlvl_scores = batch_mlvl_scores * (
                batch_mlvl_centerness.unsqueeze(2))
            max_output_boxes_per_class = cfg.nms.get(
                'max_output_boxes_per_class', 200)
            iou_threshold = cfg.nms.get('iou_threshold', 0.5)
            score_threshold = cfg.score_thr
            nms_pre = cfg.get('deploy_nms_pre', -1)
            return add_dummy_nms_for_onnx(batch_mlvl_bboxes, batch_mlvl_scores,
                                          max_output_boxes_per_class,
                                          iou_threshold, score_threshold,
                                          nms_pre, cfg.max_per_img)
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        padding = batch_mlvl_scores.new_zeros(batch_size,
                                              batch_mlvl_scores.shape[1], 1)
        batch_mlvl_scores = torch.cat([batch_mlvl_scores, padding], dim=-1)

        if with_nms:
            det_results = []
            for (mlvl_bboxes, mlvl_scores,
                 mlvl_centerness) in zip(batch_mlvl_bboxes, batch_mlvl_scores,
                                         batch_mlvl_centerness):
                det_bbox, det_label, det_inds = multiclass_nms(
                    mlvl_bboxes,
                    mlvl_scores,
                    cfg.score_thr,
                    cfg.nms,
                    cfg.max_per_img,
                    score_factors=mlvl_centerness,
                    return_inds=True)#需要返回索引，从而得到每个框相对应生成的richness值
                #richness_det = mlvl_richness.view(-1, 1).expand(mlvl_scores.size(0), num_classes)
                #richness_det = richness_det.reshape(-1)
                #det_richness = richness_det[det_inds]
                #myj 进行修改,需要将richness加入到det_bbox中
                #det_bbox = torch.cat([det_bbox, det_richness[:, None]], -1)
                det_results.append(tuple([det_bbox, det_label]))
        else:
            det_results = [
                tuple(mlvl_bs)
                for mlvl_bs in zip(batch_mlvl_bboxes, batch_mlvl_scores,
                                   batch_mlvl_centerness)
            ]
        return det_results

    def _get_points_single(self,
                           featmap_size,
                           stride,
                           dtype,
                           device,
                           flatten=False):
        """Get points according to feature map sizes."""
        y, x = super()._get_points_single(featmap_size, stride, dtype, device)
        points = torch.stack((x.reshape(-1) * stride, y.reshape(-1) * stride),
                             dim=-1) + stride // 2
        return points

    def get_targets(self, points, gt_bboxes_list, gt_labels_list):
        """Compute regression, classification and centerness targets for points
        in multiple images.

        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points, 2).
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels_list (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).

        Returns:
            tuple:
                concat_lvl_labels (list[Tensor]): Labels of each level. \
                concat_lvl_bbox_targets (list[Tensor]): BBox targets of each \
                    level.
        """
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)

        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]

        # get labels and bbox_targets of each image
        labels_list, bbox_targets_list = multi_apply(
            self._get_target_single,
            gt_bboxes_list,
            gt_labels_list,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points)

        # split to per img, per level
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]
        # richness_targets_list = [
        #     richness_targets.split(num_points, 0)
        #     for richness_targets in richness_targets_list
        # ]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        #concat_lvl_richness_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            bbox_targets = torch.cat(
                [bbox_targets[i] for bbox_targets in bbox_targets_list])
            # concat_lvl_richness_targets.append(
            #     torch.cat([richness_targets[i] for richness_targets in richness_targets_list]))
            if self.norm_on_bbox:
                # if not down_train:
                bbox_targets = bbox_targets / self.strides[i]
                # else:
                #     bbox_targets = bbox_targets / self.down_strides[i]
            concat_lvl_bbox_targets.append(bbox_targets)
        return concat_lvl_labels, concat_lvl_bbox_targets#, concat_lvl_richness_targets

    def _get_target_single(self, gt_bboxes, gt_labels, points, regress_ranges,
                           num_points_per_lvl):
        """Compute regression and classification targets for a single image."""
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.num_classes), \
                   gt_bboxes.new_zeros((num_points, 4))

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
            gt_bboxes[:, 3] - gt_bboxes[:, 1])
        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        if self.center_sampling:
            # condition1: inside a `center bbox`
            radius = self.center_sample_radius
            center_xs = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) / 2
            center_ys = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) / 2
            center_gts = torch.zeros_like(gt_bboxes)
            stride = center_xs.new_zeros(center_xs.shape)

            # project the points on current lvl back to the `original` sizes
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                lvl_begin = lvl_end

            x_mins = center_xs - stride
            y_mins = center_ys - stride
            x_maxs = center_xs + stride
            y_maxs = center_ys + stride
            center_gts[..., 0] = torch.where(x_mins > gt_bboxes[..., 0],
                                             x_mins, gt_bboxes[..., 0])
            center_gts[..., 1] = torch.where(y_mins > gt_bboxes[..., 1],
                                             y_mins, gt_bboxes[..., 1])
            center_gts[..., 2] = torch.where(x_maxs > gt_bboxes[..., 2],
                                             gt_bboxes[..., 2], x_maxs)
            center_gts[..., 3] = torch.where(y_maxs > gt_bboxes[..., 3],
                                             gt_bboxes[..., 3], y_maxs)

            cb_dist_left = xs - center_gts[..., 0]
            cb_dist_right = center_gts[..., 2] - xs
            cb_dist_top = ys - center_gts[..., 1]
            cb_dist_bottom = center_gts[..., 3] - ys
            center_bbox = torch.stack(
                (cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom), -1)
            inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        else:
            # condition1: inside a gt bbox
            inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            (max_regress_distance >= regress_ranges[..., 0])
            & (max_regress_distance <= regress_ranges[..., 1]))

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)

        #myj 对areas每个每个特征点所在行的非INF数量进行统计。
        #richness_target = torch.sum(areas != INF, dim=1)
        #对rihness进行归一化，需要探头richness.max为0的情况
        # if richness_target.max() != 0:
        #     richness_target = richness_target / richness_target.max()

        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = self.num_classes  # set as BG
        bbox_targets = bbox_targets[range(num_points), min_area_inds]

        return labels, bbox_targets#, richness_target

    def centerness_target(self, pos_bbox_targets):
        """Compute centerness targets.

        Args:
            pos_bbox_targets (Tensor): BBox targets of positive bboxes in shape
                (num_pos, 4)

        Returns:
            Tensor: Centerness target.
        """
        # only calculate pos centerness targets, otherwise there may be nan
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]
        if len(left_right) == 0:
            centerness_targets = left_right[..., 0]
        else:
            centerness_targets = (
                left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
                    top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness_targets)
#需要修改centerness的目标回归值，对于负数的情况进行处理，需要对于值加入绝对值,这里目前是采用原本来的做法，使用了跃迁函数和使用了防止梯度爆炸的小因子，现在进行更改。
    #使用高斯函数修改中心度损失，要求特征点和真实框的中心点距离重合的时候，中心度值为1
    # def centerness_target_to(self, pos_bbox_targets):
    #     left_right = pos_bbox_targets[:, [0, 2]]
    #     top_bottom = pos_bbox_targets[:, [1, 3]]
    #     if len(left_right) == 0:
    #         centerness_targets = left_right[..., 0]
    #     else:
    #         # centerness_targets = (
    #         #      left_right.min(dim=-1)[0].clamp(min=0.01) / left_right.max(dim=-1)[0]) * (
    #         #      top_bottom.min(dim=-1)[0].clamp(min=0.01) / top_bottom.max(dim=-1)[0])
    #         #没有保证严格
    #         centerness_targets = (
    #             torch.abs(left_right.min(dim=-1)[0]) / left_right.max(dim=-1)[0]) * (
    #             torch.abs(top_bottom.min(dim=-1)[0]) / top_bottom.max(dim=-1)[0])
    #     return torch.sqrt(centerness_targets)
    def centerness_target_to(self, pos_points_xy, gt_points_xy):
        if len(pos_points_xy) == 0:
            centerness_targets = pos_points_xy[..., 0]
        else:
            # centerness_targets = (
            #      left_right.min(dim=-1)[0].clamp(min=0.01) / left_right.max(dim=-1)[0]) * (
            #      top_bottom.min(dim=-1)[0].clamp(min=0.01) / top_bottom.max(dim=-1)[0])
            #使用高斯距离
            eudis = torch.norm(pos_points_xy - gt_points_xy, 2, dim = 1)
            gausdis = torch.exp(-0.5 * (eudis / 1)  ** 2)

        return gausdis