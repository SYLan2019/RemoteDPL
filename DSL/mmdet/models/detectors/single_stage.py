import warnings

import numpy as np
import torch
import torch.nn as nn

from mmdet.core import bbox2result
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector
from ..plugins import CA_Block
from ..plugins import CBAM

@DETECTORS.register_module()
class SingleStageDetector(BaseDetector):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(SingleStageDetector, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        #myj 修改 增加特征融合模块=======================================很重要，不需要进行多特征融合的时候就不要定义
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.SE_fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.Sigmoid()
        )
        #增加CA模块
        self.CA = CA_Block(512)
        #怎加CBAM模块
        self.CBAM = CBAM(512)


    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs
#myj修改
    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      down_batch=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(SingleStageDetector, self).forward_train(img, img_metas)
        #myj修改
        if down_batch:
            batch_input_shape = tuple(down_batch['img'].size()[-2:])
            for img_meta in down_batch['img_metas']:
                img_meta['batch_input_shape'] = batch_input_shape
            #print('down_batch[''img'']的形状为：', down_batch['img'].shape)
            down_x = self.extract_feat(down_batch['img'])
        x = self.extract_feat(img)
        #进行特征融合
        if down_batch:
            mix_x = []
            mix_x.append(x[0])
            for i in range(1, len(x)):
                #对原尺度图像和下采样图像进行拼接
                tmp_mix = torch.cat((x[i], down_x[i-1]), 1)#tensor(2, 512, h, w)
                #将拼接特征向量送入ca模块
                tmp_mix = self.CA(tmp_mix)
                # tmp_mix = x[i] * s_h * s_w + (1 - s_h * s_w) * down_x[i-1]
                #CBAM的模块
                # tmp_mix = self.CBAM(tmp_mix)
                #将拼接内容送入SE模块
                # b_s, c, _, _ = tmp_mix.shape
                # y = self.avg_pool(tmp_mix).view(b_s, c)
                # y = self.SE_fc(y).view(b_s, c//2, 1, 1)#tensor(2, 256, 1, 1)
                # tmp_mix = y*x[i] + (1-y)*down_x[i-1]
                mix_x.append(tmp_mix)
            #myj 7.26在融合阶段，我们应该生成对多样性的预测,返回的损失要进行修改====================================
            # losses_fuse = self.bbox_head.forward_TO(mix_x, img_metas, gt_bboxes,
            #                                            gt_labels, gt_bboxes_ignore)
            losses_fuse = self.bbox_head.forward_train(mix_x, img_metas, gt_bboxes,
                                                  gt_labels, gt_bboxes_ignore)
            # losses_regular = self.bbox_head.forward_TO(x, img_metas, gt_bboxes,
            #                                       gt_labels, gt_bboxes_ignore)
            losses_regular = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                                          gt_labels, gt_bboxes_ignore)
            #下采样在后面的损失计算以及推理过程中，它的步长是不需要改变的,下采样部分进行小尺度损失计算
            # losses_down = self.bbox_head.forward_TO(down_x, down_batch['img_metas'], down_batch['gt_bboxes'],
            #                                       down_batch['gt_labels'], down_batch['gt_bboxes_ignore'])
            losses_down = self.bbox_head.forward_train(down_x, down_batch['img_metas'], down_batch['gt_bboxes'],
                                                    down_batch['gt_labels'], down_batch['gt_bboxes_ignore'])
            losses = {}
            losses['loss_cls'] = losses_fuse['loss_cls'] + losses_regular['loss_cls'] + losses_down['loss_cls']
            losses['loss_bbox'] = losses_fuse['loss_bbox'] + losses_regular['loss_bbox'] + losses_down['loss_bbox']
            #losses['loss_centerness'] = losses_fuse['loss_centerness'] + losses_regular['loss_centerness']
            losses['loss_centerness'] = losses_down['loss_centerness'] + losses_regular['loss_centerness'] + losses_fuse['loss_centerness']
            # losses['regular_loss_cls'] = losses_regular['loss_cls']
            # losses['regular_loss_bbox'] = losses_regular['loss_bbox']
            # losses['regular_loss_centerness'] = losses_regular['loss_centerness']
            # losses['down_loss_cls'] = losses_down['loss_cls']
            # losses['down_loss_bbox'] = losses_down['loss_bbox']
            #losses['down_loss_centerness'] = losses_down['loss_centerness']
            # losses['fuse_loss_cls'] = losses_fuse['loss_cls']
            # losses['fuse_loss_bbox'] = losses_fuse['loss_bbox']
            # losses['fuse_loss_centerness'] = losses_fuse['loss_centerness']
            # losses['loss_richness'] = losses_fuse['loss_richness'] + losses_regular['loss_richness'] + losses_down['loss_richness']
            return losses

        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
        return losses


    def simple_test(self, img, img_metas, rescale=False, **kwargs):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        feat = self.extract_feat(img)
        results_list = self.bbox_head.simple_test(
            feat, img_metas, rescale=rescale)
        #myj result_list返回的是一个列表 [(det_bbox, det_label)]，det_bbox(总nms的点个数，6)，6为(x1, y1, x2, y2, score, richness)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        #上面的目的就是将tensor变为array，且按照种类进行排开，形状为，最外层的为图片个数[[tensor(第一类别框的个数，6), tensor(), ...]]
        return bbox_results

    #增加对于所需特征层的输出
    def visual_feature(self, img, img_metas, **kwargs):
        feat = self.extract_feat(img)
        _, _, h, w = img.shape
        down_img = torch.nn.functional.interpolate(img[:, :, :, :].clone(), (int(h / 2), int(w / 2)), mode='bilinear')
        down_feat = self.extract_feat(down_img)
        mix_feat = []
        mix_feat.append(feat[0])
        for i in range(1, len(feat)):
            tmp_mix =torch.cat((feat[i], down_feat[i-1]), 1)
            # b_s, c, _, _ = tmp_mix.shape
            # y = self.avg_pool(tmp_mix).view(b_s, c)
            # y = self.SE_fc(y).view(b_s, c//2, 1, 1)
            # tmp_mix = y * feat[i] + (1-y) * down_feat[i-1]
            # 改成ca模块
            # tmp_mix = self.CA(tmp_mix)
            # 改成CBAM模块
            # tmp_mix = self.CBAM(tmp_mix)
            # 改成CA模块
            tmp_mix = self.CA(tmp_mix)
            # for reg_layer in self.bbox_head.reg_convs:
            #     tmp_mix = reg_layer(tmp_mix)
            mix_feat.append(tmp_mix)

        return mix_feat



    #myj 新增加融合推理函数，用来计算融合比原始图片的提升值，需要将测试阶段检测头的功能完全实现,11.27要进行完全修改
    def fuse_test(self, img, img_metas, rescale=False):
        feat = self.extract_feat(img)
        #myj 将图片进行下采样
        _, _, h, w = img.shape
        down_img = torch.nn.functional.interpolate(img[:, :, :, :].clone(), (int(h / 2), int(w / 2)), mode='bilinear')
        down_feat = self.extract_feat(down_img)
        mix_feat = []
        mix_feat.append(feat[0])
        for i in range(1, len(feat)):
            tmp_mix =torch.cat((feat[i], down_feat[i-1]), 1)
            # b_s, c, _, _ = tmp_mix.shape
            # y = self.avg_pool(tmp_mix).view(b_s, c)
            # y = self.SE_fc(y).view(b_s, c//2, 1, 1)
            # tmp_mix = y * feat[i] + (1-y) * down_feat[i-1]
            # 改成ca模块
            tmp_mix = self.CA(tmp_mix)
            mix_feat.append(tmp_mix)
        #正常图片的前向传播
        out_feat = self.bbox_head.forward(feat)
        cls_scores, bbox_preds, centernesses, richness_scores = out_feat

        #混合图片的前向传播值,会返回分类、box、中心度、丰富度值
        out_mix_feat = self.bbox_head.forward(mix_feat)
        mix_cls_scores, mix_bbox_preds, mix_centernesses, mix_richness_scores = out_mix_feat
        #现在需要将正常图片最终保留的点映射到混合图片的点上，下面开始处理正常图片保留的点，需要进行nms操作
        img_shapes = [
            img_metas[i]['img_shape'] for i in range(cls_scores[0].shape[0])
        ]
        scale_factors = [
            img_metas[i]['scale_factor'] for i in range(cls_scores[0].shape[0])
        ]
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_points = self.bbox_head.get_points(featmap_sizes, bbox_preds[0].dtype,
                                      bbox_preds[0].device)
        cfg = self.bbox_head.test_cfg
        device = cls_scores[0].device
        batch_size = cls_scores[0].shape[0]
        num_classes = cls_scores[0].shape[1]
        nms_pre_tensor = torch.tensor(  # 1000
            cfg.get('nms_pre', -1), device=device, dtype=torch.long)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_centerness = []
        # 和centerness一样
        mlvl_richness = []

        mlvl_mix_bboxes = []
        mlvl_mix_scores = []
        mlvl_mix_centerness = []
        mlvl_mix_richness = []
        for cls_score, bbox_pred, centerness, richness, points, mix_cls_score, mix_bbox_pred, mix_centerness, mix_richness in zip(
                cls_scores, bbox_preds, centernesses, richness_scores, mlvl_points, mix_cls_scores, mix_bbox_preds, mix_centernesses, mix_richness_scores):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:], "原图特征图尺度不正确"
            assert mix_cls_score.size()[-2:] == mix_bbox_pred.size()[-2:], "混合尺度图像特征图尺寸不正确"
            scores = cls_score.permute(0, 2, 3, 1).reshape(
                batch_size, -1, self.bbox_head.cls_out_channels).sigmoid()
            centerness = centerness.permute(0, 2, 3,
                                            1).reshape(batch_size,
                                                       -1).sigmoid()
            richness = richness.permute(0, 2, 3, 1).reshape(batch_size, -1).sigmoid()
            bbox_pred = bbox_pred.permute(0, 2, 3,
                                          1).reshape(batch_size, -1, 4)
            points = points.expand(batch_size, -1, 2)
            #下面开始混合尺度部分
            mix_scores = mix_cls_score.permute(0, 2, 3, 1).reshape(
                batch_size, -1, self.bbox_head.cls_out_channels).sigmoid()
            mix_centerness = mix_centerness.permute(0, 2, 3,
                                            1).reshape(batch_size,
                                                       -1).sigmoid()
            mix_richness = mix_richness.permute(0, 2, 3, 1).reshape(batch_size, -1).sigmoid()
            mix_bbox_pred = mix_bbox_pred.permute(0, 2, 3,
                                          1).reshape(batch_size, -1, 4)

            # Get top-k prediction
            from mmdet.core.export import get_k_for_topk
            nms_pre = get_k_for_topk(nms_pre_tensor, bbox_pred.shape[1])
            if nms_pre > 0:
                max_scores, _ = (scores * centerness[..., None]).max(-1)
                #这里得到的远图像的索引值，应该将其映射到混合图片上
                _, topk_inds = max_scores.topk(nms_pre)
                batch_inds = torch.arange(batch_size).view(
                    -1, 1).expand_as(topk_inds).long()
                #这里原来要进行torch.onnx.is_in_onnx_export()判断，但是在实验中，不需要进行这个状态的考虑，因此直接进行点的选择
                points = points[batch_inds, topk_inds, :]
                bbox_pred = bbox_pred[batch_inds, topk_inds, :]
                scores = scores[batch_inds, topk_inds, :]
                centerness = centerness[batch_inds, topk_inds]
                richness = richness[batch_inds, topk_inds]
                #混合尺度的部分,原图对应的部分
                mix_bbox_pred = mix_bbox_pred[batch_inds, topk_inds, :]
                mix_scores = mix_scores[batch_inds, topk_inds, :]
                mix_centerness = mix_centerness[batch_inds, topk_inds]
                mix_richness = mix_richness[batch_inds, topk_inds]
            from mmdet.core import distance2bbox
            #这里只需要原始图像的bboxes值进行还原
            bboxes = distance2bbox(points, bbox_pred, max_shape=img_shapes)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_centerness.append(centerness)
            mlvl_richness.append(richness)
            #这里有一个改进点，在于可以将混合尺度图像对应的原始图像的框进行计算，但是这里我们只需要得到提升值，这个提升值为richness和centerness
            #可以省略
            #mix_bboxes = distance2bbox(points, mix_bbox_pred, max_shape=img_shapes)
            #mlvl_mix_bboxes.append(mix_bboxes)
            mlvl_mix_scores.append(mix_scores)
            mlvl_mix_centerness.append(mix_centerness)
            mlvl_mix_richness.append(mix_richness)

        batch_mlvl_bboxes = torch.cat(mlvl_bboxes, dim=1)
        if rescale:
            batch_mlvl_bboxes /= batch_mlvl_bboxes.new_tensor(
                np.array(scale_factors)).unsqueeze(1)
        batch_mlvl_scores = torch.cat(mlvl_scores, dim=1)
        batch_mlvl_centerness = torch.cat(mlvl_centerness, dim=1)
        batch_mlvl_richness = torch.cat(mlvl_richness, dim=1)

        #下面开始进行混合尺度特征图的部分
        batch_mlvl_mix_scores = torch.cat(mlvl_mix_scores, dim=1)
        batch_mlvl_mix_centerness = torch.cat(mlvl_mix_centerness, dim=1)
        batch_mlvl_mix_richness = torch.cat(mlvl_mix_richness, dim=1)

        padding = batch_mlvl_scores.new_zeros(batch_size, batch_mlvl_scores.shape[1], 1)
        batch_mlvl_scores = torch.cat([batch_mlvl_scores, padding], dim=-1)
        batch_mlvl_mix_scores = torch.cat([batch_mlvl_mix_scores, padding], dim=-1)

        #默认是要进行nms操作，因为在进行推理过程中，一定要省略掉许多多余的框
        from mmdet.core import multiclass_nms
        det_results = []
        for (mlvl_bboxes, mlvl_scores,
                 mlvl_centerness, mlvl_richness, mlvl_mix_scores, mlvl_mix_centerness, mlvl_mix_richness) in zip(batch_mlvl_bboxes, batch_mlvl_scores,
                                         batch_mlvl_centerness, batch_mlvl_richness,
                                            batch_mlvl_mix_scores, batch_mlvl_mix_centerness, batch_mlvl_mix_richness):
            det_bbox, det_label, det_inds = multiclass_nms(
                mlvl_bboxes,
                mlvl_scores,
                cfg.score_thr,
                cfg.nms,
                cfg.max_per_img,
                score_factors=mlvl_centerness,
                return_inds=True)  # 需要返回索引，从而得到每个框相对应生成的richness值
            richness_det = mlvl_richness.view(-1, 1).expand(mlvl_scores.size(0), num_classes)
            richness_det = richness_det.reshape(-1)
            det_richness = richness_det[det_inds]
            #下面开始计算相对应的提升值
            mix_tmp_scores = mlvl_mix_scores[:, :-1]
            mix_tmp_scores = mix_tmp_scores.reshape(-1)
            mix_tmp_centerness = mlvl_mix_centerness.view(-1, 1).expand(mlvl_scores.size(0), num_classes)
            mix_tmp_centerness = mix_tmp_centerness.reshape(-1)
            s_mix_tmp_scores = mix_tmp_scores * mix_tmp_centerness
            s_mix_scores = s_mix_tmp_scores[det_inds]
            mix_tmp_richness = mlvl_mix_richness.view(-1, 1).expand(mlvl_scores.size(0), num_classes)
            mix_tmp_richness = mix_tmp_richness.reshape(-1)
            s_mix_richness = mix_tmp_richness[det_inds]

            alpha = 0
            #正常图片总得分
            det_total_scores = torch.sqrt(det_bbox[:, 4] * det_bbox[:, 4] + det_richness * det_richness)
            #混合图片总得分
            det_total_mix_scores = torch.sqrt(s_mix_scores * s_mix_scores + s_mix_richness * s_mix_richness)
            #对应点的得分提升值
            det_boost = torch.relu(det_total_mix_scores - det_total_scores)

            det_bbox = torch.cat([det_bbox, det_boost[:, None]], dim=-1)
            det_results.append(tuple([det_bbox, det_label]))
        #上面已经完成simple_test操作，下面要继续数据的整理
        bbox_results = [
            bbox2result(det_bboxes, det_labels, num_classes)
            for det_bboxes, det_labels in det_results
        ]

        return bbox_results

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            imgs (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        assert hasattr(self.bbox_head, 'aug_test'), \
            f'{self.bbox_head.__class__.__name__}' \
            ' does not support test-time augmentation'

        feats = self.extract_feats(imgs)
        results_list = self.bbox_head.aug_test(
            feats, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def onnx_export(self, img, img_metas):
        """Test function without test time augmentation.

        Args:
            img (torch.Tensor): input images.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple[Tensor, Tensor]: dets of shape [N, num_det, 5]
                and class labels of shape [N, num_det].
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        # get origin input shape to support onnx dynamic shape

        # get shape as tensor
        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape
        # get pad input shape to support onnx dynamic shape for exporting
        # `CornerNet` and `CentripetalNet`, which 'pad_shape' is used
        # for inference
        img_metas[0]['pad_shape_for_onnx'] = img_shape
        # TODO:move all onnx related code in bbox_head to onnx_export function
        det_bboxes, det_labels = self.bbox_head.get_bboxes(*outs, img_metas)

        return det_bboxes, det_labels
