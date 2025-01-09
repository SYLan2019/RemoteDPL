import mmcv
import torch
from mmcv import Config
from mmcv.parallel import collate, scatter
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import get_dist_info,init_dist
#from mmdet.runner.hooks.unlabel_pred_hook import inference_model
from mmdet.models import build_detector
from mmdet.datasets.pipelines import Compose
class LoadImage(object):
    def __call__(self, results):
        if isinstance(results['img'], str):
            results['filename'] = results['img']
        else:
            results['filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # liangting.zl 08.20 add default pad_shape
        results['pad_shape'] = img.shape
        results['ori_filename'] = results['filename']
        return results

def inference_model(model, img, config, task_type, iou):#iou为0.6
    """Inference image(s) with the model.

        Args:
            model (nn.Module): The loaded detector/segmentor.
            imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
                images.

        Returns:
            If imgs is a str, a generator will be returned, otherwise return the
            detection results directly.
        """
    # build the data pipeline
    test_pipeline = [LoadImage()] + config.data.unlabel_pred.pipeline[1:]
    test_pipeline = Compose(test_pipeline)#目的是创造一系列transformer实例来对图片进行处理，这里的transformer是指对图片进行resize，pad等操作
    ## bhchen add mirror testing 06/02/2021
    flip = config.data.unlabel_pred.get("eval_flip", False)#False
    # prepare data
    data = dict(img=img)#img为根路径
    data = test_pipeline(data)
    print('data的数据为:\n', data)
    image_height, image_width, _ = data['img_metas'][0].data['ori_shape']

    data = scatter(#collate函数将不同样本大小同一，放在一个batch里面
        collate([data], samples_per_gpu=1),
        [torch.cuda.current_device()])[0]
    # forward the model
    with torch.no_grad():
        if task_type in {'Det', 'Sem'}:
            if flip:
                data_mirror = torch.flip(data['img'][0], [3])
                data_mirror_flop = torch.flip(data_mirror, [2])
                data_flop = torch.flip(data_mirror_flop, [3])
                data['img_metas'][0].append(data['img_metas'][0][0])
                data['img_metas'][0].append(data['img_metas'][0][0])
                data['img_metas'][0].append(data['img_metas'][0][0])
                data['img'][0] = torch.cat([data['img'][0], data_mirror, data_mirror_flop, data_flop], dim=0)
                result_tmp = model(return_loss = False, rescale = True, **data)
                result = result_tmp[0]
                result_mirror = result_tmp[1]
                result_mirror_flop = result_tmp[2]
                result_flop = result_tmp[3]
            else:
                result = model(return_loss=False, rescale=True, **data)[0]#极为重要
        elif task_type == 'Cls':
            result = model(return_loss=False, **data)
        else:
            raise Exception()

    return result, image_height, image_width


model = dict(
    type='FCOS',
    backbone=dict(
        type='RLA_ResNet',
        layers=[3,4,6,3],
        #depth=50,
        #num_stages=4,
        #out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        #norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        pretrained='/public/home/lsy/myj/code_dsl/resnet50_rla_2283.pth.tar'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='FCOSHead',
        #种类数目一定需要修改
        num_classes=15,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        norm_on_bbox=True,
        centerness_on_reg=True,
        dcn_on_last_conv=False,
        #中心采样关
        center_sampling=False,
        conv_bias=True,
        # for unlabel loss weight
        loss_weight = 2.0,
        soft_weight = 1.0,
        soft_warm_up = 0,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100)
)
config = 'configs/fcos_semi/voc/RLA_r50_caffe_mslonger_tricks_0.Xdata_unlabel_dynamic_lw_nofuse_iterlabel_si-soft_singlestage.py'
cfg = Config.fromfile(config)

init_dist('pytorch', **cfg.dist_params)
rank, world_size = get_dist_info()
model = build_detector(model)
model.init_weights()
model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=False)


image_path = '/public/home/lsy/myj/code_dsl/data/semivoc/images/full/P0000_0450_2700.png'

if rank == 0:
    result, image_height, image_width = inference_model(model, image_path, cfg, 'Det', iou=0.6)
    print('result的数据为:\n', result)

