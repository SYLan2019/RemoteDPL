import random
import re
import warnings

import numpy as np
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (HOOKS, DistSamplerSeedHook, EpochBasedRunner,
                         Fp16OptimizerHook, OptimizerHook, build_optimizer,
                         build_runner)
from ..runner import DistSamplerSeedHook_semi
from mmcv.utils import build_from_cfg
from collections import OrderedDict


import sys
sys.path.append('/public/home/lsy/myj/code_dsl/DSL')
from mmdet.core import DistEvalHook, EvalHook
from mmdet.datasets import (build_dataloader, build_dataset, build_multi_dataloader,
                            replace_ImageToTensor)
from mmdet.utils import get_root_logger
from mmdet.runner.hooks import UnlabelPredHook, SemiEpochBasedRunner#, MetaNetHook


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_detector(model,
                   dataset,
                   cfg,
                   distributed=False,
                   validate=False,
                   timestamp=None,
                   meta=None,
                   ema_model=None):
    logger = get_root_logger(log_level=cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    if 'imgs_per_gpu' in cfg.data:
        logger.warning('"imgs_per_gpu" is deprecated in MMDet V2.0. '
                       'Please use "samples_per_gpu" instead')
        if 'samples_per_gpu' in cfg.data:
            logger.warning(
                f'Got "imgs_per_gpu"={cfg.data.imgs_per_gpu} and '
                f'"samples_per_gpu"={cfg.data.samples_per_gpu}, "imgs_per_gpu"'
                f'={cfg.data.imgs_per_gpu} is used in this experiments')
        else:
            logger.warning(
                'Automatically set "samples_per_gpu"="imgs_per_gpu"='
                f'{cfg.data.imgs_per_gpu} in this experiments')
        cfg.data.samples_per_gpu = cfg.data.imgs_per_gpu
    if 'unlabel_train' not in cfg.data:
        data_loaders = [
            build_dataloader(
                ds,
                cfg.data.samples_per_gpu,
                cfg.data.workers_per_gpu,
                # cfg.gpus will be ignored if distributed
                len(cfg.gpu_ids),
                dist=distributed,
                seed=cfg.seed) for ds in dataset
        ]
    else:
        data_loaders = [build_multi_dataloader(
                                 dataset,
                                 cfg.data.samples_per_gpu,
                                 cfg.data.workers_per_gpu,
                                 cfg.data.batch_config,
                                 len(cfg.gpu_ids),
                                 dist=distributed,
                                 seed=cfg.seed)]

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
        if ema_model is not None:
            ema_model = MMDistributedDataParallel(
                ema_model.cuda(),
                device_ids=[torch.cuda.current_device()],
                broadcast_buffers=False,
                find_unused_parameters=find_unused_parameters)
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)
        if ema_model is not None:
            ema_model = MMDataParallel(ema_model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)


    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    if 'runner' not in cfg:
        cfg.runner = {
            'type': 'EpochBasedRunner',
            'max_epochs': cfg.total_epochs
        }
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)
    else:
        if 'total_epochs' in cfg:
            assert cfg.total_epochs == cfg.runner.max_epochs
    if ema_model is not None:
            scale_invariant = cfg.get("scale_invariant",False)
            down_fuse = cfg.get("down_fuse", False)
            runner = build_runner(
                cfg.runner,
                default_args=dict(
                    model=model,
                    optimizer=optimizer,
                    work_dir=cfg.work_dir,
                    logger=logger,
                    meta=meta,
                    ema_model=ema_model,
                    down_fuse=down_fuse,
                    scale_invariant = scale_invariant
                ))
    else:
            runner = build_runner(
                cfg.runner,
                default_args=dict(
                    model=model,
                    optimizer=optimizer,
                    work_dir=cfg.work_dir,
                    logger=logger,
                    meta=meta))

    runner.ema_flag=False
    runner.ITER=None
    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp
    #runner.ema_model = ema_model
    ema_config = cfg.get('ema_config', None)
    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    if ema_config is not None:
            runner.register_training_hooks(cfg.lr_config, optimizer_config, ema_config, cfg.checkpoint_config, cfg.log_config, cfg.get('momentum_config', None))
    else:
            runner.register_training_hooks(cfg.lr_config, optimizer_config,  cfg.checkpoint_config, cfg.log_config, cfg.get('momentum_config', None))
    if distributed:
        if isinstance(runner, EpochBasedRunner):
            if 'unlabel_train' not in cfg.data:
                runner.register_hook(DistSamplerSeedHook())
            else:
                runner.register_hook(DistSamplerSeedHook_semi())

    # register eval hooks
    if validate:
        # Support batch_size > 1 in validation
        val_samples_per_gpu = cfg.data.val.pop('samples_per_gpu', 1)
        if val_samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.val.pipeline = replace_ImageToTensor(
                cfg.data.val.pipeline)
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=val_samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_hook = DistEvalHook if distributed else EvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))
    # for unlabel pred
    unlabel_pred_cfg = cfg.data.get('unlabel_pred',None)
    if unlabel_pred_cfg is not None:
            runner.register_hook(UnlabelPredHook(unlabel_pred_cfg, cfg, 'Det', interval_mode=unlabel_pred_cfg.eval_checkpoint_config.mode, interval=unlabel_pred_cfg.eval_checkpoint_config.interval))
            if cfg.get("deleteFP_margin",None) is not None:
                runner.register_hook(DeleteFPHook(unlabel_pred_cfg, cfg, 'Det', interval_mode=unlabel_pred_cfg.eval_checkpoint_config.mode, interval=unlabel_pred_cfg.eval_checkpoint_config.interval))

    # user-defined hooks
    if cfg.get('custom_hooks', None):
        custom_hooks = cfg.custom_hooks
        assert isinstance(custom_hooks, list), \
            f'custom_hooks expect list type, but got {type(custom_hooks)}'
        for hook_cfg in cfg.custom_hooks:
            assert isinstance(hook_cfg, dict), \
                'Each item in custom_hooks expects dict type, but got ' \
                f'{type(hook_cfg)}'
            hook_cfg = hook_cfg.copy()
            priority = hook_cfg.pop('priority', 'NORMAL')
            hook = build_from_cfg(hook_cfg, HOOKS)
            runner.register_hook(hook, priority=priority)

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow)
