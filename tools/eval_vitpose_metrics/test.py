# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from mmpose.apis import multi_gpu_test, single_gpu_test
from mmpose.datasets import build_dataloader, build_dataset
from mmpose.models import build_posenet
from mmpose.utils import setup_multi_processes

try:
    from mmcv.runner import wrap_fp16_model
except ImportError:
    warnings.warn('auto_fp16 from mmpose will be deprecated from v0.15.0'
                  'Please install mmcv>=1.1.4')
    from mmpose.core import wrap_fp16_model

from openpyxl import Workbook
import re 


def parse_args():
    parser = argparse.ArgumentParser(description='mmpose test model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--work-dir', help='the dir to save evaluation results')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--eval',
        default=None,
        nargs='+',
        help='evaluation metric, which depends on the dataset,'
        ' e.g., "mAP" for MSCOCO')
    parser.add_argument(
        '--gpu_collect',
        action='store_true',
        help='whether to use gpu to collect results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)

    # ----------------------- by xwj    
    parser.add_argument(
        '--src-root',
        default=None,
        help='src data root to collect images.')  

    parser.add_argument(
        '--pose-excel-root',
        type=str, 
        default=r"/dat03/xuanwenjie/code/animal_pose/mmpose-main/output_dir/evaluation/vitpose_eval/debug", 
        help='pose excels save root.') 

    # ------------------------ end

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def merge_configs(cfg1, cfg2):
    # Merge cfg2 into cfg1
    # Overwrite cfg1 if repeated, ignore if value is None.
    cfg1 = {} if cfg1 is None else cfg1.copy()
    cfg2 = {} if cfg2 is None else cfg2
    for k, v in cfg2.items():
        if v:
            cfg1[k] = v
    return cfg1


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
        
    if not os.path.exists(args.pose_excel_root):
        os.makedirs(args.pose_excel_root, exist_ok=True)
    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir, exist_ok=True)
    
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    # step 1: give default values and override (if exist) from cfg.data
    loader_cfg = {
        **dict(seed=cfg.get('seed'), drop_last=False, dist=distributed),
        **({} if torch.__version__ != 'parrots' else dict(
               prefetch_num=2,
               pin_memory=False,
           )),
        **dict((k, cfg.data[k]) for k in [
                   'seed',
                   'prefetch_num',
                   'pin_memory',
                   'persistent_workers',
               ] if k in cfg.data)
    }
    # step2: cfg.data.test_dataloader has higher priority
    test_loader_cfg = {
        **loader_cfg,
        **dict(shuffle=False, drop_last=False),
        **dict(workers_per_gpu=cfg.data.get('workers_per_gpu', 1)),
        **dict(samples_per_gpu=cfg.data.get('samples_per_gpu', 1)),
        **cfg.data.get('test_dataloader', {})
    }
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the model and load checkpoint
    model = build_posenet(cfg.model)
    fp16_cfg = cfg.get('fp16', None)

    # __import__("ipdb").set_trace()
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    if not distributed:
        model = MMDataParallel(model, device_ids=[args.gpu_id])
        # final call here: /dat03/xuanwenjie/code/animal_pose/ViTPose-main/mmpose/models/detectors/top_down.py
        outputs = single_gpu_test(model, data_loader)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)

    rank, _ = get_dist_info()
    eval_config = cfg.get('evaluation', {})
    eval_config = merge_configs(eval_config, dict(metric=args.eval))

    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)

        results = dataset.evaluate(outputs, cfg.work_dir, **eval_config)
        for k, v in sorted(results.items()):
            print(f'{k}: {v}')

        # __import__("ipdb").set_trace()

        # save xlsx results 
        wb = Workbook()
        ws = wb.active
        for k,v in results.items():
            ws.append([k, v])
         
        task_name = os.path.basename(args.src_root) if args.src_root is not None else "default" 
        if "cat_annotations" in data_loader.dataset.ann_file:
            cat_id, cat_name = re.findall(r"val_split1_([\d]+)_([\w]+)\b", data_loader.dataset.ann_file)[0]
            save_path = os.path.join(args.pose_excel_root, "{}_{}_".format(cat_id, cat_name) + task_name + ".xlsx")
        else:
            save_path = os.path.join(args.pose_excel_root, task_name + ".xlsx")
        
        wb.save(save_path)
        print("INFO -- [metric xlsx] xlsx results save in: {}".format(save_path))


def copy2dir():
    import shutil 
    from tqdm import tqdm
    
    args = parse_args()

    # src_root = r"/dat03/xuanwenjie/code/animal_pose/mmpose-main/output_dir/controlnet_gen_eval/sd-controlnet-openpose"
    src_root = args.src_root
    dst_root = r"/dat03/xuanwenjie/code/animal_pose/mmpose-main/output_dir/evaluation/test_data_cache"

    if "raw_ap10k" in src_root:
        src_root = "/dat03/xuanwenjie/datasets/AP-10K_triplet/data_splits/ap10k_val_split1"

        pbar = tqdm(os.listdir(src_root))
        for afile in pbar:
            # info 
            pbar.set_description("copy from: {}".format(src_root))
            # get name
            src_img_path = os.path.join(src_root, afile)
            dst_img_path = os.path.join(dst_root, afile)

            shutil.copy(src_img_path, dst_img_path)
            # print("==> {} \n\t --> {}".format(src_img_path, dst_img_path))
            # break
            
        return 0

    if os.path.exists(dst_root):
        shutil.rmtree(dst_root)
        os.mkdir(dst_root)

    pbar = tqdm(os.listdir(src_root))
    for adir in pbar:
        # info 
        pbar.set_description("copy from: {}".format(src_root))

        adir_path = os.path.join(src_root, adir)
        if not os.path.isdir(adir_path):
            continue

        src_img_path = os.path.join(adir_path, "0.jpg")
        dst_img_path = os.path.join(dst_root, "{}.jpg".format(adir))

        shutil.copy(src_img_path, dst_img_path)
        # print("==> {} \n\t --> {}".format(src_img_path, dst_img_path))
        # break
    
    print("====> test images [0] in: {}".format(src_root))
    print("--------- build tmp_data in: {}".format(dst_root))



if __name__ == '__main__':

    # move data to tmp_dir
    copy2dir()

    # run test
    main()
