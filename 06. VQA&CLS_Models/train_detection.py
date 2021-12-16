from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmcv.runner import load_checkpoint
import argparse
from tqdm.notebook import tqdm
import torch
from mmcv.runner import load_checkpoint

if __name__ == '__main__':

    # set argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed (default: 42)')
    parser.add_argument('--gpu_ids', type=int, default=0,
                        help='your possible gpu id (default: 0)')
    parser.add_argument('--epochs', type=int, default=15,
                        help='number of epochs to train (default: 15)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=64,
                        help='input batch size for validing (default: 64)')
    parser.add_argument(
        '--work_dir', default='./work_dirs/swin+mask_kfashion', help='model save at work_dir')
    parser.add_argument(
        '--config', default='./configs/swin/mask_rcnn_swin.py', help='config_dir for training')
    parser.add_argument('--prefix', type=str, default='./data/')
    args = parser.parse_args()

    # inform your torch version and device
    print("PyTorch version:[%s]." % (torch.__version__))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("your current device:[%s]." % (device))

    # config file 들고오기
    cfg = Config.fromfile(args.config)

    # dataset 바꾸기
    PREFIX = args.prefix
    cfg.data.train.img_prefix = PREFIX
    cfg.data.train.ann_file = PREFIX + 'train.json'
    cfg.data.train.seg_prefix = PREFIX
    cfg.data.val.img_prefix = PREFIX
    cfg.data.val.ann_file = PREFIX + 'val.json'

    # 실험 parameter 수정
    cfg.seed = args.seed
    cfg.gpu_ids = [args.gpu_ids]
    cfg.work_dir = args.work_dir
    cfg.data.samples_per_gpu = args.batch_size

    # build model & dataset with mmdetection
    model = build_detector(cfg.model)
    datasets = [build_dataset(cfg.data.train)]

    # train detection model
    train_detector(model, datasets[0], cfg, distributed=False, validate=True)
