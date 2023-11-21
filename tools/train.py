import os
from os.path import join as pjoin

import sys
sys.path.append('/code/EMoG')

from options.train_options import TrainCompOptions
from utils.plot_script import *

from models import GestureTransformer
from trainers import DDPMTrainer
from dataloaders.beat_new import CustomDataset

from dataloaders.build_vocab import *

from mmcv.runner import get_dist_info
from mmcv.parallel import MMDistributedDataParallel, MMDataParallel
import torch
import torch.distributed as dist


def build_models(opt):
    encoder = GestureTransformer(
        input_feats=opt.dim_pose,
        num_frames=opt.max_motion_length,
        num_layers=opt.num_layers,
        latent_dim=opt.latent_dim)
    return encoder

if __name__ == '__main__':
    parser = TrainCompOptions()
    opt = parser.parse()
    rank, world_size = get_dist_info()

    opt.device = torch.device("cuda")
    torch.autograd.set_detect_anomaly(True)

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.meta_dir = pjoin(opt.save_root, 'meta')

    if rank == 0:
        os.makedirs(opt.model_dir, exist_ok=True)
        os.makedirs(opt.meta_dir, exist_ok=True)
    if world_size > 1:
        dist.barrier()

    if opt.dataset_name == 'beat':
        opt.data_root = '/data/Liany/BEAT/beat_cache/beat_4english_15_141/train/'
        opt.max_motion_length = opt.pose_length  # 15fps * 10
        opt.joints_num = 47
        dim_pose = 141

    else:
        raise KeyError('Dataset Does Not Exist')

    encoder = build_models(opt)
    if world_size > 1:
        encoder = MMDistributedDataParallel(
            encoder.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=True)
    elif opt.data_parallel:
        encoder = MMDataParallel(
            encoder.cuda(opt.gpu_id[0]), device_ids=opt.gpu_id)
    else:
        encoder = encoder.cuda()

    trainer = DDPMTrainer(opt, encoder)
    train_dataset = CustomDataset(opt, "train")
    test_dataset = CustomDataset(opt, "test")
    
    trainer.train(train_dataset, test_dataset)
