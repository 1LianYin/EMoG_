import os
from os.path import join as pjoin

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

import time
import logging
import numpy as np
from collections import OrderedDict
from utils.utils import print_current_loss
from models.motion_autoencoder import *


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from mmcv.runner import get_dist_info
from models.gaussian_diffusion import(
    GaussianDiffusion,
    get_named_beta_schedule,
    create_named_schedule_sampler,
    ModelMeanType,
    ModelVarType,
    LossType
)

from datasets import build_dataloader

class HuberLoss(nn.Module):
    def __init__(self, beta=0.1, reduction="mean"):
        super(HuberLoss, self).__init__()
        self.beta = beta
        self.reduction = reduction
    
    def forward(self, outputs, targets):
        final_loss = F.smooth_l1_loss(outputs / self.beta, targets / self.beta, reduction=self.reduction) * self.beta
        return final_loss

def load_checkpoints(model, save_path, load_name='model'):
    states = torch.load(save_path)
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in states['model_state'].items():
        name = k[7:] 
        new_state_dict[name] = v  
    # load params
    model.load_state_dict(new_state_dict)

    logging.info(f"load self-pretrained checkpoints for {load_name}")

class DDPMTrainer(object):

    def __init__(self, args, encoder):
        self.opt = args
        self.device = args.device
        self.encoder = encoder
        self.diffusion_steps = args.diffusion_steps
        self.pre_frames = 4
        self.lambda_vel = args.lambda_vel
        self.rec_loss = HuberLoss()
        self.LossType = LossType.MSE
        sampler = 'uniform'
        beta_scheduler = 'linear'
        betas = get_named_beta_schedule(beta_scheduler, self.diffusion_steps)
        self.diffusion = GaussianDiffusion(
            betas=betas,
            model_mean_type=ModelMeanType.EPSILON, #EPSILON, START_X
            model_var_type=ModelVarType.FIXED_SMALL,
            loss_type=self.LossType, # MSE
            lambda_vel=args.lambda_vel,
        )
        self.sampler = create_named_schedule_sampler(sampler, self.diffusion)
        self.sampler_name = sampler
        self.one_hot_labels = np.eye(args.emotion_f)

        self.eval_model = HalfEmbeddingNet(args)
        load_checkpoints(self.eval_model, args.e_path, args.e_name)

        if args.is_train:
            self.mse_criterion = torch.nn.MSELoss(reduction='none')
        self.to(self.device)

    @staticmethod
    def zero_grad(opt_list):
        for opt in opt_list:
            opt.zero_grad()

    @staticmethod
    def clip_norm(network_list):
        for network in network_list:
            clip_grad_norm_(network.parameters(), 0.5)

    @staticmethod
    def step(opt_list):
        for opt in opt_list:
            opt.step()

    def forward(self, batch_data, eval_mode=False):
        tar_pose, m_lens, in_audio, in_facial, in_word, in_id, emo, in_sem = batch_data['pose'], batch_data['length'],\
        batch_data['audio'], batch_data['facial'], batch_data['word'], batch_data['id'], batch_data['emo'],batch_data['sem']

        motions = tar_pose.detach().to(self.device).float()
        in_audio = in_audio.detach().to(self.device).float() # [128, 160000]
        in_emo = self.one_hot_labels[emo] # [128, 150] -> [128, 150, 8]

        tar_pose = tar_pose.cuda()
        in_facial = in_facial.cuda() 
        
        in_pre_pose = tar_pose.new_zeros((tar_pose.shape[0], tar_pose.shape[1], tar_pose.shape[2] + 1)).cuda()
        in_pre_pose[:, 0:self.pre_frames, :-1] = tar_pose[:, 0:self.pre_frames]
        in_pre_pose[:, 0:self.pre_frames, -1] = 1
        
        self.audio = in_audio
        self.motions = motions
        self.pre_pose = in_pre_pose
        self.in_word = in_word
        self.in_id = in_id.cuda()
        self.in_emo = emo.cuda()
        self.in_sem = in_sem.cuda()
        
        x_start = motions

        B, T = x_start.shape[:2]
        cur_len = torch.LongTensor([min(T, m_len) for m_len in m_lens]).to(self.device)
        t, _ = self.sampler.sample(B, x_start.device)

        output = self.diffusion.training_losses(
            model=self.encoder,
            x_start=x_start,
            t=t,
            model_kwargs={"audio": self.audio, "length": cur_len, "in_emo": self.in_emo, \
            "in_id": self.in_id,"is_train": True},
        )
        self.real_noise = output['target']
        self.fake_noise = output['pred_noise']
        self.real_xstart = x_start
        self.pred_xstart = output['pred_xstart']

        try:
            self.src_mask = self.encoder.module.generate_src_mask(T, cur_len).to(x_start.device)
        except:
            self.src_mask = self.encoder.generate_src_mask(T, cur_len).to(x_start.device)

    def generate_batch(self, audio, m_lens, pre_seq=None, pre_emo=None, in_id=None, ref_seq=None, dim_pose=1024):
        audio_proj, audio_out = self.encoder.encode_audio(audio, m_lens[0], self.device)
        emo_proj = None
        if pre_emo is not None:
            emo_proj = self.encoder.encode_emo(pre_emo)
        
        B = len(audio)
        T = m_lens.max() 

        output = self.diffusion.p_sample_loop(
            self.encoder,
            (B, T, dim_pose),
            pre_seq = pre_seq,
            clip_denoised=False,
            progress=True,
            resizers=None,
            range_t=self.opt.range_t,
            model_kwargs={
                'audio_proj': audio_proj,
                'audio_out': audio_out,
                'pre_seq':pre_seq,
                'length': m_lens,
                'emo_proj': emo_proj,
                'in_id': in_id,
                'ref_seq':ref_seq
                })
        return output

    def generate(self, audio, m_lens, dim_pose, pre_seq=None, pre_emo=None, in_id=None, ref_seq=None, batch_size=1024):
        N = len(audio)
        cur_idx = 0
        self.encoder.eval()
        all_output = []
        while cur_idx < N:
            if cur_idx + batch_size >= N:
                batch_audio = audio[cur_idx:]
                batch_m_lens = m_lens[cur_idx:]
            else:
                batch_audio = audio[cur_idx: cur_idx + batch_size]
                batch_m_lens = m_lens[cur_idx: cur_idx + batch_size]
            output = self.generate_batch(batch_audio, batch_m_lens, pre_seq, pre_emo=pre_emo, in_id=in_id, ref_seq=ref_seq, dim_pose=dim_pose)
            B = output.shape[0]

            for i in range(B):
                all_output.append(output[i])
            cur_idx += batch_size
        return all_output

    def backward_G(self):
        # noise_mse
        B, T, *_ = self.fake_noise.shape
        
        loss_mot_rec = self.mse_criterion(self.fake_noise.reshape(B, T, -1), self.real_noise.reshape(B, T, -1)).mean(dim=-1) # (B, 47, 512)
        loss_mot_rec = (loss_mot_rec* self.src_mask).sum() / self.src_mask.sum()
        self.loss_mot_rec = loss_mot_rec

        loss_rec = self.mse_criterion(self.pred_xstart, self.real_xstart).mean(dim=-1) # (B, T)
        loss_rec = (loss_rec* self.src_mask).sum() / self.src_mask.sum()
        self.loss_rec = loss_rec*0.05 
        self.pred_logits = None

        self.total_loss = self.loss_mot_rec + self.loss_rec

        loss_logs = OrderedDict({})
        loss_logs['loss_mot_rec'] = self.loss_mot_rec.item()
        loss_logs['total_loss'] = self.total_loss.item()

        return loss_logs

    def update(self):
        self.zero_grad([self.opt_encoder])
        loss_logs = self.backward_G()
        self.loss_mot_rec.backward()
        self.clip_norm([self.encoder])
        self.step([self.opt_encoder])

        return loss_logs

    def to(self, device):
        if self.opt.is_train:
            self.mse_criterion.to(device)
        self.encoder = self.encoder.to(device)

    def train_mode(self):
        self.encoder.train()

    def eval_mode(self):
        self.encoder.eval()

    def save(self, file_name, ep, total_it):
        state = {
            'opt_encoder': self.opt_encoder.state_dict(),
            'ep': ep,
            'total_it': total_it
        }
        try:
            state['encoder'] = self.encoder.module.state_dict()
        except:
            state['encoder'] = self.encoder.state_dict()
        torch.save(state, file_name)
        return

    def load(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        if self.opt.is_train:
            self.opt_encoder.load_state_dict(checkpoint['opt_encoder'])
        self.encoder.load_state_dict(checkpoint['encoder'], strict=True)
        return checkpoint['ep'], checkpoint.get('total_it', 0)
    
    def get_parameter_number(self, model):
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('Total:', total_num / 1e9, 'Trainable:', trainable_num / 1e9)
        return {'Total': total_num, 'Trainable': trainable_num}

    def train(self, train_dataset, test_dataset=None):
        rank, world_size = get_dist_info()
        self.to(self.device)
        self.opt_encoder = optim.Adam(self.encoder.parameters(), lr=self.opt.lr)
        it = 0
        cur_epoch = 0
        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')
            cur_epoch, it = self.load(model_dir)

        start_time = time.time()
        train_loader = build_dataloader(
            train_dataset,
            samples_per_gpu=self.opt.batch_size,
            drop_last=True,
            workers_per_gpu=4,
            shuffle=True,
            dist=self.opt.distributed,
            num_gpus=len(self.opt.gpu_id))

        logs = OrderedDict()
        for epoch in range(cur_epoch, self.opt.num_epochs):
            self.train_mode()
            for i, batch_data in enumerate(train_loader):
                self.forward(batch_data)

                log_dict = self.update()
                for k, v in log_dict.items():
                    if k not in logs:
                        logs[k] = v
                    else:
                        logs[k] += v
                it += 1
                if it % self.opt.log_every == 0 and rank == 0:
                    mean_loss = OrderedDict({})
                    for tag, value in logs.items():
                        mean_loss[tag] = value / self.opt.log_every
                    logs = OrderedDict()
                    print_current_loss(start_time, it, mean_loss, epoch, inner_iter=i)

                if it % self.opt.save_latest == 0 and rank == 0:
                    self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            if rank == 0:
                self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            if epoch % self.opt.save_every_e == 0 and rank == 0:
                self.save(pjoin(self.opt.model_dir, 'ckpt_e%03d.tar'%(epoch)),
                            epoch, total_it=it)
