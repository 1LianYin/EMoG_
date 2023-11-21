import os
from argparse import Namespace
import re
from os.path import join as pjoin


def is_float(numStr):
    flag = False
    numStr = str(numStr).strip().lstrip('-').lstrip('+')
    try:
        reg = re.compile(r'^[-+]?[0-9]+\.[0-9]+$')
        res = reg.match(str(numStr))
        if res:
            flag = True
    except Exception as ex:
        print("is_float() - error: " + str(ex))
    return flag


def is_number(numStr):
    flag = False
    numStr = str(numStr).strip().lstrip('-').lstrip('+')
    if str(numStr).isdigit():
        flag = True
    return flag


def get_opt(opt_path, device):
    opt = Namespace()
    opt_dict = vars(opt)

    skip = ('-------------- End ----------------',
            '------------ Options -------------',
            '\n')
    print('Reading', opt_path)
    with open(opt_path) as f:
        for line in f:
            if line.strip() not in skip:
                # print(line.strip())
                key, value = line.strip().split(': ')
                if value in ('True', 'False'):
                    opt_dict[key] = True if value == 'True' else False
                elif is_float(value):
                    opt_dict[key] = float(value)
                elif is_number(value):
                    opt_dict[key] = int(value)
                else:
                    opt_dict[key] = str(value)

    opt_dict['which_epoch'] = 'latest'
    if 'num_layers' not in opt_dict:
        opt_dict['num_layers'] = 8
    if 'latent_dim' not in opt_dict:
        opt_dict['latent_dim'] = 512
    if 'diffusion_steps' not in opt_dict:
        opt_dict['diffusion_steps'] = 1000

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.meta_dir = pjoin(opt.save_root, 'meta')

    if opt.dataset_name == 'beat':
        opt.data_root = '/data/Liany/BEAT/beat_cache/beat_4english_15_141/'
        opt.joints_num = 47
        opt.dim_pose = 141
        opt.max_motion_length = 150
    else:
        raise KeyError('Dataset not recognized')

    opt.is_train = False
    opt.is_continue = False
    opt.device = device

    return opt