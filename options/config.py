import configargparse
import time
import json
import yaml

def str2bool(v):
    """ from https://stackoverflow.com/a/43357954/1361529 """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise configargparse.ArgumentTypeError('Boolean value expected.')
        

def parse_args():
    parser = configargparse.ArgParser()
    parser.add("-c", "--config", required=True, is_config_file=True)
    # save the objective score
    parser.add("--trainer", default="DDPM", type=str)
    
    # ------------- path and save name ---------------- #
    parser.add("--is_train", default=True, type=str2bool)
    # different between environments
    parser.add("--root_path", default="")
    parser.add("--out_root_path", default="./outputs/audio2gesture/", type=str)
    parser.add("--train_data_path", default="/datasets/beat/train/", type=str)
    parser.add("--val_data_path", default="/datasets/beat/val/", type=str)
    parser.add("--test_data_path", default="/datasets/beat/test/", type=str)
    parser.add("--mean_pose_path", default="./datasets/beat/train/", type=str)
    parser.add("--std_pose_path", default="./datasets/beat/train/", type=str)
    
    # for pretrian weights
    parser.add("--torch_hub_path", default="../../datasets/checkpoints/", type=str)

    # pretrained vae for evaluation
    parser.add("--model_name_last", default="last.pth", type=str)
    parser.add("--model_name_best", default="best.pth", type=str)
    parser.add("--eval_model", default="motion_autoencoder", type=str)
    parser.add("--e_name", default="HalfEmbeddingNet", type=str)
    parser.add("--e_path", default="./datasets/beat/ae_300.bin")
    parser.add("--variational_encoding", default=False, type=str2bool) 
    parser.add("--vae_length", default=300, type=int)

    # --------------- data ---------------------------- #
    parser.add("--dataset", default="beat", type=str)
    parser.add("--new_cache", default=False, type=str2bool)
    parser.add("--use_aug", default=False, type=str2bool)
    parser.add("--disable_filtering", default=False, type=str2bool)
    parser.add("--clean_first_seconds", default=0, type=int)
    parser.add("--clean_final_seconds", default=0, type=int)

    parser.add("--audio_rep", default="wave16k", type=str)
    parser.add("--word_rep", default="None", type=str)
    parser.add("--emo_rep", default="None", type=str)
    parser.add("--sem_rep", default="None", type=str)
    parser.add("--audio_fps", default=16000, type=int)
    parser.add("--facial_rep", default="facial39", type=str)
    parser.add("--facial_fps", default=15, type=int)
    parser.add("--facial_dims", default=39, type=int)
    parser.add("--pose_rep", default="fps15_trinity_rot_123", type=str)
    parser.add("--pose_fps", default=15, type=int)
    parser.add("--pose_dims", default=123, type=int)
    parser.add("--speaker_id", default=False, type=str2bool)
    parser.add("--speaker_dims", default=30, type=int)
    parser.add("--audio_norm", default=False, type=str2bool)
    
    parser.add("--pose_length", default=34, type=int)
    parser.add("--pre_frames", default=4, type=int)
    parser.add("--stride", default=10, type=int)
    parser.add("--pre_type", default="zero", type=str)
    
    
    # --------------- model ---------------------------- #
    parser.add("--pretrain", default=False, type=str2bool)
    parser.add("--model", default="EMoG", type=str)
    parser.add("--dropout_prob", default=0.3, type=float)
    parser.add("--n_layer", default=4, type=int)
    parser.add("--hidden_size", default=300, type=int)
    parser.add("--audio_f", default=128, type=int)
    parser.add("--facial_f", default=128, type=int)
    parser.add("--speaker_f", default=0, type=int)
    parser.add("--word_f", default=0, type=int)
    parser.add("--emotion_f", default=0, type=int)

    # --------------- diversity sampling model ---------------------------- #
    parser.add("--z_dim", default=64, type=int)
    parser.add("--dct_n", default=47, type=int)
    parser.add("--hidden_dim", default=256, type=int)
    parser.add("--node_n", default=47, type=int)
    parser.add("--base_dim", default=128, type=int)
    parser.add("--base_num_p1", default=40, type=int)
    parser.add("--dropout_rate", default=0, type=int)
    parser.add("--multi_modal_head", default=3, type=int)
    
    
    # --------------- training ------------------------- #
    parser.add("--epochs", default=120, type=int)
    parser.add("--batch_size", default=64, type=int)
    parser.add("--opt", default="adam", type=str)
    parser.add("--lr_base", default=0.0001, type=float)
    parser.add("--weight_decay", default=0., type=float)
    # for warmup and cosine
    parser.add("--lr_min", default=5e-4, type=float)
    # for sgd
    parser.add("--momentum", default=0.8, type=float)
    parser.add("--nesterov", default=True, type=str2bool)
    # for adam
    parser.add("--opt_betas", default=[0.5, 0.999], type=list)
    parser.add("--amsgrad", default=False, type=str2bool)
    parser.add("--lr_policy", default="none", type=str)
    parser.add("--d_lr_weight", default=0.2, type=float)
    parser.add("--rec_weight", default=500, type=float)
    parser.add("--vel_weight", default=0.1, type=float)
    parser.add("--acc_weight", default=0.1, type=float)

    # --------------- device -------------------------- #
    parser.add("--random_seed", default=2021, type=int)
    parser.add("--benchmark", default=True, type=str2bool)
    parser.add("--cudnn_enabled", default=True, type=str2bool)
    # mix precision
    parser.add("--apex", default=False, type=str2bool)
    parser.add("--gpus", default=[0], type=list)
    parser.add("--loader_workers", default=0, type=int)
    # logging
    parser.add("--log_period", default=10, type=int)
    parser.add("--test_period", default=20, type=int)

    args = parser.parse_args()
    idc = 0
    for i, char in enumerate(args.config):
        if char == "/": idc = i
    args.name = args.config[idc+1:-5]
    
    is_train = args.is_train

    if is_train:
        time_local = time.localtime()
        name_expend = "%02d%02d_%02d%02d%02d_"%(time_local[1], time_local[2],time_local[3], time_local[4], time_local[5])
        args.name = name_expend + args.name
        
    return args