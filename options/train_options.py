from options.base_options import BaseOptions
import argparse

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

class TrainCompOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        # self.parser.add("-c", "--config", required=True, is_config_file=True)
        self.parser.add_argument('--num_layers', type=int, default=8, help='num_layers of transformer')
        self.parser.add_argument('--latent_dim', type=int, default=512, help='latent_dim of transformer')
        self.parser.add_argument('--dim_pose', type=int, default=141, help='latent_dim of transformer')
        self.parser.add_argument('--diffusion_steps', type=int, default=1000, help='diffusion_steps of transformer')
        self.parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
        self.parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
        self.parser.add_argument('--batch_size', type=int, default=32, help='Batch size per GPU')
        self.parser.add_argument('--is_continue', action="store_true", help='Is this trail continued from previous trail?')
        self.parser.add_argument('--log_every', type=int, default=50, help='Frequency of printing training progress (by iteration)')
        self.parser.add_argument('--save_every_e', type=int, default=5, help='Frequency of saving models (by epoch)')
        self.parser.add_argument('--eval_every_e', type=int, default=5, help='Frequency of animation results (by epoch)')
        self.parser.add_argument('--save_latest', type=int, default=500, help='Frequency of saving models (by iteration)')
        self.is_train = True
        
        # ------------- path and save name ---------------- #
        self.parser.add("--pose_rep", default="bvh_rot", type=str)
        self.parser.add("--pose_cache", default="bvh_rot_cache", type=str)

        self.parser.add("--out_root_path", default="./outputs/audio2pose/", type=str)
        self.parser.add("--train_data_path", default="/data/Liany/BEAT/beat_cache/beat_4english_15_141/train/", type=str)
        self.parser.add("--val_data_path", default="/data/Liany/BEAT/beat_cache/beat_4english_15_141/val/", type=str)
        self.parser.add("--test_data_path", default="/data/Liany/BEAT/beat_cache/beat_4english_15_141/test/", type=str)
        self.parser.add("--mean_pose_path", default="/data/Liany/BEAT/beat_cache/beat_4english_15_141/train/", type=str)
        self.parser.add("--std_pose_path", default="/data/Liany/BEAT/beat_cache/beat_4english_15_141/train/", type=str)
        
        # for pretrian weights
        self.parser.add("--torch_hub_path", default="../../datasets/checkpoints/", type=str)
        
        # --------------- data ---------------------------- #
        self.parser.add("--new_cache", default=False, type=str2bool)
        self.parser.add("--use_aug", default=False, type=str2bool)
        self.parser.add("--disable_filtering", default=False, type=str2bool)
        self.parser.add("--clean_first_seconds", default=0, type=int)
        self.parser.add("--clean_final_seconds", default=0, type=int)

        self.parser.add("--audio_rep", default="wave16k", type=str)
        self.parser.add("--word_rep", default="text", type=str)
        self.parser.add("--emo_rep", default="emo", type=str)
        self.parser.add("--sem_rep", default="sem", type=str)
        self.parser.add("--audio_fps", default=16000, type=int)
        #self.parser.add("--audio_dims", default=1, type=int)
        self.parser.add("--facial_rep", default="facial52", type=str)
        self.parser.add("--facial_fps", default=15, type=int)
        self.parser.add("--facial_dims", default=39, type=int)
        
        self.parser.add("--pose_fps", default=15, type=int)
        self.parser.add("--pose_dims", default=141, type=int)
        self.parser.add("--speaker_id", default=True, type=str2bool)
        self.parser.add("--speaker_dims", default=30, type=int)
        self.parser.add("--audio_norm", default=False, type=str2bool)
        
        self.parser.add("--pose_length", default=150, type=int)
        self.parser.add("--pre_frames", default=4, type=int)
        self.parser.add("--stride", default=10, type=int)
        self.parser.add("--pre_type", default="zero", type=str)
        
        
        # --------------- model ---------------------------- #
        self.parser.add("--lambda_vel", default=0.05, type=float)
        self.parser.add("--pretrain", default=False, type=str2bool)
        self.parser.add("--dropout_prob", default=0.3, type=float)
        self.parser.add("--audio_f", default=128, type=int)
        self.parser.add("--facial_f", default=32, type=int)
        self.parser.add("--speaker_f", default=8, type=int)
        self.parser.add("--word_f", default=8, type=int)
        self.parser.add("--emotion_f", default=8, type=int)
        self.parser.add("--emotion_dims", default=8, type=int)
        self.parser.add("--multi_length_training", default=[1.0], type=float, nargs="+")

        # --------------- device -------------------------- #
        self.parser.add("--random_seed", default=1234, type=int)
        self.parser.add("--benchmark", default=True, type=str2bool)
        self.parser.add("--cudnn_enabled", default=True, type=str2bool)
        # mix precision
        self.parser.add("--apex", default=False, type=str2bool)
        self.parser.add("--gpus", default=[0], type=list)
        self.parser.add("--loader_workers", default=0, type=int)
        # logging
        self.parser.add("--log_period", default=10, type=int)
        self.parser.add("--test_period", default=20, type=int)
