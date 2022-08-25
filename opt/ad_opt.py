import torch
import torch.cuda
import torch.optim
import torch.nn.functional as F
import svox
import json
import imageio
import os
from os import path
import shutil
import gc
import numpy as np
import math
import argparse
import cv2
from util.dataset import datasets
from util.util import Timing, get_expon_lr_func, generate_dirs_equirect, viridis_cmap
from util import config_util

from warnings import warn
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
config_util.define_common_args(parser)

group = parser.add_argument_group("general")
group.add_argument('--train_dir', '-t', type=str, default='ckpt',
                     help='checkpoint and logging directory')
group.add_argument('--basis_type',
                    choices=['sh', '3d_texture', 'mlp'],
                    default='sh',
                    help='Basis function type')
group.add_argument('--sh_dim', type=int, default=9, help='SH/learned basis dimensions (at most 10)')

group.add_argument('--background_nlayers', type=int, default=0,#32,
                   help='Number of background layers (0=disable BG model)')
group.add_argument('--background_reso', type=int, default=512, help='Background resolution')

roup = parser.add_argument_group("optimization")
group.add_argument('--batch_size', type=int, default=
                     640000*5,
                     #100000,
                     #  2000,
                   help='batch size')
group.add_argument('--sigma_optim', choices=['sgd', 'rmsprop'], default='rmsprop', help="Density optimizer")
group.add_argument('--lr_sigma', type=float, default=3e1, help='SGD/rmsprop lr for sigma')
group.add_argument('--lr_sigma_final', type=float, default=5e-2)
group.add_argument('--lr_sigma_decay_steps', type=int, default=250000)
group.add_argument('--lr_sigma_delay_steps', type=int, default=15000,
                   help="Reverse cosine steps (0 means disable)")
group.add_argument('--lr_sigma_delay_mult', type=float, default=1e-2)#1e-4)#1e-4)
group.add_argument('--sh_optim', choices=['sgd', 'rmsprop'], default='rmsprop', help="SH optimizer")
group.add_argument('--lr_sh', type=float, default=
                    1e-2,
                   help='SGD/rmsprop lr for SH')
group.add_argument('--lr_sh_final', type=float,
                      default=
                    5e-6
                    )
group.add_argument('--lr_sh_decay_steps', type=int, default=250000)
group.add_argument('--lr_sh_delay_steps', type=int, default=0, help="Reverse cosine steps (0 means disable)")
group.add_argument('--lr_sh_delay_mult', type=float, default=1e-2)

group.add_argument('--lr_fg_begin_step', type=int, default=0, help="Foreground begins training at given step number")

# BG LRs
group.add_argument('--bg_optim', choices=['sgd', 'rmsprop'], default='rmsprop', help="Background optimizer")
group.add_argument('--lr_sigma_bg', type=float, default=3e0,
                    help='SGD/rmsprop lr for background')
group.add_argument('--lr_sigma_bg_final', type=float, default=3e-3,
                    help='SGD/rmsprop lr for background')
group.add_argument('--lr_sigma_bg_decay_steps', type=int, default=250000)
group.add_argument('--lr_sigma_bg_delay_steps', type=int, default=0, help="Reverse cosine steps (0 means disable)")
group.add_argument('--lr_sigma_bg_delay_mult', type=float, default=1e-2)

group.add_argument('--lr_color_bg', type=float, default=1e-1,
                    help='SGD/rmsprop lr for background')
group.add_argument('--lr_color_bg_final', type=float, default=5e-6,#1e-4,
                    help='SGD/rmsprop lr for background')
group.add_argument('--lr_color_bg_decay_steps', type=int, default=250000)
group.add_argument('--lr_color_bg_delay_steps', type=int, default=0, help="Reverse cosine steps (0 means disable)")
group.add_argument('--lr_color_bg_delay_mult', type=float, default=1e-2)
# END BG LRs

group.add_argument('--basis_optim', choices=['sgd', 'rmsprop'], default='rmsprop', help="Learned basis optimizer")
group.add_argument('--lr_basis', type=float, default=#2e6,
                      1e-6,
                   help='SGD/rmsprop lr for SH')
group.add_argument('--lr_basis_final', type=float,
                      default=
                      1e-6
                    )
group.add_argument('--lr_basis_decay_steps', type=int, default=250000)
group.add_argument('--lr_basis_delay_steps', type=int, default=0,#15000,
                   help="Reverse cosine steps (0 means disable)")
group.add_argument('--lr_basis_begin_step', type=int, default=0)#4 * 12800)
group.add_argument('--lr_basis_delay_mult', type=float, default=1e-2)

group.add_argument('--rms_beta', type=float, default=0.95, help="RMSProp exponential averaging factor")

group.add_argument('--print_every', type=int, default=20, help='print every')
group.add_argument('--save_every', type=int, default=5,
                   help='save every x epochs')
group.add_argument('--eval_every', type=int, default=1,
                   help='evaluate every x epochs')

group.add_argument('--init_sigma', type=float,
                   default=0.1,
                   help='initialization sigma')
group.add_argument('--init_sigma_bg', type=float,
                   default=0.1,
                   help='initialization sigma (for BG)')

# Extra logging
group.add_argument('--log_mse_image', action='store_true', default=False)
group.add_argument('--log_depth_map', action='store_true', default=False)
group.add_argument('--log_depth_map_use_thresh', type=float, default=None,
        help="If specified, uses the Dex-neRF version of depth with given thresh; else returns expected term")

group = parser.add_argument_group("misc experiments")
group.add_argument('--thresh_type',
                    choices=["weight", "sigma"],
                    default="weight",
                   help='Upsample threshold type')
group.add_argument('--weight_thresh', type=float,
                    default=0.0005 * 512,
                    #  default=0.025 * 512,
                   help='Upsample weight threshold; will be divided by resulting z-resolution')
group.add_argument('--density_thresh', type=float,
                    default=5.0,
                   help='Upsample sigma threshold')
group.add_argument('--background_density_thresh', type=float,
                    default=1.0+1e-9,
                   help='Background sigma threshold for sparsification')
group.add_argument('--max_grid_elements', type=int,
                    default=44_000_000,
                   help='Max items to store after upsampling '
                        '(the number here is given for 22GB memory)')

group.add_argument('--tune_mode', action='store_true', default=False,
                   help='hypertuning mode (do not save, for speed)')
group.add_argument('--tune_nosave', action='store_true', default=False,
                   help='do not save any checkpoint even at the end')

group = parser.add_argument_group("losses")
group.add_argument('--weight_decay_sigma', type=float, default=1.0)
group.add_argument('--weight_decay_sh', type=float, default=1.0)

group.add_argument('--lr_decay', action='store_true', default=True)

group.add_argument('--n_train', type=int, default=None, help='Number of training images. Defaults to use all avaiable.')

group.add_argument('--nosphereinit', action='store_true', default=False,
                     help='do not start with sphere bounds (please do not use for 360)')

args = parser.parse_args()
config_util.maybe_merge_config_file(args)

assert args.lr_sigma_final <= args.lr_sigma, "lr_sigma must be >= lr_sigma_final"
assert args.lr_sh_final <= args.lr_sh, "lr_sh must be >= lr_sh_final"
assert args.lr_basis_final <= args.lr_basis, "lr_basis must be >= lr_basis_final"

os.makedirs(args.train_dir, exist_ok=True)
summary_writer = SummaryWriter(args.train_dir)

reso_list = json.loads(args.reso)
reso_id = 0

with open(path.join(args.train_dir, 'args.json'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)
    # Changed name to prevent errors
    shutil.copyfile(__file__, path.join(args.train_dir, 'opt_frozen.py'))

torch.manual_seed(20200823)
np.random.seed(20200823)

factor = 1
dset = datasets[args.dataset_type](
               args.data_dir,
               split="train",
               device=device,
               factor=factor,
               n_images=args.n_train,
               **config_util.build_data_options(args))

if args.background_nlayers > 0 and not dset.should_use_background:
    warn('Using a background model for dataset type ' + str(type(dset)) + ' which typically does not use background')

dset_test = datasets[args.dataset_type](
        args.data_dir, split="test", **config_util.build_data_options(args))

global_start_time = datetime.now()

