import torch
import torch.cuda
import torch.optim
import torch.nn.functional as F
import svox
import json
import os
from os import path
import shutil
import gc
import numpy as np
import math
import argparse
from model.mcost import Mcost
from util.dataset import datasets
from util.util import get_expon_lr_func
from util import adConfig
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
adConfig.define_common_args(parser)

group = parser.add_argument_group("general")
group.add_argument('--train_dir', '-t', type=str, default='ckpt',
                     help='checkpoint and logging directory')
group.add_argument('--sh_dim', type=int, default=9, help='SH/learned basis dimensions (at most 10)')
# group.add_argument('--background_nlayers', type=int, default=0,#32,
#                    help='Number of background layers (0=disable BG model)')
# group.add_argument('--background_reso', type=int, default=512, help='Background resolution')
roup = parser.add_argument_group("optimization")
group.add_argument('--batch_size', type=int, default=
                     640000*5,
                     #100000,
                     #  2000,
                   help='batch size')
group.add_argument('--sigma_optim', choices=['sgd', 'rmsprop'], default='rmsprop', help="Density optimizer")
group.add_argument('--lr_sigma', type=float, default=1e-1, help='SGD/rmsprop lr for sigma')
group.add_argument('--lr_sigma_final', type=float, default=5e-2)
group.add_argument('--lr_sigma_decay_steps', type=int, default=250000)
group.add_argument('--lr_sigma_delay_steps', type=int, default=15000,
                   help="Reverse cosine steps (0 means disable)")
group.add_argument('--lr_sigma_delay_mult', type=float, default=1e-2)#1e-4)#1e-4)

group.add_argument('--hessian_mse', type=float, default=5e-4, help='Hessian check for updating models')
group.add_argument('--hessian_mse_final', type=float, default=5e-5)
group.add_argument('--hessian_mse_decay_steps', type=int, default=250000)
group.add_argument('--hessian_mse_delay_steps', type=int, default=0,
                   help="Reverse cosine steps (0 means disable)")
group.add_argument('--hessian_mse_delay_mult', type=float, default=1e-2)#1e-4)#1e-4)
group.add_argument('--hessian_tolerance', type=int, default=5,
                   help="the limit number of counter for hessian mse check")

group.add_argument('--sampling_rate', type=float, default=1e-1, help='Sampling rate on child nodes.')
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

# group.add_argument('--lr_fg_begin_step', type=int, default=0, help="Foreground begins training at given step number")

# BG LRs
# group.add_argument('--bg_optim', choices=['sgd', 'rmsprop'], default='rmsprop', help="Background optimizer")
# group.add_argument('--lr_sigma_bg', type=float, default=3e0,
#                     help='SGD/rmsprop lr for background')
# group.add_argument('--lr_sigma_bg_final', type=float, default=3e-3,
#                     help='SGD/rmsprop lr for background')
# group.add_argument('--lr_sigma_bg_decay_steps', type=int, default=250000)
# group.add_argument('--lr_sigma_bg_delay_steps', type=int, default=0, help="Reverse cosine steps (0 means disable)")
# group.add_argument('--lr_sigma_bg_delay_mult', type=float, default=1e-2)

# group.add_argument('--lr_color_bg', type=float, default=1e-1,
#                     help='SGD/rmsprop lr for background')
# group.add_argument('--lr_color_bg_final', type=float, default=5e-6,#1e-4,
#                     help='SGD/rmsprop lr for background')
# group.add_argument('--lr_color_bg_decay_steps', type=int, default=250000)
# group.add_argument('--lr_color_bg_delay_steps', type=int, default=0, help="Reverse cosine steps (0 means disable)")
# group.add_argument('--lr_color_bg_delay_mult', type=float, default=1e-2)
# END BG LRs

group.add_argument('--rms_beta', type=float, default=0.95, help="RMSProp exponential averaging factor")
group.add_argument('--eval_every', type=int, default=1,
                   help='evaluate every x epochs')

group.add_argument('--init_sigma', type=float,
                   default=0.1,
                   help='initialization sigma')
# group.add_argument('--init_sigma_bg', type=float,
#                    default=0.1,
#                    help='initialization sigma (for BG)')

# Extra logging
group.add_argument('--log_mse_image', action='store_true', default=False)
# group.add_argument('--log_depth_map', action='store_true', default=False)
# group.add_argument('--log_depth_map_use_thresh', type=float, default=None,
#         help="If specified, uses the Dex-neRF version of depth with given thresh; else returns expected term")



group = parser.add_argument_group("misc experiments")
group.add_argument('--policy',
                   choices=['pareto', 'greedy'],
                   default='greedy',
                   help='strategy to apply')
group.add_argument('--thresh_type',
                    choices=["weight", "sigma"],
                    default="weight",
                   help='Upsample threshold type')
group.add_argument('--depth_limit', type=int,
                    default=10,
                   help='Maximum number of tree depth')
group.add_argument('--max_nodes', type=int,
                   default=1e7,
                   help='(the number here is given for 22GB memory)'
                   )
group.add_argument('--init_refine', type=int,
                   default=1,
                   help='the number of times to refine entire tree initially'
                   )
group.add_argument('--tune_mode', action='store_true', default=False,
                   help='hypertuning mode (do not save, for speed)')
group.add_argument('--tune_nosave', action='store_true', default=False,
                   help='do not save any checkpoint even at the end')

group = parser.add_argument_group("losses")
group.add_argument('--weight_decay_sigma', type=float, default=1.0)
group.add_argument('--weight_decay_sh', type=float, default=1.0)
group.add_argument('--decay', action='store_true', default=True)
group.add_argument('--n_train', type=int, default=None, help='Number of training images. Defaults to use all avaiable.')

# group.add_argument('--nosphereinit', action='store_true', default=False,
#                      help='do not start with sphere bounds (please do not use for 360)')

args = parser.parse_args()
adConfig.maybe_merge_config_file(args)

assert args.lr_sigma_final <= args.lr_sigma, "lr_sigma must be >= lr_sigma_final"
assert args.lr_sh_final <= args.lr_sh, "lr_sh must be >= lr_sh_final"

os.makedirs(args.train_dir, exist_ok=True)
summary_writer = SummaryWriter(args.train_dir)

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
               **adConfig.build_data_options(args))

norms = np.linalg.norm(dset.rays.dirs, axis=-1, keepdims=True)
viewdirs = dset.rays.dirs / norms
# if args.background_nlayers > 0 and not dset.should_use_background:
#     warn('Using a background model for dataset type ' + str(type(dset)) + ' which typically does not use background')

dset_test = datasets[args.dataset_type](
        args.data_dir, split="test", **adConfig.build_data_options(args))

global_start_time = datetime.now()

mcost = Mcost(center=dset.scene_center,
              radius=dset.scene_radius,
              step_size=args.step_size,
              init_refine=args.init_refine,
              depth_limit=args.depth_limit,
              sigma_thresh=args.sigma_thresh,
              stop_thresh=args.stop_thresh,
              data_format='SH'+str(args.sh_dim),
              policy=args.policy,
              device=device,
              )

lr_sigma_func = get_expon_lr_func(args.lr_sigma, args.lr_sigma_final, args.lr_sigma_delay_steps,
                                  args.lr_sigma_delay_mult, args.lr_sigma_decay_steps)
lr_sh_func = get_expon_lr_func(args.lr_sh, args.lr_sh_final, args.lr_sh_delay_steps,
                               args.lr_sh_delay_mult, args.lr_sh_decay_steps)
hessian_func = get_expon_lr_func(args.hessian_mse, args.hessian_mse_final, args.hessian_mse_delay_steps,
                               args.hessian_mse_delay_mult, args.hessian_mse_decay_steps)
lr_sigma_factor = 1.0
lr_sh_factor = 1.0
hessian_factor = 1.0

epoch_id = -1
gstep_id_base = 0
delta_depth = 0

cam_trans = torch.diag(torch.tensor([1, -1, -1, 1], dtype=torch.float32, device=device)).inverse()

def eval_step():
    global gstep_id_base
    # Put in a function to avoid memory leak
    print('Eval step')
    with torch.no_grad():
        stats_test = {'psnr' : 0.0, 'mse' : 0.0}
        depth = player.get_depth()

        # Standard set
        N_IMGS_TO_EVAL = min(20 if epoch_id > 0 else 5, dset_test.n_images)
        N_IMGS_TO_SAVE = N_IMGS_TO_EVAL # if not args.tune_mode else 1
        img_eval_interval = dset_test.n_images // N_IMGS_TO_EVAL
        img_save_interval = (N_IMGS_TO_EVAL // N_IMGS_TO_SAVE)
        img_ids = range(0, dset_test.n_images, img_eval_interval)   
        n_images_gen = 0
        
        for i, img_id in enumerate(img_ids):
            # OpenCV to original 
            c2w = dset_test.c2w[img_id].to(device=device)@cam_trans
            rgb_pred_test = render.render_persp(c2w=c2w,
                              width=dset_test.get_image_size(img_id)[1],
                              height=dset_test.get_image_size(img_id)[0],
                              fx=dset_test.intrins.get('fx', img_id),
                              fy=dset_test.intrins.get('fy', img_id),).clamp_(0.0, 1.0)
            rgb_gt_test = dset_test.gt[img_id].to(device=device)
            all_mses = ((rgb_gt_test - rgb_pred_test) ** 2).cpu()
            if i % img_save_interval == 0:
              img_pred = rgb_pred_test.cpu()
              img_pred.clamp_max_(1.0)
              summary_writer.add_image(f'test/image_depth_{depth}_{img_id:04d}',
                      img_pred, global_step=gstep_id_base, dataformats='HWC')
              if args.log_mse_image:
                  mse_img = all_mses / all_mses.max()
                  summary_writer.add_image(f'test/mse_map_depth_{depth}_{img_id:04d}',
                          mse_img, global_step=gstep_id_base, dataformats='HWC')
              
            rgb_pred_test = rgb_gt_test = None      
            mse_num : float = all_mses.mean().item()
            psnr = -10.0 * math.log10(mse_num)
            if math.isnan(psnr):
                print('NAN PSNR', i, img_id, mse_num)
                assert False
            stats_test['mse'] += mse_num
            stats_test['psnr'] += psnr
            n_images_gen += 1
            
        stats_test['mse'] /= n_images_gen
        stats_test['psnr'] /= n_images_gen
        for stat_name in stats_test:
            summary_writer.add_scalar('test/' + stat_name,
                    stats_test[stat_name], global_step=gstep_id_base)
        summary_writer.add_scalar('epoch_id', float(epoch_id), global_step=gstep_id_base)
        print('eval stats:', stats_test)    
        
def train_step():
  global gstep_id_base
  print('Train step')    
  stats = {"mse" : 0.0, "psnr" : 0.0, "invsqr_mse" : 0.0}
  pre_delta_mse = 0
  pre_mse = 0
  counter = 0
  
  # stimulate 
  while True:
    instant_weights = torch.zeros(player.child.size(), device=device)
    for iter_id, batch_begin in enumerate(range(0, num_rays, args.batch_size)):
        gstep_id = iter_id + gstep_id_base
        lr_sigma = lr_sigma_func(gstep_id) * lr_sigma_factor
        lr_sh = lr_sh_func(gstep_id) * lr_sh_factor
        hessian_mse = hessian_func(gstep_id) * hessian_factor
        
        if not args.decay:
            lr_sigma = args.lr_sigma * lr_sigma_factor
            lr_sh = args.lr_sh * lr_sh_factor
            hessian_mse = args.hessian_mse * hessian_factor
            
        batch_end = min(batch_begin + args.batch_size, num_rays)
        batch_origins = dset.rays.origins[batch_begin: batch_end].to(device)
        batch_dirs = dset.rays.dirs[batch_begin: batch_end].to(device)
        batch_viewdir = viewdirs[batch_begin: batch_end].to(device)
        rgb_gt = dset.rays.gt[batch_begin: batch_end]
        rays = svox.Rays(batch_origins, batch_dirs, batch_viewdir)
        
        with player.accumulate_weights(op="sum") as accum:
          rgb_pred = render.forward(rays, cuda=device=='cuda')    
                  
        weight = accum.value
        instant_weights += weight/weight.sum()

        mse = F.mse_loss(rgb_gt, rgb_pred)
        mse.backward()
        mcost.optim_basis_step(lr_sigma, lr_sh, beta=args.rms_beta, optim=args.sh_optim)

        mse_num : float = mse.detach().item()
        psnr = -10.0 * math.log10(mse_num)
        stats['mse'] += mse_num
        stats['psnr'] += psnr
        stats['invsqr_mse'] += 1.0 / mse_num ** 2
        
        # log
        summary_writer.add_scalar("train/lr_sh", lr_sh, global_step=gstep_id)
        summary_writer.add_scalar("train/lr_sigma", lr_sigma, global_step=gstep_id)
        summary_writer.add_scalar('train/thred_mse', hessian_mse, gstep_id)
          
    # check if the model gets stable by hessian mse
    delta_mse = np.abs(stats['mse']-pre_mse)
    _hessian_mse = np.abs(delta_mse-pre_delta_mse)
    pre_mse= stats['mse']
    pre_delta_mse = delta_mse
    summary_writer.add_scalar('train/hessian_mse', _hessian_mse, gstep_id) 
         
    for stat_name in stats:
      stat_val = stats[stat_name] / rays_per_batch
      summary_writer.add_scalar(f'train/{stat_name}', stat_val, gstep_id)
      stats[stat_name] = 0 
       
    if _hessian_mse < hessian_mse:
          counter += 1
          if counter > args.hessian_tolerance:
                break
              
    gstep_id_base += rays_per_batch
  
  # select 
  sample_k = int(max(1, player.n_leaves*args.sampling_rate))
  idxs = mcost.select(sample_k, instant_weights)
  
  summary_writer.add_scalar(f'train/num_nodes', player.n_leaves, gstep_id)
  summary_writer.add_scalar(f'train/depth', player.get_depth(), gstep_id)  
  
  # expand 
  return mcost.expand(idxs)
                  
with tqdm(total=args.depth_limit) as pbar:
  while True:
    dset.shuffle_rays()
    epoch_id += 1
    num_rays = dset.rays.origins.size(0)
    rays_per_batch = (num_rays-1)//args.batch_size+1
    player = mcost.player
    render = mcost._volumeRenderer()
    depth = player.get_depth()

    if delta_depth != 0: 
      # NOTE: the evaluation is launched when the tree depth is changed. 
      eval_step()
      ckpt_path = path.join(args.train_dir, f'ckpt_depth_{depth}.npz')
      print('Saving', ckpt_path)
      player.save(ckpt_path)  
      gc.collect()
      pbar.update(delta_depth)
        
    res = train_step()
    delta_depth = (player.get_depth()-depth).item()
    
    if not res:
      break
    gc.collect()
    
      
