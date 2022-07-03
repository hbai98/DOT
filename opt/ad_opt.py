# Copyright 2021 Alex Yu

# First, install svox2
# Then, python ./opt/ad_opt.py /hpc/users/CONNECT/wangxikm/humandi/nerf/nerf_synthetic/chair
# or use launching script:   sh launch.sh <EXP_NAME> <GPU> <DATA_DIR>
import torch
import torch.cuda
import torch.optim
import torch.nn.functional as F
import svox2
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

group.add_argument('--reso',
                        type=str,
                        default=
                        "[[256, 256, 256], [512, 512, 512]]",
                       help='List of grid resolution (will be evaled as json);'
                            'resamples to the next one every upsamp_every iters, then ' +
                            'stays at the last one; ' +
                            'should be a list where each item is a list of 3 ints or an int')




group.add_argument('--basis_type',
                    choices=['sh', '3d_texture', 'mlp'],
                    default='sh',
                    help='Basis function type')

group.add_argument('--sh_dim', type=int, default=9, help='SH/learned basis dimensions (at most 10)')


group.add_argument('--lambda_sparsity', type=float, default=
                    0.0,
                    help="Weight for sparsity loss as in SNeRG/PlenOctrees " +
                         "(but applied on the ray)")
group.add_argument('--lambda_beta', type=float, default=
                    0.0,
                    help="Weight for beta distribution sparsity loss as in neural volumes")


group = parser.add_argument_group("optimization")
group.add_argument('--n_iters', type=int, default=2 * 12800, help='total number of iters to optimize for')
group.add_argument('--batch_size', type=int, default=
                     5000,
                     #100000,
                     #  2000,
                   help='batch size')


# TODO: make the lr higher near the end
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

group.add_argument('--rms_beta', type=float, default=0.95, help="RMSProp exponential averaging factor")

group.add_argument('--print_every', type=int, default=20, help='print every')
group.add_argument('--save_every', type=int, default=5,
                   help='save every x epochs')
group.add_argument('--eval_every', type=int, default=1,
                   help='evaluate every x epochs')

group.add_argument('--init_sigma', type=float,
                   default=0.1,
                   help='initialization sigma')


# Extra logging
group.add_argument('--log_mse_image', action='store_true', default=False)
group.add_argument('--log_depth_map', action='store_true', default=False)
group.add_argument('--log_depth_map_use_thresh', type=float, default=None,
        help="If specified, uses the Dex-neRF version of depth with given thresh; else returns expected term")


group = parser.add_argument_group("misc experiments")



group.add_argument('--tune_mode', action='store_true', default=True,
                   help='hypertuning mode (do not save, for speed)')
group.add_argument('--tune_nosave', action='store_true', default=True,
                   help='do not save any checkpoint even at the end')



group = parser.add_argument_group("losses")


group.add_argument('--n_train', type=int, default=None, help='Number of training images. Defaults to use all avaiable.')

args = parser.parse_args()
config_util.maybe_merge_config_file(args)

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


dset_test = datasets[args.dataset_type](
        args.data_dir, split="test", **config_util.build_data_options(args))

global_start_time = datetime.now()

grid = svox2.SparseGrid(reso=reso_list[reso_id],
                        center=dset.scene_center,
                        radius=dset.scene_radius,
                        use_sphere_bound=dset.use_sphere_bound and not False,
                        basis_dim=args.sh_dim,
                        use_z_order=True,
                        device=device,
                        basis_type=svox2.__dict__['BASIS_TYPE_' + args.basis_type.upper()])

# DC -> gray; mind the SH scaling!
grid.sh_data.data[:] = 0.0
grid.density_data.data[:] = args.init_sigma


lr_sigma = args.lr_sigma
lr_sh = args.lr_sh

#  grid.sh_data.data[:, 0] = 4.0
#  osh = grid.density_data.data.shape
#  den = grid.density_data.data.view(grid.links.shape)
#  #  den[:] = 0.00
#  #  den[:, :256, :] = 1e9
#  #  den[:, :, 0] = 1e9
#  grid.density_data.data = den.view(osh)




grid.requires_grad_(True)
config_util.setup_render_opts(grid.opt, args)
print('Render options', grid.opt)

gstep_id_base = 0

resample_cameras = [
        svox2.Camera(c2w.to(device=device),
                     dset.intrins.get('fx', i),
                     dset.intrins.get('fy', i),
                     dset.intrins.get('cx', i),
                     dset.intrins.get('cy', i),
                     width=dset.get_image_size(i)[1],
                     height=dset.get_image_size(i)[0],
                     ndc_coeffs=dset.ndc_coeffs) for i, c2w in enumerate(dset.c2w)
    ]
ckpt_path = path.join(args.train_dir, 'ckpt.npz')

lr_sigma_func = get_expon_lr_func(args.lr_sigma, args.lr_sigma_final, args.lr_sigma_delay_steps,
                                  args.lr_sigma_delay_mult, args.lr_sigma_decay_steps)
lr_sh_func = get_expon_lr_func(args.lr_sh, args.lr_sh_final, args.lr_sh_delay_steps,
                               args.lr_sh_delay_mult, args.lr_sh_decay_steps)
lr_sigma_factor = 1.0
lr_sh_factor = 1.0
lr_basis_factor = 1.0

epoch_id = -1
while True:
    dset.shuffle_rays()
    epoch_id += 1
    epoch_size = dset.rays.origins.size(0)
    batches_per_epoch = (epoch_size-1)//args.batch_size+1
    # Test
    def eval_step():
        # Put in a function to avoid memory leak
        print('Eval step')
        with torch.no_grad():
            stats_test = {'psnr' : 0.0, 'mse' : 0.0}

            # Standard set
            N_IMGS_TO_EVAL = min(20 if epoch_id > 0 else 5, dset_test.n_images)
            N_IMGS_TO_SAVE = N_IMGS_TO_EVAL # if not args.tune_mode else 1
            img_eval_interval = dset_test.n_images // N_IMGS_TO_EVAL
            img_save_interval = (N_IMGS_TO_EVAL // N_IMGS_TO_SAVE)
            img_ids = range(0, dset_test.n_images, img_eval_interval)

            # Special 'very hard' specular + fuzz set
            #  img_ids = [2, 5, 7, 9, 21,
            #             44, 45, 47, 49, 56,
            #             80, 88, 99, 115, 120,
            #             154]
            #  img_save_interval = 1

            n_images_gen = 0
            for i, img_id in tqdm(enumerate(img_ids), total=len(img_ids)):
                c2w = dset_test.c2w[img_id].to(device=device)
                cam = svox2.Camera(c2w,
                                   dset_test.intrins.get('fx', img_id),
                                   dset_test.intrins.get('fy', img_id),
                                   dset_test.intrins.get('cx', img_id),
                                   dset_test.intrins.get('cy', img_id),
                                   width=dset_test.get_image_size(img_id)[1],
                                   height=dset_test.get_image_size(img_id)[0],
                                   ndc_coeffs=dset_test.ndc_coeffs)
                rgb_pred_test = grid.volume_render_image(cam, use_kernel=True)
                rgb_gt_test = dset_test.gt[img_id].to(device=device)
                all_mses = ((rgb_gt_test - rgb_pred_test) ** 2).cpu()
                if i % img_save_interval == 0:
                    img_pred = rgb_pred_test.cpu()
                    img_pred.clamp_max_(1.0)
                    summary_writer.add_image(f'test/image_{img_id:04d}',
                            img_pred, global_step=gstep_id_base, dataformats='HWC')
                    if args.log_mse_image:
                        mse_img = all_mses / all_mses.max()
                        summary_writer.add_image(f'test/mse_map_{img_id:04d}',
                                mse_img, global_step=gstep_id_base, dataformats='HWC')
                    if args.log_depth_map:
                        depth_img = grid.volume_render_depth_image(cam,
                                    args.log_depth_map_use_thresh if
                                    args.log_depth_map_use_thresh else None
                                )
                        depth_img = viridis_cmap(depth_img.cpu())
                        summary_writer.add_image(f'test/depth_map_{img_id:04d}',
                                depth_img,
                                global_step=gstep_id_base, dataformats='HWC')

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
    if epoch_id % max(factor, args.eval_every) == 0: #and (epoch_id > 0 or not args.tune_mode):
        # NOTE: we do an eval sanity check, if not in tune_mode
        eval_step()
        gc.collect()

    def train_step():
        print('Train step')
        pbar = tqdm(enumerate(range(0, epoch_size, args.batch_size)), total=batches_per_epoch)
        stats = {"mse" : 0.0, "psnr" : 0.0, "invsqr_mse" : 0.0}
        for iter_id, batch_begin in pbar:
            gstep_id = iter_id + gstep_id_base
            lr_sigma = lr_sigma_func(gstep_id) * lr_sigma_factor
            lr_sh = lr_sh_func(gstep_id) * lr_sh_factor


            batch_end = min(batch_begin + args.batch_size, epoch_size)
            batch_origins = dset.rays.origins[batch_begin: batch_end]
            batch_dirs = dset.rays.dirs[batch_begin: batch_end]
            rgb_gt = dset.rays.gt[batch_begin: batch_end]
            rays = svox2.Rays(batch_origins, batch_dirs)

            #  with Timing("volrend_fused"):
            rgb_pred = grid.volume_render_fused(rays, rgb_gt,
                    beta_loss=args.lambda_beta,
                    sparsity_loss=args.lambda_sparsity,
                    randomize=args.enable_random)

            #  with Timing("loss_comp"):
            mse = F.mse_loss(rgb_gt, rgb_pred)

            # Stats
            mse_num : float = mse.detach().item()
            psnr = -10.0 * math.log10(mse_num)
            stats['mse'] += mse_num
            stats['psnr'] += psnr
            stats['invsqr_mse'] += 1.0 / mse_num ** 2

            if (iter_id + 1) % args.print_every == 0:
                # Print averaged stats
                pbar.set_description(f'epoch {epoch_id} psnr={psnr:.2f}')
                for stat_name in stats:
                    stat_val = stats[stat_name] / args.print_every
                    summary_writer.add_scalar(stat_name, stat_val, global_step=gstep_id)
                    stats[stat_name] = 0.0
                
                summary_writer.add_scalar("lr_sh", lr_sh, global_step=gstep_id)
                summary_writer.add_scalar("lr_sigma", lr_sigma, global_step=gstep_id)

 

                if gstep_id >= 0:
                    grid.optim_density_step(lr_sigma, beta=args.rms_beta, optim=args.sigma_optim)
                    grid.optim_sh_step(lr_sh, beta=args.rms_beta, optim=args.sh_optim)




    train_step()
    gc.collect()
    gstep_id_base += batches_per_epoch

    if args.save_every > 0 and (epoch_id + 1) % max(
            factor, args.save_every) == 0 and not args.tune_mode:
        print('Saving', ckpt_path)
        grid.save(ckpt_path)



        if factor > 1 and reso_id < len(reso_list) - 1:
            print('* Using higher resolution images due to large grid; new factor', factor)
            factor //= 2
            dset.gen_rays(factor=factor)
            dset.shuffle_rays()

    if gstep_id_base >= args.n_iters:
        print('* Final eval and save')
        eval_step()
        global_stop_time = datetime.now()
        secs = (global_stop_time - global_start_time).total_seconds()
        timings_file = open(os.path.join(args.train_dir, 'time_mins.txt'), 'a')
        timings_file.write(f"{secs / 60}\n")
        if not args.tune_nosave:
            grid.save(ckpt_path)
        break
