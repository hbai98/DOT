from secrets import choice
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
from model.mcot import MCOT
from model.utils import _SOFTPLUS_M1, threshold
from util.dataset import datasets
from util.util import get_expon_lr_func
from util import adConfig
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
import os
torch.cuda.set_device(6) 
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
adConfig.define_common_args(parser)

group = parser.add_argument_group("general")
group.add_argument('--train_dir', '-t', type=str, default='ckpt',
                   help='checkpoint and logging directory')
group.add_argument('--sh_dim', type=int, default=9,
                   help='SH/learned basis dimensions (at most 10)')
# group.add_argument('--background_nlayers', type=int, default=0,#32,
#                    help='Number of background layers (0=disable BG model)')
# group.add_argument('--background_reso', type=int, default=512, help='Background resolution')
roup = parser.add_argument_group("optimization")
group.add_argument('--batch_size', type=int, default=640000,
                   # 100000,
                   #  2000,
                   help='batch size')
group.add_argument(
    '--sigma_optim', choices=['sgd', 'rmsprop'], default='rmsprop', help="Density optimizer")
group.add_argument('--lr_sigma', type=float, default=1e-1,
                   help='SGD/rmsprop lr for sigma')
group.add_argument('--lr_sigma_final', type=float, default=5e-2)
group.add_argument('--lr_sigma_decay_steps', type=int, default=250000)
group.add_argument('--lr_sigma_delay_steps', type=int, default=15000,
                   help="Reverse cosine steps (0 means disable)")
group.add_argument('--lr_sigma_delay_mult', type=float,
                   default=1e-2)  # 1e-4)#1e-4)

group.add_argument('--hessian_mse', type=float, default=5e-4,
                   help='Hessian check for updating models')
group.add_argument('--hessian_mse_final', type=float, default=5e-5)
group.add_argument('--hessian_mse_decay_steps', type=int, default=250000)
group.add_argument('--hessian_mse_delay_steps', type=int, default=0,
                   help="Reverse cosine steps (0 means disable)")
group.add_argument('--hessian_mse_delay_mult', type=float,
                   default=1e-2)  # 1e-4)#1e-4)
group.add_argument('--hessian_tolerance', type=int, default=5,
                   help="the limit number of counter for hessian mse check")

group.add_argument('--sampling_rate', type=float,
                   default=1e-1, help='Sampling on child nodes.')
group.add_argument('--sampling_rate_final', type=float, default=3e-1)
group.add_argument('--sampling_rate_decay_steps', type=int, default=250000)
group.add_argument('--sampling_rate_delay_steps', type=int, default=0,
                   help="Reverse cosine steps (0 means disable)")
group.add_argument('--sampling_rate_delay_mult',
                   type=float, default=1e-2)  # 1e-4)#1e-4)

group.add_argument(
    '--sh_optim', choices=['sgd', 'rmsprop'], default='rmsprop', help="SH optimizer")
group.add_argument('--lr_sh', type=float, default=1e-2,
                   help='SGD/rmsprop lr for SH')
group.add_argument('--lr_sh_final', type=float,
                   default=5e-6
                   )
group.add_argument('--lr_sh_decay_steps', type=int, default=250000)
group.add_argument('--lr_sh_delay_steps', type=int, default=0,
                   help="Reverse cosine steps (0 means disable)")
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

group.add_argument('--rms_beta', type=float, default=0.95,
                   help="RMSProp exponential averaging factor")
group.add_argument('--eval_every', type=int, default=1,
                   help='evaluate every x epochs')

group.add_argument('--init_sigma', type=float,
                   default=0.1,
                   help='initialization sigma')
group.add_argument('--density_softplus', type=bool,
                   default=True,
                   )

# group.add_argument('--init_sigma_bg', type=float,
#                    default=0.1,
#                    help='initialization sigma (for BG)')

# Extra logging
group.add_argument('--log_mse_image', action='store_true', default=False)
group.add_argument('--log_sigma', action='store_true', default=False)
group.add_argument('--log_weight', action='store_true', default=False)
# group.add_argument('--log_depth_map', action='store_true', default=False)
# group.add_argument('--log_depth_map_use_thresh', type=float, default=None,
#         help="If specified, uses the Dex-neRF version of depth with given thresh; else returns expected term")
group.add_argument('--log-video', action='store_true', default=False)


group = parser.add_argument_group("mcot experiments")
group.add_argument('--policy',
                   choices=['pareto', 'greedy', 'hybrid'],
                   default='greedy',
                   help='strategy to apply')
group.add_argument('--hybrid_to_pareto',
                   type=int,
                   default=5,
                   help='after how many epochs the policy is changed to make step-wise decisions.')
group.add_argument('--pareto_signals_num', type=int,
                   default=1)
group.add_argument('--use_threshold', type=bool,
                   default=True,
                   )
group.add_argument('--recursive_thred', type=bool,
                    default=True,
                    )
group.add_argument('--use_variance', type=bool,
                   default=False)
group.add_argument('--thresh_type',
                   choices=["weight", "sigma", "num_visits"],
                   default="weight",
                   help='threshold type')
group.add_argument('--thresh_epochs',
                   type=int,
                   default=2,
                   help='threshold after every epoches')
group.add_argument('--thresh_method',
                   choices=['li', 'otsu', 'local'],
                   default='li',
                   help='dynamic threshold algorithms')
group.add_argument('--thresh_tol',
                   type=float,
                   default=0.7,
                   help='tolerance of threshold')
group.add_argument('--record_tree',
                   type=bool,
                   default=True,
                   help='whether to record number of visits and rewards on nodes.')
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
group.add_argument('--n_train', type=int, default=None,
                   help='Number of training images. Defaults to use all avaiable.')

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

pareto_signals = ['ctb', 'var']

if args.policy == 'hybrid':
    policy = 'greedy'
else:
    policy = args.policy

mcot = MCOT(center=dset.scene_center,
            radius=dset.scene_radius,
            step_size=args.step_size,
            init_refine=args.init_refine,
            depth_limit=args.depth_limit,
            sigma_thresh=args.sigma_thresh,
            stop_thresh=args.stop_thresh,
            data_format='SH'+str(args.sh_dim),
            policy=policy,
            device=device,
            p_sel=pareto_signals[:args.pareto_signals_num],
            density_softplus=args.density_softplus,
            record=args.record_tree
            )

lr_sigma_func = get_expon_lr_func(args.lr_sigma, args.lr_sigma_final, args.lr_sigma_delay_steps,
                                  args.lr_sigma_delay_mult, args.lr_sigma_decay_steps)
lr_sh_func = get_expon_lr_func(args.lr_sh, args.lr_sh_final, args.lr_sh_delay_steps,
                               args.lr_sh_delay_mult, args.lr_sh_decay_steps)
hessian_func = get_expon_lr_func(args.hessian_mse, args.hessian_mse_final, args.hessian_mse_delay_steps,
                                 args.hessian_mse_delay_mult, args.hessian_mse_decay_steps)
sampling_rate_func = get_expon_lr_func(args.sampling_rate, args.sampling_rate_final, args.sampling_rate_delay_steps,
                                       args.sampling_rate_delay_mult, args.sampling_rate_decay_steps)

global delta_depth

lr_sigma_factor = 1.0
lr_sh_factor = 1.0
hessian_factor = 1.0
sampling_factor = 1.0


epoch_id = -1
gstep_id_base = 0
gstep_id = 0
delta_depth = 0

cam_trans = torch.diag(torch.tensor(
    [1, -1, -1, 1], dtype=torch.float32, device=device)).inverse()


def eval_step():
    global gstep_id, gstep_id_base

    # Put in a function to avoid memory leak
    print('Eval step')
    with torch.no_grad():
        stats_test = {'psnr': 0.0, 'mse': 0.0}
        depth = player.get_depth()

        # Standard set
        N_IMGS_TO_EVAL = min(20 if epoch_id > 0 else 5, dset_test.n_images)
        N_IMGS_TO_SAVE = N_IMGS_TO_EVAL  # if not args.tune_mode else 1
        img_eval_interval = dset_test.n_images // N_IMGS_TO_EVAL
        img_save_interval = (N_IMGS_TO_EVAL // N_IMGS_TO_SAVE)
        img_ids = range(0, dset_test.n_images, img_eval_interval)
        n_images_gen = 0

        for i, img_id in enumerate(img_ids):
            # OpenCV to original
            c2w = dset_test.c2w[img_id].to(device=device)@cam_trans
            rgb_pred_test = render.render_persp(c2w=c2w,
                                                width=dset_test.get_image_size(img_id)[
                                                    1],
                                                height=dset_test.get_image_size(img_id)[
                                                    0],
                                                fx=dset_test.intrins.get(
                                                    'fx', img_id),
                                                fy=dset_test.intrins.get('fy', img_id),).clamp_(0.0, 1.0)
            rgb_gt_test = dset_test.gt[img_id].to(device=device)
            all_mses = ((rgb_gt_test - rgb_pred_test) ** 2).cpu()

            if args.log_video:
                pass
            else:
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
            mse_num: float = all_mses.mean().item()
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
        summary_writer.add_scalar('epoch_id', float(
            epoch_id), global_step=gstep_id_base)
        print('eval stats:', stats_test)


def train_step():
    global gstep_id, gstep_id_base, delta_depth

    print('Train step')
    stats = {"mse": 0.0, "psnr": 0.0, "invsqr_mse": 0.0}
    pre_delta_mse = 0
    pre_mse = 0
    counter = 0
    sampling_rate = sampling_rate_func(gstep_id) * sampling_factor
    leaves = mcot.tree._all_leaves()
    
    # stimulate
    while True:
        # important to make the learned properties stable.
        dset.shuffle_rays()
        # updata params
        instant_weights = torch.zeros_like(player.child, device=device, dtype=player.data.dtype)
        weights = []

        for iter_id, batch_begin in enumerate(range(0, num_rays, args.batch_size)):
            gstep_id = iter_id + gstep_id_base
            batch_end = min(batch_begin + args.batch_size, num_rays)
            batch_origins = dset.rays.origins[batch_begin: batch_end].to(
                device)
            batch_dirs = dset.rays.dirs[batch_begin: batch_end].to(device)
            batch_viewdir = viewdirs[batch_begin: batch_end].to(device)
            rgb_gt = dset.rays.gt[batch_begin: batch_end]
            rays = svox.Rays(batch_origins, batch_dirs, batch_viewdir)

            lr_sigma = lr_sigma_func(gstep_id) * lr_sigma_factor
            lr_sh = lr_sh_func(gstep_id) * lr_sh_factor
            hessian_mse = hessian_func(gstep_id) * hessian_factor

            with player.accumulate_weights(op="sum") as accum:
                rgb_pred = render.forward(rays, cuda=device == 'cuda')

            mse = F.mse_loss(rgb_gt, rgb_pred)
            mcot.tree.zero_grad()
            mse.backward()
            mcot.optim_basis_all_step(
                lr_sigma, lr_sh, beta=args.rms_beta, optim=args.sh_optim)

            mse_num: float = mse.detach().item()
            psnr = -10.0 * math.log10(mse_num)
            stats['mse'] += mse_num
            stats['psnr'] += psnr
            stats['invsqr_mse'] += 1.0 / mse_num ** 2

            weight = accum.value
            # sel = [*mcot.tree._all_leaves().long().T, ]
            # weight = weight[sel]
            # mse reduction
            instant_weights += weight

            if args.use_variance:
                # weight varibility
                weights.append(weight/weight.sum())
                
        # log
        summary_writer.add_scalar(
            "train/lr_sh", lr_sh, global_step=gstep_id)
        summary_writer.add_scalar(
            "train/lr_sigma", lr_sigma, global_step=gstep_id)
        summary_writer.add_scalar('train/thred_mse', hessian_mse, gstep_id)
        summary_writer.add_scalar(
            'train/sampling_rate', sampling_rate, gstep_id)

        # check if the model gets stable by hessian mse
        delta_mse = np.abs(stats['mse']-pre_mse)
        _hessian_mse = np.abs(delta_mse-pre_delta_mse)
        pre_mse = stats['mse']
        pre_delta_mse = delta_mse
        summary_writer.add_scalar('train/hessian_mse', _hessian_mse, gstep_id)

        for stat_name in stats:
            stat_val = stats[stat_name] / rays_per_batch
            summary_writer.add_scalar(f'train/{stat_name}', stat_val, gstep_id)
            stats[stat_name] = 0

        if _hessian_mse < hessian_mse:
            counter += 1
            if counter > args.hessian_tolerance:
                instant_weights /= instant_weights.sum()
                # calculate val_weights
                if args.use_variance:
                    var_weights = torch.var(torch.stack(weights, dim=0), dim=0)
                    instant_weights = torch.stack(
                        [instant_weights, var_weights], dim=-1)
                break

        gstep_id_base += rays_per_batch

    if delta_depth != 0:
        eval_step()
        depth = player.get_depth()
        ckpt_path = path.join(args.train_dir, f'ckpt_depth_{depth}.npz')
        print('Saving', ckpt_path)
        player.save(ckpt_path)
        gc.collect()
        pbar.update(delta_depth)
    # threshold
   # the flag used to skip selection and expansion; to give more time to
    # obtain stable properties.
    prune = False
    sel = (*leaves.long().T, )
    
    if args.thresh_type == 'sigma':
        val = mcot.tree.data[sel][..., -1]
        if args.density_softplus:
            val = _SOFTPLUS_M1(val)
    elif args.thresh_type == 'num_visits':
        assert args.record_tree, 'Pruning by num_visits is only accessible when record_tree option is true.'
    elif args.thresh_type == 'weight':
        val = instant_weights[sel]
    
    val = torch.nan_to_num(val, nan=0)
        
    if args.use_threshold and epoch_id % args.thresh_epochs == 0:
        prune = True
        thred = threshold(val, args.thresh_method)

        print(f'Prunning at {thred}/{val.max()}')
        pre_sel = None

        while True:
            sel = leaves[val < thred]
            nids, counts = torch.unique(sel[:, 0], return_counts=True)
            # discover the fronts whose all children are included in sel
            # sel_nids = torch.masked_select(nids, counts>=int(mcot.tree.N**3*args.thresh_tol))
            # print(counts)
            mask = (counts>=int(mcot.tree.N**3*args.thresh_tol)).numpy()
            # print(mask.shape)
            # print(mask)

            sel_nids = nids[mask]
            parent_sel = (*mcot.tree._unpack_index(
                mcot.tree.parent_depth[sel_nids, 0]).long().T,)
            print(f'sel_nids:{sel_nids.shape}')

            if pre_sel is not None:
                if torch.equal(pre_sel, sel_nids) or sel_nids.size(0) == 0:
                    break

            pre_sel = sel_nids
            # print(nids.shape)
            mcot.merge(sel_nids)
            mcot.tree.check_integrity()

            if not args.recursive_thred:
                break

            print(f'Prune {len(sel_nids)*mcot.tree.N ** 3}/{leaves.size(0)}')

            reduced = instant_weights[sel_nids].view(-1, mcot.tree.N ** 3).sum(-1)


            # nids, idxs, counts = torch.unique(torch.tensor(
            #     leaves[:,0], device=device), return_inverse=True, return_counts=True)
            
            # reduced = torch.zeros(sel_nids.size(
            #     0), dtype=val.dtype, device=device).scatter_add_(0, idxs, val)
            # reduced /= counts
            # print(mcot.tree.parent_depth.shape)
            # print(nids.shape)

            # print(leaves.shape)
            # print(val.shape)
            # print((leaves[val >= thred]).shape)
            # print(parent_sel.shape)
            # print((val[val >= thred].shape))
            # print(reduced.shape)

            leaves = mcot.tree._all_leaves()
            instant_weights[parent_sel] = reduced

            if args.thresh_type == 'weight':
                val = instant_weights[(*leaves.long().T, )]
            elif args.thresh_type == 'sigma':
                val = mcot.tree.data[(*leaves.long().T, )][..., -1]
                if args.density_softplus:
                    val = _SOFTPLUS_M1(val)
            val = torch.nan_to_num(val, nan=0)
    

    if args.policy == 'hybrid' and args.record_tree and epoch_id == args.hybrid_to_pareto:
        print('Change the policy on selection...')
        mcot.policy = 'pareto'

    if not prune:
        print('Start sampling...')
        sample_k = int(max(1, player.n_leaves*sampling_rate))
        idxs = mcot.select(sample_k, val, leaves)
        # print(f'{idxs.size(0)}/{sample_k}')
        if args.record_tree:
            mcot.backtrace(val, leaves)
        mcot.expand(idxs)

    summary_writer.add_scalar(f'train/num_nodes', player.n_leaves, gstep_id)
    summary_writer.add_scalar(f'train/depth', player.get_depth(), gstep_id)

    if args.log_sigma:
        sel = [*mcot.tree._all_leaves().long().T, ]
        sigma = mcot.tree.data[sel][..., -1]
        if args.density_softplus:
            sigma = _SOFTPLUS_M1(sigma)
        sigma = torch.nan_to_num(sigma, nan=0)
        summary_writer.add_histogram(f'train/sigma_iter_{gstep_id}', sigma, 0)

    if args.log_weight:
        summary_writer.add_histogram(f'train/weight_iter_{gstep_id}', val, 0)



with tqdm(total=args.depth_limit) as pbar:
    player = mcot.tree
    depth = player.get_depth()
    pbar.update(depth.item())

    while True:
        epoch_id += 1
        num_rays = dset.rays.origins.size(0)
        rays_per_batch = (num_rays-1)//args.batch_size+1
        player = mcot.tree
        render = mcot._volumeRenderer()
        depth = player.get_depth()

        train_step()
        delta_depth = (player.get_depth()-depth).item()

        if depth >= args.depth_limit:
            break
        # gc.collect()
