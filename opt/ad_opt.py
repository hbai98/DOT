from secrets import choice
from einops import rearrange
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
# torch.cuda.set_device(6)
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
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
group.add_argument('--batch_size', type=int, default=5,
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
group.add_argument('--mse_weights', type=float,
                   default=50,
                   )
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

group.add_argument('--prune_tol', type=float,
                   default=1e-2, help='Sampling on child nodes.')
group.add_argument('--prune_tol_final', type=float, default=3e-1)
group.add_argument('--prune_tol_decay_steps', type=int, default=250000)
group.add_argument('--prune_tol_delay_steps', type=int, default=0,
                   help="Reverse cosine steps (0 means disable)")
group.add_argument('--prune_tol_delay_mult',
                   type=float, default=1e-2)  # 1e-4)#1e-4)


group.add_argument('--repeats', type=int, default=2)
group.add_argument('--pruneSampleRepeats', type=int, default=1)


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
group.add_argument('--lr_sparsity_loss', type=float, default=1e-3)
group.add_argument('--lr_tv_loss', type=float, default=1e-3)
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
group.add_argument('--use_sparsity_loss',
                   type=bool,
                   default=True,
                   )
group.add_argument('--init_weight_sparsity_loss',
                   type=float,
                   default=None)
group.add_argument('--use_tv_loss',
                   type=bool,
                   default=True,
                   )
group.add_argument('--init_weight_tv_sigma_loss',
                   type=float,
                   default=None)
group.add_argument('--init_weight_tv_color_loss',
                   type=float,
                   default=None)

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
group.add_argument('--use_variance', type=bool,
                   default=False)
group.add_argument('--var_weight', type=float, default=1e1)
group.add_argument('--thresh_type',
                   choices=["weight", "sigma", "num_visits"],
                   default="weight",
                   help='threshold type')
group.add_argument('--thresh_epochs',
                   type=int,
                   default=2,
                   help='threshold after every epoches')
group.add_argument('--thresh_method',
                   choices=['li', 'otsu', 'minimum', 'constant', 'triangle'],
                   default='li',
                   help='dynamic threshold algorithms')
group.add_argument('--thresh_val',
                   type=float,
                   default=0.256,
                   help='constant threshold value')
group.add_argument('--thresh_gaussian_sigma',
                   type=int,
                   default=3,
                   )
group.add_argument('--thresh_tol',
                   type=float,
                   default=0.7,
                   help='tolerance of threshold')
group.add_argument('--sample_after_prune',
                   type=bool,
                   default=True)
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

torch.manual_seed(20221009)
np.random.seed(20221009)

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
            p_sel=pareto_signals[:1],
            density_softplus=args.density_softplus,
            record=args.record_tree,
            init_weight_sparsity_loss=args.init_weight_sparsity_loss,
            init_tv_sigma_loss=args.init_weight_tv_sigma_loss,
            init_tv_color_loss=args.init_weight_tv_color_loss)

lr_sigma_func = get_expon_lr_func(args.lr_sigma, args.lr_sigma_final, args.lr_sigma_delay_steps,
                                  args.lr_sigma_delay_mult, args.lr_sigma_decay_steps)
lr_sh_func = get_expon_lr_func(args.lr_sh, args.lr_sh_final, args.lr_sh_delay_steps,
                               args.lr_sh_delay_mult, args.lr_sh_decay_steps)
hessian_func = get_expon_lr_func(args.hessian_mse, args.hessian_mse_final, args.hessian_mse_delay_steps,
                                 args.hessian_mse_delay_mult, args.hessian_mse_decay_steps)
sampling_rate_func = get_expon_lr_func(args.sampling_rate, args.sampling_rate_final, args.sampling_rate_delay_steps,
                                       args.sampling_rate_delay_mult, args.sampling_rate_decay_steps)

prune_tol_func = get_expon_lr_func(args.prune_tol, args.prune_tol_final, args.prune_tol_delay_steps,
                                       args.prune_tol_delay_mult, args.prune_tol_decay_steps)

global delta_depth

lr_sigma_factor = 1.0
lr_sh_factor = 1.0
hessian_factor = 1.0
sampling_factor = 1.0


epoch_id = -1
gstep_id_base = 0
gstep_id = 0
delta_depth = 0

max_psnr = 0
tol_delta_val = None

cam_trans = torch.diag(torch.tensor(
    [1, -1, -1, 1], dtype=torch.float32, device=device)).inverse()

param_groups = []
if args.use_sparsity_loss:
    param_groups.append({'params': mcot.w_sparsity,
                        'lr': args.lr_sparsity_loss, 'name': 'sparsity_loss'})
if args.use_tv_loss:
    param_groups.append({'params': [mcot.w_sigma_tv, mcot.w_color_tv],
                        'lr': args.lr_tv_loss, 'name': 'tv_loss'})
if len(param_groups) != 0:
    optim = torch.optim.Adam(param_groups)


def eval_step():
    global gstep_id, gstep_id_base, max_psnr

    # Put in a function to avoid memory leak
    print('Eval step')
    with torch.no_grad():
        stats_test = {'psnr': 0.0, 'mse': 0.0}
        depth = mcot.tree.get_depth()

        # Standard set
        N_IMGS_TO_EVAL = min(20 if epoch_id > 0 else 5, dset_test.n_images)
        N_IMGS_TO_SAVE = N_IMGS_TO_EVAL  # if not args.tune_mode else 1
        img_eval_interval = dset_test.n_images // N_IMGS_TO_EVAL
        img_save_interval = (N_IMGS_TO_EVAL // N_IMGS_TO_SAVE)
        img_ids = range(0, dset_test.n_images, img_eval_interval)
        n_images_gen = 0

        for i, img_id in enumerate(img_ids):
            # OpenCV to original
            c2w = dset_test.c2w[img_id].to(device=device)
            c2w = torch.mm(c2w, cam_trans)
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
        if stats_test['psnr'] > max_psnr:
            max_psnr = stats_test['psnr']
            ckpt_path = path.join(args.train_dir, f'ckpt_best.npz')
            print('Saving best:', ckpt_path)
            player.save(ckpt_path)
        print('eval stats:', stats_test)


def update_val_leaves(instant_weights):
    leaves = mcot.tree._all_leaves()
    if args.thresh_type == 'weight':
        val = instant_weights[(*leaves.long().T, )]
    elif args.thresh_type == 'sigma':
        val_ = mcot.tree.data[(*leaves.long().T, )][..., -1]
        if args.density_softplus:
            val = _SOFTPLUS_M1(val_)
    val = torch.nan_to_num(val, nan=0)
    return val, leaves


def prune_func(instant_weights, prune_tol_ratio):
    with torch.no_grad():
        leaves = mcot.tree._all_leaves()
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

        if args.thresh_method == 'constant':
            thred = args.thresh_val
        else:
            thred = threshold(val, args.thresh_method, args.thresh_gaussian_sigma)
            # thred = min(thred, args.prune_max)
        if thred is None:
            assert False, 'Threshold is wrong.'
        contrast = (val[val>thred].mean()-val[val<=thred].mean())*prune_tol_ratio
        summary_writer.add_scalar(f'train/thred', thred, gstep_id)
        summary_writer.add_scalar(f'train/contrast_upperbound_for_thred', contrast, gstep_id)

        # if thred >= contrast:
        #     print(f'thred:{thred}, goal:{contrast}')
        #     return None, None

        pre_sel = None
        print(f'Prunning at {thred}/{val.max()}')
        while True:
            sel = leaves[val < thred]
            nids, counts = torch.unique(sel[:, 0], return_counts=True)
            # discover the fronts whose all children are included in sel
            mask = (counts >= int(mcot.tree.N**3*args.thresh_tol)).numpy()

            sel_nids = nids[mask]
            parent_sel = (*mcot.tree._unpack_index(
                mcot.tree.parent_depth[sel_nids, 0]).long().T,)

            if pre_sel is not None:
                if sel_nids.size(0) == 0 or torch.equal(pre_sel, sel_nids):
                    break

            pre_sel = sel_nids
            mcot.merge(sel_nids)
            print(f'Prune {len(sel_nids)*mcot.tree.N ** 3}/{leaves.size(0)}')

            reduced = instant_weights[sel_nids].view(-1, mcot.tree.N ** 3).sum(-1)
            instant_weights[parent_sel] = reduced

            val, leaves = update_val_leaves(instant_weights)

        return val, leaves


def train_step():
    global gstep_id, gstep_id_base, delta_depth, thred

    print('Train step')
    stats = {"mse": 0.0, "psnr": 0.0, "invsqr_mse": 0.0}
    pre_delta_mse = 0
    pre_mse = 0
    counter = 0
    batch_size = dset.h*dset.w*args.batch_size
    num_rays = dset.rays.origins.size(0)
    rays_per_batch = (num_rays-1)//batch_size+1

    prune = False
    # stimulate
    while True:
        # important to make the learned properties stable.
        # dset.shuffle_rays()
        indexer = torch.randperm(dset.rays.origins.size(0))
        norms = np.linalg.norm(dset.rays.dirs, axis=-1, keepdims=True)
        viewdirs = dset.rays.dirs / norms
        rays = svox.Rays(dset.rays.origins[indexer],
                         dset.rays.dirs[indexer], viewdirs[indexer])
        gt = dset.rays.gt[indexer]
        # updata params
        s1 = torch.zeros_like(player.child, device=device,
                              dtype=player.data.dtype)  # E(x)

        for iter_id, batch_begin in enumerate(range(0, num_rays, batch_size)):
            gstep_id = iter_id + gstep_id_base
            batch_end = min(batch_begin + batch_size, num_rays)
            batch_origins = rays.origins[batch_begin: batch_end].to(
                device)
            batch_dirs = rays.dirs[batch_begin: batch_end].to(device)
            batch_viewdir = rays.viewdirs[batch_begin: batch_end].to(device)
            rgb_gt = gt[batch_begin: batch_end].to(device)
            b_rays = svox.Rays(batch_origins, batch_dirs, batch_viewdir)

            lr_sigma = lr_sigma_func(gstep_id) * lr_sigma_factor
            lr_sh = lr_sh_func(gstep_id) * lr_sh_factor
            hessian_mse = hessian_func(gstep_id) * hessian_factor
            sampling_rate = sampling_rate_func(gstep_id) * sampling_factor
            prune_tol_ratio = prune_tol_func(gstep_id)
            rgb_pred = render.forward(b_rays, cuda=device == 'cuda')
            mse = F.mse_loss(rgb_gt, rgb_pred)
            loss = mse.unsqueeze(0)

            if args.use_sparsity_loss:
                sigma = mcot._sigma()
                # loss_sparsity = torch.abs(mcot.w_sparsity)*torch.abs(1-args.sparse_weight*torch.exp(-sigma)).mean()
                # Cauchy version (from SNeRG)
                loss_sparsity = mcot._w_sparsity*torch.log(1+2*sigma*sigma).mean()
                loss += loss_sparsity

            if args.use_tv_loss:
                sel = mcot.tree._frontier
                color = mcot.tree.data[sel][..., :-1]
                sigma = mcot.tree.data[sel][..., -1]
                if args.density_softplus:
                    sigma = _SOFTPLUS_M1(sigma)
                color = rearrange(color, 'n x y z d -> (n d) (x y z)')
                sigma = rearrange(sigma, 'n x y z -> n (x y z)')
                loss_tv =  mcot._w_color_tv*torch.var(color, dim=-1).mean()+\
                    mcot._w_sigma_tv*torch.var(sigma, dim=-1).mean()
                loss += loss_tv

            loss.backward()
            mcot.optim_basis_all_step(
                lr_sigma, lr_sh, beta=args.rms_beta, optim=args.sh_optim)
            # weight = accum.value
            # weights.append(weight)
            with torch.no_grad():
                dif = rgb_gt-rgb_pred
                error = torch.exp(-args.mse_weights*(dif*dif).sum(-1))
                weight = mcot.reweight_rays(b_rays, error, render._get_options())
                # weight = accum.value*torch.exp(-args.mse_weights*mse)
                s1 += weight
            mcot.tree.data.grad.zero_()

            if args.use_sparsity_loss or args.use_tv_loss:
                optim.step()
                optim.zero_grad()

            mse_num = mse.detach().item()
            if mse_num < 0:
                assert False, 'Invalid mse'
            psnr = -10.0 * np.log10(mse_num)
            if math.isnan(psnr):
                assert False, 'NAN PSNR'
            stats['mse'] += mse_num
            stats['psnr'] += psnr
            stats['invsqr_mse'] += 1.0 / mse_num ** 2
            
            if args.use_sparsity_loss:
                summary_writer.add_scalar(
                    "train/w_sparsity", mcot._w_sparsity, global_step=gstep_id
                )
            if args.use_tv_loss:
                summary_writer.add_scalar(
                    "train/w_tv_color",  mcot._w_color_tv, global_step=gstep_id
                )         
                summary_writer.add_scalar(
                    "train/w_tv_sigma", mcot._w_sigma_tv, global_step=gstep_id
                )      
                
        # check if the model gets stable by hessian mse
        delta_mse = stats['mse']-pre_mse
        abs_dmse = np.abs(delta_mse)
        _hessian_mse = np.abs(abs_dmse-pre_delta_mse)
        pre_mse = stats['mse']
        pre_delta_mse = abs_dmse

        s1 /= rays_per_batch

        # calculate val_weights
        VAL = s1

        sigma = mcot._sigma()
        if sigma.max().isinf():
            assert False, 'Inf density.'
        sigma = torch.nan_to_num(sigma, nan=0)
        summary_writer.add_histogram(f'train/sigma', sigma, gstep_id)
        summary_writer.add_histogram(f'train/weight', VAL, gstep_id)
        summary_writer.add_scalar(
            f'train/num_nodes', player.n_leaves, gstep_id)
        summary_writer.add_scalar(f'train/depth', player.get_depth(), gstep_id)
        summary_writer.add_scalar(
            "train/lr_sh", lr_sh, global_step=gstep_id)
        summary_writer.add_scalar(
            "train/lr_sigma", lr_sigma, global_step=gstep_id)
        summary_writer.add_scalar('train/thred_mse', hessian_mse, gstep_id)
        summary_writer.add_scalar(
            'train/sampling_rate', sampling_rate, gstep_id)

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
    # obtain stable properties

    if args.policy == 'hybrid' and args.record_tree and epoch_id == args.hybrid_to_pareto:
        print('Change the policy on selection...')
        mcot.policy = 'pareto'
        
    val, leaves = prune_func(VAL, prune_tol_ratio)
    # to thrust sampling with refine_numb
    if args.sample_after_prune:
        if args.pruneSampleRepeats == 0:
            print('Stabalize it...')
        else:
            print('Sample globally after prunning...')
        mcot.tree.refine(repeats=args.pruneSampleRepeats)
        
    sel = leaves
    if val is not None:
        prune = True            

    continue_ = True
    if not prune:
        with torch.no_grad():
            print('Start sampling...')
            val, leaves = update_val_leaves(VAL)
            sample_k = int(max(1, player.n_leaves*sampling_rate))
            idxs = mcot.select(sample_k, val, leaves)
            # print(f'{idxs.size(0)}/{sample_k}')
            if args.record_tree:
                mcot.backtrace(val, leaves)
            continue_ = mcot.expand(idxs, args.repeats)

    summary_writer.add_scalar(f'train/num_nodes', player.n_leaves, gstep_id)
    summary_writer.add_scalar(f'train/depth', player.get_depth(), gstep_id)

    if args.log_weight:
        summary_writer.add_histogram(
            f'train/weight_iter_{gstep_id}', s1[sel], 0)
    return continue_


with tqdm(total=args.depth_limit) as pbar:
    player = mcot.tree
    depth = player.get_depth()
    pbar.update(depth.item())

    while True:
        epoch_id += 1
        player = mcot.tree
        # render = mcot._volumeRenderer(dset.ndc_coeffs)
        render = mcot._volumeRenderer()

        depth = player.get_depth()

        # eval_step()
        flag = train_step()
        delta_depth = (player.get_depth()-depth).item()

        if depth >= args.depth_limit or not flag:
            break
        gc.collect()
