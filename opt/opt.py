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
from model.dot import DOT
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
group.add_argument('--pre_trained_dir', '-p', type=str,
                   help="Input the octree path for the extracted .npz file",)
group.add_argument('--train_dir', '-t', type=str, default='ckpt',
                   help='checkpoint and logging directory')
group.add_argument('--num_epoches', type=int, default=80)
group = parser.add_argument_group("optimization")
group.add_argument('--batch_size', type=int, default=5,
                   help='batch size')
group.add_argument('--val_interval', default=2,
                   type=int, help='validation interval')
group.add_argument('--lr', type=float, default=1e1,
                   help='SGD/rmsprop lr for sigma')

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

group.add_argument('--sample_after_prune',
                   type=bool,
                   default=False)
group.add_argument('--sampling_rate', type=float,
                   default=1e-1, help='Sampling on child nodes.')
group.add_argument('--sampling_rate_final', type=float, default=3e-1)
group.add_argument('--sampling_rate_decay_steps', type=int, default=250000)
group.add_argument('--sampling_rate_delay_steps', type=int, default=0,
                   help="Reverse cosine steps (0 means disable)")
group.add_argument('--sampling_rate_delay_mult',
                   type=float, default=1e-2)  # 1e-4)#1e-4)
group.add_argument('--lr_sparsity_loss', type=float, default=1e-3)
group.add_argument('--density_softplus', type=bool,
                   default=True,)
group.add_argument('--postier_weight', type=bool, default=True)
group.add_argument('--rays_with_replacement', type=bool, default=True)
group.add_argument('--mse_weights', type=float,
                   default=50,
                   )
group.add_argument('--prune_tolerance', type=int, default=5,
                   help="the limit number of counter for hessian mse check")
# Extra logging
group.add_argument('--log_mse_image', action='store_true', default=True)
group.add_argument('--log-video', action='store_true', default=False)

group = parser.add_argument_group("mcot experiments")
group.add_argument('--use_sparsity_loss',
                   type=bool,
                   default=True,
                   )
group.add_argument('--init_weight_sparsity_loss',
                   type=float,
                   default=0.01)
group.add_argument('--use_variance', type=bool,
                   default=False)
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
group.add_argument('--sh_dim', type=int, default=-1,
                   help='SH/learned basis dimensions (at most 10)')
group.add_argument('--thresh_type',
                   choices=["weight", "sigma"],
                   default="weight",
                   help='threshold type')
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
group.add_argument('--tv_thred',
                   type=float,
                   default=1e-2)
group.add_argument('--lr_tv_loss', type=float, default=1e-3)
group.add_argument('--prune_node_var_thred', type=float, default=1e-3)
group.add_argument('--pruneSampleRepeats', type=int, default=1)
group.add_argument('--rms_beta', type=float, default=0.95,
                   help="RMSProp exponential averaging factor")
group.add_argument('--distance_thred', type=float, default=1e-2)
group.add_argument('--repeats', type=int, default=2)
group.add_argument('--thresh_tol',
                   type=float,
                   default=0.7,
                   help='tolerance of threshold')
group.add_argument('--depth_limit', type=int,
                   default=10,
                   help='Maximum number of tree depth')
group.add_argument('--max_nodes', type=int,
                   default=1e7,
                   help='(the number here is given for 22GB memory)'
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

args = parser.parse_args()
adConfig.maybe_merge_config_file(args)

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

dset_name  = dset.__class__.__name__

if dset_name == 'LLFFDataset':
    ndc_conf = svox.NDCConfig(width=dset.w, height=dset.h, focal=dset.fx)
else: 
    ndc_conf = None
norms = np.linalg.norm(dset.rays.dirs, axis=-1, keepdims=True)
dset_test = datasets[args.dataset_type](
    args.data_dir, split="test", **adConfig.build_data_options(args))
global_start_time = datetime.now()

data_format = None
if args.sh_dim!=-1:
    data_format = 'SH'+str(args.sh_dim)
DOT = DOT(center=dset.scene_center,
          radius=dset.scene_radius,
          pre_train_pth=args.pre_trained_dir,
          step_size=args.step_size,
          depth_limit=args.depth_limit,
          sigma_thresh=args.sigma_thresh,
          stop_thresh=args.stop_thresh,
          data_format=data_format,
          device=device,
          density_softplus=args.density_softplus,
          init_weight_sparsity_loss=args.init_weight_sparsity_loss,
          init_tv_sigma_loss=args.init_weight_tv_sigma_loss,
          init_tv_color_loss=args.init_weight_tv_color_loss)


hessian_func = get_expon_lr_func(args.hessian_mse, args.hessian_mse_final, args.hessian_mse_delay_steps,
                                 args.hessian_mse_delay_mult, args.hessian_mse_decay_steps)
sampling_rate_func = get_expon_lr_func(args.sampling_rate, args.sampling_rate_final, args.sampling_rate_delay_steps,
                                       args.sampling_rate_delay_mult, args.sampling_rate_decay_steps)

global delta_depth

hessian_factor = 1.0
sampling_factor = 1.0

epoch_id = -1
gstep_id_base = 0
gstep_id = 0
delta_depth = 0

max_psnr = 0

cam_trans = torch.diag(torch.tensor(
    [1, -1, -1, 1], dtype=torch.float32, device=device)).inverse()

param_groups = []
if args.use_sparsity_loss:
    param_groups.append({'params': DOT.w_sparsity,
                        'lr': args.lr_sparsity_loss, 'name': 'sparsity_loss'})
if args.use_tv_loss:
    param_groups.append({'params': [DOT.w_sigma_tv, DOT.w_color_tv],
                        'lr': args.lr_tv_loss, 'name': 'tv_loss'})

param_groups.append({'params': DOT.tree.data,
                    'lr': args.lr, 'name': 'tree'})



def get_lr(optimizer, name):
    for param_group in optimizer.param_groups:
        if param_group == name:
            return param_group['lr']

if len(param_groups) != 0:
    optim = torch.optim.AdamW(param_groups)


def eval_step(save=True):
    global gstep_id, gstep_id_base, max_psnr

    # Put in a function to avoid memory leak
    print('Eval step')
    with torch.no_grad():
        stats_test = {'psnr': 0.0, 'mse': 0.0}
        depth = DOT.get_depth()

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
            if save:
                ckpt_path = path.join(args.train_dir, f'ckpt_best.npz')
                print('Saving best:', ckpt_path)
                player.save(ckpt_path)
        print('eval stats:', stats_test)
    
    return stats_test['psnr']


def update_val_leaves(instant_weights):
    leaves = DOT.tree._all_leaves()
    if args.thresh_type == 'weight':
        val = instant_weights[(*leaves.long().T, )]
    elif args.thresh_type == 'sigma':
        val = DOT.tree.data[(*leaves.long().T, )][..., -1]
    val = torch.nan_to_num(val, nan=0)
    return val, leaves


def prune_func(instant_weights):
    global pre_distance, distance_thred_count
    with torch.no_grad():
        leaves = DOT.tree._all_leaves()
        sel = (*leaves.long().T, )

        if args.thresh_type == 'sigma':
            val = DOT.tree.data[sel][..., -1]
        elif args.thresh_type == 'num_visits':
            assert args.record_tree, 'Pruning by num_visits is only accessible when record_tree option is true.'
        elif args.thresh_type == 'weight':
            val = instant_weights[sel]

        val = torch.nan_to_num(val, nan=0)
        # smoothed =
        if args.thresh_method == 'constant':
            thred = args.thresh_val
        else:
            thred = threshold(val, args.thresh_method,
                              args.thresh_gaussian_sigma)
            # thred = min(thred, args.prune_max)
        if thred is None:
            assert False, 'Threshold is wrong.'
        s1 = val[val > thred]
        s2 = val[val <= thred]

        distance = torch.abs(s1.mean()-s2.mean())/torch.sqrt(s1.var()+s2.var())
        dif = torch.abs(distance-pre_distance)
        pre_distance = distance
        summary_writer.add_scalar(f'train/d_delta', dif, gstep_id)

        if dif >= args.distance_thred:
            distance_thred_count = 0
            return None, None, None
        else:
            distance_thred_count += 1

        if distance_thred_count < args.prune_tolerance:
            return None, None, None

        pre_sel = None
        toltal = 0
        print(f'Prunning at {thred}/{val.max()}')
        while True:
            # smoothed = gaussian(val.cpu().detach().numpy(), sigma=args.thresh_gaussian_sigma)
            sel = leaves[val < thred]
            nids, counts = torch.unique(sel[:, 0], return_counts=True)
            # discover the fronts whose all children are included in sel
            mask = (counts >= int(DOT.tree.N**3*args.thresh_tol)).numpy()

            sel_nids = nids[mask]
            parent_sel = (*DOT.tree._unpack_index(
                DOT.tree.parent_depth[sel_nids, 0]).long().T,)

            if pre_sel is not None:
                if sel_nids.size(0) == 0 or torch.equal(pre_sel, sel_nids):
                    break

            pre_sel = sel_nids
            DOT.merge(sel_nids)
            n = len(sel_nids)*DOT.tree.N ** 3
            toltal += n
            print(f'Prune {n}/{leaves.size(0)}')

            reduced = instant_weights[sel_nids].view(
                -1, DOT.tree.N ** 3).sum(-1)
            instant_weights[parent_sel] = reduced

            val, leaves = update_val_leaves(instant_weights)

        summary_writer.add_scalar(f'train/number_prune', toltal, gstep_id)
        return val, leaves, toltal


def train_step():
    global gstep_id, gstep_id_base, delta_depth, thred, pre_distance, distance_thred_count
    print('Train step')
    stats = {"mse": 0.0, "psnr": 0.0, "invsqr_mse": 0.0}
    pre_delta_mse = 0
    pre_mse = 0
    counter = 0
    batch_size = int(dset.h*dset.w*args.batch_size)
    num_rays = dset.rays.origins.size(0)
    rays_per_batch = (num_rays-1)//batch_size+1
    pre_prune = 0
    pre_distance = 0
    distance_thred_count = 0
    prune_delta = torch.inf
    prune = False

    # stimulate
    while True:
        # important to make the learned properties stable.
        # dset.shuffle_rays()
        if args.rays_with_replacement:
            indexer = torch.randint(dset.rays.origins.size(
                0), (dset.rays.origins.size(0),))
        else:
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

            hessian_mse = hessian_func(gstep_id) * hessian_factor
            sampling_rate = sampling_rate_func(gstep_id) * sampling_factor
            rgb_pred = render.forward(b_rays, cuda=device == 'cuda')
            mse = F.mse_loss(rgb_gt, rgb_pred)
            loss = mse.unsqueeze(0)

            with torch.no_grad():
                if not args.postier_weight:
                    with player.accumulate_weights(op="sum") as accum:
                        rgb_pred = render.forward(
                            b_rays, cuda=device == 'cuda')
                        weight = accum.value
                else:
                    rgb_pred = render.forward(b_rays, cuda=device == 'cuda')
                    dif = rgb_gt-rgb_pred
                    error = args.mse_weights*(dif*dif).sum(-1)
                    weight = DOT.reweight_rays(
                        b_rays, error, render._get_options())
            s1 += weight
            mse = F.mse_loss(rgb_gt, rgb_pred)

            if args.use_sparsity_loss:
                sigma = DOT._sigma()
                # loss_sparsity = torch.abs(mcot.w_sparsity)*torch.abs(1-args.sparse_weight*torch.exp(-sigma)).mean()
                # Cauchy version (from SNeRG)
                loss_sparsity = DOT._w_sparsity * \
                    torch.log(1+2*sigma*sigma).mean()
                loss += loss_sparsity

            if args.use_tv_loss:
                sel = DOT.tree._frontier
                sel = np.random.choice(
                    sel.cpu().numpy(), int(args.tv_thred*sel.size(0)))
                depths = DOT.tree.parent_depth[sel, 1]
                # color = mcot.tree.data[sel][..., :-1]*depths[]
                sigma = DOT.tree.data[sel][..., -1]
                color = rearrange(color, 'n x y z d -> (n d) (x y z)')
                sigma = rearrange(sigma, 'n x y z -> n (x y z)')

                loss_tv = DOT._w_color_tv*torch.std(color, dim=-1).mean() +\
                    DOT._w_sigma_tv*torch.std(sigma, dim=-1).mean()
                loss += loss_tv

            loss.backward()
            optim.step()
            optim.zero_grad()

            mse_num: float = mse.detach().item()
            psnr = -10.0 * math.log10(mse_num)
            if math.isnan(psnr):
                assert False, 'NAN PSNR'
            stats['mse'] += mse_num
            stats['psnr'] += psnr
            stats['invsqr_mse'] += 1.0 / mse_num ** 2

            if args.use_variance:
                pass
                # # weight varibility
                # weights.append(weight)

            # log
            if args.use_sparsity_loss:
                summary_writer.add_scalar(
                    "train/w_sparsity", DOT._w_sparsity, global_step=gstep_id
                )
            if args.use_tv_loss:
                summary_writer.add_scalar(
                    "train/w_tv_color",  DOT._w_color_tv, global_step=gstep_id
                )
                summary_writer.add_scalar(
                    "train/w_tv_sigma", DOT._w_sigma_tv, global_step=gstep_id
                )

        # check if the model gets stable by hessian mse
        delta_mse = stats['mse']-pre_mse
        abs_dmse = np.abs(delta_mse)
        _hessian_mse = np.abs(abs_dmse-pre_delta_mse)
        pre_mse = stats['mse']
        pre_delta_mse = abs_dmse

        VAL = s1

        if not prune:
            val, leaves, prune_num = prune_func(VAL)
            if prune_num is not None:
                prune_delta = np.abs(pre_prune-prune_num)
                pre_prune = prune_num
        # to thrust sampling with refine_numb
            if args.sample_after_prune:
                if args.pruneSampleRepeats == 0:
                    print('Stabalize it...')
                else:
                    print('Sample globally after prunning...')
                DOT.tree.refine(repeats=args.pruneSampleRepeats)

        if val is not None:
            prune = True

        # sigma = mcot._sigma()
        # if sigma.max().isinf():
        #     assert False, 'Inf density.'
        # sigma = torch.nan_to_num(sigma, nan=0)
        # summary_writer.add_histogram(f'train/sigma', sigma, gstep_id)
        summary_writer.add_histogram(f'train/weight', VAL, gstep_id)
        summary_writer.add_scalar(
            f'train/num_nodes', player.n_leaves, gstep_id)
        summary_writer.add_scalar(f'train/depth', DOT.get_depth(), gstep_id)
        lr = get_lr(optim, 'tree')

        summary_writer.add_scalar(
            "train/lr", lr, global_step=gstep_id)
        summary_writer.add_scalar('train/thred_mse', hessian_mse, gstep_id)
        summary_writer.add_scalar(
            'train/sampling_rate', sampling_rate, gstep_id)

        summary_writer.add_scalar('train/hessian_mse', _hessian_mse, gstep_id)

        for stat_name in stats:
            stat_val = stats[stat_name] / rays_per_batch
            summary_writer.add_scalar(f'train/{stat_name}', stat_val, gstep_id)
            stats[stat_name] = 0

        if _hessian_mse <= hessian_mse:
            counter += 1
            if counter > args.hessian_tolerance:
                break
        else:
            counter = 0

        gstep_id_base += rays_per_batch

    if delta_depth != 0 and not args.tune_mode:
        eval_step()
        depth = DOT.get_depth()
        ckpt_path = path.join(args.train_dir, f'ckpt_depth_{depth}.npz')
        print('Saving', ckpt_path)
        player.save(ckpt_path)
        gc.collect()
        pbar.update(delta_depth)
    # threshold
    # the flag used to skip selection and expansion; to give more time to
    # obtain stable properties
    continue_ = True

    with torch.no_grad():
        print('Start sampling...')
        val, leaves = update_val_leaves(VAL)
        sample_k = int(max(1, player.n_leaves*sampling_rate))
        idxs = DOT.select(sample_k, val, leaves)
        interval = idxs.size(0)//args.repeats
        for i in range(1, args.repeats+1):
            start = (i-1)*interval
            end = i*interval
            sel = idxs[start:end]
            continue_ = DOT.expand(sel, i)

    summary_writer.add_scalar(f'train/num_nodes', player.n_leaves, gstep_id)
    summary_writer.add_scalar(f'train/depth', DOT.get_depth(), gstep_id)

    # prune check
    var = prune_delta/DOT.tree.n_leaves
    if prune_delta != 0 and var < args.prune_node_var_thred:
        print(
            f'The tree becomes stable with nodes number variance {var} less than {args.prune_node_var_thred}. Stop optimization.')
        eval_step()
        continue_ = False
    return continue_


with tqdm(total=args.depth_limit) as pbar:
    player = DOT.tree
    depth = DOT.get_depth()
    pbar.update(depth.item())
    render = DOT._volumeRenderer(ndc_conf)
    # render = svox.VolumeRenderer(t, step_size=FLAGS.renderer_step_size, ndc=ndc_config)
    best_validation_psnr = eval_step(save=False)
    print('** initial val psnr ', best_validation_psnr)
    assert 0
    while True:
        epoch_id += 1
        player = DOT.tree
        # render = mcot._volumeRenderer(dset.ndc_coeffs)
        render = DOT._volumeRenderer(ndc_conf)
        depth = player.get_depth()

        flag = train_step()
        delta_depth = (player.get_depth()-depth).item()

        if depth >= args.depth_limit or not flag:
            break
        gc.collect()
