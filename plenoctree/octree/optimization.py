#  Copyright 2021 The PlenOctree Authors.
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#  this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
#  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
#  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
#  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
#  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
#  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
#  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.
"""Optimize a plenoctree through finetuning on train set.

Usage:

export DATA_ROOT=./data/NeRF/nerf_synthetic/
export CKPT_ROOT=./data/PlenOctree/checkpoints/syn_sh16
export SCENE=chair
export CONFIG_FILE=nerf_sh/config/blender

python -m octree.optimization \
    --input $CKPT_ROOT/$SCENE/tree.npz \
    --config $CONFIG_FILE \
    --data_dir $DATA_ROOT/$SCENE/ \
    --output $CKPT_ROOT/$SCENE/octrees/tree_opt.npz
"""
import svox
import torch
import torch.cuda
import numpy as np
import imageio
import os.path as osp
import os
from tqdm import tqdm
from torch.optim import SGD, Adam
from warnings import warn

from absl import app
from absl import flags

from octree.nerf import datasets
from octree.nerf import utils
from DOT.utils import *

from torch.utils.tensorboard import SummaryWriter

FLAGS = flags.FLAGS

utils.define_flags()

flags.DEFINE_string(
    "input",
    "./tree.npz",
    "Input octree npz from extraction.py",
)
flags.DEFINE_string(
    "output",
    "./tree_opt.npz",
    "Output octree npz",
)
flags.DEFINE_integer(
    'render_interval',
    0,
    'render interval')
flags.DEFINE_integer(
    'val_interval',
    2,
    'validation interval')
flags.DEFINE_integer(
    'num_epochs',
    20,
    'epochs to train for')
flags.DEFINE_integer(
    'drop_dot',
    50,
    'epoch to drop DOT')
flags.DEFINE_integer(
    'thred_count',
    2,
    'number for tolerance check'
)
flags.DEFINE_bool(
    'sgd',
    True,
    'use SGD optimizer instead of Adam')
flags.DEFINE_float(
    'lr',
    1e7,
    'optimizer step size')
flags.DEFINE_float(
    'sgd_momentum',
    0.0,
    'sgd momentum')
flags.DEFINE_float(
    "mse_weight",
    1,
    'mse weight for postierior calibration'
)
flags.DEFINE_float(
    "sample_rate",
    0.05,
    'sampling rate in each epoch'
)

flags.DEFINE_float(
    "stable_thred",
    1e-3,
    'check if the structure is stable'
)


flags.DEFINE_bool(
    'sgd_nesterov',
    False,
    'sgd nesterov momentum?')
flags.DEFINE_string(
    "write_vid",
    None,
    "If specified, writes rendered video to given path (*.mp4)",
)

# Manual 'val' set
flags.DEFINE_bool(
    "split_train",
    None,
    "If specified, splits train set instead of loading val set",
)
flags.DEFINE_float(
    "split_holdout_prop",
    0.2,
    "Proportion of images to hold out if split_train is set",
)

# Do not save since it is slow
flags.DEFINE_bool(
    "nosave",
    False,
    "If set, does not save (for speed)",
)

flags.DEFINE_bool(
    "continue_on_decrease",
    True,
    "If set, continues training even if validation PSNR decreases",
)

flags.DEFINE_bool(
    "use_postierior",
    True,
    "If set, the reward will be revised with postierior information"
)



device = "cuda" if torch.cuda.is_available() else "cpu"
torch.autograd.set_detect_anomaly(True)

def main(unused_argv):
    utils.set_random_seed(20200823)
    utils.update_flags(FLAGS)
    summary_writer = SummaryWriter(osp.dirname(FLAGS.output))

    def get_data(stage):
        assert stage in ["train", "val", "test"]
        dataset = datasets.get_dataset(stage, FLAGS)
        focal = dataset.focal
        all_c2w = dataset.camtoworlds
        all_gt = dataset.images.reshape(-1, dataset.h, dataset.w, 3)
        all_c2w = torch.from_numpy(all_c2w).float().to(device)
        all_gt = torch.from_numpy(all_gt).float()
        return focal, all_c2w, all_gt

    focal, train_c2w, train_gt = get_data("train")
    if FLAGS.split_train:
        test_sz = int(train_c2w.size(0) * FLAGS.split_holdout_prop)
        print('Splitting train to train/val manually, holdout', test_sz)
        perm = torch.randperm(train_c2w.size(0))
        test_c2w = train_c2w[perm[:test_sz]]
        test_gt = train_gt[perm[:test_sz]]
        train_c2w = train_c2w[perm[test_sz:]]
        train_gt = train_gt[perm[test_sz:]]
    else:
        print('Using given val set')
        test_focal, test_c2w, test_gt = get_data("val")
        assert focal == test_focal
    H, W = train_gt[0].shape[:2]

    vis_dir = osp.splitext(FLAGS.input)[0] + '_render'
    os.makedirs(vis_dir, exist_ok=True)

    print('N3Tree load')
    t = DOT_N3Tree.load(FLAGS.input, map_location=device)

    if 'llff' in FLAGS.config:
        ndc_config = svox.NDCConfig(width=W, height=H, focal=focal)
    else:
        ndc_config = None
    
    r = svox.VolumeRenderer(t, step_size=FLAGS.renderer_step_size, ndc=ndc_config)

    if FLAGS.sgd:
        print('Using SGD, lr', FLAGS.lr)
        if FLAGS.lr < 1.0:
            warn('For SGD please adjust LR to about 1e7')
        optimizer = SGD(t.parameters(), lr=FLAGS.lr, momentum=FLAGS.sgd_momentum,
                        nesterov=FLAGS.sgd_nesterov)
    else:
        adam_eps = 1e-4 if t.data.dtype is torch.float16 else 1e-8
        print('Using Adam, eps', adam_eps, 'lr', FLAGS.lr)
        optimizer = Adam(t.parameters(), lr=FLAGS.lr, eps=adam_eps)

    n_train_imgs = len(train_c2w)
    n_test_imgs = len(test_c2w)

    def run_test_step(i):
        print('Evaluating')
        with torch.no_grad():
            tpsnr = 0.0
            for j, (c2w, im_gt) in enumerate(zip(test_c2w, test_gt)):
                im = r.render_persp(c2w, height=H, width=W, fx=focal, fast=False)
                im = im.cpu().clamp_(0.0, 1.0)

                mse = ((im - im_gt) ** 2).mean()
                psnr = -10.0 * np.log(mse) / np.log(10.0)
                tpsnr += psnr.item()

                if FLAGS.render_interval > 0 and j % FLAGS.render_interval == 0:
                    vis = torch.cat((im_gt, im), dim=1)
                    vis = (vis * 255).numpy().astype(np.uint8)
                    imageio.imwrite(f"{vis_dir}/{i:04}_{j:04}.png", vis)
            tpsnr /= n_test_imgs
            summary_writer.add_scalar(
                f'test/tpsnr', tpsnr, i)             
            return tpsnr
        
    r = svox.VolumeRenderer(t, step_size=FLAGS.renderer_step_size, ndc=ndc_config)
    best_validation_psnr = run_test_step(0)
    print('** initial val psnr ', best_validation_psnr)
    best_t = None
    pre_mse = 0
    pre_delta_mse = 0
    delta_mse_count = 0
    
    for i in range(FLAGS.num_epochs):
        print('epoch', i)
        tpsnr = 0.0
        s1 = torch.zeros_like(t.child, dtype=t.data.dtype)  # E(x)
        all_mse = np.zeros(1)
        for j, (c2w, im_gt) in tqdm(enumerate(zip(train_c2w, train_gt)), total=n_train_imgs):
            with t.accumulate_weights(op="sum") as accum:
                im = r.render_persp(c2w, height=H, width=W, fx=focal, cuda=True)
            # dif = im-im_gt
            # error = flags.mse_weight*(dif*dif).sum(-1)
            # weight = reweight_rays(b_rays, error, render._get_options())
            with torch.no_grad():
                s1 += accum.value
            im_gt_ten = im_gt.to(device=device)
            im = torch.clamp(im, 0.0, 1.0)
            mse = ((im - im_gt_ten) ** 2).mean()
            im_gt_ten = None

            optimizer.zero_grad()
            t.data.grad = None  # This helps save memory weirdly enough
            mse.backward()
            optimizer.step()
            mse_val = mse.detach().cpu()
            if mse_val < 0:
                assert False, 'Invalid mse'
            all_mse += mse_val.item()
            psnr = -10.0 * np.log(mse_val) / np.log(10.0)
            tpsnr += psnr.item()
        tpsnr /= n_train_imgs
        print('** train_psnr', tpsnr)
        summary_writer.add_scalar(
            f'train/train_psnr', tpsnr, i) 

        delta_mse = all_mse-pre_mse
        abs_dmse = np.abs(delta_mse)
        _hessian_mse = np.abs(abs_dmse-pre_delta_mse)
        summary_writer.add_scalar(
            f'train/hessian_mse', _hessian_mse, i
        )
        pre_mse = all_mse
        if _hessian_mse <= FLAGS.stable_thred:
            delta_mse_count += 1
        else:
            delta_mse_count = 0
            
       
        # if i % FLAGS.val_interval == FLAGS.val_interval - 1 or i == FLAGS.num_epochs - 1:
        if i==2:
            validation_psnr = run_test_step(i + 1)
            print('** val psnr ', validation_psnr, 'best', best_validation_psnr)
            # if validation_psnr > best_validation_psnr:
            best_validation_psnr = validation_psnr
            best_t = t.clone(device='cpu')  # SVOX 0.2.22
            best_t.save(FLAGS.output, compress=False)
            print('')
            assert 0
            # elif not FLAGS.continue_on_decrease:
            #     print('Stop since overfitting')
            #     break            
        
        s1 = prune_func(t, s1, summary_writer=summary_writer, gstep_id=i)
        # sample_func(t, FLAGS.sample_rate, s1)   
        
        if delta_mse_count >= FLAGS.thred_count:
            delta_mse_count = 0
        
        summary_writer.add_scalar(
            f'train/num_nodes', t.n_leaves, i)     
        summary_writer.add_scalar(
            f'train/lr', get_lr(optimizer), i)
        

  
          
    if not FLAGS.nosave:
        if best_t is not None:
            print('Saving best model to', FLAGS.output)
            best_t.save(FLAGS.output, compress=False)
        else:
            print('Did not improve upon initial model')

if __name__ == "__main__":
    app.run(main)
