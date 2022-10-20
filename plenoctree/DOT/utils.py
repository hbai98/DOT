from svox.svox import _get_c_extension
from svox.renderer import _rays_spec_from_rays
import torch
from skimage.filters.thresholding import threshold_li, threshold_otsu, threshold_yen, threshold_minimum, threshold_triangle
from skimage.filters._gaussian import gaussian


_C = _get_c_extension()

def reweight_rays(tree, rays, error, opt):
    assert error.size(0) == rays.origins.size(0)
    assert error.is_cuda 
    tree._weight_accum = None
    with tree.accumulate_weights(op="sum") as accum:
        _C.reweight_rays(tree._spec(), _rays_spec_from_rays(rays), opt, error)
    return accum.value 

def prune_func(DOT, instant_weights, 
               distance_thred_count,
               pre_distance,
               thresh_type='weight', 
               thresh_method='triangle',
               thresh_val=0.25,
               thresh_gaussian_sigma=3,
               distance_thred=0.01,
               prune_tolerance=3,
               thresh_tol=0.8,
               summary_writer=None,
               gstep_id = None
               ):
    non_writer = summary_writer is None
    if not non_writer:
        assert gstep_id is not None
    
    with torch.no_grad():
        leaves = DOT._all_leaves()
        sel = (*leaves.long().T, )

        if thresh_type == 'sigma':
            val = DOT.data[sel][..., -1]
        elif thresh_type == 'weight':
            val = instant_weights[sel]

        val = torch.nan_to_num(val, nan=0)
        
        if thresh_method == 'constant':
            thred = thresh_val
        else:
            thred = threshold(val, thresh_method, thresh_gaussian_sigma)
            # thred = min(thred, args.prune_max)
        if thred is None:
            assert False, 'Threshold is wrong.'
        s1 = val[val > thred]
        s2 = val[val <= thred]

        distance = torch.abs(s1.mean()-s2.mean())/torch.sqrt(s1.var()+s2.var())
        dif = torch.abs(distance-pre_distance)
        pre_distance = distance
        
        if not non_writer:
            summary_writer.add_scalar(f'train/d_delta', dif, gstep_id)

        if dif >= distance_thred:
            distance_thred_count = 0
            return distance_thred_count, pre_distance
        else:
            distance_thred_count += 1

        if distance_thred_count < prune_tolerance:
           return distance_thred_count, pre_distance

        pre_sel = None
        toltal = 0
        print(f'Prunning at {thred}/{val.max()}')
        while True:
            # smoothed = gaussian(val.cpu().detach().numpy(), sigma=args.thresh_gaussian_sigma)
            sel = leaves[val < thred]
            nids, counts = torch.unique(sel[:, 0], return_counts=True)
            # discover the fronts whose all children are included in sel
            mask = (counts >= int(DOT.N**3*thresh_tol)).numpy()

            sel_nids = nids[mask]
            parent_sel = (*DOT._unpack_index(
                DOT.parent_depth[sel_nids, 0]).long().T,)

            if pre_sel is not None:
                if sel_nids.size(0) == 0 or torch.equal(pre_sel, sel_nids):
                    break

            pre_sel = sel_nids
            DOT.merge(sel_nids)
            DOT.shrink_to_fit()
            n = len(sel_nids)*DOT.N ** 3
            toltal += n
            # print(f'Prune {n}/{leaves.size(0)}')

            reduced = instant_weights[sel_nids].view(
                -1, DOT.N ** 3).sum(-1)
            instant_weights[parent_sel] = reduced

            val, leaves = update_val_leaves(instant_weights)
        print(f'Purne {toltal} nodes in toltal.')
        if not non_writer:
            summary_writer.add_scalar(f'train/number_prune', toltal, gstep_id)
        return distance_thred_count, pre_distance
    
def update_val_leaves(DOT, instant_weights, thresh_type='weight'):
    leaves = DOT._all_leaves()
    if thresh_type == 'weight':
        val = instant_weights[(*leaves.long().T, )]
    elif thresh_type == 'sigma':
        val = DOT.data[(*leaves.long().T, )][..., -1]
    val = torch.nan_to_num(val, nan=0)
    return val, leaves

def sample_func(tree, sampling_rate, VAL, repeats=1):
    with torch.no_grad():
        val, leaves = update_val_leaves(tree, VAL)
        sample_k = int(max(1, tree.n_leaves*sampling_rate))
        print(f'Start sampling {sample_k} nodes.')
        idxs = select(tree, sample_k, val, leaves)
        interval = idxs.size(0)//repeats
        for i in range(1, repeats+1):
            start = (i-1)*interval
            end = i*interval
            sel = idxs[start:end]
            continue_ = expand(tree, sel, i)
            if not continue_:
                return False
    return True 

def expand(tree, idxs, repeats):
    # group expansion
    i = idxs.size(0)
    idxs = (*idxs.long().T,)
    res = tree.refine(sel=idxs, repeats=repeats)
    return res
        
def select(t, max_sel, reward, rw_idxs):
    p_val = reward  # only MSE
    sel = min(p_val.size(0), max_sel)
    _, idxs = torch.topk(p_val, sel)
    idxs = rw_idxs[idxs]
    return idxs

def threshold(data, method, sigma=3):
    device = data.device
    data = gaussian(data.cpu().detach().numpy(), sigma=sigma)
    if method == 'li':
        return torch.tensor(threshold_li(data), device=device)
    elif method == 'otsu':
        return torch.tensor(threshold_otsu(data), device=device)
    elif method == 'yen':
        return torch.tensor(threshold_yen(data), device=device)
    elif method == 'minimum':
        return torch.tensor(threshold_minimum(data), device=device)
    elif method == 'triangle':
        return torch.tensor(threshold_triangle(data), device=device)
    else:
        assert False, f'the method {method} is not implemented.'


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']