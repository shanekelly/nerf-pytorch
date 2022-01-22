import os
import imageio

from pathlib import Path
from pickle import dump
from time import time, perf_counter
from typing import Callable, Dict, Optional

import numpy as np
import torch

from ipdb import launch_ipdb_on_exception, set_trace
from torch.nn.functional import relu
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from run_nerf_helpers import (sample_section_rays, get_all_rays, get_embedder, get_rays,
                              img2mse, load_data, mse2psnr, NeRF, ndc_rays, sample_pdf, to8b)


cpu = torch.device('cpu')
gpu_if_available = torch.device('cuda') if torch.cuda.is_available() else cpu
np.random.seed(0)
DEBUG = False


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """

    if chunk is None:
        return fn

    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)

    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:, None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])

    return outputs


def batchify_rays(rays_flat: torch.Tensor, chunk: int = 1024*32, **kwargs) -> Dict:
    """
    @brief - Render rays in smaller minibatches to avoid OOM.
    @param rays_flat: Shape (N, 8). [rays_o (Nx3), rays_d (Nx3), near (Nx1), far (Nx1)]
    @param chunk - Maximum number of rays to process simultaneously.
    """
    all_ret = {}

    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)

        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    # sk: Merge outputs from batches.
    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}

    return all_ret


def render(H: int, W: int, K: torch.Tensor, chunk: int = 1024*32,
           rays: Optional[torch.Tensor] = None, c2w: Optional[torch.Tensor] = None,
           ndc: bool = True, near: float = 0., far: float = 1.,
           use_viewdirs: bool = False, c2w_staticcam: Optional[torch.Tensor] = None,
           **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      K: array of shape [3, 3]. Camera intrinsics matrix.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, K, c2w, gpu_if_available)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d

        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam, gpu_if_available)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

    sh = rays_d.shape  # [N, 3]

    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()

    near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(rays_d[..., :1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)  # shape: (N, 8)

    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)

    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}

    return ret_list + [ret_dict]


def render_path(render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None,
                render_factor=0):

    H, W, focal = hwf

    if render_factor != 0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    disps = []

    t = time()

    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time() - t)
        t = time()
        rgb, disp, acc, _ = render(H, W, K, chunk=chunk, c2w=c2w[:3, :4], **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())

        if i == 0:
            print(rgb.shape, disp.shape)

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    # sk: If savedir is specified, then save additional debug information inside savedir.
    if savedir is not None:
        output_dir = Path(savedir)

        rgbs_file = open(output_dir / 'rgbs.pickle', 'wb')
        dump(rgbs, rgbs_file)
        rgbs_file.close()

        disps_file = open(output_dir / 'disps.pickle', 'wb')
        dump(disps, disps_file)
        disps_file.close()

        render_poses_file = open(output_dir / 'render_poses.pickle', 'wb')
        dump(render_poses.cpu().numpy(), render_poses_file)
        render_poses_file.close()

    return rgbs, disps


def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)

    input_ch_views = 0
    embeddirs_fn = None

    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(gpu_if_available)
    grad_vars = list(model.parameters())

    model_fine = None

    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views,
                          use_viewdirs=args.use_viewdirs).to(gpu_if_available)
        grad_vars += list(model_fine.parameters())

    def network_query_fn(inputs, viewdirs, network_fn):
        return run_network(inputs, viewdirs, network_fn, embed_fn=embed_fn,
                           embeddirs_fn=embeddirs_fn, netchunk=args.netchunk)

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start_iter_idx = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints

    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(
            os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    print('Found ckpts', ckpts)

    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start_iter_idx = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])

        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    ##########################

    render_kwargs_train = {
        'network_query_fn': network_query_fn,
        'perturb': args.perturb,
        'N_importance': args.N_importance,
        'network_fine': model_fine,
        'N_samples': args.N_samples,
        'network_fn': model,
        'use_viewdirs': args.use_viewdirs,
        'white_bkgd': args.white_bkgd,
        'raw_noise_std': args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data

    if args.dataset_type != 'llff' or args.no_ndc:
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start_iter_idx, grad_vars, optimizer


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    def raw2alpha(raw, dists, act_fn=relu): return 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.tensor([1e10], device=gpu_if_available).expand(
        dists[..., :1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
    noise = 0.

    if raw_noise_std > 0.:
        noise = torch.randn(raw[..., 3].shape, device=gpu_if_available) * raw_noise_std

        # Overwrite randomly sampled data if pytest

        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[..., 3].shape)) * raw_noise_std
            noise = torch.tensor(noise, device=gpu_if_available)

    alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * \
        torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=gpu_if_available),
                      1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[..., None])

    return rgb_map, disp_map, acc_map, weights, depth_map


def render_rays(ray_batch: torch.Tensor, network_fn: NeRF, network_query_fn: Callable,
                N_samples: int, retraw: bool = False, lindisp: bool = False, perturb: float = 0.,
                N_importance: int = 0, network_fine: Optional[NeRF] = None,
                white_bkgd: bool = False, raw_noise_std: float = 0., verbose: bool = False,
                pytest: bool = False
                ) -> Dict:
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each
    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]

    t_vals = torch.linspace(0., 1., steps=N_samples, device=gpu_if_available)

    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape, device=gpu_if_available)

        # Pytest, overwrite u with numpy's fixed random numbers

        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.tensor(t_rand, device=gpu_if_available)

        z_vals = lower + (upper - lower) * t_rand

    # sk: Points to query the volume at.
    pts = rays_o[..., None, :] + rays_d[..., None, :] * \
        z_vals[..., :, None]  # [N_rays, N_samples, 3]


#     raw = run_network(pts)
    raw = network_query_fn(pts, viewdirs, network_fn)  # Shape: (N, N_samples ** 2, 5)
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
        raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    if N_importance > 0:

        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1],
                               N_importance, gpu_if_available, det=(perturb == 0.), pytest=pytest)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * \
            z_vals[..., :, None]  # [N_rays, N_samples + N_importance, 3]

        run_fn = network_fn if network_fine is None else network_fine
#         raw = run_network(pts, fn=run_fn)
        raw = network_query_fn(pts, viewdirs, run_fn)

        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
            raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    ret = {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map}

    if retraw:
        ret['raw'] = raw

    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/',
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern',
                        help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8,
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256,
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250,
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*32,
                        help='number of rays processed in parallel, decrease if running out of '
                        'memory')
    parser.add_argument("--netchunk", type=int, default=1024*64,
                        help='number of pts sent through network in parallel, decrease if running '
                        'out of memory')
    parser.add_argument("--no_batching", action='store_true',
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true',
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0,
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true',
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast '
                        'preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff',
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8,
                        help='will load 1/N images from test/val sets, useful for large datasets '
                        'like deepvoxels')

    # deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek',
                        help='options : armchair / cube / greek / vase')

    # blender flags
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for '
                        'dvoxels)')
    parser.add_argument("--half_res", action='store_true',
                        help='load blender synthetic data at 400x400 instead of 800x800')

    # llff flags
    parser.add_argument("--factor", type=int, default=8,
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true',
                        help='do not use normalized device coordinates (set for non-forward facing '
                        'scenes)')
    parser.add_argument("--lindisp", action='store_true',
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true',
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8,
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_scalars",   type=int, default=100,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000,
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=50000,
                        help='frequency of render_poses video saving')

    # sk: My options.
    parser.add_argument('--img_grid_side_len',   type=int, default=8,
                        help='The side length of the grid used to split up training images for '
                        'image active sampling.')
    parser.add_argument('--i_sampling_vis', type=int, default=500, help='The frequency of '
                        'logging visualizations about ray sampling to tensorboard.')
    parser.add_argument('--verbose', action='store_true', help='True to print additional info.')
    parser.add_argument('--no_active_sampling', action='store_true', help='Set to disable active '
                        'sampling.')
    parser.add_argument('--n_training_iters', type=int, default=50000, help='The number of '
                        'training iterations to run for.')

    return parser


def train() -> None:
    parser = config_parser()
    args = parser.parse_args()

    tensorboard = SummaryWriter(log_dir=Path(args.basedir) / args.expname, flush_secs=10)

    # Load data from specified dataset.
    images, hwf, H, W, focal, K, poses, render_poses, train_idxs, test_idxs, val_idxs, near, far = \
        load_data(args, gpu_if_available)

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    args_file = os.path.join(basedir, expname, 'args.txt')
    with open(args_file, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))

    if args.config is not None:
        config_file = os.path.join(basedir, expname, 'config.txt')
        with open(config_file, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start_iter_idx, grad_vars, optimizer = create_nerf(
        args)
    global_step = start_iter_idx

    bds_dict = {
        'near': near,
        'far': far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Short circuit if only rendering out from trained model

    if args.render_only:
        print('RENDER ONLY')
        with torch.no_grad():
            if args.render_test:
                # render_test switches to test poses
                images = images[test_idxs]
            else:
                # Default is smoother render_poses path
                images = None

            testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format(
                'test' if args.render_test else 'path', start_iter_idx))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            rgbs, _ = \
                render_path(render_poses, hwf, K, args.chunk, render_kwargs_test, gt_imgs=images,
                            savedir=testsavedir, render_factor=args.render_factor)
            print('Done rendering', testsavedir)
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

            return

    # Parse command line arguments.
    N_rand = args.N_rand
    use_batching = not args.no_batching
    assert use_batching is True
    verbose = args.verbose
    n_train_imgs = len(train_idxs)
    grid_size = args.img_grid_side_len
    do_active_sampling = not args.no_active_sampling

    assert H % grid_size == 0
    assert W % grid_size == 0
    section_height = H // grid_size
    section_width = W // grid_size
    n_pixels_per_section = section_height * section_width

    n_sections_per_img = grid_size ** 2
    if do_active_sampling:
        n_rays_to_uniformly_sample_per_img = 200
        n_total_rays_to_uniformly_sample = n_rays_to_uniformly_sample_per_img * n_train_imgs
        min_n_rays_to_uniformly_sample_per_section = \
            n_rays_to_uniformly_sample_per_img // n_sections_per_img
        remaining_n_rays_to_uniformly_sample = \
            n_rays_to_uniformly_sample_per_img % n_sections_per_img
        section_uniform_sampling_prob_dist = torch.full((n_train_imgs, grid_size, grid_size),
                                                        1 / (n_train_imgs * grid_size * grid_size))

        n_total_rays_to_actively_sample = N_rand
        n_rays_to_actively_sample_per_img = n_total_rays_to_actively_sample // n_train_imgs
    else:
        n_total_rays_to_sample = N_rand
        n_rays_to_sample_per_img = n_total_rays_to_sample // n_train_imgs
        n_rays_to_sample_per_section = n_rays_to_sample_per_img // n_sections_per_img

    sw_total_n_sampled = torch.zeros(n_train_imgs, grid_size, grid_size, dtype=torch.int64)

    # Get all rays from all training images.
    section_rays, _ = get_all_rays(images[train_idxs], H, W, K, poses[train_idxs], n_train_imgs,
                                   grid_size, section_height, section_width, gpu_if_available,
                                   verbose=True)

    if do_active_sampling:
        sw_uniform_sampling_prob_dist = \
            torch.tensor(1 / n_total_rays_to_uniformly_sample).repeat(n_train_imgs,
                                                                      grid_size, grid_size)
    else:
        section_n_rays_to_sample = torch.broadcast_to(torch.tensor(n_rays_to_sample_per_section,
                                                                   dtype=int),
                                                      (n_train_imgs, grid_size, grid_size))

    start_iter_idx += 1
    n_training_iters = args.n_training_iters + 1
    tqdm_bar = trange(start_iter_idx, n_training_iters)
    for train_iter_idx in tqdm_bar:
        t_train_iter_start = perf_counter()

        log_sampling_vis = train_iter_idx % args.i_sampling_vis == 0

        if do_active_sampling:
            # Uniform ray sampling.
            (uniformly_sampled_rays, uniformly_sampled_pw_idxs, t_uniform_sampling) = \
                sample_section_rays(section_rays, sw_uniform_sampling_prob_dist,
                                    n_total_rays_to_uniformly_sample,
                                    n_train_imgs, H, W, grid_size, section_height, section_width,
                                    tensorboard, 'train/uniform_sampling/sampled_pixels', train_iter_idx,
                                    log_sampling_vis=log_sampling_vis, verbose=verbose)

            # Rendering uniformly sampled rays.
            t_uniform_batching_start = perf_counter()
            batch = torch.transpose(uniformly_sampled_rays, 0, 1).to(gpu_if_available)
            batch_rays, gt_rgbs = batch[:2], batch[2]
            t_uniform_batching = perf_counter() - t_uniform_batching_start

            t_uniform_rendering_start = perf_counter()
            with torch.no_grad():
                rendered_rgbs, _, _, _ = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                                verbose=train_iter_idx < 10, retraw=True,
                                                **render_kwargs_train)
            t_uniform_rendering = perf_counter() - t_uniform_rendering_start

            # Computing section-wise loss for uniformly sampled rays.
            t_uniform_loss_start = perf_counter()
            sw_n_rays_uniformly_sampled = torch.zeros((n_train_imgs, grid_size, grid_size))
            sw_cumu_squared_diff = torch.zeros((n_train_imgs, grid_size, grid_size))
            rendered_rgbs = rendered_rgbs.to(cpu)
            gt_rgbs = gt_rgbs.to(cpu)
            for rendered_rgb, gt_rgb, sampled_pw_idx in zip(rendered_rgbs, gt_rgbs,
                                                            uniformly_sampled_pw_idxs):
                img_idx, grid_row_idx, grid_col_idx, _, _ = sampled_pw_idx
                sw_cumu_squared_diff[img_idx, grid_row_idx, grid_col_idx] += \
                    torch.mean((rendered_rgb - gt_rgb) ** 2)
                sw_n_rays_uniformly_sampled[img_idx, grid_row_idx, grid_col_idx] += 1
            sw_loss = torch.nan_to_num(sw_cumu_squared_diff / sw_n_rays_uniformly_sampled)
            t_uniform_loss = perf_counter() - t_uniform_loss_start

            # Active ray sampling.
            sw_active_sampling_prob_dist = sw_loss / torch.sum(sw_loss)
            (actively_sampled_rays, actively_sampled_pw_idxs, t_active_sampling) = \
                sample_section_rays(section_rays, sw_active_sampling_prob_dist,
                                    n_total_rays_to_actively_sample,
                                    n_train_imgs, H, W, grid_size, section_height, section_width,
                                    tensorboard, 'train/active_sampling/sampled_pixels', train_iter_idx,
                                    log_sampling_vis=log_sampling_vis, verbose=verbose)

            sampled_rays = actively_sampled_rays
            sampled_pw_idxs = actively_sampled_pw_idxs
        else:
            (sampled_rays, sampled_pw_idxs, t_sampling) = \
                sample_section_rays(section_rays, section_n_rays_to_sample, n_total_rays_to_sample,
                                    n_train_imgs, H, W, grid_size, section_height, section_width,
                                    tensorboard, 'train/sampling/sampled_pixels', train_iter_idx,
                                    log_sampling_vis=log_sampling_vis, verbose=verbose)

        # Accumulate the number of times each section has been sampled from.
        unique_sw_idxs, sw_n_sampled = \
            torch.unique(sampled_pw_idxs[:, :3], dim=0, return_counts=True)
        sw_total_n_sampled[unique_sw_idxs[:, 0], unique_sw_idxs[:, 1], unique_sw_idxs[:, 2]] += \
            sw_n_sampled

        t_batching_start = perf_counter()
        batch = torch.transpose(sampled_rays.to(gpu_if_available), 0, 1)
        batch_rays, target_s = batch[:2], batch[2]
        t_batching = perf_counter() - t_batching_start

        # Core optimization loop! #
        t_rendering_start = perf_counter()
        rgb, disp, acc, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                        verbose=train_iter_idx < 10, retraw=True,
                                        **render_kwargs_train)
        t_rendering = perf_counter() - t_rendering_start

        t_loss_start = perf_counter()
        optimizer.zero_grad()
        img_loss = img2mse(rgb, target_s)
        loss = img_loss
        psnr = mse2psnr(img_loss)
        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss += img_loss0
            # psnr0 = mse2psnr(img_loss0)
        t_loss = perf_counter() - t_loss_start

        t_backprop_start = perf_counter()
        loss.backward()
        optimizer.step()
        t_backprop = perf_counter() - t_backprop_start

        # NOTE: IMPORTANT!
        # Update learning rate.
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))

        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate

        # Rest is logging.

        if train_iter_idx % args.i_weights == 0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(train_iter_idx))
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        if train_iter_idx % args.i_video == 0 and train_iter_idx > 0:
            # Turn on testing mode
            with torch.no_grad():
                rgbs, disps = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname,
                                                                                  train_iter_idx))
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

        if train_iter_idx % args.i_testset == 0 and train_iter_idx > 0:
            testsavedir = os.path.join(basedir, expname, f'testset_{train_iter_idx:06d}')
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', poses[test_idxs].shape)
            with torch.no_grad():
                render_path(
                    torch.tensor(poses[test_idxs], device=gpu_if_available), hwf, K, args.chunk,
                    render_kwargs_test,
                    gt_imgs=images[test_idxs], savedir=testsavedir)
            print('Saved test set')

        if train_iter_idx % args.i_scalars == 0:
            tensorboard.add_scalar('train/loss', loss, train_iter_idx)
            tensorboard.add_scalar('train/psnr', psnr, train_iter_idx)

        if log_sampling_vis:
            sw_total_n_sampled_scaled = sw_total_n_sampled / torch.max(sw_total_n_sampled)
            sw_total_n_sampled_imgs = torch.ones((n_train_imgs, grid_size, grid_size, 3))
            sw_total_n_sampled_imgs[:, :, :, 1] -= sw_total_n_sampled_scaled
            sw_total_n_sampled_imgs[:, :, :, 2] -= sw_total_n_sampled_scaled

            # Add white border around each image.
            padding_color = torch.tensor([0.6]).expand(3)
            sw_total_n_sampled_imgs = torch.cat((
                padding_color.expand(n_train_imgs, 1, grid_size, 3),
                sw_total_n_sampled_imgs,
                padding_color.expand(n_train_imgs, 1, grid_size, 3)),
                dim=1)
            sw_total_n_sampled_imgs = torch.cat((
                padding_color.expand(n_train_imgs, grid_size + 2, 1, 3),
                sw_total_n_sampled_imgs,
                padding_color.expand(n_train_imgs, grid_size + 2, 1, 3)),
                dim=2)

            sampling_name = 'active_sampling' if do_active_sampling else 'sampling'
            tensorboard.add_images(f'train/{sampling_name}/cumulative_samples_per_section',
                                   sw_total_n_sampled_imgs, train_iter_idx, dataformats='NHWC')

        if train_iter_idx % args.i_img == 0:
            # Log a rendered validation view to Tensorboard
            # img_i = np.random.choice(val_idxs)
            img_i = val_idxs[-2] if len(val_idxs) > 1 else val_idxs[0]
            target = images[img_i]
            c2w = poses[img_i, :3, :4].to(gpu_if_available)
            with torch.no_grad():
                rgb, disp, acc, extras = render(H, W, K, chunk=args.chunk, c2w=c2w,
                                                **render_kwargs_test)

            psnr = mse2psnr(img2mse(rgb, target))

            tensorboard.add_image('validation/rgb/estimate', rgb,
                                  train_iter_idx, dataformats='HWC')
            if train_iter_idx == args.i_img:
                tensorboard.add_image('validation/rgb/groundtruth', target,
                                      train_iter_idx, dataformats='HWC')
            tensorboard.add_image('validation/disp/estimate', disp,
                                  train_iter_idx, dataformats='HW')
            tensorboard.add_scalar('validation/psnr', psnr, train_iter_idx)

        t_train_iter = perf_counter() - t_train_iter_start

        if do_active_sampling:
            tqdm_bar.set_postfix_str(
                f't{t_train_iter:.3f},'
                f'u:(s{t_uniform_sampling:.3f},b{t_uniform_batching:.3f},'
                f'l{t_uniform_loss:.3f},r{t_uniform_rendering:.3f}),'
                f'a:(s{t_active_sampling:.3f},b{t_batching:.3f},r{t_rendering:.3f}),'
                f'b{t_backprop:.3f}')
        else:
            tqdm_bar.set_postfix_str(
                f't{t_train_iter:.3f},s{t_sampling:.3f},b{t_batching:.3f},r{t_rendering:.3f},'
                f'b{t_backprop:.3f}')

        global_step += 1


if __name__ == '__main__':
    with launch_ipdb_on_exception():
        train()
