import os
import imageio

from pathlib import Path
from pickle import dump
from time import time, perf_counter
from typing import Callable, Dict, Optional

import numpy as np
import open3d as o3d
import torch

from ipdb import launch_ipdb_on_exception, set_trace
from open3d.visualization.tensorboard_plugin import summary
from open3d.visualization.tensorboard_plugin.util import to_dict_batch
from torch.nn import Parameter
from torch.nn.functional import relu
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from run_nerf_helpers import (add_1d_imgs_to_tensorboard, get_coordinate_frames, get_idxs_tuple,
                              get_sw_n_sampled, get_sw_rays, get_embedder, get_rays, get_sw_loss,
                              img2mse, load_data, mse2psnr, NeRF, ndc_rays, pad_imgs, sample_pdf,
                              sample_sw_rays, to8b, tfmats_from_minreps, minreps_from_tfmats)


cpu = torch.device('cpu')
if torch.cuda.is_available():
    gpu_if_available = torch.device('cuda')
    print('GPU is available!')
else:
    gpu_if_available = cpu
    print('GPU not available!')
np.random.seed(0)
DEBUG = False

white_rgb = torch.tensor([1.0]).expand(3)


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
        # print(i, time() - t)
        t = time()
        rgb, disp, acc, _ = render(H, W, K, chunk=chunk, c2w=c2w[:3, :4], **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())

        # if i == 0:
        #     print(rgb.shape, disp.shape)

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


def create_nerf(args, initial_poses: torch.Tensor):
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

    if args.no_pose_optimization:
        train_poses_params = None
    else:
        # Convert the initial pose transformation matrices into 6-element minimal representations,
        # then add them to grad_vars, which will get passed to the optimizer.
        train_poses_params = Parameter(minreps_from_tfmats(initial_poses, gpu_if_available))
        with torch.no_grad():
            grad_vars.extend([train_poses_params])

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

    return (render_kwargs_train, render_kwargs_test, start_iter_idx, grad_vars, optimizer,
            train_poses_params)


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
    parser.add_argument("--i_train_scalars",   type=int, default=100,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_val",     type=int, default=500,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_test", type=int, default=50000,
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=50000,
                        help='frequency of render_poses video saving')

    # sk: My options.
    parser.add_argument('--img_grid_side_len',   type=int, default=8,
                        help='The side length of the grid used to split up training images for '
                        'image active sampling.')
    parser.add_argument('--n_training_iters', type=int, default=50000, help='The number of '
                        'training iterations to run for.')

    parser.add_argument('--i_sampling_vis', type=int, default=500, help='The frequency of '
                        'logging visualizations about ray sampling to tensorboard.')
    parser.add_argument('--i_pose_vis', type=int, default=500, help='The frequency of '
                        'logging visualizations about the camera frame poses to tensorboard.')

    parser.add_argument('--verbose', action='store_true', help='True to print additional info.')
    parser.add_argument('--no_active_sampling', action='store_true', help='Set to disable active '
                        'sampling.')
    parser.add_argument('--uniformly_sample_every_iteration', action='store_true', help='Set to '
                        'uniformly sampled over every image on every iteration in order to '
                        'maintain the estimated probability distribution of loss over image. If '
                        'not set, then all images will be uniformly sampled once at the start, '
                        'but then only active samples will update the estimated probability '
                        'distribution of loss over images.')
    parser.add_argument('--no_pose_optimization', action='store_true', help='Set to make initial '
                        'poses static and disable learning refined poses.')

    return parser


def train() -> None:
    parser = config_parser()
    args = parser.parse_args()

    tensorboard = SummaryWriter(Path(args.basedir) / args.expname, flush_secs=10)

    # Load data from specified dataset.
    (rgb_imgs, depth_imgs, hwf, H, W, focal, K, poses, render_poses, train_idxs, test_idxs,
     val_idxs, near, far) = load_data(args, gpu_if_available)
    train_rgb_imgs = rgb_imgs[train_idxs]
    test_rgb_imgs = rgb_imgs[test_idxs]
    initial_train_poses = poses[train_idxs]
    test_poses = poses[test_idxs]

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
    (render_kwargs_train, render_kwargs_test, start_iter_idx, grad_vars, optimizer,
     train_poses_params) = create_nerf(args, initial_train_poses)
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
                rgb_imgs = rgb_imgs[test_idxs]
            else:
                # Default is smoother render_poses path
                rgb_imgs = None

            testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format(
                'test' if args.render_test else 'path', start_iter_idx))
            os.makedirs(testsavedir, exist_ok=True)
            print('test poses shape', render_poses.shape)

            rgbs, _ = \
                render_path(render_poses, hwf, K, args.chunk, render_kwargs_test, gt_imgs=rgb_imgs,
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
    do_uniform_sampling_every_iter = args.uniformly_sample_every_iteration
    do_pose_optimization = not args.no_pose_optimization

    assert H % grid_size == 0
    assert W % grid_size == 0
    section_height = H // grid_size
    section_width = W // grid_size

    n_sections_per_img = grid_size ** 2
    n_total_sections = n_train_imgs * n_sections_per_img
    dims_sw = (n_train_imgs, grid_size, grid_size)
    if do_active_sampling:
        n_rays_to_uniformly_sample_per_img = 200
        n_total_rays_to_uniformly_sample = n_rays_to_uniformly_sample_per_img * n_train_imgs
        sw_uniform_sampling_prob_dist = torch.tensor(1 / n_total_sections).repeat(dims_sw)
        sw_train_loss = torch.zeros(dims_sw)
        n_total_rays_to_actively_sample = N_rand
    else:
        n_total_rays_to_sample = N_rand
        sw_sampling_prob_dist = \
            torch.tensor(1 / n_total_sections).expand(dims_sw)

    sw_total_n_sampled = torch.zeros(dims_sw, dtype=torch.int64)

    # Get all rays from all training images.
    train_poses = initial_train_poses.clone()

    if not do_pose_optimization:
        # Poses are not being optimized, so we only need to compute the rays once since they will
        # stay the same throughout all training iterations.
        sw_rays, t_get_rays = \
            get_sw_rays(train_rgb_imgs, H, W, K, train_poses, n_train_imgs,
                        grid_size, section_height, section_width, gpu_if_available)

    start_iter_idx += 1
    n_training_iters = args.n_training_iters + 1
    tqdm_bar = trange(start_iter_idx, n_training_iters)
    for train_iter_idx in tqdm_bar:
        t_train_iter_start = perf_counter()

        log_sampling_vis = train_iter_idx % args.i_sampling_vis == 0
        log_pose_vis = train_iter_idx % args.i_pose_vis == 0
        is_first_iter = train_iter_idx == start_iter_idx

        if not do_pose_optimization and not is_first_iter:
            t_get_rays = 0.0
        if do_pose_optimization:
            # Poses are being optimized, so we need to compute the rays on every training iteration
            # since they will change as the poses change.

            # Unpack the minimal representations of the poses from the optimizer's parameters, then
            # convert them into 4x4 transformation matrices.
            train_poses = tfmats_from_minreps(train_poses_params, initial_train_poses[0])
            # Compute the rays from the optimized poses.
            sw_rays, t_get_rays = \
                get_sw_rays(train_rgb_imgs, H, W, K, train_poses, n_train_imgs,
                            grid_size, section_height, section_width, gpu_if_available)

        if do_active_sampling:
            # If we aren't uniformly sampling on every iteration, then we should only uniformly
            # sample on the first iteration of this run.
            t_uniform_sampling = 0.0
            t_uniform_batching = 0.0
            t_uniform_loss = 0.0
            t_uniform_rendering = 0.0
            if do_uniform_sampling_every_iter or is_first_iter:
                # Uniform ray sampling.
                enforce_min_samples = False if do_uniform_sampling_every_iter else True
                with torch.no_grad():
                    (uniformly_sampled_rays, uniformly_sampled_pw_idxs, t_uniform_sampling) = \
                        sample_sw_rays(sw_rays, sw_uniform_sampling_prob_dist,
                                       n_total_rays_to_uniformly_sample,
                                       n_train_imgs, H, W, grid_size, section_height, section_width,
                                       tensorboard, 'train/uniform_sampling/sampled_pixels', train_iter_idx,
                                       log_sampling_vis=log_sampling_vis, verbose=verbose,
                                       enforce_min_samples=enforce_min_samples)

                # Render the uniformly sampled rays.
                t_uniform_batching_start = perf_counter()
                batch = torch.transpose(uniformly_sampled_rays, 0, 1).to(gpu_if_available)
                batch_rays, unif_gt_rgbs = batch[:2], batch[2]
                t_uniform_batching = perf_counter() - t_uniform_batching_start

                t_uniform_rendering_start = perf_counter()
                with torch.no_grad():
                    unif_rendered_rgbs, _, _, _ = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                                         verbose=train_iter_idx < 10, retraw=True,
                                                         **render_kwargs_train)
                t_uniform_rendering = perf_counter() - t_uniform_rendering_start

                # Computing section-wise loss for uniformly sampled rays.
                sampled_sw_idxs_tuple = get_idxs_tuple(uniformly_sampled_pw_idxs[:, :3])
                sw_n_newly_sampled = get_sw_n_sampled(sampled_sw_idxs_tuple, dims_sw)
                sw_train_loss, t_uniform_loss = \
                    get_sw_loss(unif_rendered_rgbs, unif_gt_rgbs, sw_n_newly_sampled, sampled_sw_idxs_tuple,
                                dims_sw, cpu)

            # Active ray sampling.
            sw_active_sampling_prob_dist = sw_train_loss / torch.sum(sw_train_loss)
            (actively_sampled_rays, actively_sampled_pw_idxs, t_active_sampling) = \
                sample_sw_rays(sw_rays, sw_active_sampling_prob_dist,
                               n_total_rays_to_actively_sample,
                               n_train_imgs, H, W, grid_size, section_height, section_width,
                               tensorboard, 'train/active_sampling/sampled_pixels',
                               train_iter_idx, log_sampling_vis=log_sampling_vis,
                               verbose=verbose)

            sampled_rays = actively_sampled_rays
            sampled_pw_idxs = actively_sampled_pw_idxs
        else:
            # Active sampling is disabled, so simply sample uniformly over all sections.
            (sampled_rays, sampled_pw_idxs, t_sampling) = \
                sample_sw_rays(sw_rays, sw_sampling_prob_dist, n_total_rays_to_sample,
                               n_train_imgs, H, W, grid_size, section_height, section_width,
                               tensorboard, 'train/sampling/sampled_pixels', train_iter_idx,
                               log_sampling_vis=log_sampling_vis, verbose=verbose)

        # Accumulate the total number of times each section has been sampled from so that it can be
        # visualized when log_sampling_vis is True.
        sampled_sw_idxs_tuple = get_idxs_tuple(sampled_pw_idxs[:, :3])
        sw_n_newly_sampled = get_sw_n_sampled(sampled_sw_idxs_tuple, dims_sw)
        sw_total_n_sampled += sw_n_newly_sampled

        t_batching_start = perf_counter()
        batch = torch.transpose(sampled_rays.to(gpu_if_available), 0, 1)
        batch_rays, train_gt_rgbs = batch[:2], batch[2]
        t_batching = perf_counter() - t_batching_start

        # Core optimization loop! #
        t_rendering_start = perf_counter()
        train_rendered_rgbs, _, _, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
                                                   verbose=train_iter_idx < 10, retraw=True,
                                                   **render_kwargs_train)
        t_rendering = perf_counter() - t_rendering_start

        t_loss_start = perf_counter()
        optimizer.zero_grad()
        train_loss = img2mse(train_rendered_rgbs, train_gt_rgbs)
        train_psnr = mse2psnr(train_loss)
        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], train_gt_rgbs)
            train_loss += img_loss0
            # psnr0 = mse2psnr(img_loss0)
        t_loss = perf_counter() - t_loss_start

        t_backprop_start = perf_counter()
        train_loss.backward()
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

        if train_iter_idx % args.i_test == 0 and train_iter_idx > 0:
            testsavedir = os.path.join(basedir, expname, f'testset_{train_iter_idx:06d}')
            os.makedirs(testsavedir, exist_ok=True)
            with torch.no_grad():
                test_rendered_rgbs_np, _ = render_path(
                    test_poses.to(gpu_if_available), hwf, K, args.chunk,
                    render_kwargs_test,
                    gt_imgs=test_rgb_imgs, savedir=testsavedir)
            test_rendered_rgbs = torch.from_numpy(test_rendered_rgbs_np)
            test_loss = img2mse(test_rendered_rgbs, test_rgb_imgs)
            test_psnr = mse2psnr(test_loss)

            tensorboard.add_images('test/rgb', pad_imgs(test_rendered_rgbs, white_rgb, 5),
                                   train_iter_idx, dataformats='NHWC')
            tensorboard.add_scalar('test/loss', test_loss, train_iter_idx)
            tensorboard.add_scalar('test/psnr', test_psnr, train_iter_idx)

        if train_iter_idx % args.i_train_scalars == 0:
            tensorboard.add_scalar('train/loss', train_loss, train_iter_idx)
            tensorboard.add_scalar('train/psnr', train_psnr, train_iter_idx)

        if log_sampling_vis:
            sampling_name = 'active_sampling' if do_active_sampling else 'sampling'
            add_1d_imgs_to_tensorboard(sw_total_n_sampled, torch.Tensor([1, 0, 0]), tensorboard,
                                       f'train/{sampling_name}/cumulative_samples_per_section',
                                       train_iter_idx)

            add_1d_imgs_to_tensorboard(sw_train_loss, torch.Tensor([0, 0.8, 0]), tensorboard,
                                       'train/estimated_loss_distribution', train_iter_idx)

        if train_iter_idx % args.i_val == 0:
            # Log a rendered validation view to Tensorboard.
            img_i = val_idxs[-2] if len(val_idxs) > 1 else val_idxs[0]
            val_gt_rgbs = rgb_imgs[img_i]
            c2w = poses[img_i, :3, :4].to(gpu_if_available)
            with torch.no_grad():
                val_rendered_rgb, val_rendered_disp, _, extras = \
                    render(H, W, K, chunk=args.chunk, c2w=c2w, **render_kwargs_test)

            val_loss = img2mse(val_rendered_rgb, val_gt_rgbs)
            val_psnr = mse2psnr(val_loss)

            tensorboard.add_image('validation/rgb/estimate', val_rendered_rgb,
                                  train_iter_idx, dataformats='HWC')
            if train_iter_idx == args.i_val:
                tensorboard.add_image('validation/rgb/groundtruth', val_gt_rgbs,
                                      train_iter_idx, dataformats='HWC')
            tensorboard.add_image('validation/depth/estimate', 1 / val_rendered_disp,
                                  train_iter_idx, dataformats='HW')
            tensorboard.add_scalar('validation/loss', val_loss, train_iter_idx)
            tensorboard.add_scalar('validation/psnr', val_psnr, train_iter_idx)

        if log_pose_vis:
            # Create a set of Open3D XYZ coordinate axes at the location of every initial pose and
            # optimized pose, then log them to tensorboard for visualization.
            initial_train_poses_np = initial_train_poses.cpu().numpy()
            train_poses_np = train_poses.detach().cpu().numpy()
            initial_coordinate_frames = get_coordinate_frames(initial_train_poses_np, gray_out=True)
            optimized_coordinate_frames = get_coordinate_frames(train_poses_np)
            for idx in range(len(initial_coordinate_frames)):
                # TODO: Efficiently store initial poses, since they are always the same at every
                # iteration.
                tensorboard.add_3d(f'train/pose{idx:03d}-initial',
                                   to_dict_batch([initial_coordinate_frames[idx]]), step=train_iter_idx)
                tensorboard.add_3d(f'train/pose{idx:03d}-optimized',
                                   to_dict_batch([optimized_coordinate_frames[idx]]), step=train_iter_idx)

        if do_active_sampling and not do_uniform_sampling_every_iter:
            # Update the estimated section-wise loss probability distribution using the loss from
            # the rays that were just actively sampled.
            with torch.no_grad():
                new_loss, _ = get_sw_loss(train_rendered_rgbs, train_gt_rgbs, sw_n_newly_sampled,
                                          sampled_sw_idxs_tuple, dims_sw, cpu)
                # Only update the loss of sections that were sampled from.
                sw_train_loss[sampled_sw_idxs_tuple] = new_loss[sampled_sw_idxs_tuple]

        t_train_iter = perf_counter() - t_train_iter_start

        if do_active_sampling:
            tqdm_bar.set_postfix_str(
                f't{t_train_iter:.3f},gr{t_get_rays:.3f},'
                f'u:(s{t_uniform_sampling:.3f},b{t_uniform_batching:.3f},'
                f'l{t_uniform_loss:.3f},r{t_uniform_rendering:.3f}),'
                f'a:(s{t_active_sampling:.3f},b{t_batching:.3f},r{t_rendering:.3f}),'
                f'b{t_backprop:.3f}')
        else:
            tqdm_bar.set_postfix_str(
                f't{t_train_iter:.3f},gr{t_get_ray:.3f},s{t_sampling:.3f},b{t_batching:.3f},'
                f'r{t_rendering:.3f},b{t_backprop:.3f}')

        global_step += 1


if __name__ == '__main__':
    with launch_ipdb_on_exception():
        train()
