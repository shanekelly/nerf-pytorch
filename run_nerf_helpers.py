from argparse import Namespace
from itertools import product
from time import perf_counter
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d
import torch
import torch.nn as nn

from cv2 import circle, FILLED
from ipdb import set_trace
from torch.nn.functional import relu
from torch.utils.tensorboard import SummaryWriter

from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data
from load_LINEMOD import load_LINEMOD_data
from load_bonn import load_bonn_data


white_rgb = torch.tensor([1.0]).expand(3)


def img2mse(x, y): return torch.mean((x - y) ** 2)
def mse2psnr(x): return -10. * torch.log(x) / torch.log(torch.tensor([10.], device=x.device))


def to8b(x): return (255*np.clip(x, 0, 1)).astype(np.uint8)


# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']  # sk: hard-coded to 3
        out_dim = 0

        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']  # sk: 9 for pos, 3 for view dir
        N_freqs = self.kwargs['num_freqs']  # sk: 10 for pos, 4 for view dir

        if self.kwargs['log_sampling']:  # sk: hard-coded to True
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:  # sk: hard-coded to [torch.sin, torch.cos]
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    def embed(x, eo=embedder_obj): return eo.embed(x)

    return embed, embedder_obj.out_dim


# Model
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4],
                 use_viewdirs=False):
        """
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips  # sk: layers to skip
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] +
            [nn.Linear(W, W) if i not in self.skips
             else nn.Linear(W + input_ch, W) for i in range(D-1)])

        # Implementation according to the official code release
        # https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

        # Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts

        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = relu(h)

            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"

        # Load pts_linears

        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(
                np.transpose(weights[idx_pts_linears]))
            self.pts_linears[i].bias.data = torch.from_numpy(
                np.transpose(weights[idx_pts_linears+1]))

        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(
            np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(
            np.transpose(weights[idx_feature_linear+1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(
            np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(
            np.transpose(weights[idx_views_linears+1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear+1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear+1]))


# Ray helpers
def get_rays(H: int, W: int, K: torch.Tensor, c2w: torch.Tensor, gpu_if_available: torch.device):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H), indexing='ij')
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)],
                       -1).to(gpu_if_available)
    # Rotate ray directions from camera frame to the world frame
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3, -1].expand(rays_d.shape)

    return rays_o, rays_d


def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))

    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1./(H/(2.*focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1./(W/(2.*focal)) * (rays_d[..., 0]/rays_d[..., 2] - rays_o[..., 0]/rays_o[..., 2])
    d1 = -1./(H/(2.*focal)) * (rays_d[..., 1]/rays_d[..., 2] - rays_o[..., 1]/rays_o[..., 2])
    d2 = -2. * near / rays_o[..., 2]

    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)

    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, gpu_if_available: torch.device, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples

    if det:
        u = torch.linspace(0., 1., steps=N_samples, device=gpu_if_available)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=gpu_if_available)

    # Pytest, overwrite u with numpy's fixed random numbers

    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]

        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.tensor(u, device=gpu_if_available)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1]-cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[..., 0])/denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1]-bins_g[..., 0])

    return samples


def render(H: int, W: int, K: torch.Tensor, gpu_if_available, chunk: int = 1024*32,
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
    all_ret = batchify_rays(rays, gpu_if_available, chunk, **kwargs)

    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}

    return ret_list + [ret_dict]


def batchify_rays(rays_flat: torch.Tensor, gpu_if_available, chunk: int = 1024*32, **kwargs) -> Dict:
    """
    @brief - Render rays in smaller minibatches to avoid OOM.
    @param rays_flat: Shape (N, 8). [rays_o (Nx3), rays_d (Nx3), near (Nx1), far (Nx1)]
    @param chunk - Maximum number of rays to process simultaneously.
    """
    all_ret = {}

    for i in range(0, rays_flat.shape[0], chunk):
        kwargs['gpu_if_available'] = gpu_if_available
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)

        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    # sk: Merge outputs from batches.
    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}

    return all_ret


def render_rays(ray_batch: torch.Tensor, network_fn: NeRF, network_query_fn: Callable,
                N_samples: int, gpu_if_available, retraw: bool = False, lindisp: bool = False, perturb: float = 0.,
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
        raw, z_vals, rays_d, gpu_if_available, raw_noise_std, white_bkgd, pytest=pytest)

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
            raw, z_vals, rays_d, gpu_if_available, raw_noise_std, white_bkgd, pytest=pytest)

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


def raw2outputs(raw, z_vals, rays_d, gpu_if_available, raw_noise_std=0, white_bkgd=False, pytest=False):
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


def load_data(args: Namespace, gpu_if_available: torch.device
              ) -> Tuple[torch.Tensor, torch.Tensor, List[int], int, int, float, torch.Tensor,
                         torch.Tensor, torch.Tensor, List[int], List[int], List[int], float, float]:
    # Load data
    K = None

    if args.dataset_type == 'llff':
        rgb_imgs, poses, bds, render_poses, test_idxs = load_llff_data(args.datadir, args.factor,
                                                                       recenter=True, bd_factor=.75,
                                                                       spherify=args.spherify)
        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]
        print('Loaded llff', rgb_imgs.shape, render_poses.shape, hwf, args.datadir)

        if not isinstance(test_idxs, list):
            test_idxs = [test_idxs]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            test_idxs = np.arange(rgb_imgs.shape[0])[::args.llffhold]

        val_idxs = test_idxs
        train_idxs = np.array([i for i in np.arange(int(rgb_imgs.shape[0])) if
                               (i not in test_idxs and i not in val_idxs)])

        print('DEFINING BOUNDS')

        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.

        else:
            near = 0.
            far = 1.
        print('NEAR FAR', near, far)

    elif args.dataset_type == 'blender':
        rgb_imgs, poses, render_poses, hwf, i_split = load_blender_data(
            args.datadir, args.half_res, args.testskip)
        print('Loaded blender', rgb_imgs.shape, render_poses.shape, hwf, args.datadir)
        train_idxs, val_idxs, test_idxs = i_split

        near = 2.
        far = 6.

        if args.white_bkgd:
            rgb_imgs = rgb_imgs[..., :3]*rgb_imgs[..., -1:] + (1.-rgb_imgs[..., -1:])
        else:
            rgb_imgs = rgb_imgs[..., :3]

    elif args.dataset_type == 'LINEMOD':
        rgb_imgs, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(
            args.datadir, args.half_res, args.testskip)
        print(f'Loaded LINEMOD, rgb_imgs shape: {rgb_imgs.shape}, hwf: {hwf}, K: {K}')
        print(f'[CHECK HERE] near: {near}, far: {far}.')
        train_idxs, val_idxs, test_idxs = i_split

        if args.white_bkgd:
            rgb_imgs = rgb_imgs[..., :3]*rgb_imgs[..., -1:] + (1.-rgb_imgs[..., -1:])
        else:
            rgb_imgs = rgb_imgs[..., :3]

    elif args.dataset_type == 'deepvoxels':

        rgb_imgs, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
                                                                   basedir=args.datadir,
                                                                   testskip=args.testskip)

        print('Loaded deepvoxels', rgb_imgs.shape, render_poses.shape, hwf, args.datadir)
        train_idxs, val_idxs, test_idxs = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:, :3, -1], axis=-1))
        near = hemi_R-1.
        far = hemi_R+1.

    elif args.dataset_type == 'bonn':
        rgb_imgs, depth_imgs, hwf, poses, render_poses, train_idxs, test_idxs = load_bonn_data(
            args.datadir, downsample_factor=args.factor)

        val_idxs = test_idxs
        near = 0.
        far = 1.

    else:
        raise RuntimeError(f'Unknown dataset type {args.dataset_type}. Exiting.')

    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    if K is None:
        K = np.array([
            [focal, 0, 0.5*W],
            [0, focal, 0.5*H],
            [0, 0, 1]
        ])

    if args.render_test:
        render_poses = np.array(poses[test_idxs])

    K = torch.tensor(K)

    # Attempt to move data to the GPU.
    rgb_imgs = torch.tensor(rgb_imgs, device=gpu_if_available, dtype=torch.float32)
    poses = torch.tensor(poses, device=gpu_if_available, dtype=torch.float32)
    render_poses = torch.tensor(render_poses, device=gpu_if_available, dtype=torch.float32)

    print('TRAIN views are', train_idxs)
    print('TEST views are', test_idxs)
    print('VAL views are', val_idxs)

    train_idxs = torch.tensor(train_idxs)
    test_idxs = torch.tensor(test_idxs)
    val_idxs = torch.tensor(val_idxs)

    return (rgb_imgs, depth_imgs, hwf, H, W, focal, K, poses, render_poses, train_idxs, test_idxs,
            val_idxs, near, far)


def get_sw_rays(images: torch.Tensor, H: int, W: int, K: torch.Tensor, poses: torch.Tensor,
                n_kfs: int,  grid_size: int, section_height: int,
                section_width: int, gpu_if_available: torch.device, verbose=False
                ) -> Tuple[torch.Tensor, float]:
    """
    @param images - Training images.
    @param poses - Pose for each training image.
    """
    if verbose:
        print('Getting all rays...', end='')

    t_start = perf_counter()

    # sk: A ray for every pixel of every camera image. Shape (N, ro+rd, H, W, 3).
    rays = torch.stack([torch.stack(get_rays(H, W, K, p, gpu_if_available))
                       for p in poses[:, :3, :4]])

    rays = torch.cat([rays, images[:, None]], 1)  # (N, ro+rd+rgb, H, W, 3)
    rays = torch.permute(rays, (0, 2, 3, 1, 4))  # (N, H, W, ro+rd+rgb, 3)

    sw_rays = torch.empty((n_kfs, grid_size, grid_size,
                           section_height, section_width, 3, 3))
    for grid_row_idx, grid_col_idx in product(range(grid_size), range(grid_size)):
        img_start_row_idx = grid_row_idx * section_height
        img_stop_row_idx = (grid_row_idx + 1) * section_height
        img_start_col_idx = grid_col_idx * section_width
        img_stop_col_idx = (grid_col_idx + 1) * section_width
        sw_rays[:, grid_row_idx, grid_col_idx, :, :, :] = \
            rays[:, img_start_row_idx:img_stop_row_idx, img_start_col_idx:img_stop_col_idx, :, :]

    t_delta = perf_counter() - t_start

    if verbose:
        print(f' done in {t_delta:.3f} seconds.')

    return sw_rays, t_delta


def get_initial_section_rand_pixel_idxs(num_train_imgs: int, grid_size: int, section_height: int,
                                        section_width: int, verbose: bool = False
                                        ) -> Tuple[torch.Tensor, float]:
    """
    @param num_train_imgs - The number of training images.
    @param grid_size - The number of sections along one side of the grid that images are split into
        (the grid is assumed square).
    @param section_height - The height of a single grid section, in pixels.
    @param section_width - The width of a single grid section, in pixels.
    @param verbose - True to print extra info.
    @returns - A tuple of (section_rand_pixel_idxs, t_delta).
        section_rand_pixels - Numpy array of shape (num_train_imgs, grid_size, grid_size,
            section_height * section_width) that stores a random list of all pixel indices for each
            grid section.
        t_delta - The execution time of this function.
    """
    t_start = perf_counter()

    if verbose:
        print('Getting initial section random pixel indices...', end='')

    num_pixels_per_section = section_height * section_width
    single_section_pixel_idxs = torch.tensor(list(product(range(section_height),
                                                          range(section_width))))
    section_rand_pixel_idxs = \
        single_section_pixel_idxs.repeat((num_train_imgs, grid_size, grid_size, 1, 1))
    for img_idx, grid_row_idx, grid_col_idx in product(range(num_train_imgs),
                                                       range(grid_size),
                                                       range(grid_size)):
        rand_pixel_idxs = torch.randperm(num_pixels_per_section)
        section_rand_pixel_idxs[img_idx, grid_row_idx, grid_col_idx, :, :] = \
            section_rand_pixel_idxs[img_idx, grid_row_idx, grid_col_idx, rand_pixel_idxs, :]

    t_delta = perf_counter() - t_start

    if verbose:
        print(f' done in {t_delta:.3f} seconds.')

    return section_rand_pixel_idxs, t_delta


def nd_idxs_from_1d_idxs(flat_idxs: torch.Tensor, n_elems_per_chunks: torch.Tensor):
    n_elems_to_chunk = flat_idxs.clone()
    nd_idxs = torch.empty((flat_idxs.shape[0], n_elems_per_chunks.shape[0]), dtype=torch.int64)
    for idx, n_elems_per_chunk in enumerate(n_elems_per_chunks):
        chunk_idxs = torch.div(n_elems_to_chunk, n_elems_per_chunk, rounding_mode='floor')
        n_elems_to_chunk -= chunk_idxs * n_elems_per_chunk
        nd_idxs[:, idx] = chunk_idxs

    return nd_idxs


def pad_imgs(imgs: torch.Tensor, padding_rgb: torch.Tensor, padding_width: int
             ) -> torch.Tensor:
    assert imgs.dim() == 4  # Shape (N, H, W, C).
    assert imgs.shape[-1] == 3  # C is RGB.

    padded_imgs = imgs.clone()

    padding_top_bottom = padding_rgb.expand(imgs.shape[0], padding_width, imgs.shape[2], 3)
    padded_imgs = torch.cat((padding_top_bottom, padded_imgs, padding_top_bottom), dim=1)
    padding_left_right = \
        padding_rgb.expand(imgs.shape[0], imgs.shape[1] + 2 * padding_width, padding_width, 3)
    padded_imgs = torch.cat((padding_left_right, padded_imgs, padding_left_right), dim=2)

    return padded_imgs


def pad_sections(imgs: torch.Tensor, dims_pw: Tuple[int, int, int, int, int],
                 padding_rgb: torch.Tensor, padding_width: int
                 ) -> torch.Tensor:
    n_imgs, img_height, img_width, _ = imgs.shape
    _, grid_size, _, section_height, section_width = dims_pw

    n_pads = grid_size + 2
    n_pad_pixels_per_axis = padding_width * n_pads
    padded_imgs = padding_rgb.repeat(n_imgs, img_height + n_pad_pixels_per_axis,
                                     img_width + n_pad_pixels_per_axis, 1)

    # Draw all of the image sections on the sampling visualization images.
    for grid_row_idx, grid_col_idx in product(range(grid_size), range(grid_size)):
        section_row_start_idx = grid_row_idx * section_height
        padded_section_row_start_idx = section_row_start_idx + (grid_row_idx + 1) * padding_width
        section_row_stop_idx = section_row_start_idx + section_height
        padded_section_row_stop_idx = padded_section_row_start_idx + section_height

        section_col_start_idx = grid_col_idx * section_width
        padded_section_col_start_idx = section_col_start_idx + (grid_col_idx + 1) * padding_width
        section_col_stop_idx = section_col_start_idx + section_width
        padded_section_col_stop_idx = padded_section_col_start_idx + section_width

        padded_imgs[:, padded_section_row_start_idx:padded_section_row_stop_idx,
                    padded_section_col_start_idx:padded_section_col_stop_idx] = \
            imgs[:, section_row_start_idx:section_row_stop_idx,
                 section_col_start_idx:section_col_stop_idx, :]

    return padded_imgs


def sample_sw_rays(sw_rays: torch.Tensor, sw_sampling_prob_dist: torch.Tensor,
                   n_total_rays_to_sample: int, kf_rgb_imgs: torch.Tensor, img_height: int, img_width: int,
                   dims_kf_pw: Tuple[int, int, int, int, int], dims_pw: Tuple[int, int, int, int,
                                                                              int],
                   sampled_from_kf_idxs, tensorboard: SummaryWriter,
                   tensorboard_tag: str, train_iter_idx: int, log_sampling_vis: bool = False,
                   verbose: bool = False, enforce_min_samples: bool = False
                   ) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    @param dims_kf_pw - Pixel-wise dimensions of all keyframes. At least needed for visualizing all
    keyframes during the sampling visualization.
    @param dims_pw - Pixel-wise dimensions of whatever frames are being sampled from. If sampling
    from all keyframes, then this will be equal to dims_kf_pw. If sampling from selected keyframes,
    then this will be equal to dims_kf_pw except for the first element, which will store the number
    of selected keyframes.
    """
    t_start = perf_counter()

    if verbose:
        print('Sampling rays across grid sections...', end='')

    n_kfs, grid_size, _, section_height, section_width = dims_kf_pw
    dims_kf_sw = dims_kf_pw[:3]

    dims_sw = dims_pw[:3]

    # Values for converting between flat idxs and section-wise / pixel-wise indices.
    n_pixels_per_section = section_height * section_width
    n_pixels_per_img = img_height * img_width
    n_pixels_per_grid_row_in_img = grid_size * n_pixels_per_section
    n_pixels_per_grid_col_in_grid_row = n_pixels_per_section
    n_pixels_per_pixel_row_in_grid_col = section_width
    n_pixels_per_pixel_col_in_pixel_row = 1

    # A tensor that contains, for every pixel, the probability that pixel should be sampled
    # with.  Shape (n_kfs, grid_size, grid_size, section_height, section_width).
    pw_sampling_prob_dist = sw_sampling_prob_dist.float().unsqueeze(-1).unsqueeze(-1).repeat(
        (1, 1, 1, section_height, section_width))

    if enforce_min_samples:
        sampled_flat_idxs = torch.empty((n_total_rays_to_sample), dtype=torch.int64)
        sw_min_n_to_sample = \
            torch.floor(n_total_rays_to_sample * sw_sampling_prob_dist).to(torch.int64)
        n_remaining_samples = n_total_rays_to_sample - torch.sum(sw_min_n_to_sample)
        sw_additional_n_to_sample, _ = get_n_to_sample(sw_sampling_prob_dist,
                                                       n_remaining_samples.item())
        sw_n_to_sample = sw_min_n_to_sample + sw_additional_n_to_sample
        start_insert_idx = 0
        for img_idx, grid_row_idx, grid_col_idx in product(range(n_kfs), range(grid_size),
                                                           range(grid_size)):
            start_sampled_idx = (img_idx * n_pixels_per_img +
                                 grid_row_idx * n_pixels_per_grid_row_in_img +
                                 grid_col_idx * n_pixels_per_grid_col_in_grid_row)
            sampling_prob_dist = pw_sampling_prob_dist[img_idx, grid_row_idx, grid_col_idx, :, :]
            n_to_sample = sw_n_to_sample[img_idx, grid_row_idx, grid_col_idx]
            new_sampled_flat_idxs = \
                torch.multinomial(sampling_prob_dist.view(-1), n_to_sample) + start_sampled_idx
            stop_insert_idx = start_insert_idx + n_to_sample
            sampled_flat_idxs[start_insert_idx:stop_insert_idx] = new_sampled_flat_idxs

            start_insert_idx = stop_insert_idx
    else:
        # A 1D tensor filled with indices into a flat version of all pixels. Thus, the minimum index
        # value is 0 and the maximum index value is (n_kfs * grid_size * grid_size *
        # section_height * section_width - 1). Each index refers to a pixel that should be sampled.
        # Shape (n_total_rays_to_sample, ).
        sampled_flat_idxs = \
            torch.multinomial(pw_sampling_prob_dist.view(-1), n_total_rays_to_sample)

    # Use sampled_flat_idxs to index into the rays. Now have obtained the sampled rays. Shape
    # (n_total_rays_to_sample, 3, 3).
    sampled_rays = sw_rays.view(-1, 3, 3)[sampled_flat_idxs]

    n_pixels_per_chunks = torch.tensor([n_pixels_per_img, n_pixels_per_grid_row_in_img,
                                        n_pixels_per_grid_col_in_grid_row, n_pixels_per_pixel_row_in_grid_col,
                                        n_pixels_per_pixel_col_in_pixel_row])
    sampled_pw_idxs = nd_idxs_from_1d_idxs(sampled_flat_idxs, n_pixels_per_chunks)

    sampled_sw_idxs_tuple = get_idxs_tuple(sampled_pw_idxs[:, :3])
    sw_n_newly_sampled = get_sw_n_sampled(sampled_sw_idxs_tuple, dims_sw)

    if log_sampling_vis:
        section_padding_width = 2
        sampling_vis_imgs = pad_sections(kf_rgb_imgs, dims_kf_pw, white_rgb,
                                         section_padding_width)
        sampling_vis_imgs_np = sampling_vis_imgs.numpy()

        # Draw a dot at the location of every sampled pixel.
        for img_idx, grid_row_idx, grid_col_idx, pixel_row_idx, pixel_col_idx in sampled_pw_idxs:
            kf_idx = sampled_from_kf_idxs[img_idx]
            section_row_start_idx = (grid_row_idx + 1) * section_padding_width + \
                grid_row_idx * section_height
            section_col_start_idx = (grid_col_idx + 1) * section_padding_width + \
                grid_col_idx * section_width

            pixel_row_idx = (pixel_row_idx + section_row_start_idx).item()
            pixel_col_idx = (pixel_col_idx + section_col_start_idx).item()

            sampling_vis_img_np = sampling_vis_imgs_np[kf_idx]
            # circle(sampling_vis_img_np, (pixel_col_idx, pixel_row_idx), 3, (0, 0, 0), FILLED)
            # circle(sampling_vis_img_np, (pixel_col_idx, pixel_row_idx), 2, (1, 1, 1), FILLED)
            circle(sampling_vis_img_np, (pixel_col_idx, pixel_row_idx), 1, (1, 0, 0), FILLED)

        tensorboard.add_images(tensorboard_tag, sampling_vis_imgs, train_iter_idx,
                               dataformats='NHWC')

    t_delta = perf_counter() - t_start
    if verbose:
        print(f' done in {t_delta:.3f} seconds.')

    return sampled_rays, sampled_pw_idxs, sw_n_newly_sampled, t_delta


def get_n_to_sample(prob_dist: torch.Tensor, n_samples: int
                    ) -> Tuple[torch.Tensor, float]:
    """
    @param prob_dist
    @param n_samples
    @returns - (section_n_samples, t_delta)
        section_n_samples - Same shape as prob_dist. The number of times each element from prob_dist
            should be sampled.
        t_delta - Total execution time of this function.
    """
    t_start = perf_counter()

    # Create a tensor in the same shape as the input probability distribution that will be returned
    # and dictate how many times each element should be sampled.
    section_n_samples = torch.zeros(prob_dist.shape, dtype=int)
    # Create a flat view of section_n_samples for easier manipulation. Note that when this flat view
    # is modified the appropriate element of the unflattened version will also be modified.
    section_n_samples_flat = section_n_samples.view(-1)

    # Flatten the input probability distribution and compute the cumulative probability
    # distribution.
    cumu_prob_dist_flat = torch.cumsum(torch.flatten(prob_dist), 0)

    # Create n_samples random values between 0 and 1.
    rand_vals = torch.rand(n_samples)

    # For each value in rand_vals, find the index of the first value in the cumulative probability
    # distribution that is greater than the value.
    idxs_to_sample_flat = torch.bucketize(rand_vals, cumu_prob_dist_flat)
    # Get the number of times each index was sampled.
    unique_vals, unique_val_counts = torch.unique(idxs_to_sample_flat, return_counts=True)
    # Update the return datastructure with the number of times to sample each element.
    section_n_samples_flat[unique_vals] = unique_val_counts

    t_delta = perf_counter() - t_start

    return section_n_samples, t_delta


def get_idxs_tuple(idxs: torch.Tensor
                   ) -> Tuple[torch.Tensor, ...]:
    """
    @param idxs - Shape (num indices, num dimensions that the indices index into).
    @returns - A tuple, which can be directly used to index into a tensor, eg tensor[idxs_tuple].
        The tuple has (num dimensions that the indices index into) tensors, where each tensor has
        (num indices) elements.
    """
    return tuple(torch.transpose(idxs, 0, 1))


def get_sw_n_sampled(sampled_sw_idxs_tuple: Tuple[torch.Tensor, ...], dims_kf_sw: Tuple[int, int, int]
                     ) -> torch.Tensor:
    sw_n_sampled = torch.index_put(torch.zeros(dims_kf_sw, dtype=torch.int64), sampled_sw_idxs_tuple,
                                   torch.tensor([1], dtype=torch.int64), accumulate=True)

    return sw_n_sampled


def get_sw_loss(rendered_rgbs: torch.Tensor, gt_rgbs: torch.Tensor, sw_n_sampled: torch.Tensor,
                sampled_sw_idxs_tuple: Tuple[torch.Tensor, ...], extras, dims_sw: Tuple[int, int, int],
                cpu: torch.device,
                ) -> Tuple[torch.Tensor, float]:
    t_start = perf_counter()

    rendered_rgbs = rendered_rgbs.to(cpu)
    gt_rgbs = gt_rgbs.to(cpu)

    def get_sw_coarse_or_fine_loss(get_sw_coarse_or_fine_rendered_rgbs):
        mean_squared_diffs = torch.mean(
            (get_sw_coarse_or_fine_rendered_rgbs - gt_rgbs) ** 2, axis=1)
        sw_cumu_mean_squared_diffs = torch.index_put(torch.zeros(dims_sw), sampled_sw_idxs_tuple,
                                                     mean_squared_diffs, accumulate=True)
        # For each section, the average mean squared difference between rendered RGB and groundtruth RGB
        # for each sampled pixel within the section.
        sw_coarse_or_fine_loss = torch.nan_to_num(sw_cumu_mean_squared_diffs / sw_n_sampled, nan=0)

        return sw_coarse_or_fine_loss

    # Get section-wise coarse loss.
    sw_loss = get_sw_coarse_or_fine_loss(rendered_rgbs)

    if 'rgb0' in extras:
        # Get section-wise fine loss.
        sw_loss += get_sw_coarse_or_fine_loss(extras['rgb0'].to(cpu))

    t_delta = perf_counter() - t_start

    return sw_loss, t_delta


def add_1d_imgs_to_tensorboard(imgs: torch.Tensor, img_rgb: torch.Tensor,
                               tensorboard: SummaryWriter, tag: str, iter_idx: int,
                               padding_rgb: torch.Tensor = torch.tensor([0.6]).expand(3),
                               padding_width: int = 1
                               ) -> torch.Tensor:
    assert imgs.dim() == 3

    # Scale all pixels between 0 and 1.
    imgs_scaled = imgs / torch.max(imgs)

    # If an element has the value 0, then it should receive the RGB value of zero_rgb.
    zero_rgb = torch.Tensor([1]).expand(3)
    # If an element has the value 1 (the max value after scaling), then it should receive the RGB
    # value of max_rgb.
    max_rgb = img_rgb
    diff_rgb = max_rgb - zero_rgb

    # Linear interpolation between zero_rgb and max_rgb based on the value of each pixel.
    output_imgs = (zero_rgb.repeat(imgs.shape[0], imgs.shape[1], imgs.shape[2], 1) +
                   imgs_scaled.unsqueeze(-1).repeat(1, 1, 1, 3) * diff_rgb)

    # Add border around each image.
    output_imgs = pad_imgs(output_imgs, padding_rgb, padding_width)

    tensorboard.add_images(tag, output_imgs, iter_idx, dataformats='NHWC')


def skew_symmetric(w):
    w0, w1, w2 = w.unbind(dim=-1)
    O = torch.zeros_like(w0)
    wx = torch.stack([torch.stack([O, -w2, w1], dim=-1),
                      torch.stack([w2, O, -w0], dim=-1),
                      torch.stack([-w1, w0, O], dim=-1)], dim=-2)
    return wx


def taylor_A(x, nth=10):
    # Taylor expansion of sin(x)/x
    ans = torch.zeros_like(x)
    denom = 1.
    for i in range(nth+1):
        if i > 0:
            denom *= (2*i)*(2*i+1)
        ans = ans+(-1)**i*x**(2*i)/denom
    return ans


def taylor_B(x, nth=10):
    # Taylor expansion of (1-cos(x))/x**2
    ans = torch.zeros_like(x)
    denom = 1.
    for i in range(nth+1):
        denom *= (2*i+1)*(2*i+2)
        ans = ans+(-1)**i*x**(2*i)/denom
    return ans


def taylor_C(x, nth=10):
    # Taylor expansion of (x-sin(x))/x**3
    ans = torch.zeros_like(x)
    denom = 1.
    for i in range(nth+1):
        denom *= (2*i+2)*(2*i+3)
        ans = ans+(-1)**i*x**(2*i)/denom
    return ans


def SO3_to_so3(R, eps=1e-7):  # [...,3,3]
    trace = R[..., 0, 0]+R[..., 1, 1]+R[..., 2, 2]
    # ln(R) will explode if theta==pi
    theta = ((trace-1)/2).clamp(-1+eps, 1-eps).acos_()[..., None, None] % np.pi
    lnR = 1/(2*taylor_A(theta)+1e-8) * \
        (R-R.transpose(-2, -1))  # FIXME: wei-chiu finds it weird
    w0, w1, w2 = lnR[..., 2, 1], lnR[..., 0, 2], lnR[..., 1, 0]
    w = torch.stack([w0, w1, w2], dim=-1)
    return w


def tfmats_from_minreps(wu, world_from_camera1, gpu_if_available):  # [...,3]
    w, u = wu.split([3, 3], dim=-1)
    wx = skew_symmetric(w)
    theta = w.norm(dim=-1)[..., None, None]
    I = torch.eye(3, device=w.device, dtype=torch.float32)
    A = taylor_A(theta)
    B = taylor_B(theta)
    C = taylor_C(theta)
    R = I+A*wx+B*wx@wx
    V = I+B*wx+C*wx@wx
    Rt = torch.cat([R, (V@u[..., None])], dim=-1)
    camera1_from_cameras = torch.cat((Rt, torch.tensor([0, 0, 0, 1], device=gpu_if_available).expand(
        Rt.shape[0], 1, 4)), axis=1)
    # Convert N - 1 poses relative to first pose into N poses relative to world frame.
    world_from_cameras = torch.cat((world_from_camera1.unsqueeze(0),
                                    world_from_camera1.matmul(camera1_from_cameras)))

    return world_from_cameras


def minreps_from_tfmats(Rt, gpu_if_available, eps=1e-8):  # [...,3,4]
    # Transform the N poses relative to the world frame to N - 1 poses relative to the first pose.
    camera1_from_world = Rt[0].inverse()
    world_from_cameras = Rt[1:]
    camera1_from_cameras = camera1_from_world.matmul(world_from_cameras)

    Rt = camera1_from_cameras[:, :3, :]
    R, t = Rt.split([3, 1], dim=-1)
    w = SO3_to_so3(R)
    wx = skew_symmetric(w)
    theta = w.norm(dim=-1)[..., None, None]
    I = torch.eye(3, device=w.device, dtype=torch.float32)
    A = taylor_A(theta)
    B = taylor_B(theta)
    invV = I-0.5*wx+(1-A/(2*B))/(theta**2+eps)*wx@wx
    u = (invV@t)[..., 0]
    wu = torch.cat([w, u], dim=-1).to(gpu_if_available)
    return wu


def get_coordinate_frames(poses, coordinate_frame_size: float = 0.05, gray_out: bool = False):
    coordinate_frames = []
    for pose in poses:
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            coordinate_frame_size).transform(pose)
        if gray_out:
            coordinate_frame.vertex_colors = o3d.utility.Vector3dVector(
                np.clip(np.asarray(coordinate_frame.vertex_colors) + 0.5, 0, 1))
        coordinate_frames.append(coordinate_frame)

    return coordinate_frames


def create_keyframes(rgb_imgs: torch.Tensor, initial_poses: torch.Tensor,
                     train_idxs: List[int], keyframe_creation_strategy, every_Nth
                     ) -> Tuple[torch.Tensor, torch.Tensor, List[int], int]:
    if keyframe_creation_strategy == 'all':
        kf_idxs = train_idxs
    elif keyframe_creation_strategy == 'every_Nth':
        assert every_Nth is not None, ('If --keyframe_creation_strategy is set to "every_Nth", '
                                       'then --every_Nth must also be defined.')
        kf_idxs = train_idxs[::every_Nth]
    else:
        raise RuntimeError(
            f'Unknown keyframe creation strategy "{keyframe_creation_strategy}". Exiting.')

    kf_rgb_imgs = rgb_imgs[kf_idxs]
    kf_initial_poses = initial_poses[kf_idxs]
    n_kfs = len(kf_idxs)

    print('Keyframe views are ', kf_idxs)

    return kf_rgb_imgs, kf_initial_poses, kf_idxs, n_kfs


def get_kf_poses(kf_initial_poses: torch.Tensor, kf_poses_params: torch.nn.Parameter,
                 do_pose_optimization: bool
                 ) -> torch.Tensor:
    if do_pose_optimization:
        # Unpack the minimal representations of the poses from the optimizer's parameters, then
        # convert them into 4x4 transformation matrices.
        kf_poses = tfmats_from_minreps(kf_poses_params, kf_initial_poses[0])
    else:
        kf_poses = kf_initial_poses

    return kf_poses


def render_and_compute_loss(rays, intrinsics_matrix, render_kwargs_train, img_height, img_width,
                            dims_kf_sw, chunk, sampled_sw_idxs_tuple, sw_n_newly_sampled, optimizer,
                            train_iter_idx, cpu, gpu_if_available
                            ) -> Tuple[torch.Tensor, torch.Tensor, Dict, torch.Tensor, torch.Tensor,
                                       torch.Tensor, float, float, float]:
    t_batching_start = perf_counter()
    batch = torch.transpose(rays.to(gpu_if_available), 0, 1)
    batch_rays, gt_rgbs = batch[:2], batch[2]
    t_batching = perf_counter() - t_batching_start

    t_rendering_start = perf_counter()
    rendered_rgbs, rendered_disps, _, extras = render(img_height, img_width, intrinsics_matrix, gpu_if_available, chunk=chunk, rays=batch_rays,
                                                      verbose=train_iter_idx < 10, retraw=True, **render_kwargs_train)
    t_rendering = perf_counter() - t_rendering_start

    t_loss_start = perf_counter()
    optimizer.zero_grad()

    sw_loss, t_loss = get_sw_loss(rendered_rgbs, gt_rgbs, sw_n_newly_sampled, sampled_sw_idxs_tuple, extras,
                                  dims_kf_sw, cpu)

    loss = torch.mean(sw_loss)
    psnr = mse2psnr(loss)

    t_loss = perf_counter() - t_loss_start

    return (rendered_rgbs, rendered_disps, extras, sw_loss, loss, psnr, t_batching, t_rendering,
            t_loss)


def select_keyframes(kf_rgb_imgs, kf_poses, kf_idxs, sw_kf_loss, img_height, img_width,
                     intrinsics_matrix, is_first_iter,
                     n_total_rays_to_uniformly_sample, dims_kf_pw,
                     keyframe_selection_strategy, n_explore, n_exploit,
                     sw_unif_sampling_prob_dist, tensorboard, verbose, train_iter_idx, optimizer,
                     chunk, render_kwargs_train, cpu, gpu_if_available):
    n_kfs, grid_size, _, section_height, section_width = dims_kf_pw
    dims_kf_sw = dims_kf_pw[:3]

    if keyframe_selection_strategy == 'all':
        skf_rgb_imgs = kf_rgb_imgs
        skf_poses = kf_poses
        skf_idxs = kf_idxs
        n_skfs = n_kfs
        sw_skf_loss = sw_kf_loss
        skf_from_kf_idxs = torch.arange(n_kfs)

    elif keyframe_selection_strategy == 'explore_exploit':
        assert n_explore is not None, ('If --keyframe_selection_strategy is set to '
                                       '"explore_exploit", then --n_explore must also be defined.')
        assert n_exploit is not None, ('If --keyframe_selection_strategy is set to '
                                       '"explore_exploit", then --n_exploit must also be defined.')
        assert n_explore + n_exploit <= n_kfs
        if is_first_iter:
            # This is the first training iteration, so the section-wise loss estimate over the
            # keyframes will need to be initialized via uniform sampling over all keyframes.

            with torch.no_grad():
                sw_kf_rays, _ = \
                    get_sw_rays(kf_rgb_imgs, img_height, img_width, intrinsics_matrix, kf_poses,
                                n_kfs, grid_size, section_height, section_width, gpu_if_available)
            (sampled_rays, sampled_pw_idxs, sw_n_newly_sampled, t_uniform_sampling) = \
                sample_sw_rays(sw_kf_rays, sw_unif_sampling_prob_dist,
                               n_total_rays_to_uniformly_sample, kf_rgb_imgs, img_height, img_width,
                               dims_kf_pw, dims_kf_pw, torch.arange(
                                   dims_kf_pw[0]), tensorboard, 'train/uniform_sampling/sampled_pixels',
                               train_iter_idx, log_sampling_vis=True, verbose=verbose,
                               enforce_min_samples=True)
            sampled_sw_idxs_tuple = get_idxs_tuple(sampled_pw_idxs[:, :3])
            _, _, _, sw_kf_loss, _, _, _, _, _ = \
                render_and_compute_loss(sampled_rays, intrinsics_matrix, render_kwargs_train,
                                        img_height, img_width, dims_kf_sw, chunk,
                                        sampled_sw_idxs_tuple, sw_n_newly_sampled, optimizer,
                                        train_iter_idx, cpu, gpu_if_available)

        kf_losses = torch.sum(sw_kf_loss, dim=(1, 2))
        idxs_descending_loss = torch.argsort(kf_losses, descending=True)
        idxs_exploit = idxs_descending_loss[:n_exploit]
        idxs_remaining = torch.tensor(list(set(range(len(kf_idxs))) - set(idxs_exploit.tolist())),
                                      dtype=torch.int64)
        idxs_explore = idxs_remaining[torch.randperm(idxs_remaining.shape[0])[:n_explore]]
        skf_from_kf_idxs, _ = torch.sort(torch.cat((idxs_explore, idxs_exploit)))

        skf_rgb_imgs = kf_rgb_imgs[skf_from_kf_idxs]
        skf_poses = kf_poses[skf_from_kf_idxs]
        skf_idxs = kf_idxs[skf_from_kf_idxs]
        n_skfs = skf_idxs.shape[0]
        sw_skf_loss = torch.zeros_like(sw_kf_loss)
        sw_skf_loss[skf_from_kf_idxs] = sw_kf_loss[skf_from_kf_idxs]

    else:
        raise RuntimeError(
            f'Unknown keyframe selection strategy "{keyframe_selection_strategy}". Exiting.')

    dims_skf_pw = tuple([n_skfs] + list(dims_kf_pw[1:]))

    return (skf_rgb_imgs, skf_poses, skf_idxs, n_skfs, dims_skf_pw, sw_kf_loss, sw_skf_loss,
            skf_from_kf_idxs)
