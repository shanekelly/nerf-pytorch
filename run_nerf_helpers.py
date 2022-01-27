from argparse import Namespace
from itertools import product
from time import perf_counter
from typing import List, Optional, Tuple

import numpy as np
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


# Misc
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

    return (rgb_imgs, depth_imgs, hwf, H, W, focal, K, poses, render_poses, train_idxs, test_idxs,
            val_idxs, near, far)


def get_sw_rays(images: torch.Tensor, H: int, W: int, K: torch.Tensor, poses: torch.Tensor,
                n_train_imgs: int,  grid_size: int, section_height: int,
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

    sw_rays = torch.empty((n_train_imgs, grid_size, grid_size,
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


def sample_sw_rays(sw_rays: torch.Tensor, sw_sampling_prob_dist: torch.Tensor,
                   n_total_rays_to_sample: int, n_train_imgs: int, img_height: int,
                   img_width: int, grid_size: int, section_height: int, section_width: int,
                   tensorboard: SummaryWriter, tensorboard_tag: str, train_iter_idx: int,
                   log_sampling_vis: bool = False, verbose: bool = False,
                   enforce_min_samples: bool = False
                   ) -> Tuple[torch.Tensor, torch.Tensor, float]:
    t_start = perf_counter()

    if verbose:
        print('Sampling rays across grid sections...', end='')

    # Values for converting between flat idxs and section-wise / pixel-wise indices.
    n_pixels_per_section = section_height * section_width
    n_pixels_per_img = img_height * img_width
    n_pixels_per_grid_row_in_img = grid_size * n_pixels_per_section
    n_pixels_per_grid_col_in_grid_row = n_pixels_per_section
    n_pixels_per_pixel_row_in_grid_col = section_width
    n_pixels_per_pixel_col_in_pixel_row = 1

    # A tensor that contains, for every pixel, the probability that pixel should be sampled
    # with.  Shape (n_train_imgs, grid_size, grid_size, section_height, section_width).
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
        for img_idx, grid_row_idx, grid_col_idx in product(range(n_train_imgs), range(grid_size),
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
        # value is 0 and the maximum index value is (n_train_imgs * grid_size * grid_size *
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

    if log_sampling_vis:
        # The number of white pixels between each grid section.
        section_padding_size = 5
        n_pads = grid_size + 2
        n_pad_pixels_per_axis = section_padding_size * n_pads
        sampling_vis_imgs = torch.ones((n_train_imgs, img_height + n_pad_pixels_per_axis,
                                        img_width + n_pad_pixels_per_axis, 3), device='cpu')
        sampling_vis_imgs_np = sampling_vis_imgs.numpy()

        # Draw all of the image sections on the sampling visualization images.
        for img_idx, grid_row_idx, grid_col_idx in product(range(n_train_imgs), range(grid_size),
                                                           range(grid_size)):
            section_row_start_idx = (grid_row_idx + 1) * section_padding_size + \
                grid_row_idx * section_height
            section_row_stop_idx = section_row_start_idx + section_height
            section_col_start_idx = (grid_col_idx + 1) * section_padding_size + \
                grid_col_idx * section_width
            section_col_stop_idx = section_col_start_idx + section_width

            sampling_vis_imgs[img_idx, section_row_start_idx:section_row_stop_idx,
                              section_col_start_idx:section_col_stop_idx] = sw_rays[img_idx, grid_row_idx, grid_col_idx, :, :, 2]

        # Draw a dot at the location of every sampled pixel.
        for img_idx, grid_row_idx, grid_col_idx, pixel_row_idx, pixel_col_idx in sampled_pw_idxs:
            section_row_start_idx = (grid_row_idx + 1) * section_padding_size + \
                grid_row_idx * section_height
            section_col_start_idx = (grid_col_idx + 1) * section_padding_size + \
                grid_col_idx * section_width

            pixel_row_idx = (pixel_row_idx + section_row_start_idx).item()
            pixel_col_idx = (pixel_col_idx + section_col_start_idx).item()

            sampling_vis_img_np = sampling_vis_imgs_np[img_idx]
            circle(sampling_vis_img_np, (pixel_col_idx, pixel_row_idx), 3, (0, 0, 0), FILLED)
            circle(sampling_vis_img_np, (pixel_col_idx, pixel_row_idx), 2, (1, 1, 1), FILLED)
            circle(sampling_vis_img_np, (pixel_col_idx, pixel_row_idx), 1, (1, 0, 0), FILLED)

        tensorboard.add_images(tensorboard_tag, sampling_vis_imgs, train_iter_idx,
                               dataformats='NHWC')

    t_delta = perf_counter() - t_start
    if verbose:
        print(f' done in {t_delta:.3f} seconds.')

    return sampled_rays, sampled_pw_idxs, t_delta


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


def get_sw_n_sampled(sampled_sw_idxs_tuple: Tuple[torch.Tensor, ...], dims_sw: Tuple[int, int, int]
                     ) -> torch.Tensor:
    sw_n_sampled = torch.index_put(torch.zeros(dims_sw, dtype=torch.int64), sampled_sw_idxs_tuple,
                                   torch.tensor([1], dtype=torch.int64), accumulate=True)

    return sw_n_sampled


def get_sw_loss(rendered_rgbs: torch.Tensor, gt_rgbs: torch.Tensor, sw_n_sampled: torch.Tensor,
                sampled_sw_idxs_tuple: Tuple[torch.Tensor, ...], dims_sw: Tuple[int, int, int],
                cpu: torch.device,
                ) -> Tuple[torch.Tensor, float]:
    t_start = perf_counter()

    rendered_rgbs = rendered_rgbs.to(cpu)
    gt_rgbs = gt_rgbs.to(cpu)

    mean_squared_diffs = torch.mean((rendered_rgbs - gt_rgbs) ** 2, axis=1)
    sw_cumu_mean_squared_diffs = torch.index_put(torch.zeros(dims_sw), sampled_sw_idxs_tuple,
                                                 mean_squared_diffs, accumulate=True)
    # For each section, the average mean squared difference between rendered RGB and groundtruth RGB
    # for each sampled pixel within the section.
    sw_loss = torch.nan_to_num(sw_cumu_mean_squared_diffs / sw_n_sampled, nan=0)

    t_delta = perf_counter() - t_start

    return sw_loss, t_delta


def add_1d_imgs_to_tensorboard(imgs: torch.Tensor, img_rgb: torch.Tensor,
                               tensorboard: SummaryWriter, tag: str, iter_idx: int,
                               padding_rgb: torch.Tensor = torch.tensor([0.6]).expand(3)
                               ) -> torch.Tensor:
    assert imgs.dim() == 3

    # Scale all elements between 0 and 1.
    imgs_scaled = imgs / torch.max(imgs)

    # If an element has the value 0, then it should receive the RGB value of zero_rgb.
    zero_rgb = torch.Tensor([1]).expand(3)
    # If an element has the value 1 (the max value after scaling), then it should receive the RGB
    # value of max_rgb.
    max_rgb = img_rgb
    diff_rgb = max_rgb - zero_rgb

    # Scale the
    output_imgs = (zero_rgb.repeat(imgs.shape[0], imgs.shape[1], imgs.shape[2], 1) +
                   imgs_scaled.unsqueeze(-1).repeat(1, 1, 1, 3) * diff_rgb)

    # Add gray border around each image.
    output_imgs = torch.cat((
        padding_rgb.expand(imgs.shape[0], 1, imgs.shape[2], 3),
        output_imgs,
        padding_rgb.expand(imgs.shape[0], 1, imgs.shape[2], 3)),
        dim=1)
    output_imgs = torch.cat((
        padding_rgb.expand(imgs.shape[0], imgs.shape[1] + 2, 1, 3),
        output_imgs,
        padding_rgb.expand(imgs.shape[0], imgs.shape[1] + 2, 1, 3)),
        dim=2)

    tensorboard.add_images(tag, output_imgs, iter_idx, dataformats='NHWC')
