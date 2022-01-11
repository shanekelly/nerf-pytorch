import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ipdb import set_trace
from itertools import product
from matplotlib.patches import Circle
from time import perf_counter
from torch.utils.tensorboard import SummaryWriter
from typing import List, Tuple

from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data
from load_LINEMOD import load_LINEMOD_data
from load_bonn import load_bonn_data


# Misc
def img2mse(x, y): return torch.mean((x - y) ** 2)
def mse2psnr(x): return -10. * torch.log(x) / torch.log(torch.Tensor([10.]))


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
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
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
            [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])

        # Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
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
            h = F.relu(h)

            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

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
def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H), indexing='ij'
                          )  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
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
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples

    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers

    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]

        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

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


def load_data(args):
    # Load data
    K = None

    if args.dataset_type == 'llff':
        images, poses, bds, render_poses, test_idxs = load_llff_data(args.datadir, args.factor,
                                                                     recenter=True, bd_factor=.75,
                                                                     spherify=args.spherify)
        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)

        if not isinstance(test_idxs, list):
            test_idxs = [test_idxs]

        if args.llffhold > 0:
            print('Auto LLFF holdout,', args.llffhold)
            test_idxs = np.arange(images.shape[0])[::args.llffhold]

        val_idxs = test_idxs
        train_idxs = np.array([i for i in np.arange(int(images.shape[0])) if
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
        images, poses, render_poses, hwf, i_split = load_blender_data(
            args.datadir, args.half_res, args.testskip)
        print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
        train_idxs, val_idxs, test_idxs = i_split

        near = 2.
        far = 6.

        if args.white_bkgd:
            images = images[..., :3]*images[..., -1:] + (1.-images[..., -1:])
        else:
            images = images[..., :3]

    elif args.dataset_type == 'LINEMOD':
        images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(
            args.datadir, args.half_res, args.testskip)
        print(f'Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}')
        print(f'[CHECK HERE] near: {near}, far: {far}.')
        train_idxs, val_idxs, test_idxs = i_split

        if args.white_bkgd:
            images = images[..., :3]*images[..., -1:] + (1.-images[..., -1:])
        else:
            images = images[..., :3]

    elif args.dataset_type == 'deepvoxels':

        images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
                                                                 basedir=args.datadir,
                                                                 testskip=args.testskip)

        print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, args.datadir)
        train_idxs, val_idxs, test_idxs = i_split

        hemi_R = np.mean(np.linalg.norm(poses[:, :3, -1], axis=-1))
        near = hemi_R-1.
        far = hemi_R+1.

    elif args.dataset_type == 'bonn':
        images, hwf, poses, render_poses, train_idxs, test_idxs = load_bonn_data(
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

    return images, hwf, H, W, focal, K, poses, render_poses, train_idxs, test_idxs, val_idxs, near, far


def get_all_rays_np(images: np.ndarray, H: int, W: int, K: np.ndarray, poses: np.ndarray,
                    n_train_imgs: int, train_idxs: List[int], grid_size: int, section_height: int,
                    section_width: int, verbose=False) -> Tuple[np.ndarray, float]:
    if verbose:
        print('Getting rays...', end='')

    t_start = perf_counter()

    # sk: A ray for every pixel of every camera image. Shape (N, ro+rd, H, W, 3).
    rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:, :3, :4]], 0)

    rays = np.concatenate([rays, images[:, None]], 1)  # (N, ro+rd+rgb, H, W, 3)
    rays = np.transpose(rays, [0, 2, 3, 1, 4])  # (N, H, W, ro+rd+rgb, 3)
    rays = np.stack([rays[i] for i in train_idxs], 0)  # (N train, H, W, ro+rd+rgb, 3)

    section_rays = np.empty((n_train_imgs, grid_size, grid_size,
                             section_height, section_width, 3, 3))
    for grid_row_idx, grid_col_idx in product(range(grid_size), range(grid_size)):
        img_start_row_idx = grid_row_idx * section_height
        img_stop_row_idx = (grid_row_idx + 1) * section_height
        img_start_col_idx = grid_col_idx * section_width
        img_stop_col_idx = (grid_col_idx + 1) * section_width
        section_rays[:, grid_row_idx, grid_col_idx, :, :, :] = \
            rays[:, img_start_row_idx:img_stop_row_idx, img_start_col_idx:img_stop_col_idx, :, :]

    # sk: shape (num training images, num grid rows, num grid cols, num pixels per section, ro+rd+rgb, 3)
    section_rays = section_rays.astype(np.float32)

    t_delta = perf_counter() - t_start

    if verbose:
        print(f' done in {t_delta:.4f} seconds.')

    return section_rays, t_delta


def get_initial_section_rand_pixel_idxs(num_train_imgs: int, grid_size: int, section_height: int,
                                        section_width: int, verbose: bool = False
                                        ) -> Tuple[np.ndarray, float]:
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
        print('Getting initial grid section random pixel indices...', end='')

    num_pixels_per_section = section_height * section_width
    single_section_pixel_idxs = list(product(range(section_height), range(section_width)))
    section_rand_pixel_idxs = np.broadcast_to(single_section_pixel_idxs,
                                              (num_train_imgs, grid_size, grid_size,
                                               num_pixels_per_section, 2)).copy()
    for img_idx, grid_row_idx, grid_col_idx in product(range(num_train_imgs),
                                                       range(grid_size),
                                                       range(grid_size)):
        rand_pixel_idxs = np.random.permutation(num_pixels_per_section)
        section_rand_pixel_idxs[img_idx, grid_row_idx, grid_col_idx, :, :] = \
            section_rand_pixel_idxs[img_idx, grid_row_idx, grid_col_idx, rand_pixel_idxs, :]

    t_delta = perf_counter() - t_start

    if verbose:
        print(f' done in {t_delta:.4f} seconds.')

    return section_rand_pixel_idxs, t_delta


def sample_section_rays(section_rays: np.ndarray, section_rand_pixel_idxs: np.ndarray,
                        section_sampling_start_idxs: np.ndarray, section_n_rays_to_sample: np.ndarray,
                        n_train_imgs: int, grid_size: int, n_pixels_per_section: int,
                        tensorboard: SummaryWriter, tensorboard_tag: str, train_iter_idx: int,
                        log_sampling_fig: bool = False, verbose: bool = False
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    if verbose:
        print('Sampling rays across grid sections...', end='')

    t_start = perf_counter()

    if log_sampling_fig:
        fig = plt.figure(figsize=(3.5, n_train_imgs * 2.5))
        sub_figs = fig.subfigures(n_train_imgs, 1)

    sampled_rays = np.empty((0, 3, 3))
    section_sampled_bounding_idxs = np.empty((n_train_imgs, grid_size, grid_size, 2), dtype=int)
    for img_idx in range(n_train_imgs):
        if log_sampling_fig:
            img_sub_figs = sub_figs[img_idx]
            axs = img_sub_figs.subplots(grid_size, grid_size)
        for grid_row_idx, grid_col_idx in product(range(grid_size), range(grid_size)):
            n_rays_to_sample = section_n_rays_to_sample[img_idx, grid_row_idx, grid_col_idx]
            pixel_idxs_to_sample = np.empty((0, 2), dtype=int)
            while pixel_idxs_to_sample.shape[0] < n_rays_to_sample:
                n_pixels_to_add = n_rays_to_sample - pixel_idxs_to_sample.shape[0]
                start_idx = section_sampling_start_idxs[img_idx, grid_row_idx, grid_col_idx]
                stop_idx = start_idx + n_pixels_to_add
                pixel_idxs_to_sample = np.concatenate(
                    (pixel_idxs_to_sample, section_rand_pixel_idxs[img_idx, grid_row_idx, grid_col_idx,
                                                                   start_idx:stop_idx, :]))

                section_sampling_start_idxs[img_idx, grid_row_idx, grid_col_idx] += n_pixels_to_add

                if (section_sampling_start_idxs[img_idx, grid_row_idx, grid_col_idx] >=
                        n_pixels_per_section):
                    rand_idxs = np.random.permutation(n_pixels_per_section)
                    section_rand_pixel_idxs[img_idx, grid_row_idx, grid_col_idx, :, :] = \
                        section_rand_pixel_idxs[img_idx, grid_row_idx, grid_col_idx, rand_idxs, :]
                    section_sampling_start_idxs[img_idx, grid_row_idx, grid_col_idx] = 0

            section_sampled_bounding_idxs[img_idx, grid_row_idx, grid_col_idx, 0] = \
                sampled_rays.shape[0]
            sampled_rays = np.concatenate((
                sampled_rays, section_rays[img_idx, grid_row_idx, grid_col_idx,
                                           pixel_idxs_to_sample[:, 0], pixel_idxs_to_sample[:, 1], :]))
            section_sampled_bounding_idxs[img_idx, grid_row_idx, grid_col_idx, 1] = \
                sampled_rays.shape[0]

            if log_sampling_fig:
                ax = axs[grid_row_idx, grid_col_idx]
                ax.imshow(section_rays[img_idx, grid_row_idx, grid_col_idx, :, :, 2, :])
                for pixel_row_idx, pixel_col_idx in pixel_idxs_to_sample:
                    ax.add_patch(Circle((pixel_col_idx, pixel_row_idx), 2, color='black'))
                    ax.add_patch(Circle((pixel_col_idx, pixel_row_idx), 1.5, color='white'))
                    ax.add_patch(Circle((pixel_col_idx, pixel_row_idx), 0.5, color='red'))
                ax.axis('off')

    if log_sampling_fig:
        tensorboard.add_figure(tensorboard_tag, fig, train_iter_idx)

    t_delta = perf_counter() - t_start
    if verbose:
        print(f' done in {t_delta:.4f} seconds.')

    return (sampled_rays, section_sampled_bounding_idxs, section_rand_pixel_idxs,
            section_sampling_start_idxs, t_delta)
