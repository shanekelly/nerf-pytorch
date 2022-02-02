import os
import imageio

from pathlib import Path
from pickle import dump
from time import time, perf_counter
from typing import Dict, Optional

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

from run_nerf_helpers import (add_1d_imgs_to_tensorboard, create_keyframes, get_coordinate_frames,
                              get_idxs_tuple, get_kf_poses, get_sw_n_sampled, get_sw_rays,
                              get_embedder, get_rays, get_sw_loss, img2mse, load_data, mse2psnr,
                              NeRF, ndc_rays, pad_imgs, pad_sections, render,
                              render_and_compute_loss, sample_pdf, sample_sw_rays, select_keyframes,
                              to8b, tfmats_from_minreps, minreps_from_tfmats, white_rgb)


cpu = torch.device('cpu')
if torch.cuda.is_available():
    gpu_if_available = torch.device('cuda')
    print('GPU is available!')
else:
    gpu_if_available = cpu
    print('GPU not available!')
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
        rgb, disp, acc, _ = render(H, W, K, gpu_if_available, chunk=chunk,
                                   c2w=c2w[:3, :4], **render_kwargs)
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
        kf_poses_params = None
    else:
        # Convert the initial pose transformation matrices into 6-element minimal representations,
        # then add them to grad_vars, which will get passed to the optimizer.
        kf_poses_params = Parameter(minreps_from_tfmats(initial_poses, gpu_if_available))
        with torch.no_grad():
            grad_vars.extend([kf_poses_params])

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
            kf_poses_params)


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
    parser.add_argument("--i_test", type=int, default=5000,
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
    parser.add_argument('--keyframe_creation_strategy', choices=['all', 'every_Nth'], default='all',
                        help='The keyframe creation strategy to use. Choose between using every '
                        'frame as a keyframe or using every Nth frame as a keyframe, where N is '
                        'defined by --every_Nth.')
    parser.add_argument('--every_Nth', type=int, help='Only used if '
                        '--keyframe_creation_strategy is "every_Nth". The value of N to use when '
                        'choosing every Nth frame as a keyframe.')
    parser.add_argument('--keyframe_selection_strategy', choices=['all', 'explore_exploit'],
                        default='all', help='The keyframe creation strategy to use. Choose between '
                        'using every frame as a keyframe or using every Nth frame as a keyframe, '
                        'where N is defined by --every_Nth.')
    parser.add_argument('--n_explore', type=int, help='Only used if '
                        '--keyframe_selection_strategy is "explore_exploit". The number of '
                        'keyframes to randomly select.')
    parser.add_argument('--n_exploit', type=int, help='Only used if '
                        '--keyframe_selection_strategy is "explore_exploit". The number of '
                        'keyframes with the highest loss to select.')

    return parser


def train() -> None:
    parser = config_parser()
    args = parser.parse_args()

    tensorboard = SummaryWriter(Path(args.basedir) / args.expname, flush_secs=10)

    # Load data from specified dataset.
    (rgb_imgs, depth_imgs, hwf, H, W, focal, K, poses, render_poses, train_idxs, test_idxs,
     val_idxs, near, far) = load_data(args, gpu_if_available)
    test_rgb_imgs = rgb_imgs[test_idxs]
    test_poses = poses[test_idxs]

    kf_rgb_imgs, kf_initial_poses, kf_idxs, n_kfs = \
        create_keyframes(rgb_imgs, poses, train_idxs, args.keyframe_creation_strategy,
                         args.every_Nth)

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
     kf_poses_params) = create_nerf(args, kf_initial_poses)
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
    grid_size = args.img_grid_side_len
    do_active_sampling = not args.no_active_sampling
    do_uniform_sampling_every_iter = args.uniformly_sample_every_iteration
    do_pose_optimization = not args.no_pose_optimization

    assert H % grid_size == 0
    assert W % grid_size == 0
    section_height = H // grid_size
    section_width = W // grid_size

    n_sections_per_img = grid_size ** 2
    n_total_sections = n_kfs * n_sections_per_img
    dims_kf_pw = (n_kfs, grid_size, grid_size, section_height, section_width)
    dims_kf_sw = dims_kf_pw[:3]
    if do_active_sampling:
        n_rays_to_uniformly_sample_per_img = 200
        n_total_rays_to_uniformly_sample = n_rays_to_uniformly_sample_per_img * n_kfs
        sw_unif_sampling_prob_dist = torch.tensor(1 / n_total_sections).repeat(dims_kf_sw)
        sw_kf_loss = torch.zeros(dims_kf_sw)
        n_total_rays_to_actively_sample = N_rand
    else:
        n_total_rays_to_sample = N_rand
        sw_sampling_prob_dist = \
            torch.tensor(1 / n_total_sections).expand(dims_kf_sw)

    sw_total_n_sampled = torch.zeros(dims_kf_sw, dtype=torch.int64)

    tensorboard.add_images('train/keyframes', pad_sections(kf_rgb_imgs, dims_kf_pw, white_rgb,
                                                           padding_width=2), global_step=1, dataformats='NHWC')

    open3d_vis_count = 1
    start_iter_idx += 1
    n_training_iters = args.n_training_iters + 1
    tqdm_bar = trange(start_iter_idx, n_training_iters)
    for train_iter_idx in tqdm_bar:
        t_train_iter_start = perf_counter()

        log_sampling_vis = train_iter_idx % args.i_sampling_vis == 0
        log_pose_vis = train_iter_idx % args.i_pose_vis == 0
        is_first_iter = train_iter_idx == start_iter_idx

        # Get the keyframe poses.
        kf_poses = get_kf_poses(kf_initial_poses, kf_poses_params, do_pose_optimization)

        # Select a subset of the keyframes to actually use in this training iteration.
        (skf_rgb_imgs, skf_poses, skf_idxs, n_skfs, dims_skf_pw, sw_kf_loss, sw_skf_loss,
         skf_from_kf_idxs) = \
            select_keyframes(kf_rgb_imgs, kf_poses, kf_idxs, sw_kf_loss, H, W, K, is_first_iter,
                             n_total_rays_to_uniformly_sample, dims_kf_pw,
                             args.keyframe_selection_strategy, args.n_explore, args.n_exploit,
                             sw_unif_sampling_prob_dist, tensorboard, verbose, train_iter_idx,
                             optimizer, args.chunk, render_kwargs_train, cpu, gpu_if_available)
        dims_skf_sw = dims_skf_pw[:3]

        # Compute the rays from the selected keyframe images and poses.
        sw_skf_rays, t_get_rays = \
            get_sw_rays(skf_rgb_imgs, H, W, K, skf_poses, n_skfs,
                        grid_size, section_height, section_width, gpu_if_available)

        if do_active_sampling:
            # If we aren't uniformly sampling on every iteration, then we should only uniformly
            # sample on the first iteration of this run.
            t_uniform_sampling = 0.0
            t_uniform_batching = 0.0
            t_uniform_loss = 0.0
            t_uniform_rendering = 0.0
            if do_uniform_sampling_every_iter:
                # Uniform ray sampling.
                enforce_min_samples = False
                with torch.no_grad():
                    (uniformly_sampled_rays, uniformly_sampled_pw_idxs, sw_n_newly_sampled, t_uniform_sampling) = \
                        sample_sw_rays(sw_skf_rays, sw_unif_sampling_prob_dist,
                                       n_total_rays_to_uniformly_sample,
                                       H, W, dims_kf_pw,
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
                    unif_rendered_rgbs, _, _, _ = render(H, W, K, gpu_if_available, chunk=args.chunk, rays=batch_rays,
                                                         verbose=train_iter_idx < 10, retraw=True,
                                                         **render_kwargs_train)
                t_uniform_rendering = perf_counter() - t_uniform_rendering_start

                # Computing section-wise loss for uniformly sampled rays.
                # TODO: should this not be sw_kf_loss?
                sw_kf_loss, t_uniform_loss = \
                    get_sw_loss(unif_rendered_rgbs, unif_gt_rgbs, sw_n_newly_sampled,
                                sampled_skf_sw_idxs_tuple,
                                dims_kf_sw, cpu)

            # Active ray sampling.
            sw_active_sampling_prob_dist = sw_skf_loss / torch.sum(sw_skf_loss)
            (sampled_rays, sampled_skf_sw_idxs, sw_n_newly_sampled, t_active_sampling) = \
                sample_sw_rays(sw_skf_rays, sw_active_sampling_prob_dist[skf_from_kf_idxs],
                               n_total_rays_to_actively_sample,
                               kf_rgb_imgs, H, W, dims_kf_pw, dims_skf_pw, skf_from_kf_idxs,
                               tensorboard, 'train/active_sampling/sampled_pixels',
                               train_iter_idx, log_sampling_vis=log_sampling_vis,
                               verbose=verbose)
        else:
            # Active sampling is disabled, so simply sample uniformly over all sections.
            (sampled_rays, sampled_skf_sw_idxs, sw_n_newly_sampled, t_sampling) = \
                sample_sw_rays(sw_skf_rays, sw_sampling_prob_dist, n_total_rays_to_sample,
                               H, W, dims_kf_pw,
                               tensorboard, 'train/sampling/sampled_pixels', train_iter_idx,
                               log_sampling_vis=log_sampling_vis, verbose=verbose)

        # Accumulate the total number of times each section has been sampled from so that it can be
        # visualized when log_sampling_vis is True.
        sw_total_n_sampled[skf_from_kf_idxs] += sw_n_newly_sampled

        sampled_skf_sw_idxs_tuple = get_idxs_tuple(sampled_skf_sw_idxs[:, :3])
        (train_rendered_rgbs, train_gt_rgbs, _, new_sw_skf_loss, train_loss, train_psnr, t_batching,
         t_rendering, t_loss) = \
            render_and_compute_loss(sampled_rays, K, render_kwargs_train, H, W, dims_skf_sw, args.chunk,
                                    sampled_skf_sw_idxs_tuple, sw_n_newly_sampled, optimizer,
                                    train_iter_idx, cpu, gpu_if_available)

#         t_batching_start = perf_counter()
#         batch = torch.transpose(sampled_rays.to(gpu_if_available), 0, 1)
#         batch_rays, train_gt_rgbs = batch[:2], batch[2]
#         t_batching = perf_counter() - t_batching_start

#         t_rendering_start = perf_counter()
#         train_rendered_rgbs, _, _, extras = render(H, W, K, chunk=args.chunk, rays=batch_rays,
#                                                    verbose=train_iter_idx < 10, retraw=True,
#                                                    **render_kwargs_train)
#         t_rendering = perf_counter() - t_rendering_start

#         t_loss_start = perf_counter()
#         optimizer.zero_grad()
#         train_loss = img2mse(train_rendered_rgbs, train_gt_rgbs)
#         train_psnr = mse2psnr(train_loss)
#         if 'rgb0' in extras:
#             img_loss0 = img2mse(extras['rgb0'], train_gt_rgbs)
#             train_loss += img_loss0
#             # psnr0 = mse2psnr(img_loss0)
#         t_loss = perf_counter() - t_loss_start

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
            test_loss = img2mse(test_rendered_rgbs, test_rgb_imgs.to(cpu))
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

            add_1d_imgs_to_tensorboard(sw_kf_loss, torch.Tensor([0, 0.8, 0]), tensorboard,
                                       'train/estimated_loss_distribution', train_iter_idx)

        if train_iter_idx % args.i_val == 0:
            # Log a rendered validation view to Tensorboard.
            img_i = val_idxs[-2] if len(val_idxs) > 1 else val_idxs[0]
            val_gt_rgbs = rgb_imgs[img_i]
            c2w = poses[img_i, :3, :4].to(gpu_if_available)
            with torch.no_grad():
                val_rendered_rgb, val_rendered_disp, _, extras = \
                    render(H, W, K, gpu_if_available, chunk=args.chunk,
                           c2w=c2w, **render_kwargs_test)

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
            kf_initial_poses_np = kf_initial_poses.cpu().numpy()
            kf_poses_np = kf_poses.detach().cpu().numpy()
            initial_coordinate_frames = get_coordinate_frames(kf_initial_poses_np, gray_out=True)
            optimized_coordinate_frames = get_coordinate_frames(kf_poses_np)
            for idx in range(len(initial_coordinate_frames)):
                # TODO: Efficiently store initial poses, since they are always the same at every
                # iteration.
                tensorboard.add_3d(f'train/pose{idx:03d}-initial',
                                   to_dict_batch([initial_coordinate_frames[idx]]),
                                   step=open3d_vis_count)
                tensorboard.add_3d(f'train/pose{idx:03d}-optimized',
                                   to_dict_batch([optimized_coordinate_frames[idx]]),
                                   step=open3d_vis_count)

            open3d_vis_count += 1

        if do_active_sampling and not do_uniform_sampling_every_iter:
            # Update the estimated section-wise loss probability distribution using the loss from
            # the rays that were just actively sampled.
            with torch.no_grad():
                # new_loss, _ = get_sw_loss(train_rendered_rgbs, train_gt_rgbs, sw_n_newly_sampled,
                #                           sampled_skf_sw_idxs_tuple, dims_skf_sw, cpu)
                # Only update the loss of sections that were sampled from.
                sampled_kf_sw_idxs_tuple = \
                    tuple([torch.tensor([skf_from_kf_idxs[idx] for idx in sampled_skf_sw_idxs_tuple[0]]),
                           sampled_skf_sw_idxs_tuple[1], sampled_skf_sw_idxs_tuple[2]])
                sw_kf_loss[sampled_kf_sw_idxs_tuple] = new_sw_skf_loss[sampled_skf_sw_idxs_tuple]
                # set_trace()

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
