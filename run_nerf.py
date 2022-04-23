import os
import imageio

from pathlib import Path
from pickle import dump, load
from time import time, perf_counter
from typing import Dict, Optional
from queue import Queue

import numpy as np
import open3d as o3d
import torch

from GPUtil import showUtilization
from ipdb import launch_ipdb_on_exception, set_trace
from open3d.visualization.tensorboard_plugin import summary
from open3d.visualization.tensorboard_plugin.util import to_dict_batch
from torch.nn import Parameter
from torch.nn.functional import relu
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from axes.util import o3d_axes_from_poses
from im_util.transforms import euler_zyx_from_tfmat

from run_nerf_helpers import (add_1d_imgs_to_tensorboard, append_to_log_file, create_keyframes,
                              extract_mesh, get_depth_loss_iters_multiplier,
                              get_idxs_tuple, get_intrinsics_matrix, get_kf_poses, get_log_fpath,
                              get_sw_rays, get_embedder, get_rays,
                              get_sw_sampling_prob_dist_modifier, get_pw_sampling_prob_modifier,
                              intrinsics_params_from_intrinsics_matrix, img2mse, initialize_sw_kf_loss,
                              iw_from_pw, load_data, load_data_and_create_keyframes,
                              log_depth_loss_meters_multiplier_function,
                              log_depth_loss_iters_multiplier_function,
                              mse2psnr, NeRF, ndc_rays, pad_imgs, pad_sections, render,
                              render_and_compute_loss, sample_pdf, sample_skf_rays,
                              save_imgs, save_poses, save_point_clouds_from_rgb_imgs_and_depth_imgs, select_keyframes,
                              should_trigger, split_into_sections, to8b, tfmats_from_minreps,
                              minreps_from_tfmats, gray_rgb, purple_rgb, black_rgb, white_rgb, GpuMonitor)


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


def render_path(render_poses, hwf, intrinsics_matrix, chunk, render_kwargs, gt_imgs=None, savedir=None,
                render_factor=0):

    img_height, img_width, _ = hwf

    if render_factor != 0:
        # Render downsampled for speed
        img_height = img_height//render_factor
        img_width = img_width//render_factor

    rgbs = []
    disps = []
    depths = []

    t = time()

    for i, c2w in enumerate(tqdm(render_poses)):
        # print(i, time() - t)
        t = time()
        rgb, disp, _, depth, _ = render(img_height, img_width, intrinsics_matrix, gpu_if_available, chunk=chunk,
                                        c2w=c2w[:3, :4], **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        depths.append(depth.cpu().numpy())

        # if i == 0:
        #     print(rgb.shape, disp.shape)

        if savedir is not None:
            if not isinstance(savedir, Path):
                savedir = Path(savedir).expanduser()
            rgb8 = to8b(rgbs[-1])
            rgb_fpath = savedir / f'{i:03d}.png'
            imageio.imwrite(rgb_fpath, rgb8)

    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)
    depths = np.stack(depths, 0)

    return rgbs, disps, depths


def create_nerf(args, initial_poses: torch.Tensor, initial_intrinsics_matrix: torch.Tensor):
    """Instantiate NeRF's MLP model.
    """
    # Load checkpoints
    basedir = args.basedir
    expname = args.expname

    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(
            os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    print('Found ckpts', ckpts)

    ckpt = None
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=gpu_if_available)

    if args.no_gaussian_positional_embedding:
        B = None
    else:
        embedding_size = 256
        scale = args.initial_gpe_scale
        B = torch.normal(0, 1, (3, embedding_size), device=gpu_if_available) * scale
        if args.B_opt:
            B = Parameter(B)
        if ckpt is not None:
            B = ckpt['B']

    embed_fn, input_ch = get_embedder(args.multires, args.i_embed, B=B)

    input_ch_views = 0
    embeddirs_fn = None

    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]
    model = NeRF(D=args.netdepth, img_width=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(gpu_if_available)
    grad_vars = [{'params': model.parameters(), 'lr': args.initial_scene_lr}]

    model_fine = None

    if args.N_importance > 0:
        model_fine = NeRF(D=args.netdepth_fine, img_width=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views,
                          use_viewdirs=args.use_viewdirs).to(gpu_if_available)
        grad_vars += [{'params': model_fine.parameters(), 'lr': args.initial_scene_lr}]

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
            grad_vars.append({'params': kf_poses_params, 'lr': args.initial_poses_lr})

    if not args.no_gaussian_positional_embedding and args.B_opt:
        with torch.no_grad():
            grad_vars.append({'params': B, 'lr': args.initial_gpe_mat_lr})

    if args.no_intrinsics_optimization:
        intrinsics_params = None
    else:
        # Convert the initial pose transformation matrices into 6-element minimal representations,
        # then add them to grad_vars, which will get passed to the optimizer.
        intrinsics_params = Parameter(intrinsics_params_from_intrinsics_matrix(
            initial_intrinsics_matrix, gpu_if_available))
        with torch.no_grad():
            grad_vars.append({'params': intrinsics_params, 'lr': args.initial_intrinsics_lr})

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, betas=(0.9, 0.999))

    start_iter_idx = 0
    open3d_vis_count = 0

    ##########################

    if ckpt is not None:
        start_iter_idx = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])

        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

        kf_poses_params = ckpt['kf_poses_params']
        intrinsics_params = ckpt['intrinsics_params']
        open3d_vis_count = ckpt['open3d_vis_count']

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
            kf_poses_params, intrinsics_params, open3d_vis_count, B)


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
    parser.add_argument("--N_rand", type=int, default=2048,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--initial_scene_lr", type=float, default=5e-4,
                        help='Initial learning rate for neural volume weights.')
    parser.add_argument("--scene_lr_10_pct_pt", type=int, default=60000,
                        help='After this many training iterations, the learning rate will have '
                        'decayed to 10 percent of its original value.')
    parser.add_argument("--initial_poses_lr", type=float, default=1e-3,
                        help='Initial learning rate for camera poses.')
    parser.add_argument("--poses_lr_10_pct_pt", type=int, default=30000,
                        help='After this many training iterations, the learning rate will have '
                        'decayed to 10 percent of its original value.')
    parser.add_argument("--initial_gpe_mat_lr", type=float, default=5e-3,
                        help='Initial learning rate for the Gaussian positional embedding matrix.')
    parser.add_argument("--gpe_mat_lr_10_pct_pt", type=int, default=60000,
                        help='After this many training iterations, the learning rate will have '
                        'decayed to 10 percent of its original value.')
    parser.add_argument("--initial_intrinsics_lr", type=float, default=5e-3,
                        help='Initial learning rate for the Gaussian positional embedding matrix.')
    parser.add_argument("--intrinsics_lr_10_pct_pt", type=int, default=30000,
                        help='After this many training iterations, the learning rate will have '
                        'decayed to 10 percent of its original value.')
    parser.add_argument("--chunk", type=int, default=8192,
                        help='number of rays processed in parallel, decrease if running out of '
                        'memory')
    parser.add_argument("--netchunk", type=int, default=32768,
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
    parser.add_argument("--raw_noise_std", type=float, default=1e0,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--img_type_to_render", choices=['test', 'train', 'render'],
                        default='train',
                        help='Only relevant if "render_only" is set. Defines which type of images '
                        'to render. "test" renders the test images with their respective '
                        'initial poses. "train" renders the training keyframes with their '
                        'respective poses (optimized poses are used if available). "render" '
                        'renders the render path, which is defined during dataset loading.')
    parser.add_argument("--render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast '
                        'preview')
    parser.add_argument("--mesh_only", action='store_true',
                        help='do not optimize, reload weights and save mesh to a file')
    parser.add_argument("--mesh_grid_size", type=int, default=100,
                        help='number of grid points to sample in each dimension for marching cubes')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='bonn',
                        help='options: llff / blender / deepvoxels / bonn')
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
    parser.add_argument("--factor", type=int, default=1,
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
    parser.add_argument('--save_logs_to_file', action='store_true', help='Set to save '
                        'some tensorboard visualizations to file in addition to logging them to '
                        'tensorboard.')
    parser.add_argument('--val_idx', type=int)
    parser.add_argument("--i_train_scalars",   type=int, default=0,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--s_train_scalars",   type=int, default=-1,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_val",     type=int, default=0,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--s_val",     type=int, default=-1,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=0,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--s_weights", type=int, default=-1,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_test", type=int, default=0,
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=0,
                        help='frequency of render_poses video saving')
    parser.add_argument('--i_sampling_vis', type=int, default=0, help='The frequency of '
                        'logging visualizations about ray sampling to tensorboard.')
    parser.add_argument('--s_sampling_vis', type=int, default=-1, help='The frequency of '
                        'logging visualizations about ray sampling to tensorboard.')
    parser.add_argument("--i_poses_vis", type=int, default=0,
                        help='Frequency of visualizing keyframe poses.')
    parser.add_argument("--s_poses_vis", type=int, default=-1,
                        help='Frequency of visualizing keyframe poses.')
    parser.add_argument("--i_pose_scalars", type=int, default=0,
                        help='Frequency of visualizing keyframe poses.')
    parser.add_argument("--s_pose_scalars", type=int, default=-1,
                        help='Frequency of visualizing keyframe poses.')
    parser.add_argument("--i_kf_renders_vis", type=int, default=0,
                        help='Frequency of visualizing RGBD results of rendering at keyframe poses.')
    parser.add_argument("--s_kf_renders_vis", type=int, default=-1,
                        help='Frequency of visualizing RGBD results of rendering at keyframe poses.')
    parser.add_argument("--i_point_cloud_vis", type=int, default=0,
                        help='Frequency of visualizing point cloud rendered from '
                        'keyframes.')
    parser.add_argument("--s_point_cloud_vis", type=int, default=-1,
                        help='Frequency of visualizing point cloud rendered from '
                        'keyframes.')
    parser.add_argument("--i_B_vis", type=int, default=0,
                        help='Frequency of visualizing the gaussian positional encoding matrix, '
                        'B.')
    parser.add_argument("--s_B_vis", type=int, default=-1,
                        help='Frequency of visualizing the gaussian positional encoding matrix, '
                        'B.')
    parser.add_argument("--i_intrinsics_vis", type=int, default=0,
                        help='Frequency of visualizing the camera intrinsics.')
    parser.add_argument("--s_intrinsics_vis", type=int, default=-1,
                        help='Frequency of visualizing the camera intrinsics.')
    parser.add_argument('--s_stop', type=int, default=-1,
                        help='When to stop training, in seconds.')

    parser.add_argument('--img_grid_side_len',   type=int, default=8,
                        help='The side length of the grid used to split up training images for '
                        'image active sampling.')
    parser.add_argument('--n_training_iters', type=int, default=50000, help='The number of '
                        'training iterations to run for.')

    parser.add_argument('--verbose', action='store_true', help='True to print additional info.')
    parser.add_argument('--no_active_sampling', action='store_true', help='Set to disable active '
                        'sampling.')
    parser.add_argument('--no_lazy_sw_loss', action='store_true', help='Set to '
                        'uniformly sampled over every image on every iteration in order to '
                        'maintain the estimated probability distribution of loss over image. If '
                        'not set, then all images will be uniformly sampled once at the start, '
                        'but then only active samples will update the estimated probability '
                        'distribution of loss over images.')
    parser.add_argument('--no_pose_optimization', action='store_true', help='Set to make initial '
                        'poses static and disable learning refined poses.')
    parser.add_argument('--no_intrinsics_optimization', action='store_true', help='Set to make initial '
                        'intrinsics static and disable learning refined intrinsics.')
    parser.add_argument('--keyframe_creation_strategy', choices=['all', 'every_Nth'],
                        default='every_Nth',
                        help='The keyframe creation strategy to use. Choose between using every '
                        'frame as a keyframe or using every Nth frame as a keyframe, where N is '
                        'defined by --every_Nth.')
    parser.add_argument('--every_Nth', type=int, default=10, help='Only used if '
                        '--keyframe_creation_strategy is "every_Nth". The value of N to use when '
                        'choosing every Nth frame as a keyframe.')
    parser.add_argument('--keyframe_selection_strategy', choices=['all', 'explore_exploit'],
                        default='explore_exploit', help='The keyframe creation strategy to use. Choose between '
                        'using every frame as a keyframe or using every Nth frame as a keyframe, '
                        'where N is defined by --every_Nth.')
    parser.add_argument('--n_explore', type=int, default=1, help='Only used if '
                        '--keyframe_selection_strategy is "explore_exploit". The number of '
                        'keyframes to randomly select.')
    parser.add_argument('--n_exploit', type=int, default=4, help='Only used if '
                        '--keyframe_selection_strategy is "explore_exploit". The number of '
                        'keyframes with the highest loss to select.')
    parser.add_argument('--sw_sampling_prob_dist_modifier_strategy',
                        choices=['fruit_detections', 'uniform', 'avg_saturation'], default='uniform',
                        help='A section-wise modifier will be element-wise multiplied with the ray '
                        'sampling probability distribution before sampling rays. This argument '
                        'allows for customizing that modifier. "uniform" effectively make this '
                        'modifier not exist / have no effect. "avg_saturation" modifies by the '
                        'average saturation of the pixels within each section.')
    parser.add_argument('--pw_sampling_prob_modifier_strategy',
                        choices=['fruit_detections', 'none'], default='none')
    parser.add_argument('--fruit_detection_model_fpath', default='', help='Path to the NN model '
                        'to use for fruit detection if sw_sampling_prob_dist_modifier_strategy '
                        'is set to "fruit_detections".')
    parser.add_argument('--no_gaussian_positional_embedding', action='store_true', help='Set to '
                        'use the standard NeRF positional embedding instead of the Gaussian '
                        'positional embedding with learned B matrix.')
    parser.add_argument('--initial_gpe_scale', type=float, default=12, help='The scale to use for '
                        'initializing the Gaussian Positional Encoding matrix.')
    parser.add_argument('--B_opt', action='store_true', help='Set to enable learning of the '
                        'gaussian positional encoding matrix, B.')
    parser.add_argument('--no_depth_measurements', action='store_true', help='Set to ignore sensor '
                        'depth measurements. Disables direct learning of depth.')
    parser.add_argument('--depth_loss_iters_diminish_point', type=int, default=5000, help='Used to tune '
                        'how quickly the depth loss will be diminished. After the specified number '
                        'of iterations, the exponentially decaying function that determines how much'
                        'the depth loss is diminished will equal 0.01 (ie, 99% of the depth loss '
                        'will be ignored).')

    return parser


def train() -> None:
    # gpu_monitor = GpuMonitor(0.1)

    parser = config_parser()
    args = parser.parse_args()

    log_dpath = Path(args.basedir) / args.expname
    vis_dpath = log_dpath / 'vis'
    vis_dpath.mkdir(parents=True, exist_ok=True)
    tensorboard = SummaryWriter(log_dpath, flush_secs=10)

    (hwf, img_height, img_width, _, initial_intrinsics_matrix, initial_poses, render_poses,
     near, far, val_rgb_img, val_depth_img, val_pose,
     kf_rgb_imgs, kf_depth_imgs, kf_initial_poses, kf_idxs, n_kfs) = \
        load_data_and_create_keyframes(args, gpu_if_available, tensorboard, cpu, args.val_idx)

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
     kf_poses_params, intrinsics_params, open3d_vis_count, B) = create_nerf(args, kf_initial_poses,
                                                                            initial_intrinsics_matrix)
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
            if args.img_type_to_render == 'test':
                # render_test switches to test poses
                assert False
                render_gt_rgb_imgs = initial_poses[test_idxs]
                poses_to_render = initial_poses[test_idxs]
            elif args.img_type_to_render == 'train':
                kf_poses, _ = \
                    get_kf_poses(kf_initial_poses, kf_poses_params, not args.no_pose_optimization,
                                 gpu_if_available)
                render_gt_rgb_imgs = kf_rgb_imgs
                poses_to_render = kf_poses
            elif args.img_type_to_render == 'render':
                # Default is smoother render_poses path
                render_gt_rgb_imgs = None
                poses_to_render = render_poses
            else:
                raise RuntimeError(
                    f'Unrecognized image type to render "{args.img_type_to_render}".')

            # testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format(
            #     'test' if args.img_type_to_render == 'test' else 'path', start_iter_idx))
            # os.makedirs(testsavedir, exist_ok=True)
            # print('test poses shape', render_poses.shape)

            intrinsics_matrix = get_intrinsics_matrix(initial_intrinsics_matrix, intrinsics_params,
                                                      not args.no_intrinsics_optimization, gpu_if_available)
            rendered_rgb_imgs, _, rendered_depth_imgs = \
                render_path(poses_to_render, hwf, intrinsics_matrix, args.chunk, render_kwargs_test,
                            gt_imgs=render_gt_rgb_imgs, savedir=log_dpath,
                            render_factor=args.render_factor)
            save_point_clouds_from_rgb_imgs_and_depth_imgs(vis_dpath,
                                                           rendered_rgb_imgs,
                                                           rendered_depth_imgs, poses_to_render,
                                                           intrinsics_matrix)
            save_imgs(vis_dpath, render_gt_rgb_imgs, rendered_rgb_imgs, rendered_depth_imgs)
            save_poses(vis_dpath, poses_to_render)
            # imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

            return

     # Short circuit if only extracting mesh from trained model
    if args.mesh_only:
        mesh = extract_mesh(render_kwargs_test, mesh_grid_size=args.mesh_grid_size, threshold=0.0)

        testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format(
            'test' if args.img_type_to_render == 'test' else 'path', start_iter_idx))
        os.makedirs(testsavedir, exist_ok=True)
        path = os.path.join(testsavedir, 'mesh.obj')

        print("saving mesh to ", path)

        mesh.export(path)

        return

    # Parse command line arguments.
    N_rand = args.N_rand
    use_batching = not args.no_batching
    assert use_batching is True
    verbose = args.verbose
    grid_size = args.img_grid_side_len
    do_active_sampling = not args.no_active_sampling
    do_lazy_sw_loss = not args.no_lazy_sw_loss
    do_pose_optimization = not args.no_pose_optimization

    assert img_height % grid_size == 0
    assert img_width % grid_size == 0
    section_height = img_height // grid_size
    section_width = img_width // grid_size

    n_sections_per_img = grid_size ** 2
    n_total_sections = n_kfs * n_sections_per_img
    dims_kf_pw = (n_kfs, grid_size, grid_size, section_height, section_width)
    dims_kf_sw = dims_kf_pw[:3]

    # For active sampling.
    n_rays_to_unif_sample_per_img = 200
    n_total_rays_to_unif_sample = n_rays_to_unif_sample_per_img * n_kfs
    sw_unif_sampling_prob_dist = torch.tensor(1 / n_total_sections).repeat(dims_kf_sw)
    n_total_rays_to_actively_sample = N_rand

    # For uniform sampling.
    n_total_rays_to_sample = N_rand
    sw_sampling_prob_dist = torch.tensor(1 / n_total_sections).expand(dims_kf_sw)

    with torch.no_grad():
        intrinsics_matrix = get_intrinsics_matrix(initial_intrinsics_matrix, intrinsics_params, not
                                                  args.no_intrinsics_optimization, gpu_if_available)
        sw_kf_loss = initialize_sw_kf_loss(kf_rgb_imgs, kf_depth_imgs, kf_initial_poses, sw_unif_sampling_prob_dist, dims_kf_pw,
                                           intrinsics_matrix, n_total_rays_to_unif_sample, render_kwargs_train,
                                           args.chunk, optimizer, do_active_sampling, tensorboard,
                                           cpu, gpu_if_available, verbose=verbose)
    sw_total_n_sampled = torch.zeros(dims_kf_sw, dtype=torch.int64)
    pw_total_n_sampled = torch.zeros(dims_kf_pw, dtype=torch.int64)
    tensorboard.add_images('train/keyframes',
                           pad_sections(kf_rgb_imgs, dims_kf_pw, white_rgb, padding_width=2,
                                        desired_img_shape=(120, 160)),
                           global_step=1, dataformats='NHWC')
    sw_sampling_prob_dist_modifier = get_sw_sampling_prob_dist_modifier(kf_rgb_imgs, grid_size,
                                                                        args.sw_sampling_prob_dist_modifier_strategy,
                                                                        tensorboard, cpu,
                                                                        Path(args.fruit_detection_model_fpath).expanduser())
    pw_sampling_prob_modifier = get_pw_sampling_prob_modifier(kf_rgb_imgs,
                                                              args.pw_sampling_prob_modifier_strategy,
                                                              tensorboard, cpu,
                                                              Path(args.fruit_detection_model_fpath).expanduser())

    log_depth_loss_iters_multiplier_function(tensorboard, not args.no_depth_measurements,
                                             args.depth_loss_iters_diminish_point)
    log_depth_loss_meters_multiplier_function(tensorboard)

    t_prev_train_scalars_log = 0.0
    t_prev_val_log = 0.0
    t_prev_sampling_vis_log = 0.0
    t_prev_poses_vis_log = 0.0
    t_prev_pose_scalars_log = 0.0
    t_prev_point_cloud_vis_log = 0.0
    t_prev_weights_log = 0.0
    t_prev_kf_renders_log = 0.0
    t_prev_B_vis_log = 0.0
    t_prev_intrinsics_vis_log = 0.0
    t_training_start = perf_counter()

    already_logged_val_gt = False

    showUtilization()

    start_iter_idx += 1
    open3d_vis_count += 1
    n_training_iters = args.n_training_iters + 1
    tqdm_bar = trange(start_iter_idx, n_training_iters)
    for train_iter_idx in tqdm_bar:
        t_train_iter_start = perf_counter()

        is_first_iter = train_iter_idx == 1
        is_start_iter = train_iter_idx == start_iter_idx

        log_train_scalars = should_trigger(train_iter_idx, args.i_train_scalars,
                                           t_prev_train_scalars_log, args.s_train_scalars)
        log_sampling_vis = should_trigger(train_iter_idx, args.i_sampling_vis, t_prev_sampling_vis_log,
                                          args.s_sampling_vis)
        log_val_vis = should_trigger(train_iter_idx, args.i_val, t_prev_val_log, args.s_val)
        log_poses_vis = should_trigger(train_iter_idx, args.i_poses_vis, t_prev_poses_vis_log,
                                       args.s_poses_vis)
        log_pose_scalars = should_trigger(train_iter_idx, args.i_pose_scalars, t_prev_pose_scalars_log,
                                          args.s_pose_scalars)
        log_point_cloud_vis = should_trigger(train_iter_idx, args.i_point_cloud_vis,
                                             t_prev_point_cloud_vis_log, args.s_point_cloud_vis)
        log_B_vis = should_trigger(train_iter_idx, args.i_B_vis, t_prev_B_vis_log, args.s_B_vis)
        log_weights = should_trigger(train_iter_idx, args.i_weights, t_prev_weights_log,
                                     args.s_weights)
        log_video = should_trigger(train_iter_idx, args.i_video) and train_iter_idx > 0
        log_test = should_trigger(train_iter_idx, args.i_test) and train_iter_idx > 0
        log_kf_renders_log = should_trigger(train_iter_idx, args.i_kf_renders_vis,
                                            t_prev_kf_renders_log, args.s_kf_renders_vis)
        log_intrinsics_vis = should_trigger(train_iter_idx, args.i_intrinsics_vis,
                                            t_prev_intrinsics_vis_log, args.s_intrinsics_vis)
        should_stop = should_trigger(train_iter_idx, 0, t_training_start, args.s_stop)

        # Get the keyframe poses.
        kf_poses, t_get_poses = get_kf_poses(
            kf_initial_poses, kf_poses_params, do_pose_optimization, gpu_if_available)
        intrinsics_matrix = get_intrinsics_matrix(initial_intrinsics_matrix, intrinsics_params, not
                                                  args.no_intrinsics_optimization, gpu_if_available)

        with torch.no_grad():
            # Select a subset of the keyframes to actually use in this training iteration.
            (skf_rgb_imgs, skf_depth_imgs, skf_poses, skf_idxs, n_skfs, dims_skf_pw, sw_skf_loss, skf_from_kf_idxs,
             t_select_keyframes) = select_keyframes(kf_rgb_imgs, kf_depth_imgs, kf_poses, kf_idxs, sw_kf_loss, img_height, img_width,
                                                    intrinsics_matrix, is_start_iter,
                                                    n_total_rays_to_unif_sample, dims_kf_pw,
                                                    args.keyframe_selection_strategy, args.n_explore, args.n_exploit,
                                                    sw_unif_sampling_prob_dist, tensorboard, verbose, train_iter_idx,
                                                    optimizer, args.chunk, render_kwargs_train, cpu, gpu_if_available)
        dims_skf_sw = dims_skf_pw[:3]

        # Compute the rays from the selected keyframe images and poses.
        sw_skf_rays, t_get_rays = get_sw_rays(skf_rgb_imgs, skf_depth_imgs, img_height, img_width, intrinsics_matrix, skf_poses, n_skfs,
                                              grid_size, section_height, section_width, gpu_if_available)

        # Sample rays to use for training.
        sampled_rays, sampled_skf_sw_idxs, sw_n_newly_sampled, pw_n_newly_sampled, t_sample_rays = sample_skf_rays(sw_skf_rays, kf_rgb_imgs, intrinsics_matrix, n_total_rays_to_sample, sw_unif_sampling_prob_dist,
                                                                                                                   n_total_rays_to_unif_sample, sw_sampling_prob_dist,
                                                                                                                   n_total_rays_to_actively_sample, sw_sampling_prob_dist_modifier,
                                                                                                                   sw_skf_loss, dims_kf_pw, dims_skf_pw,
                                                                                                                   skf_from_kf_idxs, args.chunk, render_kwargs_train, train_iter_idx,
                                                                                                                   do_active_sampling, do_lazy_sw_loss, tensorboard, log_sampling_vis,
                                                                                                                   verbose, cpu, gpu_if_available, pw_sampling_prob_modifier)

        # Accumulate the total number of times each section has been sampled from so that it can be
        # visualized when log_sampling_vis is True.
        sw_total_n_sampled[skf_from_kf_idxs] += sw_n_newly_sampled
        pw_total_n_sampled[skf_from_kf_idxs] += pw_n_newly_sampled

        # Render the sampled rays and compute the loss.
        depth_loss_iters_multiplier = get_depth_loss_iters_multiplier(not args.no_depth_measurements,
                                                                      train_iter_idx, args.depth_loss_iters_diminish_point)
        sampled_skf_sw_idxs_tuple = get_idxs_tuple(sampled_skf_sw_idxs[:, :3])
        (_, _, new_sw_skf_loss, new_sw_skf_rgb_loss,
         new_sw_skf_depth_loss, train_loss, train_psnr, t_batching,
         t_rendering, t_loss) = render_and_compute_loss(sampled_rays, intrinsics_matrix, render_kwargs_train,
                                                        img_height, img_width, dims_skf_sw, args.chunk,
                                                        sampled_skf_sw_idxs_tuple, sw_n_newly_sampled,
                                                        depth_loss_iters_multiplier, optimizer,
                                                        train_iter_idx, cpu, gpu_if_available)

        # Compute gradients for parameters via backpropagation and use them to update parameter
        # values.
        t_backprop_start = perf_counter()
        train_loss.backward()

        optimizer.step()
        t_backprop = perf_counter() - t_backprop_start

        # Update learning rate for the coarse network.
        lr_decay_rate = 0.1
        param_group_idx = 0
        new_scene_lr = args.initial_scene_lr * (lr_decay_rate ** (global_step /
                                                                  args.scene_lr_10_pct_pt))
        optimizer.param_groups[param_group_idx]['lr'] = new_scene_lr
        param_group_idx += 1

        # Update learning rate for the fine network.
        if args.N_importance > 0:
            optimizer.param_groups[param_group_idx]['lr'] = new_scene_lr
            param_group_idx += 1

        # Update learning rate for the poses.
        if not args.no_pose_optimization:
            new_poses_lr = args.initial_poses_lr * (lr_decay_rate ** (global_step /
                                                                      args.poses_lr_10_pct_pt))
            optimizer.param_groups[param_group_idx]['lr'] = new_poses_lr
            param_group_idx += 1

        # Update learning rate for the Gaussian positional embedding matrix.
        if args.B_opt:
            new_gpe_mat_lr = args.initial_gpe_mat_lr * (lr_decay_rate ** (global_step /
                                                                          args.gpe_mat_lr_10_pct_pt))
            optimizer.param_groups[param_group_idx]['lr'] = new_gpe_mat_lr
            param_group_idx += 1

        # Update learning rate for the camera intrinsics.
        if not args.no_intrinsics_optimization:
            new_intrinsics_lr = args.initial_intrinsics_lr * (lr_decay_rate ** (global_step /
                                                                                args.intrinsics_lr_10_pct_pt))
            optimizer.param_groups[param_group_idx]['lr'] = new_intrinsics_lr

        depth_train_loss = new_sw_skf_depth_loss.mean()

        # Rest is logging.

        if log_weights:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(train_iter_idx))
            save_dict = {
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'kf_poses_params': kf_poses_params,
                'intrinsics_params': intrinsics_params,
                'open3d_vis_count': open3d_vis_count,
                'B': B
            }
            if render_kwargs_train['network_fine'] is not None:
                save_dict['network_fine_state_dict'] = render_kwargs_train['network_fine'].state_dict()
            torch.save(save_dict, path)
            print('Saved checkpoints at', path)
            t_prev_weights_log = t_train_iter_start

        if log_video:
            # Turn on testing mode
            with torch.no_grad():
                rgbs, disps, _ = render_path(render_poses, hwf, intrinsics_matrix,
                                             args.chunk, render_kwargs_test)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname,
                                                                                  train_iter_idx))
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

#         if log_test:
#             testsavedir = os.path.join(basedir, expname, f'testset_{train_iter_idx:06d}')
#             os.makedirs(testsavedir, exist_ok=True)
#             with torch.no_grad():
#                 test_rendered_rgbs_np, _, _ = render_path(
#                     test_poses.to(gpu_if_available), hwf, intrinsics_matrix, args.chunk,
#                     render_kwargs_test,
#                     gt_imgs=test_rgb_imgs, savedir=testsavedir)
#                 test_rendered_rgbs = torch.from_numpy(test_rendered_rgbs_np)
#                 test_loss = img2mse(test_rendered_rgbs, test_rgb_imgs.to(cpu))
#                 test_psnr = mse2psnr(test_loss)

#             tensorboard.add_images('test/rgb/estimate', pad_imgs(test_rendered_rgbs, white_rgb, 2),
#                                    train_iter_idx, dataformats='NHWC')
#             tensorboard.add_scalar('test/loss', test_loss, train_iter_idx)
#             tensorboard.add_scalar('test/psnr', test_psnr, train_iter_idx)

        if log_train_scalars:
            with torch.no_grad():
                tensorboard.add_scalar('train/loss', train_loss, train_iter_idx)
                rgb_train_loss = new_sw_skf_rgb_loss.mean()
                tensorboard.add_scalar('train/loss_rgb', rgb_train_loss, train_iter_idx)
                tensorboard.add_scalar('train/loss_depth', depth_train_loss, train_iter_idx)
                tensorboard.add_scalar('train/loss_depth_multiplier',
                                       depth_loss_iters_multiplier, train_iter_idx)
                tensorboard.add_scalar(
                    'train/loss_percent_rgb',
                    100 * rgb_train_loss / (rgb_train_loss + depth_train_loss),
                    train_iter_idx)
                tensorboard.add_scalar('train/psnr', train_psnr, train_iter_idx)
                tensorboard.add_scalar('learning_rate/scene', new_scene_lr, train_iter_idx)
                if not args.no_pose_optimization:
                    tensorboard.add_scalar('learning_rate/poses',
                                           new_poses_lr, train_iter_idx)
                if args.B_opt:
                    tensorboard.add_scalar('learning_rate/gpe_mat',
                                           new_gpe_mat_lr, train_iter_idx)
                if not args.no_intrinsics_optimization:
                    tensorboard.add_scalar('learning_rate/intrinsics',
                                           new_intrinsics_lr, train_iter_idx)
                if (args.save_logs_to_file):
                    append_to_log_file(vis_dpath, 'loss', args.i_train_scalars, args.s_train_scalars,
                                       train_loss)
                t_prev_train_scalars_log = t_train_iter_start

        if log_sampling_vis:
            sampling_name = 'active_sampling' if do_active_sampling else 'sampling'
            with torch.no_grad():
                add_1d_imgs_to_tensorboard(sw_total_n_sampled, white_rgb, torch.tensor([1, 0, 0]), tensorboard,
                                           f'train/{sampling_name}/section-wise_cumulative_samples',
                                           train_iter_idx, cpu)
                add_1d_imgs_to_tensorboard(
                    iw_from_pw(pw_total_n_sampled, (n_kfs, img_height, img_width)), white_rgb,
                    torch.tensor([1, 0, 0]), tensorboard,
                    f'train/{sampling_name}/pixel-wise_cumulative_samples',
                    train_iter_idx, cpu)

                add_1d_imgs_to_tensorboard(sw_kf_loss, white_rgb, torch.tensor([0, 0.8, 0]), tensorboard,
                                           'train/estimated_loss_distribution', train_iter_idx, cpu)
            t_prev_sampling_vis_log = t_train_iter_start

        if log_val_vis:
            # Log a rendered validation view to Tensorboard.
            with torch.no_grad():
                val_rendered_rgb, _, _, val_rendered_depth, _ = render(img_height, img_width, intrinsics_matrix, gpu_if_available,
                                                                       chunk=args.chunk, c2w=val_pose, **render_kwargs_test)

                val_loss = img2mse(val_rendered_rgb, val_rgb_img)
                val_psnr = mse2psnr(val_loss)

            tensorboard.add_image('validation/rgb/estimate', val_rendered_rgb,
                                  train_iter_idx, dataformats='HWC')
            add_1d_imgs_to_tensorboard(val_rendered_depth, black_rgb, white_rgb, tensorboard,
                                       'validation/depth/estimate', train_iter_idx, cpu)
            if not already_logged_val_gt:
                tensorboard.add_image('validation/rgb/groundtruth', val_rgb_img,
                                      train_iter_idx, dataformats='HWC')
                add_1d_imgs_to_tensorboard(val_depth_img, black_rgb, white_rgb, tensorboard,
                                           'validation/depth/groundtruth', train_iter_idx, cpu)
                already_logged_val_gt = True
            tensorboard.add_scalar('validation/loss', val_loss, train_iter_idx)
            tensorboard.add_scalar('validation/psnr', val_psnr, train_iter_idx)
            # tb_info = (tensorboard, 'train/point-cloud', open3d_vis_count)
            # save_point_cloud_from_rgb_imgs_and_depth_imgs(
            #     vis_dpath /
            #     f'rendered-val-point-cloud-{train_iter_idx:06d}.ply', val_rendered_rgb,
            #     val_rendered_depth, val_pose, intrinsics_matrix, tb_info)
            if args.save_logs_to_file:
                if is_first_iter:
                    torch.save(intrinsics_matrix.cpu(), vis_dpath / 'intrinsics-matrix_rgb.pt')
                    torch.save(torch.cat((val_pose.cpu(), torch.tensor(
                        [[0.0, 0.0, 0.0, 1.0]])), 0), vis_dpath / 'val-pose.pt')
                append_to_log_file(vis_dpath, 'val-rgbd', args.i_val, args.s_val,
                                   torch.cat((val_rendered_rgb, val_rendered_depth.unsqueeze(-1)), -1))
            # open3d_vis_count += 1
            t_prev_val_log = t_train_iter_start

        if log_kf_renders_log:
            with torch.no_grad():
                kf_rendered_rgbs_np, _, kf_rendered_depths_np = render_path(
                    kf_poses.to(gpu_if_available), hwf, intrinsics_matrix, args.chunk,
                    render_kwargs_test, gt_imgs=kf_rgb_imgs)
            kf_rendered_rgbds = torch.cat((torch.from_numpy(kf_rendered_rgbs_np),
                                           (torch.from_numpy(kf_rendered_depths_np)).unsqueeze(-1)), -1)
            if args.save_logs_to_file:
                if is_first_iter:
                    torch.save(intrinsics_matrix, vis_dpath / 'intrinsics-matrix_rgb.pt')
                append_to_log_file(vis_dpath, 'kfs-rgbd', args.i_kf_renders_vis,
                                   args.s_kf_renders_vis, kf_rendered_rgbds)
            t_prev_kf_renders_log = t_train_iter_start

        if log_pose_scalars:
            with torch.no_grad():
                for idx, (world_from_pose, world_from_initial_pose) in enumerate(zip(kf_poses, kf_initial_poses)):
                    pose_from_initial_pose = (torch.linalg.inv(world_from_pose) @
                                              world_from_initial_pose)
                    d_x = pose_from_initial_pose[0, 3]
                    d_y = pose_from_initial_pose[1, 3]
                    d_z = pose_from_initial_pose[2, 3]
                    euler_zyx = np.rad2deg(
                        euler_zyx_from_tfmat(pose_from_initial_pose.cpu().numpy()))
                    d_z_rot = euler_zyx[0]
                    d_y_rot = euler_zyx[1]
                    d_x_rot = euler_zyx[2]
                    tensorboard.add_scalars(f'poses/diff-from-init-{idx}', {
                        'dist-x': d_x, 'dist-y': d_y, 'dist-z': d_z,
                        'rot-x': d_x_rot, 'rot-y': d_y_rot, 'rot-z': d_z_rot},
                        train_iter_idx)
            t_prev_pose_scalars_log = t_train_iter_start

        if log_poses_vis:
            with torch.no_grad():
                # Create a set of Open3D XYZ coordinate axes at the location of every initial pose and
                # optimized pose, then log them to tensorboard for visualization.
                kf_initial_poses_np = kf_initial_poses.cpu().numpy()
                kf_poses_np = kf_poses.detach().cpu().numpy()
                initial_coordinate_frames = o3d_axes_from_poses(
                    kf_initial_poses_np, gray_out=True)
                optimized_coordinate_frames = o3d_axes_from_poses(kf_poses_np)
                for idx in range(len(initial_coordinate_frames)):
                    # TODO: Efficiently store initial poses, since they are always the same at every
                    # iteration.
                    tensorboard.add_3d(f'train/pose{idx:03d}-initial',
                                       to_dict_batch([initial_coordinate_frames[idx]]),
                                       step=open3d_vis_count)
                    tensorboard.add_3d(f'train/pose{idx:03d}-optimized',
                                       to_dict_batch([optimized_coordinate_frames[idx]]),
                                       step=open3d_vis_count)
                if args.save_logs_to_file:
                    if is_first_iter:
                        append_to_log_file(vis_dpath, 'poses', args.i_poses_vis, args.s_poses_vis,
                                           kf_initial_poses)
                    append_to_log_file(vis_dpath, 'poses', args.i_poses_vis, args.s_poses_vis,
                                       kf_poses)
            open3d_vis_count += 1
            t_prev_poses_vis_log = t_train_iter_start

        if log_point_cloud_vis:
            with torch.no_grad():
                rendered_rgb_imgs, _, rendered_depth_imgs = render_path(kf_poses, hwf, intrinsics_matrix, args.chunk, render_kwargs_test,
                                                                        gt_imgs=kf_rgb_imgs, savedir=None,
                                                                        render_factor=args.render_factor)
                tb_info = (tensorboard, 'train/point-cloud', open3d_vis_count)
                save_point_cloud_from_rgb_imgs_and_depth_imgs(
                    vis_dpath /
                    f'rendered-kf-point-cloud-{train_iter_idx:06d}.ply', rendered_rgb_imgs,
                    rendered_depth_imgs, kf_poses, intrinsics_matrix, tb_info)

            open3d_vis_count += 1
            t_prev_point_cloud_vis_log = t_train_iter_start

        if log_B_vis:
            with torch.no_grad():
                add_1d_imgs_to_tensorboard(B.unsqueeze(0), white_rgb, purple_rgb, tensorboard,
                                           'train/gaussian_positional_encoding_matrix',
                                           train_iter_idx, cpu, padding_width=0)
            if args.save_logs_to_file:
                append_to_log_file(vis_dpath, 'gpe-mat', args.i_B_vis, args.s_B_vis, B)
            t_prev_B_vis_log = t_train_iter_start

        if log_intrinsics_vis:
            tensorboard.add_scalar('intrinsics/f', intrinsics_matrix[0, 0], train_iter_idx)
            tensorboard.add_scalar('intrinsics/cx', intrinsics_matrix[0, 2], train_iter_idx)
            tensorboard.add_scalar('intrinsics/cy', intrinsics_matrix[1, 2], train_iter_idx)
            t_prev_intrinsics_vis_log = t_train_iter_start

        if do_active_sampling and do_lazy_sw_loss:
            # Update the estimated section-wise loss probability distribution using the loss from
            # the rays that were just actively sampled.
            with torch.no_grad():
                # Only update the loss of sections that were sampled from.
                sampled_kf_sw_idxs_tuple = tuple([torch.tensor([skf_from_kf_idxs[idx] for
                                                                idx in sampled_skf_sw_idxs_tuple[0]]),
                                                  sampled_skf_sw_idxs_tuple[1], sampled_skf_sw_idxs_tuple[2]])
                sw_kf_loss[sampled_kf_sw_idxs_tuple] = new_sw_skf_loss[sampled_skf_sw_idxs_tuple]

        t_train_iter = perf_counter() - t_train_iter_start

        tqdm_bar.set_postfix_str(
            f't{t_train_iter:.3f},gp{t_get_poses:.3f},sk{t_select_keyframes:.3f},'
            f'gr{t_get_rays:.3f},sr{t_sample_rays:.3f},ba{t_batching:.3f},r{t_rendering:.3f},'
            f'l{t_loss:.3f},bp{t_backprop:.3f}')

        global_step += 1

        if should_stop:
            return


if __name__ == '__main__':
    # torch.autograd.set_detect_anomaly(True)
    with launch_ipdb_on_exception():
        train()
