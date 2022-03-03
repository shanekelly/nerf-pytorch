import numpy as np
import open3d as o3d
import torch
from pathlib import Path
from pickle import load

from cv2 import imread, imwrite, resize, INTER_NEAREST_EXACT

import matplotlib.pyplot as plt
from ipdb import launch_ipdb_on_exception, set_trace

from point_cloud.rgbd import o3d_coordinate_frames_from_poses, o3d_point_cloud_from_rgbd_imgs
from time import sleep

from run_nerf_helpers import color_1d_imgs
from image.util import scale_imgs


def save_rgbd_imgs(rgbd_imgs_fpath):
    output_dpath = rgbd_imgs_fpath.parent / rgbd_imgs_fpath.stem.split('_')[0]
    output_dpath.mkdir(exist_ok=True)

    rgbd_imgs = torch.load(rgbd_imgs_fpath)
    rgb_imgs = rgbd_imgs[:, :, :, :3]
    depth_imgs = rgbd_imgs[:, :, :, -1]

    # Format RGB images for saving to file.
    if rgb_imgs.max() <= 1.1:
        rgb_imgs *= 255
    rgb_imgs = rgb_imgs.flip(3)

    # Format depth images for saving to file.
    n_imgs, img_height, img_width = depth_imgs.shape
    torch.nan_to_num_(depth_imgs, nan=0.0, posinf=0.0, neginf=0.0)
    depth_imgs = (depth_imgs / depth_imgs.max() *
                  255).unsqueeze(-1).expand(n_imgs, img_height, img_width, 3)

    # RGB and depth images side-by-side.
    rgb_d_imgs = torch.cat((rgb_imgs, depth_imgs), 2)

    rgb_d_imgs_np = rgb_d_imgs.numpy()
    for idx, rgb_d_img in enumerate(rgb_d_imgs_np):
        img_fpath = output_dpath / f'rgb-d_{idx:03d}.png'
        imwrite(img_fpath.as_posix(), rgb_d_img)


def point_cloud(rgbd_imgs_fpath, rgbd_poses_fpath, intrinsics_matrix_fpath, vis_poses_fpath):
    all_rgbd_imgs = torch.load(rgbd_imgs_fpath)
    assert all_rgbd_imgs.dim() == 4 or all_rgbd_imgs.dim() == 5
    if all_rgbd_imgs.dim() == 4:
        # At each time step there is only one image, so expand a dimension that will have size one.
        all_rgbd_imgs = all_rgbd_imgs.unsqueeze(1)
    # Now all_rgbd_imgs is shape (N timesteps, N imgs per timestep, img height, img width, 4)

    all_rgbd_poses = torch.load(rgbd_poses_fpath)
    assert all_rgbd_poses.dim() == 2 or all_rgbd_poses.dim() == 4
    if all_rgbd_poses.dim() == 2:
        # There is only a single pose for all image frames, so expand it to repeat (N timesteps, N
        # imgs per timestep) times.
        n_timesteps, n_imgs_per_timestep, _, _, _ = all_rgbd_imgs.shape
        all_rgbd_poses = all_rgbd_poses.unsqueeze(0).unsqueeze(0).expand(
            n_timesteps, n_imgs_per_timestep, 4, 4)
        # initial_poses = all_rgbd_poses[0]
    else:
        # all_rgbd_poses starts with the initial poses
        # initial_poses = all_rgbd_poses[0]
        all_rgbd_poses = all_rgbd_poses[1:]
    # Now all_rgbd_poses is shape (N timesteps, N imgs per timestep, 4, 4).

    all_vis_poses = torch.load(vis_poses_fpath)
    assert all_vis_poses.dim() == 4
    # initial_vis_poses = all_vis_poses[0]
    all_vis_poses = all_vis_poses[1:]
    # all_vis_poses is shape (N timesteps, N poses to visualize per timestep, 4, 4).

    intrinsics_matrix = torch.load(intrinsics_matrix_fpath)

    point_clouds = np.array([o3d_point_cloud_from_rgbd_imgs(
        rgbd_imgs, rgbd_poses, intrinsics_matrix, z_forwards=False) for
        rgbd_imgs, rgbd_poses in zip(all_rgbd_imgs, all_rgbd_poses)])
    all_vis_pose_frames = np.array([o3d_coordinate_frames_from_poses(vis_poses, size=0.1)
                                    for vis_poses in all_vis_poses])

    screenshot_dpath = rgbd_imgs_fpath.parent / 'o3d-screenshots'
    screenshot_dpath.mkdir(exist_ok=True)

    cont = 'y'
    while cont == 'y':
        # Allow the user to set the rendering view before all geometries are iterated through.
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(point_clouds[-1])
        for vis_pose_frame in all_vis_pose_frames[-1]:
            vis.add_geometry(vis_pose_frame)
        vis.run()
        vctrl_params = vis.get_view_control().convert_to_pinhole_camera_parameters()
        point_rendering_size = vis.get_render_option().point_size
        vis.destroy_window()

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        for idx in range(point_clouds.shape[0]):
            # Render the current point cloud and visualization poses.
            vis.add_geometry(point_clouds[idx])
            for vis_pose_frame in all_vis_pose_frames[idx]:
                vis.add_geometry(vis_pose_frame)

            # Set the view as defined by the user previously.
            ctr = vis.get_view_control()
            ctr.convert_from_pinhole_camera_parameters(vctrl_params)

            # Set the size of rendered points, as defined by the user previously.
            vis.get_render_option().point_size = point_rendering_size

            # Update the renderer to show the new contents.
            vis.poll_events()
            vis.update_renderer()

            # Take a screenshot and save it to file.
            vis.capture_screen_image((screenshot_dpath /
                                      f'point-cloud-and-poses_{idx:04d}.png').as_posix())

            # Clear the renderer so that new contents can be added next iteration.
            vis.clear_geometries()
        vis.destroy_window()

        cont = input('Type "y" to redo: ')


def stack_imgs_vertically(top_imgs_glob, bot_imgs_glob, output_dname):
    output_dpath = Path(output_dname).expanduser()
    output_dpath.mkdir(exist_ok=True)

    top_imgs = np.array([imread(img_fpath.as_posix()) for img_fpath in
                         sorted(Path('.').glob(top_imgs_glob))])
    bot_imgs = np.array([imread(img_fpath.as_posix()) for img_fpath in
                         sorted(Path('.').glob(bot_imgs_glob))])

    def scale_imgs_A_to_imgs_B_width(imgs_A, imgs_B):
        _, A_height, A_width, _ = imgs_A.shape
        _, B_height, B_width, _ = imgs_B.shape

        scaling = B_width / A_width
        new_A_shape = (B_width, int(round(A_height * scaling)))  # Maintain A aspect ratio.
        scaled_imgs_A = []
        for img in imgs_A:
            scaled_imgs_A.append(resize(img, new_A_shape))

        return scaled_imgs_A

    top_width = top_imgs.shape[2]
    bot_width = bot_imgs.shape[2]
    if top_width < bot_width:
        # Shrink bot imgs to be as wide as top imgs.
        bot_imgs = scale_imgs_A_to_imgs_B_width(bot_imgs, top_imgs)
    elif top_width > bot_width:
        # Shrink top imgs to be as wide as bot imgs.
        top_imgs = scale_imgs_A_to_imgs_B_width(top_imgs, bot_imgs)

    stacked_imgs = np.concatenate((top_imgs, bot_imgs), 1)
    for idx, img in enumerate(stacked_imgs):
        imwrite((output_dpath / f'{idx:04d}.png').as_posix(), img)


def gpe_matrix(gpe_mats_fpath, scale=1) -> None:
    output_dpath = gpe_mats_fpath.parent / gpe_mats_fpath.stem.split('_')[0]
    output_dpath.mkdir(exist_ok=True)

    gpe_mats = torch.load(gpe_mats_fpath)

    colored_gpe_mats = scale_imgs(color_1d_imgs(gpe_mats, torch.tensor([1.0, 0.0, 1.0])).numpy() *
                                  255, scale)

    for idx, colored_gpe_mat in enumerate(colored_gpe_mats):
        img_fpath = output_dpath / f'{idx:04d}.png'
        imwrite(img_fpath.as_posix(), colored_gpe_mat)


def main(args) -> None:
    if args.command == 'rgbd':
        save_rgbd_imgs(Path(args.rgbd_fname).expanduser())
    elif args.command == 'point_cloud':
        point_cloud(Path(args.rgbd_fname), Path(args.rgbd_poses_fname),
                    Path(args.intrinsics_matrix_fname), Path(args.vis_poses_fname))
    elif args.command == 'img_stack':
        stack_imgs_vertically(args.top_imgs_glob, args.bot_imgs_glob, Path(args.output_dname))
    elif args.command == 'gpe_matrix':
        gpe_matrix(Path(args.gpe_mats_fname).expanduser(), args.scale)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()

    subparsers = parser.add_subparsers(dest='command', required=True)

    rgb_subparser = subparsers.add_parser('rgbd')
    rgb_subparser.add_argument('rgbd_fname')

    point_cloud_subparser = subparsers.add_parser('point_cloud')
    point_cloud_subparser.add_argument('rgbd_fname')
    point_cloud_subparser.add_argument('rgbd_poses_fname')
    point_cloud_subparser.add_argument('intrinsics_matrix_fname')
    point_cloud_subparser.add_argument('vis_poses_fname')

    img_stack_subparser = subparsers.add_parser('img_stack')
    img_stack_subparser.add_argument('top_imgs_glob')
    img_stack_subparser.add_argument('bot_imgs_glob')
    img_stack_subparser.add_argument('output_dname')

    gpe_matrix_subparser = subparsers.add_parser('gpe_matrix')
    gpe_matrix_subparser.add_argument('gpe_mats_fname')
    gpe_matrix_subparser.add_argument('scale', type=int, default=1)

    args = parser.parse_args()

    with launch_ipdb_on_exception():
        main(args)
