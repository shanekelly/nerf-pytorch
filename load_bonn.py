from json import load
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from cv2 import COLOR_BGR2RGB, cvtColor, imread, resize, IMREAD_ANYDEPTH

from axes.util import o3d_axes_from_poses
from im_util.transforms import (rotmat_from_euler_zyx,
                                tfmat_from_rotmat_and_translation,
                                tfmat_from_quat_and_translation)


def load_bonn_data(base_dir, downsample_factor):
    """
    @returns - (rgb_imgs, hwf, poses, render_poses, train_idxs, test_idxs)
      rgb_imgs: (N,H,W,3) RGB images
      hwf: (3,) image height, image width, focal distance
      poses: (N,3,4) homogeneous transform matrices corresponding to the groundtruth positions of
        each image
      render_poses: (N_render,3,4)
      train_idxs: (N_train,) indices into the images array that specify which images to use for
          training
      test_idxs: (N_test,) indices into the images array that specify which images to use for
          testing

    """
    MICROSECONDS_PER_SECOND = 1e6

    scale_factor = 1. / downsample_factor

    base_dirpath = Path(base_dir).expanduser()
    assert base_dirpath.exists()

    # Get height, width, and focal length.
    camera_info_fpath = base_dirpath / 'camera-info.json'
    assert camera_info_fpath.exists()
    camera_info_file = open(camera_info_fpath.as_posix(), 'r')
    camera_info = load(camera_info_file)
    camera_info_file.close()
    height = int(camera_info['rgb']['height'] * scale_factor)
    width = int(camera_info['rgb']['width'] * scale_factor)
    focal_length = camera_info['rgb']['focal_length'] * scale_factor
    depth_scale = camera_info['depth']['scale']
    hwf = [height, width, focal_length]
    intrinsics_matrix = np.array(camera_info['rgb']['intrinsics_matrix'])
    intrinsics_matrix_focal_length = (intrinsics_matrix[0, 0] + intrinsics_matrix[1, 1]) / 2
    intrinsics_matrix[0, 0] = intrinsics_matrix_focal_length
    intrinsics_matrix[1, 1] = intrinsics_matrix_focal_length

    # Read images and poses.
    trajectory_fpath = base_dirpath / 'traj_z-backwards.txt'
    assert trajectory_fpath.exists()
    trajectory_file = open(trajectory_fpath.as_posix(), 'r')

    rgb_imgs = []
    depth_imgs = []
    poses = []

    for line in trajectory_file.readlines():
        timestamp_s, px, py, pz, qx, qy, qz, qw = [float(w) for w in line.split(' ')]
        # Convert timestamp from seconds to microseconds.
        timestamp = int(timestamp_s * MICROSECONDS_PER_SECOND)

        rgb_img_fpath = base_dirpath / 'images' / f'rgb_{timestamp}.png'
        assert rgb_img_fpath.exists()
        rgb_img = resize(cvtColor(imread(rgb_img_fpath.as_posix()), COLOR_BGR2RGB),
                         (width, height))
        rgb_imgs.append(rgb_img)

        depth_img_fpath = base_dirpath / 'images' / f'registered-depth_{timestamp}.png'
        assert depth_img_fpath.exists()
        # IMREAD_ANYDEPTH allows the depth image to be read in a 16 bit format if the image is saved
        # with such information.
        depth_img = resize(imread(depth_img_fpath.as_posix(), IMREAD_ANYDEPTH), (width, height))
        depth_imgs.append(depth_img)

        quat = np.array([qx, qy, qz, qw])
        translation = np.array([px, py, pz])
        poses.append(tfmat_from_quat_and_translation(quat, translation))

    trajectory_file.close()

    rgb_imgs = np.array(rgb_imgs) / 255.
    depth_imgs = np.array(depth_imgs) * depth_scale  # Convert to units of meters.
    poses = np.array(poses)
    assert rgb_imgs.shape[0] == depth_imgs.shape[0] == poses.shape[0]

    # Compute render poses.
    # Move forward by 1 meter in even steps over the first half of the number of render poses, then
    # move backward by 1 meter in even steps over the last half of the render poses. End up back
    # at start.
    NUM_RENDER_POSES = 10
    render_poses = []

    # start_translation = np.array([0.11, -0.012, 0.563]) + np.array([0, 0, -0.25])
    start_translation = np.array([-0.11, 0.012, 0.563]) + np.array([0, 0, -0.25])
    # translation_motion = np.array([0, 0, 0.75])
    translation_motion = np.array([0, 0, 0.75])

    start_rotation = np.deg2rad(np.array([0, 90, 180]))
    rotation_motion = np.deg2rad(np.array([0, 0, 0]))

    for idx in range(NUM_RENDER_POSES):
        pct = idx / NUM_RENDER_POSES
        translation = start_translation + pct * translation_motion
        rotmat = rotmat_from_euler_zyx(start_rotation + pct * rotation_motion)

        render_pose = tfmat_from_rotmat_and_translation(rotmat, translation)
        render_poses.append(render_pose)

    render_poses = np.array(render_poses)

    # Compute train indices and test indices.
    all_idxs = list(range(rgb_imgs.shape[0]))
    if len(all_idxs) == 1:
        test_idxs = [0]
        train_idxs = [0]
    elif len(all_idxs) <= 5:
        test_idxs = [len(all_idxs) // 2]
        train_idxs = [idx for idx in all_idxs if idx not in test_idxs]
    else:
        test_idxs = all_idxs[::8]
        train_idxs = [idx for idx in all_idxs if idx not in test_idxs]

    assert np.all(np.isfinite(depth_imgs))

    return rgb_imgs, depth_imgs, hwf, intrinsics_matrix, poses, render_poses, train_idxs, test_idxs


if __name__ == '__main__':
    load_bonn_data('~/eth/implicit-mapping/trajectory/strawberry-0-0to10/orbslam2/front-camera', 2)
