from json import load
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from cv2 import COLOR_BGR2RGB, cvtColor, imread, resize
from ipdb import set_trace

from im_util.transforms import (rotmat_from_euler_zyx,
                                tfmat_from_rotmat_and_translation,
                                tfmat_from_quat_and_translation)


def load_bonn_data(base_dir, downsample_factor):
    """
    @returns - (images, hwf, poses, render_poses, train_idxs, test_idxs)
      images: (N,H,W,3) RGB images
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
    camera_info_fpath = base_dirpath / 'camera_info.json'
    assert camera_info_fpath.exists()
    camera_info_file = open(camera_info_fpath.as_posix(), 'r')
    camera_info = load(camera_info_file)
    height = int(camera_info['height'] * scale_factor)
    width = int(camera_info['width'] * scale_factor)
    focal_length = camera_info['focal_length'] * scale_factor
    hwf = [height, width, focal_length]
    camera_info_file.close()

    # Read images and poses.
    trajectory_fpath = base_dirpath / 'traj_z-backwards.txt'
    assert trajectory_fpath.exists()
    trajectory_file = open(trajectory_fpath.as_posix(), 'r')

    images = []
    poses = []

    for line in trajectory_file.readlines():
        timestamp_s, px, py, pz, qx, qy, qz, qw = [float(w) for w in line.split(' ')]
        # Convert timestamp from seconds to microseconds.
        timestamp = int(timestamp_s * MICROSECONDS_PER_SECOND)

        img_fpath = base_dirpath / 'images' / f'{timestamp}.png'
        assert img_fpath.exists()
        image = resize(cvtColor(imread(img_fpath.as_posix()), COLOR_BGR2RGB),
                       (width, height))
        images.append(image)

        quat = np.array([qx, qy, qz, qw])
        translation = np.array([px, py, pz])
        poses.append(tfmat_from_quat_and_translation(quat, translation))

    trajectory_file.close()

    images = np.array(images) / 255.
    poses = np.array(poses)
    assert images.shape[0] == poses.shape[0]

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
        # render_pose = np.array([
        #     [ 0.0005171 ,  0.01598652,  0.99987207,  0.11357104],
        #     [-0.02391396,  0.99958646, -0.01596959, -0.01041086],
        #     [-0.99971389, -0.02390265,  0.00089919,  0.69796815],
        #     [ 0.        ,  0.        ,  0.        ,  1.        ]])
        render_poses.append(render_pose)

    render_poses = np.array(render_poses)

    # Compute train indices and test indices.
    all_idxs = list(range(images.shape[0]))
    if len(all_idxs) <= 5:
        test_idxs = [len(all_idxs) // 2]
    else:
        test_idxs = all_idxs[::8]
    train_idxs = [idx for idx in all_idxs if idx not in test_idxs]

    return images, hwf, poses, render_poses, train_idxs, test_idxs


if __name__ == '__main__':
    load_bonn_data('~/eth/implicit-mapping/trajectory/strawberry-0-0to10/orbslam2/front-camera', 2)
