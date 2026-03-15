import bisect
from argparse import ArgumentParser
from pathlib import Path

import hydra
import imageio
import matplotlib.pyplot as plt
import numpy as np
import pytransform3d.camera as pc
import pytransform3d.transformations as pt
import torch
import torch.nn.functional as F
from einops import rearrange
from math import sqrt
from torchvision.transforms import ToPILImage
from tqdm import tqdm

from gaussiansplatting.arguments import PipelineParams
from gaussiansplatting.gaussian_renderer import render
from gaussiansplatting.scene import GaussianModel
from gaussiansplatting.scene.camera_scene import CamScene
from gaussiansplatting.scene.cameras import Simple_Camera


def so3_to_quaternion(so3):
    """Converts an SO3 rotation matrix to a quaternion

    cfg:
        so3: 3x3 rotation matrix

    Returns:
        numpy.ndarray: quaternion [w, x, y, z]

    Raises:
        ValueError: if so3 is not 3x3
    """
    if so3.shape != (3, 3):
        raise ValueError("SO3 matrix must be 3x3")

    R_xx = so3[0, 0]
    R_xy = so3[0, 1]
    R_xz = so3[0, 2]
    R_yx = so3[1, 0]
    R_yy = so3[1, 1]
    R_yz = so3[1, 2]
    R_zx = so3[2, 0]
    R_zy = so3[2, 1]
    R_zz = so3[2, 2]

    try:
        w = sqrt(so3.trace() + 1) / 2
    except(ValueError):
        # w is non-real
        w = 0

    x = sqrt(1 + R_xx - R_yy - R_zz) / 2
    y = sqrt(1 + R_yy - R_xx - R_zz) / 2
    z = sqrt(1 + R_zz - R_yy - R_xx) / 2

    max_index = max(range(4), key=[w, x, y, z].__getitem__)

    if max_index == 0:
        x = (R_zy - R_yz) / (4 * w)
        y = (R_xz - R_zx) / (4 * w)
        z = (R_yx - R_xy) / (4 * w)
    elif max_index == 1:
        w = (R_zy - R_yz) / (4 * x)
        y = (R_xy + R_yx) / (4 * x)
        z = (R_zx + R_xz) / (4 * x)
    elif max_index == 2:
        w = (R_xz - R_zx) / (4 * y)
        x = (R_xy + R_yx) / (4 * y)
        z = (R_yz + R_zy) / (4 * y)
    elif max_index == 3:
        w = (R_yx - R_xy) / (4 * z)
        x = (R_zx + R_xz) / (4 * z)
        y = (R_yz + R_zy) / (4 * z)

    return np.array([w, x, y, z])


def get_camera_SE3(cam: Simple_Camera) -> np.ndarray:
    R = cam.R
    T = cam.T
    SE3 = np.zeros((4, 4))
    SE3[:3, :3] = R
    SE3[:3, 3] = T
    SE3[3, 3] = 1.0
    return SE3


def interpolate_poses(pose_timestamps, abs_poses, requested_timestamps):
    """Interpolate between absolute poses.

    cfg:
        pose_timestamps (list[int]): Timestamps of supplied poses. Must be in ascending order.
        abs_poses (list[numpy.matrixlib.defmatrix.matrix]): SE3 matrices representing poses at the timestamps specified.
        requested_timestamps (list[int]): Timestamps for which interpolated timestamps are required.

    Returns:
        list[numpy.matrixlib.defmatrix.matrix]: SE3 matrix representing interpolated pose for each requested timestamp.

    Raises:
        ValueError: if pose_timestamps and abs_poses are not the same length
        ValueError: if pose_timestamps is not in ascending order

    """
    requested_timestamps = np.array(requested_timestamps)
    pose_timestamps = np.array(pose_timestamps)

    if len(pose_timestamps) != len(abs_poses):
        raise ValueError('Must supply same number of timestamps as poses')

    abs_quaternions = np.zeros((4, len(abs_poses)))
    abs_positions = np.zeros((3, len(abs_poses)))
    for i, pose in enumerate(abs_poses):
        if i > 0 and pose_timestamps[i - 1] >= pose_timestamps[i]:
            raise ValueError('Pose timestamps must be in ascending order')

        abs_quaternions[:, i] = so3_to_quaternion(pose[0:3, 0:3])
        abs_positions[:, i] = np.ravel(pose[0:3, 3])

    upper_indices = [bisect.bisect(pose_timestamps, pt) for pt in requested_timestamps]

    if max(upper_indices) >= len(pose_timestamps):
        upper_indices = [min(i, len(pose_timestamps) - 1) for i in upper_indices]

    lower_indices = [u - 1 for u in upper_indices]

    fractions = (requested_timestamps - pose_timestamps[lower_indices]) / \
                (pose_timestamps[upper_indices] - pose_timestamps[lower_indices])

    quaternions_lower = abs_quaternions[:, lower_indices]
    quaternions_upper = abs_quaternions[:, upper_indices]

    d_array = (quaternions_lower * quaternions_upper).sum(0)

    linear_interp_indices = np.nonzero(d_array >= 1)
    sin_interp_indices = np.nonzero(d_array < 1)

    scale0_array = np.zeros(d_array.shape)
    scale1_array = np.zeros(d_array.shape)

    scale0_array[linear_interp_indices] = 1 - fractions[linear_interp_indices]
    scale1_array[linear_interp_indices] = fractions[linear_interp_indices]

    theta_array = np.arccos(np.abs(d_array[sin_interp_indices]))

    scale0_array[sin_interp_indices] = \
        np.sin((1 - fractions[sin_interp_indices]) * theta_array) / np.sin(theta_array)
    scale1_array[sin_interp_indices] = \
        np.sin(fractions[sin_interp_indices] * theta_array) / np.sin(theta_array)

    negative_d_indices = np.nonzero(d_array < 0)
    scale1_array[negative_d_indices] = -scale1_array[negative_d_indices]

    quaternions_interp = np.tile(scale0_array, (4, 1)) * quaternions_lower \
                         + np.tile(scale1_array, (4, 1)) * quaternions_upper

    positions_lower = abs_positions[:, lower_indices]
    positions_upper = abs_positions[:, upper_indices]

    positions_interp = np.multiply(np.tile((1 - fractions), (3, 1)), positions_lower) \
                       + np.multiply(np.tile(fractions, (3, 1)), positions_upper)

    poses_mat = np.zeros((4, 4 * len(requested_timestamps)))

    poses_mat[0, 0::4] = 1 - 2 * np.square(quaternions_interp[2, :]) - \
                         2 * np.square(quaternions_interp[3, :])
    poses_mat[0, 1::4] = 2 * np.multiply(quaternions_interp[1, :], quaternions_interp[2, :]) - \
                         2 * np.multiply(quaternions_interp[3, :], quaternions_interp[0, :])
    poses_mat[0, 2::4] = 2 * np.multiply(quaternions_interp[1, :], quaternions_interp[3, :]) + \
                         2 * np.multiply(quaternions_interp[2, :], quaternions_interp[0, :])

    poses_mat[1, 0::4] = 2 * np.multiply(quaternions_interp[1, :], quaternions_interp[2, :]) \
                         + 2 * np.multiply(quaternions_interp[3, :], quaternions_interp[0, :])
    poses_mat[1, 1::4] = 1 - 2 * np.square(quaternions_interp[1, :]) \
                         - 2 * np.square(quaternions_interp[3, :])
    poses_mat[1, 2::4] = 2 * np.multiply(quaternions_interp[2, :], quaternions_interp[3, :]) - \
                         2 * np.multiply(quaternions_interp[1, :], quaternions_interp[0, :])

    poses_mat[2, 0::4] = 2 * np.multiply(quaternions_interp[1, :], quaternions_interp[3, :]) - \
                         2 * np.multiply(quaternions_interp[2, :], quaternions_interp[0, :])
    poses_mat[2, 1::4] = 2 * np.multiply(quaternions_interp[2, :], quaternions_interp[3, :]) + \
                         2 * np.multiply(quaternions_interp[1, :], quaternions_interp[0, :])
    poses_mat[2, 2::4] = 1 - 2 * np.square(quaternions_interp[1, :]) - \
                         2 * np.square(quaternions_interp[2, :])

    poses_mat[0:3, 3::4] = positions_interp
    poses_mat[3, 3::4] = 1

    poses_out = [0] * len(requested_timestamps)
    for i in range(len(requested_timestamps)):
        poses_out[i] = poses_mat[:, i * 4:(i + 1) * 4]

    return poses_out


def visualize_camera(SE3):
    # visualize
    fig = plt.figure()
    fig.set_size_inches(20, 20)
    ax = fig.add_subplot(111, projection='3d')

    sensor_size = np.array([0.036, 0.024])
    intrinsic_matrix = np.array([
        [0.05, 0, sensor_size[0] / 2.0],
        [0, 0.05, sensor_size[1] / 2.0],
        [0, 0, 1]
    ])
    virtual_image_distance = 1

    cam_T = [se3[:3, 3] for se3 in SE3]
    cam_R = [se3[:3, :3] for se3 in SE3]

    cam_qvec = [so3_to_quaternion(R) for R in cam_R]

    for p, q in zip(cam_T, cam_qvec):
        pq = np.concatenate([p, q])
        ax = pt.plot_transform(ax, A2B=pt.transform_from_pq(pq), s=0.2)
        pc.plot_camera(
            ax, cam2world=pt.transform_from_pq(pq), M=intrinsic_matrix, sensor_size=sensor_size,
            virtual_image_distance=virtual_image_distance)

    # ax.plot(cam_T[:, 0], cam_T[:, 1], cam_T[:, 2])  # camera trajectory

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()


def get_heatmap(tensor, size=None, cmap='viridis'):
    """
    Given a batch of tensor, colorize it with a palette and return a grid of heatmaps
    Args:
        tensor: A tensor of shape "B 3 H W" or "B 1 H W" or "B H W"
        size: The target shape of the heatmap, (H, W), if None, use the tensor.shape[-2:]
        cmap: The color palette

    Returns:
        A batched tensor heatmap of the tensor of shape "B 3 H W", range [0, 1]
    """
    assert tensor.dim() in [3, 4], f"Expected 3D or 4D tensor, got {tensor.dim()}D tensor"

    if size is None:
        size = tensor.shape[-2:]

    if tensor.dim() == 3:  # B H W
        tensor = tensor.unsqueeze(1)
    elif tensor.shape[1] == 3:  # B 3 H W
        tensor = tensor.mean(1, keepdim=True)
    elif tensor.shape[1] == 1:  # B 1 H W
        pass

    B = tensor.shape[0]

    tensor = tensor.abs()  # B 1 H W
    tensor = F.interpolate(tensor.float(), size=size)  # B 1 H W
    tensor = rearrange(tensor, "B 1 H W -> 1 H (B W)").expand(3, -1, -1)  # 3 H_ W_
    tensor = tensor[0]  # H_ W_

    tensor = tensor.detach().cpu().numpy()

    fig = plt.figure(frameon=False)
    fig.set_size_inches(tensor.shape[1] / 100, tensor.shape[0] / 100)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(tensor, cmap=cmap, aspect='auto')

    fig.canvas.draw()

    # extract image from the figure
    buffer = fig.canvas.buffer_rgba()
    buffer = np.array(buffer)
    buffer = buffer[:, :, :-1]  # remove alpha channel
    buffer = torch.from_numpy(buffer) / 255.0
    buffer = rearrange(buffer, "H (B W) C -> B C H W", B=B)

    plt.close(fig)

    return buffer


@hydra.main(version_base="1.3", config_path="configs")
def main(cfg):
    device = torch.device(cfg.device)

    pipe = PipelineParams(ArgumentParser(description="Dummy"))

    gaussian = GaussianModel(
        sh_degree=0,
        anchor_weight_init_g0=1.0,
        anchor_weight_init=0.1,
        anchor_weight_multiplier=2,
    )

    gaussian.load_ply(cfg.gs_source)

    scene = CamScene(cfg.colmap_dir, h=512, w=512)

    background_tensor = torch.tensor(
        [0, 0, 0], dtype=torch.float32, device=device
    )

    cameras = [scene.cameras[i] for i in cfg.key_cam_indices]

    # camera interpolation
    pose_t = list(np.linspace(0, 1, len(cameras)))
    requested_t = list(np.linspace(0, 1, cfg.num_views))

    original_SE3 = [get_camera_SE3(cam) for cam in cameras]
    interpolated_SE3 = interpolate_poses(pose_t, original_SE3, requested_t)

    # visualize_camera(original_SE3)
    # visualize_camera(interpolated_SE3)

    # copy basics
    first_cam = cameras[0]
    colmap_id = first_cam.colmap_id
    FoVx = first_cam.FoVx
    FoVy = first_cam.FoVy
    h = first_cam.image_height
    w = first_cam.image_width
    image_name = first_cam.image_name
    uid = first_cam.uid
    scale = first_cam.scale
    data_device = first_cam.data_device

    interpolated_cameras = []
    for i, SE3 in enumerate(interpolated_SE3):
        cam = Simple_Camera(
            colmap_id=colmap_id,
            R=SE3[:3, :3],
            T=SE3[:3, 3],
            FoVx=FoVx,
            FoVy=FoVy,
            h=h,
            w=w,
            image_name=image_name,
            uid=uid,
            scale=scale,
            data_device=data_device,
        )
        interpolated_cameras.append(cam)

    # render
    rendered_imgs = []
    rendered_depths = []
    for i, cam in enumerate(tqdm(interpolated_cameras, desc="Rendering images")):
        render_pkg = render(cam, gaussian, pipe, background_tensor)
        # print(f"{render_pkg['depth_3dgs'].shape}, {render_pkg['depth_3dgs'].max()}, {render_pkg['depth_3dgs'].min()}")
        rendered_imgs.append(ToPILImage()(torch.clip(render_pkg["render"], 0, 1)))
        rendered_depths.append(ToPILImage()(get_heatmap(render_pkg["depth_3dgs"])[0]))

    # save
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    if cfg.output_format == "mp4":
        imageio.mimsave(
            output_dir / f"{cfg.name}-rgb.mp4", [np.array(img) for img in rendered_imgs],
            fps=cfg.fps,
        )
        imageio.mimsave(
            output_dir / f"{cfg.name}-depth.mp4", [np.array(img) for img in rendered_depths],
            fps=cfg.fps,
        )
    elif cfg.output_format == "gif":
        imageio.mimsave(
            output_dir / f"{cfg.name}-rgb.gif", [np.array(img) for img in rendered_imgs],
            palettesize=256,
            fps=cfg.fps,
        )
        imageio.mimsave(
            output_dir / f"{cfg.name}-depth.gif", [np.array(img) for img in rendered_depths],
            palettesize=256,
            fps=cfg.fps,
        )
    else:
        print(f"Saving as default images to {output_dir / cfg.name}")
        for i, (img_rgb, img_depth) in tqdm(enumerate(zip(rendered_imgs, rendered_depths)), desc="Saving images"):
            img_rgb.save(output_dir / f"{cfg.name}-{i:04d}-rgb.png")
            img_depth.save(output_dir / f"{cfg.name}-{i:04d}-depth.png")

    print(f"Saved to {output_dir / cfg.name}.{cfg.output_format}")


if __name__ == "__main__":
    main()
