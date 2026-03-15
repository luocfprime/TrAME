import bisect
import enum
import random
from dataclasses import dataclass, field

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

import threestudio
from threestudio import register
from threestudio.utils.base import Updateable
from threestudio.utils.config import parse_structured
from threestudio.utils.schedule import schedule_functions
from threestudio.utils.typing import *


def safe_normalize(x, eps=1e-20):
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))


# def convert_camera_to_world_transform(transform):
#     converted_transform = transform.clone()
#
#     converted_transform[:, 2] *= -1
#
#     converted_transform[[0, 2], :] = converted_transform[[2, 0], :]
#
#     return converted_transform


# def circle_poses(
#     device, radius=torch.tensor([3.2]), theta=torch.tensor([60]), phi=torch.tensor([0])
# ):
#     theta = theta / 180 * np.pi
#     phi = phi / 180 * np.pi
#
#     centers = torch.stack(
#         [
#             radius * torch.sin(theta) * torch.sin(phi),
#             radius * torch.cos(theta),
#             radius * torch.sin(theta) * torch.cos(phi),
#         ],
#         dim=-1,
#     )  # [B, 3]
#
#     # lookat
#     forward_vector = safe_normalize(centers)
#     up_vector = (
#         torch.FloatTensor([0, 1, 0]).to(device).unsqueeze(0).repeat(len(centers), 1)
#     )
#     right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))
#     up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1))
#
#     poses = (
#         torch.eye(4, dtype=torch.float, device=device)
#         .unsqueeze(0)
#         .repeat(len(centers), 1, 1)
#     )
#     poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
#     poses[:, :3, 3] = centers
#
#     return poses


trans_t = lambda t: torch.Tensor(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, t], [0, 0, 0, 1]]
).float()

rot_phi = lambda phi: torch.Tensor(
    [
        [1, 0, 0, 0],
        [0, np.cos(phi), -np.sin(phi), 0],
        [0, np.sin(phi), np.cos(phi), 0],
        [0, 0, 0, 1],
    ]
).float()

rot_theta = lambda th: torch.Tensor(
    [
        [np.cos(th), 0, -np.sin(th), 0],
        [0, 1, 0, 0],
        [np.sin(th), 0, np.cos(th), 0],
        [0, 0, 0, 1],
    ]
).float()


def quaternion_angle(q1, q2):
    angle = 2 * np.arccos(np.dot(q1, q2))
    return np.degrees(angle)


# def rodrigues_mat_to_rot(R):
#     eps = 1e-16
#     trc = np.trace(R)
#     trc2 = (trc - 1.0) / 2.0
#     s = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
#     if (1 - trc2 * trc2) >= eps:
#         tHeta = np.arccos(trc2)
#         tHetaf = tHeta / (2 * (np.sin(tHeta)))
#     else:
#         tHeta = np.real(np.arccos(trc2))
#         tHetaf = 0.5 / (1 - tHeta / 6)
#     omega = tHetaf * s
#     return omega


# def rodrigues_rot_to_mat(r):
#     wx, wy, wz = r
#     theta = np.sqrt(wx * wx + wy * wy + wz * wz)
#     a = np.cos(theta)
#     b = (1 - np.cos(theta)) / (theta * theta)
#     c = np.sin(theta) / theta
#     R = np.zeros([3, 3])
#     R[0, 0] = a + b * (wx * wx)
#     R[0, 1] = b * wx * wy - c * wz
#     R[0, 2] = b * wx * wz + c * wy
#     R[1, 0] = b * wx * wy + c * wz
#     R[1, 1] = a + b * (wy * wy)
#     R[1, 2] = b * wy * wz - c * wx
#     R[2, 0] = b * wx * wz - c * wy
#     R[2, 1] = b * wz * wy + c * wx
#     R[2, 2] = a + b * (wz * wz)
#     return R


# def pose_spherical(theta, phi, radius):
#     c2w = trans_t(radius)
#     c2w = rot_phi(phi / 180.0 * np.pi) @ c2w
#     c2w = rot_theta(theta / 180.0 * np.pi) @ c2w
#     c2w = (
#         torch.Tensor(
#             np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
#         )
#         @ c2w
#     )
#     return c2w


# def convert_camera_pose(camera_pose):
#     # Clone the tensor to avoid in-place operations
#     colmap_pose = camera_pose.clone()
#
#     # Extract rotation and translation components
#     rotation = colmap_pose[:, :3, :3]
#     translation = colmap_pose[:, :3, 3]
#
#     # Change rotation orientation
#     rotation[:, 0, :] *= -1
#     rotation[:, 1, :] *= -1
#
#     # Change translation position
#     translation[:, 0] *= -1
#     translation[:, 1] *= -1
#
#     return colmap_pose


@dataclass
class GSLoadDataModuleConfig:
    # height, width, and batch_size should be Union[int, List[int]]
    # but OmegaConf does not support Union of containers
    source: str = None
    height: Any = 512
    width: Any = 512
    batch_size: Any = 1
    resolution_milestones: List[int] = field(default_factory=lambda: [])
    eval_height: int = -1
    eval_width: int = -1
    # eval_batch_size: int = 1
    max_view_num: int = 48  # max train view number
    n_val_views: int = 8
    # n_test_views: int = 120
    # elevation_range: Tuple[float, float] = (-10, 45)
    elevation_view_num: int = 2  # not used
    # azimuth_range: Tuple[float, float] = (-180, 180)
    azimuth_view_num: int = 8  # not used
    camera_distance_range: Tuple[float, float] = (4.0, 6.0)  # not used
    # fovy_range: Tuple[float, float] = (
    #     40,
    #     70,
    # )  # in degrees, in vertical direction (along height)
    # camera_perturb: float = 0.0
    # center_perturb: float = 0.0
    # up_perturb: float = 0.0
    # light_position_perturb: float = 1.0
    # light_distance_range: Tuple[float, float] = (0.8, 1.5)
    # eval_elevation_deg: float = 15.0
    eval_camera_distance: float = 6.0  # not used
    # eval_fovy_deg: float = 70.0
    light_sample_strategy: str = "dreamfusion"  # not used
    # batch_uniform_azimuth: bool = True
    # progressive_until: int = 0  # progressive ranges for elevation, azimuth, r, fovy
    random_shuffle_views: bool = False
    use_trajectory_anchored_train_dataset: bool = False
    per_editing_step_schedule_cfg: dict = field(default_factory=dict)
    num_inference_steps: int = 20  # for diffusion editing
    init_inference_step: int = 0
    # scheduler_id: str = "SimianLuo/LCM_Dreamshaper_v7"  # for diffusion editing
    context_max_angle_diff: float = 15.0  # in degrees

    # # timestep churn for trajectory anchored editing
    # churn_start_at: int = 0
    # churn_denoise_interval: int = 1
    # churn_interval: int = 0

    keyframe_shuffle: bool = False

    churn_indices: List[List[Union[int]]] = field(default_factory=list)

    # accumulated GS update step
    opacity_reset_interval: int = 500
    opacity_restore_interval: int = 250
    opacity_reset_until: int = 1500

    reset_gs_enabled: bool = True
    reset_gs_at: List[int] = field(default=list)

    keyframe_twice_mode: bool = True

class GSLoadIterableDataset(IterableDataset, Updateable):
    def __init__(self, cfg, scene) -> None:
        super().__init__()
        self.cfg: GSLoadDataModuleConfig = cfg
        self.scene = scene
        self.total_view_num = len(self.scene.cameras)
        threestudio.info(f"Total view number: {self.total_view_num}")
        # random.seed(0)  # make sure same views
        generator = torch.Generator()
        self.n2n_view_index_tensor = torch.linspace(
            0, self.total_view_num - 1,
            min(self.total_view_num, self.cfg.max_view_num), dtype=torch.int
        )
        threestudio.info(f"Selected view number: {len(self.n2n_view_index_tensor)}")

        self.n2n_view_index_tensor_unshuffled = self.n2n_view_index_tensor.clone()

        if self.cfg.random_shuffle_views:
            self.n2n_view_index_tensor = self.n2n_view_index_tensor[
                torch.randperm(len(self.n2n_view_index_tensor), generator=generator)]

        # self.n2n_view_index = random.sample(
        #     range(0, self.total_view_num),
        #     min(self.total_view_num, self.cfg.max_view_num),
        # )  # random permutation cam idx
        self.view_index_stack = self.n2n_view_index.copy()
        self.heights: List[int] = (
            [self.cfg.height] if isinstance(self.cfg.height, int) else self.cfg.height
        )
        self.widths: List[int] = (
            [self.cfg.width] if isinstance(self.cfg.width, int) else self.cfg.width
        )
        self.batch_sizes: List[int] = (
            [self.cfg.batch_size]
            if isinstance(self.cfg.batch_size, int)
            else self.cfg.batch_size
        )
        assert len(self.heights) == len(self.widths) == len(self.batch_sizes)
        self.resolution_milestones: List[int]
        if (
                len(self.heights) == 1
                and len(self.widths) == 1
                and len(self.batch_sizes) == 1
        ):
            if len(self.cfg.resolution_milestones) > 0:
                threestudio.warn(
                    "Ignoring resolution_milestones since height and width are not changing"
                )
            self.resolution_milestones = [-1]
        else:
            assert len(self.heights) == len(self.cfg.resolution_milestones) + 1
            self.resolution_milestones = [-1] + self.cfg.resolution_milestones

        self.height: int = self.heights[0]
        self.width: int = self.widths[0]
        self.batch_size: int = self.batch_sizes[0]

    @property
    def n_views(self):
        return len(self.n2n_view_index)

    @property
    def n2n_view_index(self):
        return self.n2n_view_index_tensor.tolist()

    def collate(self, batch) -> Dict[str, Any]:
        # sample elevation angles
        cam_list = []
        index_list = []
        for _ in range(self.batch_size):
            if not self.view_index_stack:
                self.view_index_stack = self.n2n_view_index.copy()
            view_index = random.choice(self.view_index_stack)
            self.view_index_stack.remove(view_index)
            cam_list.append(self.scene.cameras[view_index])
            index_list.append(view_index)

        return {
            "index": index_list,
            "camera": cam_list,
            "height": self.height,
            "width": self.width,
        }

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        size_ind = bisect.bisect_right(self.resolution_milestones, global_step) - 1
        self.height = self.heights[size_ind]
        self.width = self.widths[size_ind]
        self.batch_size = self.batch_sizes[size_ind]
        # self.directions_unit_focal = self.directions_unit_focals[size_ind]
        # threestudio.debug(
        #     f"Training height: {self.height}, width: {self.width}, batch_size: {self.batch_size}"
        # )
        # progressive view
        self.progressive_view(global_step)

    def __iter__(self):  # dummy iterator
        while True:
            yield {}

    def progressive_view(self, global_step):
        pass


class GSLoadDataset(Dataset):
    def __init__(self, cfg, split, scene, train_view_list=None) -> None:
        super().__init__()
        self.cfg: GSLoadDataModuleConfig = cfg
        self.split = split
        self.scene = scene
        self.total_view_num = len(self.scene.cameras)

        if split == "val":
            self.n_views = self.cfg.n_val_views
            self.h = self.cfg.height
            self.w = self.cfg.width
        elif split == "test":
            self.n_views = self.total_view_num
            self.h = self.cfg.eval_height if self.cfg.eval_height > 0 else self.scene.cameras[0].image_height
            self.w = self.cfg.eval_width if self.cfg.eval_width > 0 else self.scene.cameras[0].image_width
        elif split == "train":
            self.n_views = min(self.total_view_num, self.cfg.max_view_num)
            self.h = self.cfg.height
            self.w = self.cfg.width
        else:
            raise ValueError(f"Invalid split: {split}")

        if train_view_list is None:
            self.selected_views = torch.linspace(
                0, self.total_view_num - 1, self.n_views, dtype=torch.int
            )
        else:  # equal-distance sample n_views from sorted train_view_list
            train_view_list = sorted(train_view_list)
            self.selected_views = torch.linspace(
                0, len(train_view_list) - 1, self.n_views, dtype=torch.int
            )
            self.selected_views = [train_view_list[idx] for idx in self.selected_views]

        # random shuffle train views
        if split == "train" and cfg.random_shuffle_views:
            self.selected_views = self.selected_views[torch.randperm(len(self.selected_views))]

    def __len__(self):
        return self.n_views

    def __getitem__(self, index):
        if self.split == "train":
            threestudio.warn(
                "You should not use GSLoadDataset.__getitem__ in training"
            )

        return {
            "index": self.selected_views[index] if self.split == "val" else index,
            "height": self.h,
            "width": self.w,
        }

    def collate(self, batch):
        batch = torch.utils.data.default_collate(batch)
        batch.update({"height": self.h, "width": self.w})
        return batch


class TrajectoryAnchoredTrainDataset(GSLoadIterableDataset):
    @enum.unique
    class FSM(enum.Enum):
        UPDATE_KEYFRAME = enum.auto()
        UPDATE_CONTEXT = enum.auto()
        UPDATE_GS = enum.auto()
        OPACITY_RESTORE = enum.auto()

    def __init__(self, cfg, scene) -> None:
        """
        Rationale for implementing this class:
            In GaussianEditor, training dataset is an instance of `GSLoadIterableDataset`. Some batches are discarded by
            the trainer system because the 3DGS need to be updated by training_step() before update the edited
            images. This discard is controlled by `per_editing_step`.

            In trajectory anchored editing, the current editing step is maintained by the dataset rather than
            the trainer system (because it would be difficult to maintain the editing step in the training system while
            producing the indices of editing-step-synchronized views by the dataset).

            Therefore, such discard behavior controlled solely by the trainer system is not desired, since it may disrupt
            the editing step controlled by the dataset, causing skipped steps. We choose to implement this class to
            both control the editing step in the dataset, and also control when to discard the batch (to optimize the 3DGS),
            or not to discard the batch (to perform image editing).
        """
        from infedit.ddcm import get_churned_step_to_inference_step_mapper_fn, get_check_churning_fn

        super().__init__(cfg, scene=scene)

        self.num_keyframes = self.n_views // self.cfg.batch_size  # 1 keyframe per context

        assert self.n_views % (self.cfg.batch_size ** 2) == 0, \
            f"n_views={self.n_views} should be divisible by batch_size^2, perhaps alter max_view_num or the batch_size"

        self.keyframe_indices = torch.linspace(
            0, self.n_views - self.cfg.batch_size,
            self.num_keyframes, dtype=torch.int
        )
        # self.keyframe_context_queue = list(
        #     uniform(
        #         0,
        #         None,  # useless argument
        #         num_frames=self.num_keyframes,
        #         context_size=self.cfg.batch_size,
        #         context_stride=1,
        #         context_overlap=self.cfg.context_overlap,
        #         closed_loop=True
        #     )
        # )  # e.g. [[0, 1, 2, 3], [1, 2, 3, 4]...]

        self.keyframe_context_queue_ordered = torch.linspace(0, self.num_keyframes - 1, self.num_keyframes,
                                                             dtype=torch.int)

        self.reset_context_queue()
        self.current_keyframe_context_queue_idx = 0

        # self.diffusion_scheduler = LCMScheduler.from_config(self.cfg.scheduler_id, subfolder="scheduler")

        # timesteps = churn_timesteps(
        #     self.diffusion_scheduler.timesteps,
        #     start_at=self.cfg.churn_start_at,
        #     denoise_interval=self.cfg.churn_denoise_interval,
        #     churn_interval=self.cfg.churn_interval
        # )
        # self.num_inference_steps = len(timesteps)

        self.img_editing_step = self.cfg.init_inference_step  # counter for current editing step for all views
        self.gs_editing_step = 0  # how many steps the 3DGS has been updated for all views

        self.gs_update_step_accum = 0

        self.opacity_reset_start_at = 0
        self.prev_opacity_reset_end_at = 0

        self.edited = torch.zeros(self.n_views, dtype=torch.bool)
        self.gs_updated = torch.zeros(self.n_views, dtype=torch.bool)

        self.churn_step_to_inference_step_mapper_fn = get_churned_step_to_inference_step_mapper_fn(
            churn_indices=self.cfg.churn_indices, num_inference_steps=self.cfg.num_inference_steps
        )
        self.is_churning = get_check_churning_fn(churn_indices=self.cfg.churn_indices,
                                                 num_inference_steps=self.cfg.num_inference_steps)

        per_editing_step_schedule_args = self.cfg.per_editing_step_schedule_cfg.get("args", {})
        per_editing_step_schedule = schedule_functions[self.cfg.per_editing_step_schedule_cfg.name](
            **per_editing_step_schedule_args)
        self.per_editing_step_schedule = lambda step: int(per_editing_step_schedule(step))

        self.fsm_state = self.FSM.UPDATE_KEYFRAME

    # def __len__(self):
    #     total_editing_step_per_view = 0
    #     for i in range(self.cfg.num_inference_steps - 1):
    #         total_editing_step_per_view += max(self.per_editing_step_schedule(i), 1)
    #     return self.n_views * total_editing_step_per_view // self.cfg.batch_size

    def large_angle_diff(self, qvec1, qvec2) -> bool:
        """Check if the angle between two quaternions is larger than the threshold"""
        return quaternion_angle(qvec1, qvec2) > self.cfg.context_max_angle_diff

    def check_context_angle_diff(self, context_indices) -> bool:
        """
        Check if the angle between the first and frames in the context is larger than the threshold.
        Args:
            context_indices: shape (context_size,)
        Returns
            bool: True if smaller than thresh
        """
        for i in range(1, len(context_indices)):
            if self.large_angle_diff(self.scene.cameras[context_indices[0]].qvec,
                                     self.scene.cameras[context_indices[i]].qvec):
                return False
        return True

    @property
    def per_editing_step(self):
        """num of steps for 3DGS to update per edit"""
        return self.per_editing_step_schedule(
            self.churn_step_to_inference_step_mapper_fn(self.img_editing_step)
        )

    @staticmethod
    def get_indices(bitmask, state: bool):
        return bitmask.eq(state).nonzero(as_tuple=True)[0]

    def compose_return_dict(self, view_indices, edit_mode):
        assert edit_mode in ["keyframe", "context", "gs"], f"Invalid edit_mode: {edit_mode}"

        if edit_mode == "gs":
            self.gs_update_step_accum += len(view_indices)  # the accumulated GS update step

        return {
            "index": view_indices,
            "camera": [self.scene.cameras[idx] for idx in view_indices],
            "edit_mode": edit_mode,
            "trajectory_step": self.img_editing_step,
            "height": self.height,
            "width": self.width,
        }

    def _fsm_update_keyframe_to_update_keyframe(self):
        from infedit.context import roll_context_queue

        selected_indices = self.keyframe_indices[
            self.keyframe_context_queue[self.current_keyframe_context_queue_idx]
        ]
        self.edited[selected_indices] = True

        self.current_keyframe_context_queue_idx = (
                (self.current_keyframe_context_queue_idx + 1) % (self.num_keyframes // self.cfg.batch_size)
        )

        if self.current_keyframe_context_queue_idx == 0:  # reset, roll the context queue
            if self.cfg.keyframe_shuffle:
                self.reset_context_queue()  # random shuffle
            else:  # roll
                self.keyframe_context_queue = list(roll_context_queue(
                    self.keyframe_context_queue, shift=self.cfg.batch_size // 2  # roll half context window
                ))

        view_indices = self.n2n_view_index_tensor[selected_indices]
        return self.compose_return_dict(view_indices=view_indices, edit_mode="keyframe")

    def _fsm_update_context_to_update_context(self):
        not_edited_indices = self.get_indices(self.edited, False)
        selected_indices = not_edited_indices[:self.cfg.batch_size]
        self.edited[selected_indices] = True
        view_indices = self.n2n_view_index_tensor[selected_indices]
        # if angle diff is too large, use more robust "keyframe" mode
        edit_mode = "context" if self.check_context_angle_diff(view_indices) else "keyframe"
        return self.compose_return_dict(view_indices=view_indices, edit_mode=edit_mode)

    def _fsm_update_gs_to_update_gs(self):
        if self.gs_updated.all():  # if gs corresponding to all views updated, proceed to next step
            self.gs_editing_step += 1
            self.gs_updated.fill_(False)

        not_updated_indices = self.get_indices(self.gs_updated, False)
        selected_indices = not_updated_indices[:self.cfg.batch_size]
        self.gs_updated[selected_indices] = True
        view_indices = self.n2n_view_index_tensor[selected_indices]

        return self.compose_return_dict(view_indices=view_indices, edit_mode="gs")

    def reset_context_queue(self):
        if self.cfg.keyframe_shuffle:
            self.keyframe_context_queue = self.keyframe_context_queue_ordered[
                torch.randperm(self.num_keyframes)]
        else:
            self.keyframe_context_queue = self.keyframe_context_queue_ordered

        self.keyframe_context_queue = self.keyframe_context_queue.split(self.cfg.batch_size)
        self.keyframe_context_queue = [ctx.tolist() for ctx in self.keyframe_context_queue]

    def collate(self, batch) -> Dict[str, Any]:
        # FSM
        if self.fsm_state == self.FSM.UPDATE_KEYFRAME:
            # check if keyframe_twice_mode is enabled
            if not self.cfg.keyframe_twice_mode:  # skip keyframe update. keyframes are updated along with context
                self.fsm_state = self.FSM.UPDATE_CONTEXT
                return self.collate(batch)
            # 1. UPDATE_KEYFRAME -> UPDATE_KEYFRAME
            if not self.edited[self.keyframe_indices].all():  # if not all keyframes have been edited
                return self._fsm_update_keyframe_to_update_keyframe()
            # 2. UPDATE_KEYFRAME -> UPDATE_CONTEXT
            else:
                # 2.1 state transition
                self.fsm_state = self.FSM.UPDATE_CONTEXT
                # 2.2 reset edited flag, because we are going to update keyframe when updating context
                self.edited.fill_(False)
                # 2.3 prepare context indices, same as UPDATE_CONTEXT -> UPDATE_CONTEXT
                return self._fsm_update_context_to_update_context()

        elif self.fsm_state == self.FSM.UPDATE_CONTEXT:
            # 1. UPDATE_CONTEXT -> UPDATE_CONTEXT
            if not self.edited.all():
                return self._fsm_update_context_to_update_context()

            self.edited.fill_(False)  # reset edited flag

            # 2. UPDATE_CONTEXT -> UPDATE_KEYFRAME
            if self.per_editing_step == 0 or self.is_churning(self.img_editing_step):
                self.fsm_state = self.FSM.UPDATE_KEYFRAME
                self.img_editing_step += 1
                # same as UPDATE_KEYFRAME -> UPDATE_KEYFRAME
                # return self._fsm_update_keyframe_to_update_keyframe()
                return self.collate(batch)

            # 3. UPDATE_CONTEXT -> UPDATE_GS
            else:
                self.fsm_state = self.FSM.UPDATE_GS

                if self.gs_updated.all():  # if gs corresponding to all views updated, proceed to next step
                    self.gs_editing_step += 1
                    self.gs_updated.fill_(False)

                not_updated_indices = self.get_indices(self.gs_updated, False)
                selected_indices = not_updated_indices[:self.cfg.batch_size]
                self.gs_updated[selected_indices] = True
                view_indices = self.n2n_view_index_tensor[selected_indices]
                out = self.compose_return_dict(view_indices=view_indices, edit_mode="gs")
                if (
                        self.cfg.reset_gs_enabled
                        and self.churn_step_to_inference_step_mapper_fn(self.img_editing_step) in self.cfg.reset_gs_at
                ):
                    out["gs_reset_init"] = True
                    self.opacity_reset_start_at = 0
                    self.prev_opacity_reset_end_at = 0
                    self.gs_update_step_accum = 0
                return out

        elif self.fsm_state == self.FSM.OPACITY_RESTORE:  # FIXME
            # 1. OPACITY_RESTORE -> OPACITY_RESTORE
            if self.gs_update_step_accum < self.opacity_reset_start_at + self.cfg.opacity_restore_interval:
                out = self._fsm_update_gs_to_update_gs()
                out["restore_opacity"] = True
                return out
            # 2. OPACITY_RESTORE -> UPDATE_GS
            else:  # similar to UPDATE_CONTEXT -> UPDATE_GS
                self.fsm_state = self.FSM.UPDATE_GS
                self.prev_opacity_reset_end_at = self.gs_update_step_accum

                if self.gs_updated.all():  # if gs corresponding to all views updated, proceed to next step
                    self.gs_editing_step += 1
                    self.gs_updated.fill_(False)

                not_updated_indices = self.get_indices(self.gs_updated, False)
                selected_indices = not_updated_indices[:self.cfg.batch_size]
                self.gs_updated[selected_indices] = True
                view_indices = self.n2n_view_index_tensor[selected_indices]
                return self.compose_return_dict(view_indices=view_indices, edit_mode="gs")

        elif self.fsm_state == self.FSM.UPDATE_GS:
            # 1. UPDATE_GS -> OPACITY_RESTORE
            if (
                    self.gs_editing_step < self.per_editing_step
                    and self.gs_update_step_accum < self.cfg.opacity_reset_until
                    and self.gs_update_step_accum > self.cfg.opacity_reset_interval + self.prev_opacity_reset_end_at
            ):
                self.fsm_state = self.FSM.OPACITY_RESTORE
                # get update gs (similar to UPDATE_CONTEXT -> UPDATE_GS)

                if self.gs_updated.all():  # if gs corresponding to all views updated, proceed to next step
                    self.gs_editing_step += 1
                    self.gs_updated.fill_(False)

                not_updated_indices = self.get_indices(self.gs_updated, False)
                selected_indices = not_updated_indices[:self.cfg.batch_size]
                self.gs_updated[selected_indices] = True
                view_indices = self.n2n_view_index_tensor[selected_indices]
                out = self.compose_return_dict(view_indices=view_indices, edit_mode="gs")
                # add reset opacity flg
                out["reset_opacity"] = True
                out["restore_opacity"] = True
                # record current gs accum
                self.opacity_reset_start_at = self.gs_update_step_accum
                return out

            # 2. UPDATE_GS -> UPDATE_GS
            elif self.gs_editing_step < self.per_editing_step:
                return self._fsm_update_gs_to_update_gs()

            # 4. UPDATE_GS -> UPDATE_KEYFRAME
            else:
                self.fsm_state = self.FSM.UPDATE_KEYFRAME
                self.img_editing_step += 1
                self.gs_editing_step = 0
                self.edited.fill_(False)  # reset edited flag
                # same as UPDATE_KEYFRAME -> UPDATE_KEYFRAME
                # return self._fsm_update_keyframe_to_update_keyframe()
                return self.collate(batch)
        else:
            raise ValueError(f"Invalid FSM state: {self.fsm_state}")

    # def collate(self, batch) -> Dict[str, Any]:
    #     not_updated_indices = self.updated.eq(False).nonzero(as_tuple=True)[0]
    #
    #     if len(not_updated_indices) == 0:  # all views have updated 3DGS
    #         self.updated.fill_(False)  # reset updated flag
    #         self.gs_editing_step += 1  # increment the 3DGS editing step for all views
    #         not_updated_indices = self.updated.eq(False).nonzero(as_tuple=True)[0]
    #
    #         threestudio.debug(
    #             f"All views have updated 3DGS. Incrementing 3DGS editing step to {self.gs_editing_step}"
    #         )
    #
    #     if self.gs_editing_step >= max(self.per_editing_step, 1):  # do next step img editing
    #         self.gs_editing_step = 0
    #         self.img_editing_step += 1
    #
    #         self.img_editing_step = min(self.img_editing_step, self.cfg.num_inference_steps - 1)  # clip to max steps
    #
    #         if self.img_editing_step == self.cfg.num_inference_steps - 1:  # all steps done
    #             threestudio.warn(
    #                 f"Editing step {self.img_editing_step} should be completed. Double check your step config")
    #
    #         threestudio.debug(f"All views updated 3DGS. Step to next img editing: step={self.img_editing_step}")
    #
    #     selected_indices = not_updated_indices[:self.cfg.batch_size]  # select batch_size views to update
    #     self.updated[selected_indices] = True
    #
    #     view_indices = self.n2n_view_index_tensor[selected_indices]
    #
    #     return {
    #         "index": view_indices,
    #         "camera": [self.scene.cameras[idx] for idx in view_indices],
    #         "img_editing_step": self.img_editing_step,
    #         "should_update_img": self.gs_editing_step == 0,  # whether to update the edited images
    #         "gs_editing_step": self.gs_editing_step,
    #         "per_editing_step": self.per_editing_step,
    #         "edit_mode": ...,  # FIXME
    #         "height": self.height,
    #         "width": self.width,
    #     }


@register("gs-load")
class GS_load(pl.LightningDataModule):
    cfg: GSLoadDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        from gaussiansplatting.scene.camera_scene import CamScene

        super().__init__()
        self.cfg = parse_structured(GSLoadDataModuleConfig, cfg)
        self.train_scene = CamScene(
            self.cfg.source, h=self.cfg.height, w=self.cfg.width
        )
        self.eval_scene = CamScene(
            self.cfg.source, h=self.cfg.eval_height, w=self.cfg.eval_width
        )

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            if self.cfg.use_trajectory_anchored_train_dataset:
                self.train_dataset = TrajectoryAnchoredTrainDataset(self.cfg, self.train_scene)
            else:
                self.train_dataset = GSLoadIterableDataset(self.cfg, self.train_scene)
        if stage in [None, "fit", "validate"]:
            self.val_dataset = GSLoadDataset(
                self.cfg, "val", self.eval_scene, self.train_dataset.n2n_view_index
            )
        if stage in [None, "test", "predict"]:
            self.test_dataset = GSLoadDataset(self.cfg, "test", self.eval_scene)

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size, collate_fn=None) -> DataLoader:
        return DataLoader(
            dataset,
            # very important to disable multi-processing if you want to change self attributes at runtime!
            # (for example setting self.width and self.height in update_step)
            num_workers=0,  # type: ignore
            batch_size=batch_size,
            collate_fn=collate_fn,
        )

    def train_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.train_dataset, batch_size=None, collate_fn=self.train_dataset.collate
        )

    def val_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.val_dataset,
            batch_size=self.cfg.n_val_views,
            collate_fn=self.val_dataset.collate,
        )
        # return self.general_loader(self.train_dataset, batch_size=None, collate_fn=self.train_dataset.collate)

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=1, collate_fn=self.test_dataset.collate
        )

    def predict_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=1, collate_fn=self.test_dataset.collate
        )
