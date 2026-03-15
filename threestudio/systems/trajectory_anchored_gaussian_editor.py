import os
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial

import torch
from einops import rearrange, reduce
from torchvision.transforms import ToPILImage
from tqdm import tqdm

import threestudio
from gaussiansplatting.gaussian_renderer import render
from threestudio.systems.gaussian_editor import GaussianEditor
from threestudio.utils.typing import List
from threestudio.utils.visergui import RendererServer


@threestudio.register("ta-gsedit-system-edit")
class TrajectoryAnchoredGaussianEditorEdit(GaussianEditor):
    @dataclass
    class Config(GaussianEditor.Config):
        local_edit: bool = False

        seg_prompt: str = ""

        opacity_reset_max: float = 0.01

        reset_gs_enabled: bool = True
        reset_gs_at: List[int] = field(default_factory=list)
        update_learning_rate: bool = True

        masked_loss: bool = False

        gradient_clip: dict = field(default_factory=dict)

        min_scale: float = 1e-8

        stop_at: int = -1

        enable_renderer_server: bool = False
        renderer_ip: str = "localhost"
        renderer_port: int = 9077

    cfg: Config

    def configure(self) -> None:
        super().configure()
        if len(self.cfg.cache_dir) > 0:
            self.cache_dir = os.path.join("edit_cache", self.cfg.cache_dir)
        else:
            self.cache_dir = os.path.join("edit_cache", self.cfg.gs_source.replace("/", "-"))

        self.per_view_loss_record = defaultdict(list)  # key: view index, value: list of loss values

        self.gs_update_step_accum = 0
        self.img_editing_t = 999

        self._update_gaussians = False  # whether 3DGS should be updated (otherwise only edit views instead)
        self._restoring_opacity = False

        self.prev_densification_step = 0

        self.rendered_frames = {}
        self.prev_view_indices = []
        self.views_grad_clip_triggered_cnt = defaultdict(int)
        self.views_grad_clip_triggered_strength = defaultdict(float)
        self.views_accum_grad_norm = defaultdict(float)

    @property
    def stop_at(self):
        if self.cfg.stop_at == -1:
            return self.guidance.cfg.num_inference_steps
        return self.cfg.stop_at

    def on_fit_start(self) -> None:
        super().on_fit_start()
        self.render_all_view(cache_name="origin_render")

        if len(self.cfg.seg_prompt) > 0:
            self.update_mask()

        self.gaussian.save_init_gaussians()

        if len(self.cfg.prompt_processor) > 0:
            self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
                self.cfg.prompt_processor
            )

        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
        self.guidance.init_latent_store(self.origin_frames)

        if self.cfg.enable_renderer_server:
            render_fn = partial(
                torch.no_grad()(render),
                pc=self.gaussian,
                pipe=self.pipe,
                bg_color=self.background_tensor
            )

            self.render_server = RendererServer(
                renderer_ip=self.cfg.renderer_ip,
                renderer_port=self.cfg.renderer_port,
                render_fn=render_fn
            )
            self.render_server.start()

    def log_gaussian_stats(self):
        mask = self.gaussian.mask
        self.log("gaussians/num_gaussians_masked", len(self.gaussian.get_xyz[mask]))

        self.log("gaussians/opacity_masked/max", self.gaussian.get_opacity[mask].max())
        self.log("gaussians/opacity_masked/min", self.gaussian.get_opacity[mask].min())
        self.log("gaussians/opacity_masked/mean", self.gaussian.get_opacity[mask].mean())

        self.log("gaussians/scale_masked/max", self.gaussian.get_scaling[mask].max())
        # self.log("gaussians/scale_masked/min", self.gaussian.get_scaling[mask].min())
        self.log("gaussians/scale_masked/min", self.gaussian.get_scaling[mask].max(dim=1).values.min())  # min max scale
        self.log("gaussians/scale_masked/mean", self.gaussian.get_scaling[mask].mean())

    def training_step(self, batch, batch_idx):
        if self.cfg.update_learning_rate:
            self.gaussian.update_learning_rate(self.gs_update_step_accum)

        img_editing_step = batch["trajectory_step"]  # current diffusion step
        edit_mode = batch["edit_mode"]

        view_indices = batch["index"]
        self.prev_view_indices = view_indices

        reset_opacity = batch.get("reset_opacity", False)
        gs_reset_init = batch.get("gs_reset_init", False)
        self._restoring_opacity = batch.get("restore_opacity", False)

        if reset_opacity:
            threestudio.info(
                f"Reset opacity with mask at img_editing_step={img_editing_step}, self.img_editing_t={self.img_editing_t}, gs_update_step_accum={self.gs_update_step_accum}")
            self.gaussian.reset_opacity_with_mask(reset_val=self.cfg.opacity_reset_max)

        if self._restoring_opacity:
            threestudio.info(
                f"Restoring opacity at img_editing_step={img_editing_step}, self.img_editing_t={self.img_editing_t}, gs_update_step_accum={self.gs_update_step_accum}")

        inference_step = self.guidance.inference_step(img_editing_step)

        if inference_step == self.stop_at or self.img_editing_t == 0:
            threestudio.info(f"Early stopping at inference step {inference_step}")
            self.trainer.should_stop = True

        if (
                self.cfg.reset_gs_enabled
                and gs_reset_init
                and inference_step in self.cfg.reset_gs_at
        ):
            threestudio.info(
                f"Reset init gaussians at img_editing_step={img_editing_step}, self.img_editing_t={self.img_editing_t}, gs_update_step_accum={self.gs_update_step_accum}")
            self.save_all_views()
            self.gaussian.reset_init_gaussians()
            self.gs_update_step_accum = 0

        bsz = len(view_indices)

        self._update_gaussians = (edit_mode == "gs")

        # FIXME: remove this
        if self._update_gaussians:
            threestudio.info(
                f"Update 3DGS at img_editing_step={img_editing_step}, self.img_editing_t={self.img_editing_t}, view_indices={view_indices}, self.gs_update_step_accum={self.gs_update_step_accum}")
        else:
            threestudio.info(
                f"Edit image at img_editing_step={img_editing_step}, self.img_editing_t={self.img_editing_t}, view_indices={view_indices}")

        out = self(batch, local=self.cfg.local_edit)
        images = out["comp_rgb"]
        masks = out["masks"]  # out["masks"].shape = 4, 512, 512
        soft_masks = out["soft_masks"]

        for i, view_idx in enumerate(view_indices):
            self.rendered_frames[int(view_idx)] = images[i].unsqueeze(0).detach().clone()

        if (
                self.cfg.edit_begin_step
                <= self.global_step
                <= self.cfg.edit_until_step
                and not self._update_gaussians
        ):
            # threestudio.debug(f"Image edit at img_editing_step={img_editing_step}")

            result = self.guidance(
                images,
                masks,
                view_indices,
                img_editing_step,
                edit_mode=edit_mode
            )

            for idx, edit_img in zip(view_indices, result["edit_images"]):
                self.edit_frames[int(idx)] = edit_img.detach().clone().unsqueeze(0)  # HWC-> 1HWC

            self.img_editing_t = result["img_editing_t"]

            # save edit input
            if "edit_input" in result:
                edit_input = result["edit_input"]
                edit_input_raw = result["edit_input_raw"]
                # for idx, edit_img, edit_img_raw in zip(view_indices, edit_input, edit_input_raw):
                #     self.save_image(
                #         f"edit_input_enc_dec_it{self.true_global_step}_t{self.img_editing_t}_v{idx}.png",
                #        rearrange(
                #            (edit_img * 255).to(torch.uint8),
                #            "c h w -> h w c"
                #        ),
                #     )
                #     self.save_image(
                #         f"edit_input_raw_it{self.true_global_step}_t{self.img_editing_t}_v{idx}.png",
                #         rearrange(
                #             (edit_img_raw * 255).to(torch.uint8),
                #             "c h w -> h w c"
                #         ),
                #     )

                # tgt_0 = result["tgt_0"]
                # tgt_t = result["tgt_t"]
                # for idx, tgt_0_sample, tgt_t_sample in zip(view_indices, tgt_0, tgt_t):
                #     torch.save(tgt_0_sample, self.get_save_path(f"tgt_0_it{self.true_global_step}_t{self.img_editing_t}_v{idx}.pt"))
                #     torch.save(tgt_t_sample, self.get_save_path(f"tgt_t_it{self.true_global_step}_t{self.img_editing_t}_v{idx}.pt"))

        if not self._update_gaussians:  # should only update image w/o update gaussians
            # dummy_loss = self._dummy_param - 0
            # return {"loss": dummy_loss}
            return

        gt_images = torch.cat(
            [self.edit_frames[int(idx)] for idx in view_indices], dim=0
        )

        original_images = torch.cat(
            [self.origin_frames[int(idx)] for idx in view_indices], dim=0
        )

        if self.cfg.masked_loss:
            guidance_out = {
                "loss_l1": (
                        torch.nn.functional.l1_loss(images * soft_masks[..., None], gt_images * soft_masks[..., None])
                        + torch.nn.functional.l1_loss(images * ~masks[..., None], original_images * ~masks[..., None])
                ),
                "loss_p": self.perceptual_loss(
                    (images * masks[..., None]).permute(0, 3, 1, 2).contiguous(),
                    (gt_images * masks[..., None]).permute(0, 3, 1, 2).contiguous(),
                ).sum(),
            }
        else:
            guidance_out = {
                "loss_l1": torch.nn.functional.l1_loss(images, gt_images),
                "loss_p": self.perceptual_loss(
                    images.permute(0, 3, 1, 2).contiguous(),
                    gt_images.permute(0, 3, 1, 2).contiguous(),
                ).sum(),
            }

        loss = 0.0

        for name, value in guidance_out.items():
            if torch.isnan(value).any():
                threestudio.warn(f"NaN detected in {name}")
                value = torch.nan_to_num(value)
            self.log(f"train/{name}", value)
            if name.startswith("loss_"):
                loss += value * self.C(
                    self.cfg.loss[name.replace("loss_", "lambda_")]
                )

        if (
                self.cfg.loss.lambda_anchor_color > 0
                or self.cfg.loss.lambda_anchor_geo > 0
                or self.cfg.loss.lambda_anchor_scale > 0
                or self.cfg.loss.lambda_anchor_opacity > 0
        ):
            anchor_out = self.gaussian.anchor_loss()
            for name, value in anchor_out.items():
                if torch.isnan(value).any():
                    threestudio.warn(f"NaN detected in {name}")
                    value = torch.nan_to_num(value)
                self.log(f"train/{name}", value)
                if name.startswith("loss_"):
                    loss += value * self.C(
                        self.cfg.loss[name.replace("loss_", "lambda_")]
                    )

        if self.cfg.loss.lambda_scale > 0:
            # L2 regularization on scale
            loss_scale = self.gaussian.get_scaling.pow(2).sum() * self.C(self.cfg.loss.lambda_scale)
            loss += loss_scale
            self.log("train/loss_scale", loss_scale)

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        # record per-view loss
        loss_per_view = self.update_per_view_loss_record(images, gt_images, masks, view_indices)
        for i, idx in enumerate(view_indices):
            for name, value in loss_per_view.items():
                self.log(f"loss_per_view/v_{int(idx)}_{name}", value[i])

        self.gs_update_step_accum += bsz

        self.log_gaussian_stats()

        return {"loss": loss}

    @torch.no_grad()
    def update_per_view_loss_record(self, images, gt_images, masks, view_indices):
        bsz = len(view_indices)

        _loss_per_view = {
            # loss l1 with mask
            "loss_l1": reduce(torch.abs(images - gt_images) * masks[..., None], "b h w c -> b", "mean"),
            "loss_p": self.perceptual_loss(
                images.permute(0, 3, 1, 2).contiguous(),
                gt_images.permute(0, 3, 1, 2).contiguous(),
            ).squeeze(),  # (bsz,)
        }

        _loss = torch.zeros(bsz, device=self.device)
        for _name, _value in _loss_per_view.items():
            if torch.isnan(_value).any():
                threestudio.warn(f"NaN detected in {_name}")
                _value = torch.nan_to_num(_value)
            if _name.startswith("loss_"):
                _loss += _value * self.C(
                    self.cfg.loss[_name.replace("loss_", "lambda_")]
                )

        for _idx, _val in zip(view_indices, _loss):
            self.per_view_loss_record[int(_idx)].append(_val.item())

        return _loss_per_view

    def save_all_views(self):
        # save edited images
        sorted_keys = sorted(self.edit_frames.keys())

        edited_images = torch.cat(
            [self.edit_frames[int(idx)] for idx in sorted_keys], dim=0
        )
        edited_images = rearrange(edited_images, "views h w c -> c h (views w)")
        ToPILImage()(edited_images).save(
            self.get_save_path(f"edit_views_it{self.true_global_step}_t{self.img_editing_t}.png")
        )

        rendered_images = torch.cat(
            [self.rendered_frames[int(idx)] for idx in sorted_keys], dim=0
        )
        rendered_images = rearrange(rendered_images, "views h w c -> c h (views w)")
        ToPILImage()(rendered_images).save(
            self.get_save_path(
                f"rendered_views_it{self.true_global_step}_t{self.img_editing_t}_gs{self.gs_update_step_accum}.png")
        )

    # def add_dummy_param(self):
    #     optimizer = self.optimizers().optimizer
    #
    #     self._dummy_param = torch.tensor(0., device=self.device).requires_grad_(True)
    #
    #     optimizer.add_param_group(
    #         {
    #             "params": [self._dummy_param],
    #             "lr": 0.,
    #             "name": "dummy_param",
    #         }
    #     )
    #
    #     return {"optimizer": optimizer}

    # def configure_optimizers(self):
    #     """
    #     Add a dummy parameter to the optimizer to prevent AssertionError from GradScaler
    #     """
    #     self._dummy_param = torch.tensor(0., device=self.device).requires_grad_(True)
    #
    #     # self.gaussian.all_params_requires_grad_(False)  # freeze all params at beginning
    #
    #     optimizer = super().configure_optimizers()["optimizer"]
    #     optimizer.add_param_group(
    #         {
    #             "params": [self._dummy_param],
    #             "lr": 0.,
    #             "name": "dummy_param",
    #         }
    #     )
    #
    #     return {"optimizer": optimizer}

    # def on_before_optimizer_step(self, optimizer):
    #     if not self._update_gaussians:  # skip 3DGS optimization
    #         pass
    #     else:
    #         if hasattr(self, "_dummy_param"):  # remove dummy param when updating 3DGS
    #             new_param_groups = []
    #             dummy_param_group = None
    #             for group in optimizer.param_groups:
    #                 if group["name"] != "dummy_param":
    #                     new_param_groups.append(group)
    #                 else:
    #                     dummy_param_group = group
    #
    #             optimizer.param_groups = list(new_param_groups)
    #             optimizer.state.pop(dummy_param_group["params"][0])
    #             # del self._dummy_param
    #
    #             # self.gaussian.all_params_requires_grad_(True)  # unfreeze all params for 3DGS optimization
    #
    #         super().on_before_optimizer_step(optimizer)

    def on_before_optimizer_step(self, optimizer):
        if (
                self._update_gaussians
                and self.gs_update_step_accum < self.cfg.densify_until_iter
                # and not self._restoring_opacity
        ):
            with torch.no_grad():
                viewspace_point_tensor_grad = torch.zeros_like(
                    self.viewspace_point_list[0]
                )
                for idx in range(len(self.viewspace_point_list)):
                    viewspace_point_tensor_grad = (
                            viewspace_point_tensor_grad
                            + torch.abs(self.viewspace_point_list[idx].grad)  # TODO: check
                    )
                # Keep track of max radii in image-space for pruning
                self.gaussian.max_radii2D[self.visibility_filter] = torch.max(
                    self.gaussian.max_radii2D[self.visibility_filter],
                    self.radii[self.visibility_filter],
                )
                self.gaussian.add_densification_stats(
                    viewspace_point_tensor_grad, self.visibility_filter
                )
                # Densification
                if (
                        self.gs_update_step_accum >= self.cfg.densify_from_iter
                        and self.gs_update_step_accum >= self.prev_densification_step + self.cfg.densification_interval
                ):  # 500 100
                    self.prev_densification_step = self.gs_update_step_accum

                    before, clone, split, prune = self.gaussian.densify_and_prune(
                        self.cfg.max_grad,
                        self.cfg.max_densify_percent,
                        self.cfg.min_opacity,
                        self.cameras_extent,
                        5,
                        min_scale=self.cfg.min_scale,
                    )

                    self.log("gaussians/densify_before", before)
                    self.log("gaussians/densify_clone", clone)
                    self.log("gaussians/densify_split", split)
                    self.log("gaussians/densify_prune", prune)

        if self._update_gaussians:
            # do gradient clipping
            param_dict = {
                "xyz": self.gaussian._xyz,
                "opacity": self.gaussian._opacity,
                "scaling": self.gaussian._scaling,
                "rotation": self.gaussian._rotation,
                "features": self.gaussian._features_dc,
                "features_rest": self.gaussian._features_rest,
            }

            # check if there is any NaN in the parameters
            for name, param in param_dict.items():
                if torch.isnan(param).any():
                    threestudio.warn(f"NaN detected in {name} parameters")

            triggered = False
            accum_strength = 0.
            for name, param in param_dict.items():
                grad = param.grad
                if grad is None:
                    continue
                norm = grad.norm()
                self.log(f"train/grad_norm_{name}", norm.item())

                if name in self.cfg.gradient_clip:
                    if torch.isnan(norm).any():
                        threestudio.warn(f"NaN detected in {name} gradient norm")
                        norm = torch.nan_to_num(norm)

                        for idx in self.prev_view_indices:
                            self.views_accum_grad_norm[int(idx)] += norm.item()

                        if name in self.cfg.gradient_clip and norm > self.cfg.gradient_clip[name]:
                            triggered = True
                            accum_strength += norm.item() / self.cfg.gradient_clip[name]  # exceed how many times

                    torch.nn.utils.clip_grad_norm_(param, self.cfg.gradient_clip[name])

            if triggered:
                for idx in self.prev_view_indices:
                    self.views_grad_clip_triggered_cnt[int(idx)] += 1
                    self.views_grad_clip_triggered_strength[int(idx)] += accum_strength

    def validation_step(self, batch, batch_idx):
        batch["camera"] = [
            self.trainer.datamodule.train_dataset.scene.cameras[idx]
            for idx in batch["index"]
        ]
        out = self(batch)
        for idx in range(len(batch["index"])):
            cam_index = batch["index"][idx].item()
            self.save_image_grid(
                f"it{self.true_global_step}-t{self.img_editing_t}-v{batch['index'][idx]}.png",
                (
                    [
                        {
                            "type": "rgb",
                            "img": self.origin_frames[cam_index][0],
                            "kwargs": {"data_format": "HWC"},
                        },
                        {
                            "type": "rgb",
                            "img": self.edit_frames[cam_index][0]
                            if cam_index in self.edit_frames
                            else torch.zeros_like(self.origin_frames[cam_index][0]),
                            "kwargs": {"data_format": "HWC"},
                        },
                    ]
                ),
                name=f"validation_step_{idx}",
                step=self.true_global_step,
            )
            self.save_image_grid(
                f"render_it{self.true_global_step}-t{self.img_editing_t}-gs{self.gs_update_step_accum}-v{batch['index'][idx]}.png",
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][idx],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                + (
                    [
                        {
                            "type": "rgb",
                            "img": out["comp_normal"][idx],
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ]
                    if "comp_normal" in out
                    else []
                )
                + (
                    [
                        {
                            "type": "rgb",
                            "img": out["semantic"][idx].moveaxis(0, -1),
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ]
                    if "semantic" in out
                    else []
                )
                + (
                    [
                        {
                            "type": "rgb",
                            "img": self.origin_frames[cam_index][0],
                            "kwargs": {"data_format": "HWC"},
                        },
                        {
                            "type": "rgb",
                            "img": self.edit_frames[cam_index][0]
                            if cam_index in self.edit_frames
                            else torch.zeros_like(self.origin_frames[cam_index][0]),
                            "kwargs": {"data_format": "HWC"},
                        },
                    ]
                ),
                name=f"validation_step_render_{idx}",
                step=self.true_global_step,
            )

        with open(self.get_save_path(f"validation_step_{self.true_global_step}_grad_clip.txt"), "w") as f:
            for idx, count in self.views_grad_clip_triggered_cnt.items():
                f.write(f"{idx}: {count}, {self.views_grad_clip_triggered_strength[idx]}\n")

        with open(self.get_save_path(f"validation_step_{self.true_global_step}_grad_norm.txt"), "w") as f:
            for idx, norm in self.views_accum_grad_norm.items():
                f.write(f"{idx}: {norm}\n")

        self.gaussian.save_ply(
            self.get_save_path(f"it{self.true_global_step}.ply")
        )

    def on_validation_epoch_end(self):
        # save edited images
        sorted_keys = sorted(self.edit_frames.keys())
        edited_images = torch.cat(
            [self.edit_frames[int(idx)] for idx in sorted_keys], dim=0
        )
        edited_images = rearrange(edited_images, "views h w c -> c h (views w)")
        ToPILImage()(edited_images).save(
            self.get_save_path(f"edit_views_{self.true_global_step}_{self.img_editing_t}.png")
        )

    def on_test_epoch_end(self):
        super().on_test_epoch_end()

    def test_step(self, batch, batch_idx):
        super().test_step(batch, batch_idx)

        # save edited images
        for idx in range(len(batch["index"])):
            cam_index = batch['index'][idx].item()
            if cam_index in self.edit_frames:
                self.save_image_grid(
                    f"{batch['index'][idx]}.png",
                    (
                        [
                            {
                                "type": "rgb",
                                "img": self.edit_frames[cam_index][0],
                                "kwargs": {"data_format": "HWC"},
                            },
                        ]
                    ),
                    name=f"test_step_{idx}",
                    step=self.true_global_step,
                )
