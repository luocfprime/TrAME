from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from diffusers.utils.import_utils import is_xformers_available
from einops import rearrange
from torchvision.transforms import GaussianBlur

import threestudio
from infedit.ddcm import DDCM, churn, get_check_churning_fn, get_churned_step_to_inference_step_mapper_fn
from infedit.pipeline_infedit import (
    InfEditPipeline, register_attn_processor_, GlobalStepCounter, get_mapper_batched
)
from threestudio.utils.base import BaseObject
from threestudio.utils.misc import parse_version
from threestudio.utils.ops import clamp
from threestudio.utils.schedule import schedule_functions
from threestudio.utils.typing import *


class TensorToPrimitiveDict(dict):
    def __init__(self, key_type=int):
        """
        A class to convert tensor key to primitive key in a dictionary.
        """
        super().__init__()
        self.key_type = key_type

    def __getitem__(self, key):
        if isinstance(key, torch.Tensor):
            key = key.item()

        if not isinstance(key, self.key_type):
            raise KeyError(f"Key must be of type {self.key_type}, got {type(key)}")

        return super().__getitem__(key)

    def __setitem__(self, key, value):
        if isinstance(key, torch.Tensor):
            key = key.item()

        if not isinstance(key, self.key_type):
            raise KeyError(f"Key must be of type {self.key_type}, got {type(key)}")

        super().__setitem__(key, value)


def sharpen(images):
    """
    Input RGB image, first convert to YUV then sharpen using convolution
    Args:
        images: (B, H, W, C), range [0, 1], RGB

    Returns:
        sharpened images: (B, H, W, C), range [0, 1]
    """
    device = images.device
    dtype = images.dtype

    edited_images = []
    for image in images:
        image = image.detach().cpu().numpy()
        image = np.clip(image, 0, 1)
        image = (image * 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        image[:, :, 0] = cv2.filter2D(image[:, :, 0], -1, kernel)
        image = cv2.cvtColor(image, cv2.COLOR_YUV2RGB)
        image = torch.from_numpy(image).to(device, dtype=dtype)
        image = image / 255.0
        edited_images.append(image)

    return torch.stack(edited_images, dim=0)


@threestudio.register("infedit-guidance")
class InfEditGuidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        cache_dir: Optional[str] = None
        model_name_or_path: str = "SimianLuo/LCM_Dreamshaper_v7"

        enable_memory_efficient_attention: bool = False
        enable_sequential_cpu_offload: bool = False
        enable_attention_slicing: bool = False
        enable_channels_last_format: bool = False
        guidance_scale: float = 2.0
        source_guidance_scale: float = 1.0

        half_precision_weights: bool = True
        batch_size: int = 1
        fixed_size: int = -1

        num_inference_steps: int = 20

        prompt: str = ""
        source_prompt: str = ""
        negative_prompt: str = ""
        positive_prompt: str = ""
        self_kv_edit_step_thresh: float = 0.3
        self_q_edit_step_thresh: float = 0.6
        cross_edit_step_thresh: float = 0.7
        cross_attention_kwargs: Optional[Dict[str, Any]] = field(default_factory=dict)

        error_compensation_cfg: dict = field(default_factory=dict)
        ddcm_coef_schedule_cfg: dict = field(default_factory=dict)

        # timestep churn
        churn_indices: List[List[Union[int]]] = field(default_factory=list)

        sharpen_output: bool = False

        original_blend_factor: float = 0.8
        original_blend_start: int = 18
        original_blend_until: int = 20

        use_lcm_guidance: bool = False
        ddcm_until: float = 1.0

        attn_control_until: float = 1.0

        blend_until: float = 0.7

        # use_x_tgt_0_as_gt: bool = False

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading InfEdit ...")

        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )

        pipe_kwargs = {
            "safety_checker": None,
            "feature_extractor": None,
            "requires_safety_checker": False,
            "torch_dtype": self.weights_dtype,
            "cache_dir": self.cfg.cache_dir,
        }

        self.pipe = InfEditPipeline.from_pretrained(
            self.cfg.model_name_or_path, **pipe_kwargs
        ).to(self.device)

        self.scheduler = self.pipe.scheduler
        self.scheduler.set_timesteps(self.cfg.num_inference_steps)

        if self.cfg.enable_memory_efficient_attention:
            if parse_version(torch.__version__) >= parse_version("2"):
                threestudio.info(
                    "PyTorch2.0 uses memory efficient attention by default."
                )
            elif not is_xformers_available():
                threestudio.warn(
                    "xformers is not available, memory efficient attention is not enabled."
                )
            else:
                self.pipe.enable_xformers_memory_efficient_attention()

        if self.cfg.enable_sequential_cpu_offload:
            self.pipe.enable_sequential_cpu_offload()

        if self.cfg.enable_attention_slicing:
            self.pipe.enable_attention_slicing(1)

        if self.cfg.enable_channels_last_format:
            self.pipe.unet.to(memory_format=torch.channels_last)

        # Create model
        self.vae = self.pipe.vae.eval()
        self.unet = self.pipe.unet.eval()

        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)

        self.prompt_embeds = None
        self.source_prompt_embeds = None

        self.unet_attn_step_counter = GlobalStepCounter()

        self.cross_attention_kwargs = None

        # maintain src latents, and tgt latents, indexed by view index
        self.src_0 = TensorToPrimitiveDict(key_type=int)
        self.tgt_0 = TensorToPrimitiveDict(key_type=int)

        # TODO: check
        self.src_cons = TensorToPrimitiveDict(key_type=int)

        # prepare context needed for editing. mostly copied from pipeline_infedit
        self.pre_edit_setup(
            prompt=self.cfg.prompt,
            source_prompt=self.cfg.source_prompt,
            negative_prompt=self.cfg.negative_prompt,
            positive_prompt=self.cfg.positive_prompt,
            batch_size=self.cfg.batch_size,
            self_kv_edit_step_thresh=self.cfg.self_kv_edit_step_thresh,
            self_q_edit_step_thresh=self.cfg.self_q_edit_step_thresh,
            cross_edit_step_thresh=self.cfg.cross_edit_step_thresh,
            cross_attention_kwargs=self.cfg.cross_attention_kwargs,
            churn_index_mode="indices",
            churn_indices=self.cfg.churn_indices if self.cfg.churn_indices else [(0, self.cfg.num_inference_steps)],
            num_inference_steps=self.cfg.num_inference_steps,
            guidance_scale=self.cfg.guidance_scale
        )

        self._dir_zt = 0  # cached

        # error compensation schedule
        error_compensation_args = self.cfg.error_compensation_cfg.get("args", {})
        self.error_compensation_schedule = schedule_functions[self.cfg.error_compensation_cfg.name](
            **error_compensation_args)

        ddcm_coef_schedule_args = self.cfg.ddcm_coef_schedule_cfg.get("args", {})
        self.ddcm_coef_schedule = schedule_functions[self.cfg.ddcm_coef_schedule_cfg.name](
            **ddcm_coef_schedule_args)

        threestudio.info(f"Loaded InfEdit!")

    def pre_edit_setup(
            self,
            prompt: Union[str, List[str]],
            source_prompt: Union[str, List[str]],
            negative_prompt: Union[str, List[str]] = None,
            positive_prompt: Union[str, List[str]] = None,
            prompt_highlight: Union[str, List[str]] = "",
            source_prompt_highlight: Union[str, List[str]] = "",
            batch_size: int = 4,
            churn_indices=((0., 1.)),
            churn_index_mode: str = "portion",  # "portion" or "indices
            self_kv_edit_step_thresh: float = 0.3,
            self_q_edit_step_thresh: float = 0.6,
            cross_edit_step_thresh: float = 0.7,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            num_inference_steps: int = 20,
            guidance_scale: float = 1.0,
    ):
        """
        Execute once for all views before editing.
        """
        device = self.pipe._execution_device

        do_classifier_free_guidance = self.do_classifier_free_guidance

        # prepare ddcm sampler before churn
        self.ddcm = DDCM(
            timesteps=self.pipe.scheduler.timesteps,
            alphas_cumprod=self.scheduler.alphas_cumprod
        ).to(device)

        # churn check fn and map fn
        self.is_churning = get_check_churning_fn(churn_indices=churn_indices, num_inference_steps=num_inference_steps)
        self.churn_step_to_inference_step_mapper_fn = get_churned_step_to_inference_step_mapper_fn(
            churn_indices=churn_indices, num_inference_steps=num_inference_steps
        )

        # churn timesteps
        churned_timesteps = churn(
            self.pipe.scheduler.timesteps,
            churn_indices=churn_indices,
            mode=churn_index_mode
        )

        # register attn control
        (
            mappers,
            tgt_to_src_masks,
            tgt_highlight_masks,
            src_highlight_masks,
        ) = get_mapper_batched(
            src_prompts=[source_prompt],
            tgt_prompts=[prompt],
            src_prompt_highlights=[source_prompt_highlight],
            tgt_prompt_highlights=[prompt_highlight],
            tokenizer=self.pipe.tokenizer,
            device=device,
            max_len=self.pipe.tokenizer.model_max_length
        )

        _, self_attn_processors, cross_attn_processors = register_attn_processor_(
            self.unet,
            use_cfg=do_classifier_free_guidance,
            num_inference_steps=num_inference_steps,
            self_kv_edit_step_thresh=self_kv_edit_step_thresh,
            self_q_edit_step_thresh=self_q_edit_step_thresh,
            cross_edit_step_thresh=cross_edit_step_thresh,
            global_step_counter=self.unet_attn_step_counter,
            mapper=mappers,
            tgt_to_src_mask=tgt_to_src_masks,
        )

        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )

        num_images_per_prompt = batch_size

        prompt_embeds_tuple = self.pipe.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds=None,
            lora_scale=text_encoder_lora_scale,
        )
        source_prompt_embeds_tuple = self.pipe.encode_prompt(
            source_prompt, device, num_images_per_prompt, do_classifier_free_guidance, positive_prompt, None
        )  # tuple of (pos, neg) emb with shape = (bsz, seq_len, dim)
        if prompt_embeds_tuple[1] is not None:  # negative prompt is not None, do classifier-free guidance
            prompt_embeds = torch.cat([prompt_embeds_tuple[1], prompt_embeds_tuple[0]])  # (2 bsz, seq_len, dim)
        else:
            prompt_embeds = prompt_embeds_tuple[0]  # (bsz, seq_len, dim)
        if source_prompt_embeds_tuple[1] is not None:
            source_prompt_embeds = torch.cat([source_prompt_embeds_tuple[1], source_prompt_embeds_tuple[0]])
        else:
            source_prompt_embeds = source_prompt_embeds_tuple[0]

        self.prompt_embeds = prompt_embeds
        self.source_prompt_embeds = source_prompt_embeds

        self.w_embedding = self.get_guidance_scale_embedding(
            torch.tensor(guidance_scale - 1).unsqueeze(0),
            embedding_dim=self.unet.config.time_cond_proj_dim
        ).to(device, dtype=self.weights_dtype)

        self.cross_attention_kwargs = cross_attention_kwargs
        self.self_attn_processors = self_attn_processors
        self.cross_attn_processors = cross_attn_processors

        self.churned_timesteps = churned_timesteps

    # def is_churning(self, img_editing_step):
    #     """not churning: i.e. all previous t is greater than current t"""
    #     return (self.churned_timesteps[:img_editing_step] > self.churned_timesteps[img_editing_step]).all()

    @property
    def do_classifier_free_guidance(self):
        return (self.cfg.guidance_scale > 1.0 or self.cfg.source_guidance_scale > 1.0) and not self.cfg.use_lcm_guidance

    @property
    def concat_prompt_embeds(self):
        source_prompt_embeds = self.source_prompt_embeds
        prompt_embeds = self.prompt_embeds

        concat_prompt_embeds = torch.cat(
            [
                source_prompt_embeds,
                prompt_embeds,
            ],
            dim=0,
        )  # (2 2 bsz, seq_len, dim) if do_classifier_free_guidance else (2 bsz, seq_len, dim)

        if self.do_classifier_free_guidance:
            concat_prompt_embeds = rearrange(concat_prompt_embeds,
                                             "(branches conditions bsz) seq_len dim -> (conditions branches bsz) seq_len dim",
                                             branches=2, conditions=2)

        return concat_prompt_embeds

    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(
            self,
            latents: Float[Tensor, "..."],
            t: Float[Tensor, "..."],
            encoder_hidden_states: Float[Tensor, "..."],
    ) -> Float[Tensor, "..."]:
        input_dtype = latents.dtype
        return self.unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
        ).sample.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(
            self, imgs: Float[Tensor, "B 3 H W"]
    ) -> Float[Tensor, "B 4 DH DW"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def decode_latents(
            self, latents: Float[Tensor, "B 4 DH DW"]
    ) -> Float[Tensor, "B 3 H W"]:
        input_dtype = latents.dtype
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents.to(self.weights_dtype)).sample
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)

    def init_latent_store(self, origin_frames):
        for idx, img in origin_frames.items():
            img = rearrange(img, "1 H W C -> 1 C H W")
            latent = self.encode_images(img)
            self.src_0[idx] = latent.detach().clone()
            self.tgt_0[idx] = latent.detach().clone()

            # TODO: check
            self.src_cons[idx] = latent.detach().clone()

    @staticmethod
    def _dict_collate(view_indices, data_dict):
        try:
            return torch.cat([data_dict[idx] for idx in view_indices], dim=0)
        except KeyError:
            return None

    def _pack_latents(self, source_latents, target_latents):
        do_classifier_free_guidance = self.do_classifier_free_guidance
        latents = target_latents

        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        source_latent_model_input = (
            torch.cat([source_latents] * 2) if do_classifier_free_guidance else source_latents
        )  # (2 bsz, 4, h, w) if do_classifier_free_guidance else (bsz, 4, h, w)

        concat_latent_model_input = torch.cat(
            [
                source_latent_model_input,
                latent_model_input,
            ],
            dim=0,
        )  # (2 2 bsz, 4, h, w) if do_classifier_free_guidance else (2 bsz, 4, h, w)

        if do_classifier_free_guidance:
            concat_latent_model_input = rearrange(concat_latent_model_input,
                                                  "(branches conditions bsz) c h w -> (conditions branches bsz) c h w",
                                                  branches=2, conditions=2)

        return concat_latent_model_input

    def _unpack_noise_pred_and_cfg(self, concat_noise_pred):
        """
        1. Unpack the noise predictions from the concatenated tensor
        2. Apply classifier-free guidance if needed
        """
        if self.do_classifier_free_guidance:
            (
                source_noise_pred_uncond,
                noise_pred_uncond,
                source_noise_pred_text,
                noise_pred_text,
            ) = concat_noise_pred.chunk(4, dim=0)

            tgt_noise_pred = noise_pred_uncond + self.cfg.guidance_scale * (noise_pred_text - noise_pred_uncond)
            source_noise_pred = source_noise_pred_uncond + self.cfg.source_guidance_scale * (
                    source_noise_pred_text - source_noise_pred_uncond
            )

        else:
            (source_noise_pred, tgt_noise_pred) = concat_noise_pred.chunk(2, dim=0)

        return source_noise_pred, tgt_noise_pred

    @staticmethod
    def inject_and_compensate_error(rendered_latents, target_latents, factor):
        error = rendered_latents - target_latents
        return rendered_latents - factor * error

    def get_guidance_scale_embedding(self, w, embedding_dim=512, dtype=torch.float32):
        """
        See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

        Args:
            timesteps (`torch.Tensor`):
                generate embedding vectors at these timesteps
            embedding_dim (`int`, *optional*, defaults to 512):
                dimension of the embeddings to generate
            dtype:
                data type of the generated embeddings

        Returns:
            `torch.FloatTensor`: Embedding vectors with shape `(len(timesteps), embedding_dim)`
        """
        assert len(w.shape) == 1
        w = w * 1000.0

        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        assert emb.shape == (w.shape[0], embedding_dim)
        return emb

    @torch.no_grad()
    def __call__(
            self,
            rgb: Float[Tensor, "B H W C"],
            mask,  # B H W
            view_indices: Int[Tensor, "B"],
            img_editing_step: int,
            edit_mode: str,
            **kwargs,
    ):
        """

        Args:
            rgb:
            view_indices:
            img_editing_step: range [0, len(self.churned_timesteps) - 1]
            edit_mode: one of ["keyframe", "context"]
            **kwargs:

        Returns:

        """
        batch_size, H, W, _ = rgb.shape

        if img_editing_step >= len(self.churned_timesteps):
            threestudio.warn("img_editing_step exceeds len(self.churned_timesteps).")
            edit_images = self.latent_to_img(self._dict_collate(view_indices, self.tgt_0), H, W)
            if self.cfg.sharpen_output:
                edit_images = sharpen(edit_images)
            return {
                "edit_images": edit_images,
                "img_editing_t": 0,
            }

        t = self.churned_timesteps[img_editing_step]

        edit_mode = 0 if edit_mode == "keyframe" else 1  # KEYFRAME: 0, CONTEXT: 1, see class EditMode(enum.Enum)

        for processor in self.self_attn_processors.values():
            processor.set_edit_mode(edit_mode)

        do_attn_control = self.inference_step(img_editing_step) < self.cfg.attn_control_until * self.cfg.num_inference_steps
        for processor in self.self_attn_processors.values():
            processor.enabled = do_attn_control
        for processor in self.cross_attn_processors.values():
            processor.enabled = do_attn_control

        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        input_latents: Float[Tensor, "B 4 DH DW"]
        if self.cfg.fixed_size > 0:
            RH, RW = self.cfg.fixed_size, self.cfg.fixed_size
        else:
            RH, RW = H // 8 * 8, W // 8 * 8

        rgb_BCHW_HW8 = F.interpolate(
            rgb_BCHW, (RH, RW), mode="bilinear", align_corners=False
        )

        # replace rgb_BCHW_HW8 > 1.0 or < 0.0 with corresponding blurred values
        blurred = GaussianBlur(kernel_size=3)(rgb_BCHW_HW8)
        rgb_BCHW_HW8 = torch.where(
            (rgb_BCHW_HW8 > 1.0) | (rgb_BCHW_HW8 < 0.0), blurred, rgb_BCHW_HW8
        )
        rgb_BCHW_HW8 = clamp(rgb_BCHW_HW8, 0.0, 1.0)

        input_latents = self.encode_images(rgb_BCHW_HW8)

        *_, latent_H, latent_W = input_latents.shape  # bsz, dim, h, w

        self.unet_attn_step_counter.update(step=self.inference_step(img_editing_step))

        src_0 = self._dict_collate(view_indices, self.src_0)
        tgt_0 = self._dict_collate(view_indices, self.tgt_0)

        src_cons = self._dict_collate(view_indices, self.src_cons)

        noise = torch.randn_like(src_0)

        if not self.is_churning(img_editing_step):  # compensate err only when not churning
            # inject rendered image and apply error compensation
            compensation_factor = self.error_compensation_schedule(self.inference_step(img_editing_step))

            if compensation_factor < 1.:
                threestudio.info(f"compensate with factor {compensation_factor} at img_editing_step={img_editing_step}")

            # # blend original
            # if self.cfg.original_blend_start <= self.inference_step(img_editing_step) < self.cfg.original_blend_until:
            #     input_latents_ = self.cfg.original_blend_factor * input_latents + (
            #             1 - self.cfg.original_blend_factor) * src_0
            # else:
            #     input_latents_ = input_latents

            input_latents_ = input_latents

            # if self.cfg.use_x_tgt_0_as_gt:
            #     tgt_0 = self.inject_and_compensate_error(
            #         rendered_latents=input_latents_,
            #         target_latents=tgt_0,  # tgt_0 in this case is the tgt_cons from prev step
            #         factor=compensation_factor
            #     )
            #     tgt_0 = tgt_0 + self.cfg.ddcm_gamma * (src_0 - src_cons)  # obtain ddcm_next
            # else:
            #     tgt_0 = self.inject_and_compensate_error(
            #         rendered_latents=input_latents_,
            #         target_latents=tgt_0,
            #         factor=compensation_factor
            #     )

            tgt_0 = self.inject_and_compensate_error(
                rendered_latents=input_latents_,
                target_latents=tgt_0,
                factor=compensation_factor
            )

        # add noise to t
        src_t, tgt_t = self.ddcm.add_noise(
            x_src_0=src_0,
            x_tgt_0=tgt_0,
            dir_xt=self._dir_zt,
            t=t,
            noise=noise,
        )

        # inpainting
        if self.inference_step(img_editing_step) < self.cfg.blend_until * self.cfg.num_inference_steps:
            mask = rearrange(mask, "b H W -> b 1 H W").to(self.device, dtype=tgt_t.dtype)
            mask = F.interpolate(mask, size=(latent_H, latent_W), mode="bilinear", align_corners=False)
            tgt_t = mask * tgt_t + (1 - mask) * src_t

        # predict noise
        concat_latent_model_input = self._pack_latents(
            source_latents=src_t,
            target_latents=tgt_t,
        )

        concat_noise_pred = self.unet(
            concat_latent_model_input,
            t,
            timestep_cond=self.w_embedding,
            cross_attention_kwargs=self.cross_attention_kwargs,
            encoder_hidden_states=self.concat_prompt_embeds,
        ).sample

        source_noise_pred, tgt_noise_pred = self._unpack_noise_pred_and_cfg(concat_noise_pred)

        use_ddcm = self.inference_step(img_editing_step) < self.cfg.ddcm_until * self.cfg.num_inference_steps

        # denoise
        _, self._dir_zt, src_cons, tgt_cons = self.ddcm.stage_1(
            x_src=src_t,
            x_tgt=tgt_t,
            t=t,
            e_src=source_noise_pred,
            e_tgt=tgt_noise_pred,
            x_src_0=src_0,
            use_ddcm=use_ddcm,
            return_cons_pred=True
        )

        # if self.cfg.use_x_tgt_0_as_gt:
        #     tgt_0_ddcm_next = tgt_cons
        # else:
        #     tgt_0_ddcm_next = tgt_cons + self.cfg.ddcm_gamma * (src_0 - src_cons)

        ddcm_coef = self.ddcm_coef_schedule(self.inference_step(img_editing_step))
        tgt_0_ddcm_next = tgt_cons + ddcm_coef * (src_0 - src_cons)

        # else:
        #     tgt_0_ddcm_next, self._dir_zt = self.ddcm.stage_1(
        #         x_src=src_t,
        #         x_tgt=tgt_t,
        #         t=t,
        #         e_src=source_noise_pred,
        #         e_tgt=tgt_noise_pred,
        #         x_src_0=src_0,
        #         use_ddcm=use_ddcm,
        #     )

        for i, view_idx in enumerate(view_indices):
            self.tgt_0[view_idx] = tgt_0_ddcm_next[i].view(1, 4, latent_H, latent_W)

        # if self.cfg.use_x_tgt_0_as_gt:
        #     edit_latents = tgt_cons
        # else:
        #     edit_latents = tgt_0_ddcm_next

        edit_latents = tgt_0_ddcm_next

        edit_images = self.latent_to_img(edit_latents, H, W)

        # if self.cfg.sharpen_output:
        #     edit_images = sharpen(edit_images)

        return {
            "edit_images": edit_images,
            "edit_input": self.latent_to_img(tgt_0, H, W, to_bhwc=False),  # bchw
            "edit_input_raw": rgb_BCHW_HW8,
            "img_editing_t": t,
            "tgt_0": tgt_0,
            "tgt_t": tgt_t,  # with inpainting
        }

    def latent_to_img(self, latent, H, W, to_bhwc=True):
        img = self.decode_latents(latent)
        img = F.interpolate(img, (H, W), mode="bilinear")
        if to_bhwc:
            return rearrange(img, "B C H W -> B H W C")
        return img

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        pass

    def inference_step(self, img_editing_step: int):
        return self.churn_step_to_inference_step_mapper_fn(img_editing_step)

    def reached_end(self, img_editing_step: int):
        return img_editing_step >= len(self.churned_timesteps)
