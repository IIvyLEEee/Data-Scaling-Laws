from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.vision.timm_obs_encoder import TimmObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply

import os
import hook_adder.hook_manager as hm
import time
import logging
from logging.handlers import RotatingFileHandler

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 使用 RotatingFileHandler 防止日志文件过大
file_handler = RotatingFileHandler(
    'unet_timing.log', 
    maxBytes=5 * 1024 * 1024,  # 每个日志文件最大 5MB
    backupCount=3              # 保留 3 个备份日志
)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

class DiffusionUnetTimmPolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            obs_encoder: TimmObsEncoder,
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            input_pertub=0.1,
            inpaint_fixed_action_prefix=False,
            train_diffusion_n_samples=1,
            load_path=None,
            save_path=None,
            # parameters passed to step
            **kwargs
        ):
        super().__init__()

        # parse shapes
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        action_horizon = shape_meta['action']['horizon']
        # get feature dim
        obs_feature_dim = np.prod(obs_encoder.output_shape())


        # create diffusion model
        assert obs_as_global_cond
        input_dim = action_dim
        global_cond_dim = obs_feature_dim

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
            quant_layer=False,
        )

        if load_path is not None:
            self.load_quant_model(load_path)

        # ---- create hookmanager ----
        self.hook_manager = hm.HookManager()

        self.obs_encoder = obs_encoder
        self.model = model

        self.sample_data: dict = {}
        self.sample_data["xs"] = [[] for _ in range(num_inference_steps + 1)]
        self.sample_data["ts"] = [[] for _ in range(num_inference_steps + 1)]
        self.sample_data["cs"] = [[] for _ in range(num_inference_steps + 1)]

        self.noise_scheduler = noise_scheduler
        self.normalizer = LinearNormalizer()
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon # used for training
        self.obs_as_global_cond = obs_as_global_cond
        self.input_pertub = input_pertub
        self.inpaint_fixed_action_prefix = inpaint_fixed_action_prefix
        self.train_diffusion_n_samples = int(train_diffusion_n_samples)
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

    # ========= inference  ============
    def conditional_sample(self, 
            condition_data,
            condition_mask,
            local_cond=None,
            global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
        ):
        
        model = self.model
        scheduler = self.noise_scheduler

        # ---------- Hook register ----------
        self.hook_manager.register_hooks(self.model, name="conditional_unet1d")
        # -----------------------------------

        # set seed manually
        if generator is None:
            generator = torch.Generator(device=condition_data.device)
            generator.manual_seed(42)

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            torch.cuda.synchronize()
            start = time.time()
            trajectory[condition_mask] = condition_data[condition_mask]
            torch.cuda.synchronize()
            end = time.time()
            logger.info(f"[1] apply condition time: {end - start:.4f} seconds")

            # 2. predict model output
            # self.hook_manager.register_hooks(model, name="model of timestep %d" % t)
            torch.cuda.synchronize()
            start = time.time()
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)
            torch.cuda.synchronize()
            end = time.time()
            logger.info(f"[2] model predict time: {end - start:.4f} seconds")

            self.sample_data["xs"][int(t)].append(trajectory)
            self.sample_data["ts"][int(t)].append(t)
            self.sample_data["cs"][int(t)].append(cond)

            # 3. compute previous image: x_t -> x_t-1
            torch.cuda.synchronize()
            start = time.time()
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
            torch.cuda.synchronize()
            end = time.time()
            logger.info(f"[3] scheduler step time: {end - start:.4f} seconds")
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor], fixed_action_prefix: torch.Tensor=None) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        fixed_action_prefix: unnormalized action prefix
        result: must include "action" key
        """

        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        torch.cuda.synchronize()
        start = time.time()
        nobs = self.normalizer.normalize(obs_dict)
        B = next(iter(nobs.values())).shape[0]
        torch.cuda.synchronize()
        end = time.time()
        logger.info(f"normalize time: {end - start:.4f} seconds")

        # condition through global features
        torch.cuda.synchronize()
        start = time.time()
        global_cond = self.obs_encoder(nobs)
        torch.cuda.synchronize()
        end = time.time()
        logger.info(f"obs_encoder time: {end - start:.4f} seconds")

        # empty data for action
        cond_data = torch.zeros(size=(B, self.action_horizon, self.action_dim), device=self.device, dtype=self.dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        if fixed_action_prefix is not None and self.inpaint_fixed_action_prefix:
            n_fixed_steps = fixed_action_prefix.shape[1]
            cond_data[:, :n_fixed_steps] = fixed_action_prefix
            cond_mask[:, :n_fixed_steps] = True
            cond_data = self.normalizer['action'].normalize(cond_data)


        # run sampling
        torch.cuda.synchronize()
        start = time.time()
        nsample = self.conditional_sample(
            condition_data=cond_data, 
            condition_mask=cond_mask,
            local_cond=None,
            global_cond=global_cond,
            **self.kwargs)
        torch.cuda.synchronize()
        end = time.time()
        logger.info(f"diffusion con_sample time: {end - start:.4f} seconds")
        
        # unnormalize prediction
        torch.cuda.synchronize()
        start = time.time()       
        assert nsample.shape == (B, self.action_horizon, self.action_dim)
        action_pred = self.normalizer['action'].unnormalize(nsample)
        torch.cuda.synchronize()
        end = time.time()
        logger.info(f"unnormalize time: {end - start:.4f} seconds")
        
        result = {
            'action': action_pred,
            'action_pred': action_pred
        }

        # ---------- 新加的 ----------
        self.hook_manager.clear()
        # ----------------------------

        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        
        assert self.obs_as_global_cond
        global_cond = self.obs_encoder(nobs)

        # train on multiple diffusion samples per obs
        if self.train_diffusion_n_samples != 1:
            # repeat obs features and actions multiple times along the batch dimension
            # each sample will later have a different noise sample, effecty training 
            # more diffusion steps per each obs encoder forward pass
            global_cond = torch.repeat_interleave(global_cond, 
                repeats=self.train_diffusion_n_samples, dim=0)
            nactions = torch.repeat_interleave(nactions, 
                repeats=self.train_diffusion_n_samples, dim=0)

        trajectory = nactions
        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        # input perturbation by adding additonal noise to alleviate exposure bias
        # reference: https://github.com/forever208/DDPM-IP
        noise_new = noise + self.input_pertub * torch.randn(trajectory.shape, device=trajectory.device)

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (nactions.shape[0],), device=trajectory.device
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise_new, timesteps)
        
        # Predict the noise residual
        pred = self.model(
            noisy_trajectory,
            timesteps, 
            local_cond=None,
            global_cond=global_cond
        )

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()

        return loss

    def forward(self, batch):
        return self.compute_loss(batch)

    def save_quant_model(self, path):
        # 指定保存路径（确保目录存在）
        save_dir = self.save_path
        os.makedirs(save_dir, exist_ok=True)  # 自动创建目录（如果不存在）

        # 保存模型权重
        savepath = os.path.join(save_dir, "quant_unet1d.pth")
        torch.save(self.model.state_dict(), savepath)  # 推荐保存state_dict而非整个模型
        print(f"已保存量化模型权重: {savepath}")

    def load_quant_model(self, path):
        # 指定加载路径
        load_dir = self.load_path
        loadpath = os.path.join(load_dir, "quant_unet1d.pth")
        if not os.path.exists(loadpath):
            raise FileNotFoundError(f"模型权重文件不存在: {loadpath}")
        # 加载模型权重
        state_dict = torch.load(loadpath, map_location='cpu')
        self.model.load_state_dict(state_dict)
        print(f"已加载量化模型权重: {loadpath}")
        