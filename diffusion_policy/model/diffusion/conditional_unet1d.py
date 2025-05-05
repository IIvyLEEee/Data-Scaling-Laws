from typing import Union
import logging
from logging.handlers import RotatingFileHandler
import time
import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange

from diffusion_policy.model.diffusion.conv1d_components import (
    Downsample1d, Upsample1d, Conv1dBlock)
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb

from diffusion_policy.model.diffusion.quant_module.quant_linear import QuantLinear
from diffusion_policy.model.diffusion.quant_module.quant_conv1d import QuantConv1d
import hook_adder.hook_manager as hm

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

class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, 
            in_channels, 
            out_channels, 
            cond_dim,
            kernel_size=3,
            n_groups=8,
            cond_predict_scale=False,
            quant_layer=False,
            ):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups, quant_layer=quant_layer),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups, quant_layer=quant_layer),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels
        if cond_predict_scale:
            cond_channels = out_channels * 2
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        if quant_layer:
            self.cond_encoder = nn.Sequential(
                nn.Mish(),
                QuantLinear(cond_dim, cond_channels),
                Rearrange('batch t -> batch t 1'),
            )
        else:
            self.cond_encoder = nn.Sequential(
                nn.Mish(),
                nn.Linear(cond_dim, cond_channels),
                Rearrange('batch t -> batch t 1'),
            )

        # make sure dimensions compatible
        if quant_layer:
            self.residual_conv = QuantConv1d(in_channels, out_channels, 1) \  
                if in_channels != out_channels else nn.Identity()
        else:
            self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
                if in_channels != out_channels else nn.Identity()
        

    def forward(self, x, cond):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        if self.cond_predict_scale:
            embed = embed.reshape(
                embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:,0,...]
            bias = embed[:,1,...]
            out = scale * out + bias
        else:
            out = out + embed
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    def __init__(self, 
        input_dim,
        local_cond_dim=None,
        global_cond_dim=None,
        diffusion_step_embed_dim=256,
        down_dims=[256,512,1024],
        kernel_size=3,
        n_groups=8,
        cond_predict_scale=False,
        quant_layer=False,
        ):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        if quant_layer:
            diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            QuantLinear(dsed, dsed * 4),
            nn.Mish(),
            QuantLinear(dsed * 4, dsed),
            )
        else:
            diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
            )
        
        cond_dim = dsed
        if global_cond_dim is not None:
            cond_dim += global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        local_cond_encoder = None
        if local_cond_dim is not None:
            _, dim_out = in_out[0]
            dim_in = local_cond_dim
            local_cond_encoder = nn.ModuleList([
                # down encoder
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                # up encoder
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale)
            ])

        # create hookmanager
        self.hook_manager = hm.HookManager()

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            ),
        ])
        for j, sub_module in enumerate(self.mid_modules):
            self.hook_manager.register_hooks(sub_module, name=f"mid_modules_{j}")

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                Downsample1d(dim_out,, quant_layer=quant_layer) if not is_last else nn.Identity()
            ]))
            for j, sub_module in enumerate(down_modules[-1]):
                self.hook_manager.register_hooks(sub_module, name=f"down_modules_{ind}_part_{j}")

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out*2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                Upsample1d(dim_in,, quant_layer=quant_layer) if not is_last else nn.Identity()
            ]))

            for j, sub_module in enumerate(up_modules[-1]):
                self.hook_manager.register_hooks(sub_module, name=f"up_modules_{ind}_part_{j}")
        
        if quant_layer:
            final_conv = nn.Sequential(
                Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size, quant_layer=quant_layer),
                # nn.Conv1d(start_dim, input_dim, 1),
                QuantConv1d(start_dim, input_dim, 1),
            )
        else:
            final_conv = nn.Sequential(
                Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size, quant_layer=quant_layer),
                nn.Conv1d(start_dim, input_dim, 1),
            )
        self.hook_manager.register_hooks(final_conv, name="final_conv")

        self.hook_manager.register_hooks(diffusion_step_encoder, name="diffusion_step_encoder")

        self.diffusion_step_encoder = diffusion_step_encoder
        self.local_cond_encoder = local_cond_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )


    def forward(self, 
            sample: torch.Tensor, 
            timestep: Union[torch.Tensor, float, int], 
            local_cond=None, global_cond=None, **kwargs):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        local_cond: (B,T,local_cond_dim)
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        sample = einops.rearrange(sample, 'b h t -> b t h')

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)

        if global_cond is not None:
            global_feature = torch.cat([
                global_feature, global_cond
            ], axis=-1)
        
        # encode local features
        h_local = list()
        if local_cond is not None:
            local_cond = einops.rearrange(local_cond, 'b h t -> b t h')
            resnet, resnet2 = self.local_cond_encoder
            x = resnet(local_cond, global_feature)
            h_local.append(x)
            x = resnet2(local_cond, global_feature)
            h_local.append(x)
        
        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            self.hook_manager.register_hooks(resnet, name=f"down_modules_{idx}_resnet")
            self.hook_manager.register_hooks(resnet2, name=f"down_modules_{idx}_resnet2")
            self.hook_manager.register_hooks(downsample, name=f"down_modules_{idx}_downsample")
            torch.cuda.synchronize()
            start = time.time()
            x = resnet(x, global_feature)
            torch.cuda.synchronize()
            end = time.time()
            logger.info(f"down_modules_{idx}_resnet time: {end - start:.4f}s")
            if idx == 0 and len(h_local) > 0:
                x = x + h_local[0]

            torch.cuda.synchronize()
            start = time.time()
            x = resnet2(x, global_feature)
            torch.cuda.synchronize()
            end = time.time()
            logger.info(f"down_modules_{idx}_resnet2 time: {end - start:.4f}s")

            h.append(x)

            torch.cuda.synchronize()
            start = time.time()
            x = downsample(x)
            torch.cuda.synchronize()
            end = time.time()
            logger.info(f"down_modules_{idx}_downsample time: {end - start:.4f}s")

        torch.cuda.synchronize()
        start = time.time()
        for mid_module in self.mid_modules:
            self.hook_manager.register_hooks(mid_module, name=f"mid_modules")
            x = mid_module(x, global_feature)
        torch.cuda.synchronize()
        end = time.time()
        logger.info(f"mid_modules time: {end - start:.4f}s")

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            self.hook_manager.register_hooks(resnet, name=f"up_modules_{idx}_resnet")
            self.hook_manager.register_hooks(resnet2, name=f"up_modules_{idx}_resnet2")
            self.hook_manager.register_hooks(upsample, name=f"up_modules_{idx}_upsample")
            x = torch.cat((x, h.pop()), dim=1)
            torch.cuda.synchronize()
            start = time.time()
            x = resnet(x, global_feature)
            torch.cuda.synchronize()
            end = time.time()
            logger.info(f"up_modules_{idx}_resnet time: {end - start:.4f}s")

            if idx == len(self.up_modules) and len(h_local) > 0:
                x = x + h_local[1]
            
            torch.cuda.synchronize()
            start = time.time()
            x = resnet2(x, global_feature)
            torch.cuda.synchronize()
            end = time.time()
            logger.info(f"up_modules_{idx}_resnet2 time: {end - start:.4f}s")

            torch.cuda.synchronize()
            start = time.time()
            x = upsample(x)
            torch.cuda.synchronize()
            end = time.time()
            logger.info(f"up_modules_{idx}_upsample time: {end - start:.4f}s")

        torch.cuda.synchronize()
        start = time.time()
        x = self.final_conv(x)
        torch.cuda.synchronize()
        end = time.time()
        logger.info(f"final_conv time: {end - start:.4f}s")
        self.hook_manager.register_hooks(self.final_conv, name="final_conv")
        
        x = einops.rearrange(x, 'b t h -> b h t')

        self.hook_manager.clear()
        return x