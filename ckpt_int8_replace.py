import torch
from omegaconf import OmegaConf
import hydra
import sys
import os
import pathlib

from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy

def int8_replace(cfg, new_ckpt_path, device='cuda'):
    # 1. 加载ckpt
    policy: DiffusionUnetImagePolicy = hydra.utils.instantiate(cfg.policy)
    ckpt = torch.load(cfg.checkpoint.ckpt, map_location=device)

    # 2. 加载模型参数
    model = torch.load("/home/liyixuan23/Data-Scaling-Laws/17-step-model/rquant_model_8_wflat.ckpt", map_location=device)
    state_dict = model["state_dicts"] if "state_dicts" in model else model
    
    # 先将非 weight_int8 的权重转为 bfloat16
    for k, v in state_dict.items():
        if (not k.endswith("weight_flat") and not k.startswith("model.diffusion_step_encoder")) and isinstance(v, torch.Tensor):
            state_dict[k] = v.to(torch.bfloat16)
            print(f"Replaced {k} with {state_dict[k].dtype}")

    # keys_to_replace = [k for k in state_dict if k.endswith("weight_flat_8")]

    # for k in keys_to_replace:
    #     weight_int8_value = state_dict[k]
    #     print(f"{k} with {weight_int8_value.dtype}")
    #     # 构造原始的 weight 名称
    #     weight_key = k.replace("weight_flat_8", "weight")
        
    #     # 替换值
    #     state_dict[weight_key] = weight_int8_value
    #     print(f"Replaced {weight_key} with {state_dict[weight_key].dtype}")
    #     # 删除原来的两个字段
    #     # del state_dict[k]  # 删除 weight_int8
    
    if not isinstance(state_dict, dict):  # 如果加载的是模型实例
        state_dict = state_dict.state_dict()

    print("replace complete!")

    ckpt['state_dicts']['model'] = state_dict
    ckpt['state_dicts']['ema_model'] = state_dict

    torch.save(ckpt, new_ckpt_path)
    print("new ckpt save to:", new_ckpt_path)

    print("ckpt keys:")
    print(ckpt.keys())
    
    print("model keys:")
    print(ckpt['state_dicts']['model'].keys())
    print("ema_model keys:")
    print(ckpt['state_dicts']['ema_model'].keys())

OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base="1.2",
    config_path="diffusion_policy/config",
    config_name="calibration_diffusion_unet_timm_umi_workspace.yaml"
)
def main(cfg: OmegaConf):
    # 解析插值
    OmegaConf.resolve(cfg)

    int8_replace(
        cfg,
        new_ckpt_path='17-step-model/rquant_model_8_wflat_16.ckpt',
    )

    print("ckpt模型替换完成！")

if __name__ == "__main__":
    # 获取当前脚本的绝对路径
    current_script_path = os.path.abspath(__file__)
    # 获取当前脚本所在目录的路径
    current_dir = os.path.dirname(current_script_path)
    # 将当前脚本所在目录添加到系统路径中
    sys.path.append(current_dir)
    # print("current_dir:", current_dir)
    # print("sys.path:", sys.path)
    main()