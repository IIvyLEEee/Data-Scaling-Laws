import torch
from omegaconf import OmegaConf
import hydra
import sys
import os
import pathlib

from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy

def merge_model_into_ckpt(cfg, new_ckpt_path, model_path, device='cuda'):
    # 1. 加载ckpt
    policy: DiffusionUnetImagePolicy = hydra.utils.instantiate(cfg.policy)
    ckpt = torch.load(cfg.checkpoint.ckpt, map_location=device)
    # 2. 加载模型参数
    model = torch.load(model_path, map_location=device)
    print("model_path:", model_path)
    # policy.load_state_dict(model)
    # 3. 将模型参数加载到policy中
    ckpt['state_dicts']['model'] = model
    torch.save(ckpt, new_ckpt_path)
    print("new ckpt save to:", new_ckpt_path)

OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base="1.2",
    config_path="diffusion_policy/config",
    config_name="calibration_diffusion_unet_timm_umi_workspace.yaml"
)
def main(cfg: OmegaConf):
    # 解析插值
    OmegaConf.resolve(cfg)

    merge_model_into_ckpt(
        cfg,
        new_ckpt_path='checkpoint/calibrated_int8.ckpt',
        model_path='quant_model/calibrated_int8_model.pth',
    )

    print("ckpt模型合并完成！")

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