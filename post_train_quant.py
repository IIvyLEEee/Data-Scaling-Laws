import hydra
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import sys
import os
import pathlib

from diffusion_policy.dataset.base_dataset import BaseImageDataset, BaseDataset
from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from diffusion_policy.model.diffusion.quant_module.quant_linear import QuantLinear
from diffusion_policy.model.diffusion.quant_module.quant_conv1d import QuantConv1d
from diffusion_policy.model.diffusion.quant_module.quant_convtranspose1d import QuantConvTranspose1d

def calibrate_model(cfg, quant_model, device='cuda', max_batches=100):
    # 1. 加载数据集
    # 自动设置数据集路径为 dataset/dataset.zarr.zip
    if hasattr(cfg.task, 'dataset'):
        if hasattr(cfg.task.dataset, 'dataset_path'):
            cfg.task.dataset.dataset_path = 'dataset/dataset.zarr.zip'
        elif 'dataset_path' in cfg.task.dataset:
            cfg.task.dataset['dataset_path'] = 'dataset/dataset.zarr.zip'
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    assert isinstance(dataset, BaseImageDataset) or isinstance(dataset, BaseDataset)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    quant_model.eval()
    quant_model.to(device)

    # 2. 遍历数据集，前向推理，收集统计信息
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            # 把batch中的所有Tensor都转到device
            batch = {k: (v.to(device) if hasattr(v, 'to') else v) for k, v in batch.items()}
            _ = quant_model(batch)
            if i + 1 >= max_batches:
                break

    # 3. 校准完成后保存模型
    torch.save(quant_model.state_dict(), 'quant_model/calibrated_fp32_model.pth')
    print("fp32模型校准并保存完成！")

    # 打印模型参数
    print(quant_model.state_dict().keys())



OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base="1.2",
    config_path="diffusion_policy/config",
    config_name="calibration_diffusion_unet_timm_umi_workspace.yaml"
)
def main(cfg: OmegaConf):
    # 2. 解析插值
    OmegaConf.resolve(cfg)

    # # 3. 加载模型
    # quant_model: DiffusionUnetImagePolicy = hydra.utils.instantiate(cfg.policy)
    # print("1")
    # checkpoint = torch.load(cfg.checkpoint.ckpt, map_location='cpu')
    # print("2")
    # print(checkpoint.keys())
    # model = checkpoint['state_dicts']['model']
    # print("3")
    # # print(model.keys())
    # quant_model.load_state_dict(model)
    # print("4")


    # 3. 加载模型
    policy: DiffusionUnetImagePolicy = hydra.utils.instantiate(cfg.policy)
    print("1")
    model_path = 'quant_model/calibrated_fp32_model.pth'
    fp32_model = torch.load(model_path, map_location='cpu')
    print("2")
    print(fp32_model['model.down_modules.0.0.residual_conv.weight_delta'])
    print(fp32_model['model.down_modules.0.0.residual_conv.input_delta'])
    print(fp32_model['model.down_modules.0.0.residual_conv.output_delta'])
    policy.load_state_dict(fp32_model)
    print("3")
    policy_bf16 = policy.to(torch.bfloat16)
    torch.save(policy_bf16.state_dict(), 'quant_model/calibrated_bf16_model.pth')
    print("bf16模型保存完成!")
    print("4")
    
    int_bits = 8  # 可选4或8
    clip_ratio = 0.99  # 例如0.95，表示只保留绝对值前95%的权重，极值clip掉

    for name, module in policy_bf16.named_modules():
        if (
            name.startswith('model.down_modules')
            or name.startswith('model.mid_modules')
            or name.startswith('model.up_modules')
            or name.startswith('model.diffusion_step_encoder')
        ):
            if isinstance(module, (QuantLinear, QuantConv1d, QuantConvTranspose1d)):
                scale = module.weight_delta
                weight_fp = module.weight.data

                # 计算量化范围
                qmax = 2 ** (int_bits - 1) - 1
                qmin = -2 ** (int_bits - 1)

                # 归一化到整数区间
                weight_norm = weight_fp / scale

                # clip
                if clip_ratio < 1.0:
                    abs_weight = weight_norm.abs().flatten()
                    threshold = abs_weight.kthvalue(int(abs_weight.numel() * clip_ratio))[0]
                    weight_norm = weight_norm.clamp(-threshold, threshold)
                weight_norm = weight_norm.clamp(qmin, qmax)

                # 量化
                if int_bits == 8:
                    weight_int = torch.round(weight_norm).to(torch.int8)
                elif int_bits == 4:
                    # int4 PyTorch没有原生类型，通常用int8存储，推理时解包
                    weight_int = torch.round(weight_norm).to(torch.int8)
                else:
                    raise ValueError("只支持int4或int8")

                module.weight.data = weight_int

    torch.save(policy_bf16.state_dict(), 'quant_model/calibrated_bf16_model.pth')
    print("6")
    # # 4. 校准
    # calibrate_model(
    #     cfg,
    #     quant_model,
    # )


if __name__ == "__main__":
    main()

