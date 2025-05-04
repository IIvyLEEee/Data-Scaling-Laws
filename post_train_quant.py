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

def model_quantization(policy_bf16, model_path, int_bits=8, clip_ratio=0.99):
    # 计算量化范围
    qmax = 2 ** (int_bits - 1) - 1
    qmin = -2 ** (int_bits - 1)

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
                    # TODO: 需要实现int4的量化
                    weight_int = torch.round(weight_norm).to(torch.int8)
                    # weight_int = (weight_int & 0x0F) | ((weight_int & 0xF0) >> 4)
                    # weight_int = weight_int.to(torch.int8)
                    # weight_int = weight_int.view(-1, 2)
                    # weight_int = weight_int.permute(1, 0).contiguous()
                    # weight_int = weight_int.view(-1)
                    # weight_int = weight_int.to(torch.int8)
                else:
                    raise ValueError("只支持int4或int8")

                module.register_buffer(
                    name='weight_int',
                    tensor=weight_int,
                )
    print("5")

    torch.save(policy_bf16.state_dict(), model_path)
    print("int8模型保存完成!")

OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base="1.2",
    config_path="diffusion_policy/config",
    config_name="calibration_diffusion_unet_timm_umi_workspace.yaml"
)
def main(cfg: OmegaConf):
    sw = 3 # 0:从fp32 ckpt 添加delta，1:校准，2：量化，3:只量化到bf16

    # 解析插值
    OmegaConf.resolve(cfg)

    if sw == 0:
        # 执行校准代码
        print("Running FIRST calibration...")
        quant_model: DiffusionUnetImagePolicy = hydra.utils.instantiate(cfg.policy)
        print("1")
        checkpoint = torch.load(cfg.checkpoint.ckpt, map_location='cpu')
        print(checkpoint.keys())
        model = checkpoint['state_dicts']['model']
        print("2")
        ema_model = checkpoint['state_dicts']['ema_model']
        print("2.5")
        # print(model.keys())
        # quant_model.load_state_dict(model)
        # print("3")
        # calibrate_model(
        #     cfg,
        #     quant_model,
        # )
        # print("4")
        # torch.save(quant_model.state_dict(), 'quant_model/calibrated_fp32_model.pth')
        # print("fp32模型校准完成！")

        quant_model.load_state_dict(ema_model)
        print("5")
        calibrate_model(
            cfg,
            quant_model,
        )
        print("6")
        torch.save(quant_model.state_dict(), 'quant_model/calibrated_fp32_ema_model.pth')
        print("fp32 EMA模型校准完成！")
    
    elif sw == 1:
        # 执行校准代码
        print("Running calibration...")
        quant_model: DiffusionUnetImagePolicy = hydra.utils.instantiate(cfg.policy)
        print("1")
        model_path = 'quant_model/calibrated_fp32_model.pth'
        model = torch.load(model_path, map_location='cpu')
        print("model_path:", model_path)
        # checkpoint = torch.load(cfg.checkpoint.ckpt, map_location='cpu')
        # print("2")
        # print(checkpoint.keys())
        # model = checkpoint['state_dicts']['model']
        print("2")
        # print(model.keys())
        quant_model.load_state_dict(model)
        print("3")
        calibrate_model(
            cfg,
            quant_model,
        )
        print("4")
        torch.save(quant_model.state_dict(), 'quant_model/calibrated_fp32_model.pth')
        print("fp32模型校准完成！")    

    elif sw == 2:
        # 执行量化代码
        print("Running quantization...")
        policy: DiffusionUnetImagePolicy = hydra.utils.instantiate(cfg.policy)
        print("1")
        model_path = 'quant_model/calibrated_fp32_model.pth'
        ema_model_path = 'quant_model/calibrated_fp32_ema_model.pth'
        fp32_model = torch.load(model_path, map_location='cpu')
        fp32_ema_model = torch.load(ema_model_path, map_location='cpu')
        print("2")
        print(fp32_model['model.down_modules.0.0.residual_conv.weight_delta'])
        print(fp32_model['model.down_modules.0.0.residual_conv.input_delta'])
        print(fp32_model['model.down_modules.0.0.residual_conv.output_delta'])
        print(fp32_ema_model['model.down_modules.0.0.residual_conv.weight_delta'])
        print(fp32_ema_model['model.down_modules.0.0.residual_conv.input_delta'])
        print(fp32_ema_model['model.down_modules.0.0.residual_conv.output_delta'])
        policy.load_state_dict(fp32_model)
        print("3")
        policy_bf16 = policy.to(torch.bfloat16)
        torch.save(policy_bf16.state_dict(), 'quant_model/calibrated_bf16_model.pth')
        print("bf16模型保存完成!")
        policy.load_state_dict(fp32_ema_model)
        policy_bf16_ema = policy.to(torch.bfloat16)
        torch.save(policy_bf16_ema.state_dict(), 'quant_model/calibrated_bf16_ema_model.pth')
        print("bf16 EMA模型保存完成!")
        print("4")
        model_quantization(
            policy_bf16,
            model_path='quant_model/calibrated_int8_model.pth',
            int_bits=8,
            clip_ratio=0.99,
        )
        model_quantization(
            policy_bf16_ema,
            model_path='quant_model/calibrated_int8_ema_model.pth',
            int_bits=8,
            clip_ratio=0.99,
        )
        print("5")

    elif sw == 3:
        # 只量化到bf16
        print("Running bf16 quantization...")
        quant_model: DiffusionUnetImagePolicy = hydra.utils.instantiate(cfg.policy)
        print("1")
        checkpoint = torch.load(cfg.checkpoint.ckpt, map_location='cpu')
        print(checkpoint['state_dicts']['model'].keys())
        model = checkpoint['state_dicts']['model']
        ema_model = checkpoint['state_dicts']['ema_model']
        print("2")
        quant_model.load_state_dict(model)
        for name, module in quant_model.named_modules():
            if (
                name.startswith('model.down_modules')
                or name.startswith('model.mid_modules')
                or name.startswith('model.up_modules')
            ):
                module.to(torch.bfloat16)
        torch.save(quant_model.state_dict(), 'quant_model/bf16_model.pth')
        print("bf16模型保存完成!")
        bf16_model = torch.load('quant_model/bf16_model.pth', map_location='cpu')
        checkpoint['state_dicts']['model'] = bf16_model
        print("model 量化完成！")
        quant_model.load_state_dict(ema_model)
        for name, module in quant_model.named_modules():
            if (
                name.startswith('model.down_modules')
                or name.startswith('model.mid_modules')
                or name.startswith('model.up_modules')
            ):
                module.to(torch.bfloat16)
        torch.save(quant_model.state_dict(), 'quant_model/bf16_ema_model.pth')
        print("bf16 ema_model 保存完成!")
        bf16_ema_model = torch.load('quant_model/bf16_ema_model.pth', map_location='cpu')
        checkpoint['state_dicts']['ema_model'] = bf16_ema_model
        print("ema_model 量化完成！")
        print("3")
        torch.save(checkpoint, 'checkpoint/bf16_model.ckpt')
        print("bf16模型保存完成!")
    
    else:
        print("请指定 sw 参数来执行校准或量化。")

if __name__ == "__main__":
    main()

