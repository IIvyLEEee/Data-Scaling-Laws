import matplotlib.pyplot as plt
import os
import torch
import numpy as np

# 加载模型
# checkpoint = torch.load("/home/liyixuan23/Data-Scaling-Laws/checkpoint/latest.ckpt", map_location='cpu')
checkpoint = torch.load("/home/liyixuan23/Data-Scaling-Laws/17-step-model/int8_quant.ckpt", map_location='cpu')
weights = checkpoint['state_dicts']['model']
# weights = torch.load("/home/liyixuan23/Data-Scaling-Laws/quant_model/bf16_model.pth")

# 设置基础输出目录
base_output_dir = "weight_int_8"

# 所有模块前缀
module_prefixes = [
    "obs_encoder.key_model_map.camera0_rgb",
    "model.mid_modules",
    "model.up_modules",
    "model.down_modules",
    "model.final_conv",
    "normalizer.params_dict",
    "model.diffusion_step_encoder",
]

# 处理每个权重
for key, value in weights.items():
    if not isinstance(value, torch.Tensor):
        continue
    if value.dim() == 0 or value.numel() < 10:
        continue

    # 找出该key属于哪个模块
    module_dir = "others"
    for prefix in module_prefixes:
        if key.startswith(prefix):
            module_dir = prefix.replace(".", "_")
            break

    # 创建对应模块文件夹
    output_dir = os.path.join(base_output_dir, module_dir)
    os.makedirs(output_dir, exist_ok=True)

    # 画直方图
    plt.figure(figsize=(6, 4))
    plt.hist(value.cpu().numpy().flatten(), bins=50, alpha=0.75)
    plt.title(key)
    plt.xlabel("Weight value")
    plt.ylabel("Frequency")

    # 保存图像
    filename = key.replace(".", "_") + ".png"
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

print(f"Saved weight distributions into: {base_output_dir}")
