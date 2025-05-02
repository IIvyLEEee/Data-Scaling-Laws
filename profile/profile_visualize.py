# import re
# import matplotlib.pyplot as plt
# import pandas as pd
# from collections import defaultdict

# def parse_log_file(log_file_path):
#     # 初始化数据结构
#     data = defaultdict(list)
#     current_timestep = None
    
#     with open(log_file_path, 'r') as f:
#         for line in f:
#             line = line.strip()
            
#             # 检查是否是新的时间步
#             timestep_match = re.search(r'Submitted (\d+) steps of actions', line)
#             if timestep_match:
#                 current_timestep = int(timestep_match.group(1))
#                 continue
                
#             # 解析均值/方差行
#             mean_var_match = re.search(r'input mean: (-?\d+\.\d+), var: (-?\d+\.\d+)', line)
#             if mean_var_match and current_timestep is not None:
#                 mean = float(mean_var_match.group(1))
#                 var = float(mean_var_match.group(2))
#                 data[current_timestep].append((mean, var))
    
#     return data

# def visualize_data(data, save_path):
#     # 准备绘图数据
#     steps = sorted(data.keys())
#     means = []
#     vars = []
    
#     for step in steps:
#         step_data = data[step]
#         step_means = [m for m, v in step_data]
#         step_vars = [v for m, v in step_data]
        
#         means.append(step_means)
#         vars.append(step_vars)
    
#     # 创建子图
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
#     # 绘制均值热图
#     im1 = ax1.imshow(means, aspect='auto', cmap='viridis')
#     ax1.set_title('Input Mean Values Over Time')
#     ax1.set_xlabel('Layer/Module Index')
#     ax1.set_ylabel('Time Step')
#     fig.colorbar(im1, ax=ax1, label='Mean Value')
    
#     # 绘制方差热图
#     im2 = ax2.imshow(vars, aspect='auto', cmap='plasma')
#     ax2.set_title('Input Variance Values Over Time')
#     ax2.set_xlabel('Layer/Module Index')
#     ax2.set_ylabel('Time Step')
#     fig.colorbar(im2, ax=ax2, label='Variance Value')
    
#     # 保存热图图像
#     heatmap_save_path = save_path + '_heatmaps.png'
#     plt.tight_layout()
#     plt.savefig(heatmap_save_path)
#     plt.close()  # 关闭当前图像，避免实时显示
    
#     # 绘制均值和方差的折线图（第一个和最后一个时间步）
#     if steps:
#         plt.figure(figsize=(12, 6))
        
#         # 第一个时间步
#         first_step = steps[0]
#         first_means = [m for m, v in data[first_step]]
#         first_vars = [v for m, v in data[first_step]]
        
#         # 最后一个时间步
#         last_step = steps[-1]
#         last_means = [m for m, v in data[last_step]]
#         last_vars = [v for m, v in data[last_step]]
        
#         # 绘制均值
#         plt.subplot(1, 2, 1)
#         plt.plot(first_means, label=f'Step {first_step}')
#         plt.plot(last_means, label=f'Step {last_step}')
#         plt.title('Input Mean Comparison')
#         plt.xlabel('Layer/Module Index')
#         plt.ylabel('Mean Value')
#         plt.legend()
        
#         # 绘制方差
#         plt.subplot(1, 2, 2)
#         plt.plot(first_vars, label=f'Step {first_step}')
#         plt.plot(last_vars, label=f'Step {last_step}')
#         plt.title('Input Variance Comparison')
#         plt.xlabel('Layer/Module Index')
#         plt.ylabel('Variance Value')
#         plt.legend()
        
#         # 保存折线图
#         linechart_save_path = save_path + '_linecharts.png'
#         plt.tight_layout()
#         plt.savefig(linechart_save_path)
#         plt.close()  # 关闭当前图像，避免实时显示

# # 使用示例
# if __name__ == "__main__":
#     log_file_path = "/home/liyixuan23/Data-Scaling-Laws/log.txt"  # 替换为你的日志文件路径
#     save_path = "/home/liyixuan23/Data-Scaling-Laws/visualization"  # 替换为你希望保存图片的路径
#     parsed_data = parse_log_file(log_file_path)
#     visualize_data(parsed_data, save_path)
#     print(f"Visualization saved to {save_path}_heatmaps.png and {save_path}_linecharts.png")

import re
import matplotlib.pyplot as plt
from collections import defaultdict

# 读取log
log_file = 'log.txt'

# 用来保存模块名到 mean 和 var 的映射
data = defaultdict(lambda: {"mean": [], "var": []})

# 提取正则
pattern = re.compile(r'\[(.*?)\] Forward Hook - input mean: ([\-\d\.e]+), var: ([\-\d\.e]+)')

with open(log_file, 'r') as f:
    for line in f:
        match = pattern.search(line)
        if match:
            module, mean, var = match.groups()
            mean = float(mean)
            var = float(var)
            data[module]["mean"].append(mean)
            data[module]["var"].append(var)

# 画图
fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

# Plot mean
for module, values in data.items():
    axes[0].plot(values["mean"], label=module)
axes[0].set_title('Input Mean per Module')
axes[0].set_ylabel('Mean')
axes[0].legend()
axes[0].grid(True)

# Plot var
for module, values in data.items():
    axes[1].plot(values["var"], label=module)
axes[1].set_title('Input Variance per Module')
axes[1].set_ylabel('Variance')
axes[1].set_xlabel('Hook Event Index')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()

# 保存到文件而不是显示
output_path = "hook_analysis.png"
plt.savefig(output_path)

print(f"图表已保存到 {output_path}")
