import re
import matplotlib.pyplot as plt
import os
from collections import defaultdict

log_file = "sample_minmax.log"  # 你的log文件
output_dir = "plots"            # 保存图片的文件夹

# 正则表达式提取模块名、数值
pattern = re.compile(
    r'\[(.*?)\] Forward Hook - input mean: ([\-\d\.e]+), var: ([\-\d\.e]+), min:([\-\d\.e]+), max:([\-\d\.e]+)'
)

# 解析log
with open(log_file, 'r') as f:
    lines = f.readlines()

# 找到分界点"Submitted 14 steps of actions"
split_indices = [i for i, line in enumerate(lines) if "Submitted 14 steps of actions" in line]
split_indices = [-1] + split_indices + [len(lines)]  # 方便分段

# 统计信息 {module_name: {timestep: [mean,var,min,max]}}
data = defaultdict(lambda: defaultdict(list))

# 分析每一帧（每个段）
for seg_idx in range(len(split_indices) - 1):
    segment = lines[split_indices[seg_idx] + 1: split_indices[seg_idx + 1]]
    
    timestep = 0
    for line in segment:
        match = pattern.search(line)
        if match:
            module, mean, var, min_v, max_v = match.groups()
            mean = float(mean)
            var = float(var)
            min_v = float(min_v)
            max_v = float(max_v)
            # 按照顺序每16次认为是一个时间步
            data[module][timestep].append((mean, var, min_v, max_v))
            if module == "up_modules_1_upsample":
                timestep += 1  # 一个Unet结束，进入下一个时间步

# 计算每个module在每个时间步的平均值
avg_data = defaultdict(lambda: {"mean": [], "var": [], "min": [], "max": []})

for module, timestep_dict in data.items():
    for t in range(16):  # 时间步 0~15
        if t in timestep_dict:
            stats = timestep_dict[t]
            means = [s[0] for s in stats]
            vars_ = [s[1] for s in stats]
            mins = [s[2] for s in stats]
            maxs = [s[3] for s in stats]
            avg_data[module]["mean"].append(sum(means)/len(means))
            avg_data[module]["var"].append(sum(vars_)/len(vars_))
            avg_data[module]["min"].append(sum(mins)/len(mins))
            avg_data[module]["max"].append(sum(maxs)/len(maxs))
        else:
            # 没有数据填充为nan
            avg_data[module]["mean"].append(float('nan'))
            avg_data[module]["var"].append(float('nan'))
            avg_data[module]["min"].append(float('nan'))
            avg_data[module]["max"].append(float('nan'))

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 开始画图
for module, values in avg_data.items():
    timesteps = list(range(1, 17))
    plt.figure(figsize=(10, 6))
    plt.errorbar(timesteps, values["mean"], yerr=values["var"], fmt='-o', label='Mean ± Var')
    plt.fill_between(timesteps, values["min"], values["max"], alpha=0.2, label='Min-Max Range')
    plt.title(f"{module} - Statistics across Timesteps")
    plt.xlabel('Timestep')
    plt.ylabel('Values')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{module}.png"))
    plt.close()

print(f"✅ 已完成！图片保存在 {output_dir}/ 目录下。")
