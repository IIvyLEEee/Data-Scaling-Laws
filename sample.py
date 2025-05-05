import os
import torch
import torch.nn as nn
from diffusion_policy.module.quant_linear import Linear

model = torch.load(
    os.path.expanduser("~/new_diffusion/chiptest/data/cali_data/quant_model.pth")
)

cali_data = torch.load(
    os.path.expanduser("~/new_diffusion/chiptest/data/cali_data/cali_data.ckpt")
)
nsteps = len(cali_data["ts"])
timesteps = list(range(0, nsteps, 1))
cali_xs = [cali_data["xs"][i][:] for i in timesteps]
cali_ts = [cali_data["ts"][i][:] for i in timesteps]
cali_cond = [cali_data["cs"][i][:] for i in timesteps]
flattened_list = [item for sublist in cali_xs for item in sublist]
cali_xs = torch.stack(flattened_list)
flattened_list = [item for sublist in cali_ts for item in sublist]
cali_ts = torch.stack(flattened_list)
flattened_list = [item for sublist in cali_cond for item in sublist]
cali_cond = torch.stack(flattened_list)


def change(module: nn.Module, calibration=False):
    for _, child_module in module.named_children():
        if isinstance(child_module, (Linear)):  # nn.Conv1d
            child_module.calibration = calibration
        else:
            change(child_module, calibration)


with torch.no_grad():
    # inds = np.random.choice(cali_xs.shape[0], 64, replace=False)
    # _ = qnn(cali_xs[:64].cuda(), cali_ts[:64].cuda())
    change(model, True)

    # for i in range(len(cali_xs)):
    for i in range(200):
        _ = model(
            cali_xs[i].cuda(),
            cali_ts[i].cuda(),
            cali_cond[i].cuda(),
        )
        print(cali_xs[i].abs().max(), cali_xs[i].abs().min())

    change(model, False)

torch.save(
    model, os.path.expanduser("~/new_diffusion/chiptest/data/cali_data/cali_model.pth")
)
