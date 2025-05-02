import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.quantization
import copy
import zarr
import fsspec

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.vision.timm_obs_encoder import TimmObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply

backend = "fbgemm"

store = fsspec.get_mapper("zip://dataset.zarr.zip")
root = zarr.open(store, mode="r")

print(root.tree())

images = root['images'][:100]
labels = root['labels'][:100]

images_tensor = torch.tensor(images)
labels_tensor = torch.tensor(labels)

calib_dataset = TensorDataset(images_tensor, labels_tensor)
calib_loader = 	DataLoader(calib_dataset, batch_size=32, shuffle=False)

model_output = model(trajectory, t, 
    local_cond=local_cond, global_cond=global_cond)

obs_dict_np = get_real_umi_obs_dict(
        env_obs=obs, shape_meta=cfg.task.shape_meta, 
        obs_pose_repr=obs_pose_rep,
        tx_robot1_robot0=tx_robot1_robot0,
        episode_start_pose=episode_start_pose)
obs_dict = dict_apply(obs_dict_np, 
    lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
result = policy.predict_action(obs_dict)
raw_action = result['action_pred'][0].detach().to('cpu').numpy()
action = get_real_umi_action(raw_action, obs, action_pose_repr)