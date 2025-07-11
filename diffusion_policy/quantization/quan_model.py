import logging
import torch.nn as nn

from diffusion_policy.quantization.quan_block import get_specials, BaseQuantBlock
from diffusion_policy.quantization.quan_block import QuantPosEmbBlock
from diffusion_policy.quantization.quan_layer import QuantModule, StraightThrough
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D

logger = logging.getLogger(__name__)


class QuantModel(nn.Module):

    def __init__(self, model: ConditionalUnet1D):
        super().__init__()
        self.model = model
        self.specials = get_specials()
        self.quant_module_refactor(self.model)
        #self.quant_block_refactor(self.model)

    def quant_module_refactor(self, module: nn.Module):
        for name, child_module in module.named_children():
            if isinstance(child_module, (nn.Linear)) and child_module.name == "linear1":
                setattr(module, name, QuantLinear(child_module))
            else:
                self.quant_module_refactor(child_module)

    def quant_block_refactor(self, module: nn.Module):
        for name, child_module in module.named_children():
            if type(child_module) in self.specials:
                if self.specials[type(child_module)] == QuantPosEmbBlock:
                    setattr(module, name, self.specials[type(child_module)](child_module))
                else:
                    setattr(module, name, self.specials[type(child_module)](child_module))
            else:
                self.quant_block_refactor(child_module)

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        for m in self.model.modules():
            if isinstance(m, (QuantModule, BaseQuantBlock)):
                m.set_quant_state(weight_quant, act_quant)

    def forward(self, x, timesteps=None, local_cond = None, context=None):
        # import ipdb; ipdb.set_trace()
        # print(f"global_cond: {context}")
        return self.model(x, timesteps, local_cond=None, global_cond=context)
    
    def set_running_stat(self, running_stat: bool, sm_only=False):
        for m in self.model.modules():
            if isinstance(m, QuantModule) and not sm_only:
                m.set_running_stat(running_stat)

    def load_state_dict(self, state_dict, strict: bool = True):
        return super().load_state_dict(state_dict, strict)
