import logging

import torch.nn as nn

from diffusion_policy.model.diffusion.transformer_for_diffusion import (
    TransformerForDiffusion,
)
from diffusion_policy.quantization.quan_block import (
    BaseQuantBlock,
    QuantPosEmbBlock,
    QuantMultihead,
    get_specials,
)
from diffusion_policy.quantization.quan_layer_FP4 import QuantModule

logger = logging.getLogger(__name__)


class QuantModel(nn.Module):
    def __init__(self, model: TransformerForDiffusion):
        super().__init__()
        self.model = model
        self.specials = get_specials()
        self.quant_module_refactor(self.model)
        self.quant_block_refactor(self.model)

    def quant_module_refactor(self, module: nn.Module):
        """
        Recursively replace the normal layers (conv2D, conv1D, Linear etc.) to QuantModule
        :param module: nn.Module with nn.Conv2d, nn.Conv1d, or nn.Linear in its children
        :param weight_quant_params: quantization parameters like n_bits for weight quantizer
        :param act_quant_params: quantization parameters like n_bits for activation quantizer
        """
        for name, child_module in module.named_children():
            if isinstance(child_module, (nn.Conv2d, nn.Conv1d, nn.Linear)):  # nn.Conv1d
                setattr(module, name, QuantModule(child_module))

            else:
                self.quant_module_refactor(child_module)

    def quant_block_refactor(self, module: nn.Module):
        for name, child_module in module.named_children():
            if type(child_module) in self.specials:
                if self.specials[type(child_module)] == QuantPosEmbBlock:
                    setattr(
                        module, name, self.specials[type(child_module)](child_module)
                    )
                # elif self.specials[type(child_module)] == QuantMultihead:
                #     setattr(
                #         module, name, self.specials[type(child_module)](child_module)
                #     )
                else:
                    setattr(
                        module, name, self.specials[type(child_module)](child_module)
                    )
            else:
                self.quant_block_refactor(child_module)

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        for m in self.model.modules():
            if isinstance(m, (QuantModule, BaseQuantBlock)):
                m.set_quant_state(weight_quant, act_quant)

    def forward(self, x, timesteps=None, context=None):
        return self.model(x, timesteps, context)

    def set_running_stat(self, running_stat: bool, sm_only=False):
        for m in self.model.modules():
            if isinstance(m, QuantModule) and not sm_only:
                m.set_running_stat(running_stat)
