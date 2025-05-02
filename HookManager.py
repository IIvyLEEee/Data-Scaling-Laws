# def forward_wrapper(name):
#     def forward_hook(module, input, output):
#         if name == "Conv1d":
#             print("Conv1d input mean is: ", input.mean())
#             print("Conv1d input var is: ", input.var())
#         elif name == "Conv1dBlock":
#             print("Conv1dBlock input mean is: ", input.mean())
#             print("Conv1dBlock input var is: ", input.var())
#         elif name == "ConditionalResidualBlock1D":
#             print("ConditionalResidualBlock1D input mean is: ", input.mean())
#             print("ConditionalResidualBlock1D input var is: ", input.var())
#         elif name == "Downsample1d":
#             print("Downsample1d input mean is: ", input.mean())
#             print("Downsample1d input var is: ", input.var())
#         elif name == "Upsample1d":
#             print("Upsample1d input mean is: ", input.mean())
#             print("Upsample1d input var is: ", input.var())
#     return forward_hook

# def backward_wrapper(name):
#     def backward_hook(module, input, output):
#         if name == "Conv1d":
#             print("Conv1d dx is: ", input)
#             print("Conv1d dy is: ", output)
#             print("Conv1d output mean is: ", output.mean())
#             print("Conv1d output var is: ", output.var())
#         elif name == "Conv1dBlock":
#             print("Conv1dBlock dx is: ", input)
#             print("Conv1dBlock dy is: ", output)
#             print("Conv1dBlock output mean is: ", output.mean())
#             print("Conv1dBlock output var is: ", output.var())
#         elif name == "ConditionalResidualBlock1D":
#             print("ConditionalResidualBlock1D dx is: ", input)
#             print("ConditionalResidualBlock1D dy is: ", output)
#             print("ConditionalResidualBlock1D output mean is: ", output.mean())
#             print("ConditionalResidualBlock1D output var is: ", output.var())
#         elif name == "Downsample1d":
#             print("Downsample1d dx is: ", input)
#             print("Downsample1d dy is: ", output)
#             print("Downsample1d output mean is: ", output.mean())
#             print("Downsample1d output var is: ", output.var())
#         elif name == "Upsample1d":
#             print("Upsample1d dx is: ", input)
#             print("Upsample1d dy is: ", output)
#             print("Upsample1d output mean is: ", output.mean())
#             print("Upsample1d output var is: ", output.var())
#     return backward_hook

import torch

class HookManager:
    def __init__(self):
        self.handles = []

    def forward_hook(self, name):
        def hook(module, input, output):
            input = input[0]  # input是tuple，取第一个
            print(f"[{name}] Forward Hook - input mean: {input.mean():.6f}, var: {input.var():.6f}")
        return hook

    def backward_hook(self, name):
        def hook(module, grad_input, grad_output):
            grad_input = grad_input[0]  # grad_input是tuple，取第一个
            grad_output = grad_output[0]  # grad_output是tuple
            print(f"[{name}] Backward Hook - grad_output mean: {grad_output.mean():.6f}, var: {grad_output.var():.6f}")
        return hook

    def register_hooks(self, module, name, forward=True, backward=True):
        """
        - module: 需要挂hook的nn.Module
        - name: 模块的名字（打印时用）
        - forward: 是否注册forward hook
        - backward: 是否注册backward hook
        """
        if forward:
            handle_fwd = module.register_forward_hook(self.forward_hook(name))
            self.handles.append(handle_fwd)
        if backward:
            handle_bwd = module.register_full_backward_hook(self.backward_hook(name))
            self.handles.append(handle_bwd)

    def clear(self):
        """清除所有注册的hook"""
        for handle in self.handles:
            handle.remove()
        self.handles = []
        print("All hooks removed.")