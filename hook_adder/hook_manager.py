import torch

class HookManager:
    def __init__(self):
        self.handles = []

    def forward_hook(self, name):
        def hook(module, input, output):
            input = input[0]  # input是tuple，取第一个
            output = output[0]  # output是tuple
            print(f"[{name}] Forward Hook - input mean: {input.float().mean():.6f}, var: {input.float().var():.6f}, min: {input.float().min():.6f}, max: {input.float().max():.6f}")
            print(f"[{name}] Forward Hook - output mean: {output.float().mean():.6f}, var: {output.float().var():.6f}, min: {output.float().min():.6f}, max: {output.float().max():.6f}")
        return hook

    def backward_hook(self, name):
        def hook(module, grad_input, grad_output):
            grad_input = grad_input[0]  # grad_input是tuple，取第一个
            grad_output = grad_output[0]  # grad_output是tuple
            print(f"[{name}] Backward Hook - grad_output mean: {grad_output.float().mean():.6f}, var: {grad_output.float().var():.6f}, min: {grad_output.float().min():.6f}, max: {grad_output.float().max():.6f}")
            print(f"[{name}] Backward Hook - grad_input mean: {grad_input.float().mean():.6f}, var: {grad_input.float().var():.6f}, min: {grad_input.float().min():.6f}, max: {grad_input.float().max():.6f}")
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