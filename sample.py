import gc
import os

import numpy as np
import torch
from torch import nn

from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D

from diffusion_policy.quantization.adaption import AdaRoundQuantizer
from diffusion_policy.quantization.layer_recon import layer_reconstruction
from diffusion_policy.quantization.quan_block import BaseQuantBlock
from diffusion_policy.quantization.quan_layer import (
    Activation_Quantizer,
    QuantModule,
    Weight_Quantizer,
)
from diffusion_policy.quantization.quan_model import QuantModel


class Sample:
    def __init__(self, device=None):
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

    def get_train_samples(
        self, cali_n, cali_st, sample_data, custom_steps=None, cond=True
    ):
        num_samples, num_st = cali_n, cali_st
        # num_samples = len(sample_data["xs"][0])
        print(f"num_samples: {num_samples}")
        print(type(sample_data))
        print(sample_data.keys())
        print(f"sample_data['xs']: {len(sample_data['xs'])}")
        print(f"sample_data['ts']: {len(sample_data['ts'])}")
        # print(f"sample_data['ls']: {len(sample_data['ls'])}")
        print(f"sample_data['cs']: {len(sample_data['cs'])}")

        if num_st == 1:
            xs = sample_data[:num_samples]
            ts = torch.ones(num_samples) * 800

        else:
            # get the real number of timesteps (especially for DDIM)
            nsteps = len(sample_data["ts"])
            assert nsteps >= custom_steps
            timesteps = list(range(0, nsteps, nsteps))
            xs = [sample_data["xs"][i][:num_samples] for i in timesteps]
            ts = [sample_data["ts"][i][:num_samples] for i in timesteps]
            conds = [sample_data["cs"][i][:num_samples] for i in timesteps]
            print(f"len(xs): {len(xs)}")
            print(f"len(ts): {len(ts)}")
            print(f"len(conds): {len(conds)}")
            print(f"len(xs[0]): {len(xs[0])}")
            print(f"len(ts[0]): {len(ts[0])}")
            print(f"len(conds[0]): {len(conds[0])}")
            flattened_list = [item for sublist in xs for item in sublist]
            xs = torch.stack(flattened_list)
            print(f"xs: {xs.shape}")
            flattened_list = [item for sublist in ts for item in sublist]
            ts = torch.stack(flattened_list)
            print(f"ts: {ts.shape}")
            if cond:
                flattened_list = [item for sublist in conds for item in sublist]
                conds = torch.stack(flattened_list)
                print(f"conds: {conds.shape}")
                return xs, ts, conds
        return xs, ts

    def sample(self):
        model: ConditionalUnet1D
        model = torch.load("/home/liyixuan23/Data-Scaling-Laws/10-step-model/fp32_model.pth")
        
        print("fp32 model is loaded")

        model.to(self.device)
        model.eval()
        print("quantized model is being creating")
        qnn = QuantModel(model)
        qnn.to(self.device)
        qnn.eval()
        print("quantized model has been created")

        cali_n = 20
        cali_st = 20
        cali_data: torch.Tensor
        sample_data = torch.load("/home/liyixuan23/Data-Scaling-Laws/17-step-model/sample_fp32_data.ckpt")
        print(len(sample_data["ts"]))
        print(sample_data["ts"])
        print("calibration data is loaded")
        cali_data = self.get_train_samples(cali_n, cali_st, sample_data, 0, True)
        del sample_data
        gc.collect()

        # cali_xs, cali_ts, cali_lcond, cali_cond = cali_data
        cali_xs, cali_ts, cali_cond = cali_data
        # qnn.set_quant_state(
        #     True, False
        # )  # enable weight quantization, disable act quantization
        qnn.set_quant_state(False, False)
        print("weight quantizer is being initialized")
        _ = qnn(
            cali_xs[0].to(self.device),
            cali_ts[0].to(self.device),
            # cali_lcond[0].to(self.device),
            cali_cond[0].to(self.device),
        )
        print("weight quantizer has been initialized")

        # Kwargs for weight rounding calibration
        kwargs = dict(
            cali_data=cali_data,
            batch_size=1,
            iters=20000,
            weight=0.01,
            asym=True,
            b_range=(20, 2),
            warmup=0.2,
            act_quant=False,
            opt_mode="mse",
            cond=True,
        )

        def recon_model(model):
            for name, module in model.named_children():
                if isinstance(module, QuantModule):
                    print(name, " is being reconstructed")
                    layer_reconstruction(qnn, module, **kwargs)
                    break
                elif isinstance(module, BaseQuantBlock):
                    # block_reconstruction(qnn, module, **kwargs)
                    pass
                else:
                    recon_model(module)

        # recon_model(qnn)
        qnn.set_quant_state(weight_quant=True, act_quant=False)

        # Initialize activation quantization parameters
        qnn.set_quant_state(True, True)
        with torch.no_grad():
            # inds = np.random.choice(cali_xs.shape[0], 64, replace=False)
            # _ = qnn(cali_xs[:64].cuda(), cali_ts[:64].cuda())
            # _ = qnn(cali_xs[0].cuda(), cali_ts[0].cuda(), cali_lcond[0].cuda(), cali_cond[0].cuda())
            _ = qnn(cali_xs[0].cuda(), cali_ts[0].cuda(), cali_cond[0].cuda())

            qnn.set_running_stat(True)
            print(int(cali_xs.size(0)))
            for i in range(int(cali_xs.size(0))):
                _ = qnn(
                    cali_xs[i].to(self.device),
                    cali_ts[i].to(self.device),
                    # cali_lcond[i].to(self.device),
                    cali_cond[i].to(self.device),
                )
                print("in range")
                print(cali_xs[i].abs().max(), cali_xs[i].abs().min())
            qnn.set_running_stat(False)

        # kwargs = dict(
        #     cali_data=cali_data, act_quant=True, lr=4e-6, opt_mode="mse", cond=True
        # )
        # recon_model(qnn)
        qnn.set_quant_state(weight_quant=True, act_quant=True)

        # for m in qnn.model.modules():
        #     if isinstance(m, (Activation_Quantizer, Weight_Quantizer)):
        #         m.delta = nn.Parameter(m.delta)
        torch.save(
            qnn.state_dict(),
            "/home/liyixuan23/Data-Scaling-Laws/17-step-model/quantized_model_2.ckpt"
        )
        torch.save(qnn, "/home/liyixuan23/Data-Scaling-Laws/17-step-model/quantized_model_2.pth")

        model = qnn
        return model


if __name__ == "__main__":
    runner = Sample()
    runner.sample()
