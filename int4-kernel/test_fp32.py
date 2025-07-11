import torch
import int4_kernel

if __name__ == '__main__':
    torch.set_printoptions(edgeitems=8, linewidth=100)
    x = torch.randn((2, 512, 1024), dtype=torch.float32, device='cuda')
    w = torch.randn((2048, 1024), dtype=torch.float32, device='cuda')

    # comparison
    y = (x @ w.T)

    # using kernel
    yk = int4_kernel.matmul_fp32(x, w)

    print('y:', y)
    print('yk:', yk)
    print('diff:', y - yk)