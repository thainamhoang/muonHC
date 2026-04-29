"""Frequency Component Kernel – Fixed DCT decomposition. 0 learnable parameters.
Taken from GeoFAR: https://github.com/eceo-epfl/GeoFAR
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def generate_local_dct_kernels(n, k):
    """Generate n*n local DCT basis kernels of size k x k.
    Args:
        n: number of basis functions per dimension (total n*n filters)
        k: kernel size (spatial support)
    Returns: Tensor of shape (n*n, 1, k, k)
    """
    kernels = []
    for u in range(n):
        cu = math.sqrt(1/k) if u == 0 else math.sqrt(2/k)
        for v in range(n):
            cv = math.sqrt(1/k) if v == 0 else math.sqrt(2/k)
            kernel = torch.zeros(k, k)
            for x in range(k):
                for y in range(k):
                    kernel[x, y] = cu * cv * math.cos(math.pi*(2*x+1)*u/(2*k)) * math.cos(math.pi*(2*y+1)*v/(2*k))
            kernels.append(kernel)
    kernels = torch.stack(kernels).unsqueeze(1)  # (n*n, 1, k, k)
    return kernels

class LocalDCTConv(nn.Module):
    """Apply fixed DCT filters to input, one basis per channel group.
    If in_channels > 1, each input channel is convolved with all n*n filters independently
    and the results are concatenated, so out_channels = in_channels * n*n.
    """
    def __init__(self, in_channels, n=8, k=8):
        super().__init__()
        self.in_channels = in_channels
        self.n = n
        self.k = k
        pad_total = k - 1
        self.pad_left = pad_total // 2
        self.pad_right = pad_total - self.pad_left
        self.pad_top = pad_total // 2
        self.pad_bottom = pad_total - self.pad_top
        n_filters = n * n
        kernels = generate_local_dct_kernels(n, k)  # (n_filters, 1, k, k)
        if in_channels == 1:
            self.conv = nn.Conv2d(1, n_filters, k, padding=0, bias=False)
            self.conv.weight.data = kernels.clone()
        else:
            self.conv = nn.Conv2d(in_channels, in_channels * n_filters, k,
                                  padding=0, groups=in_channels, bias=False)
            # Replicate kernels for each group
            weight = kernels.repeat(in_channels, 1, 1, 1)  # (in*n_filters, 1, k, k)
            self.conv.weight.data = weight.clone()
        self.conv.weight.requires_grad = False

    def forward(self, x):
        x = F.pad(x, (self.pad_left, self.pad_right, self.pad_top, self.pad_bottom))
        return self.conv(x)
