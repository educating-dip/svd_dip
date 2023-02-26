import torch
import torch.nn as nn
import numpy as np
import tensorly as tl


class SVD_Conv2d(nn.Module):
    def __init__(self, old_weight_data, in_channels, out_channels, kernel_size,
                 stride, padding, dilation, groups, bias=True,
                 padding_mode='zeros', device=None, dtype=None, rank_frac=1, adaptive_threshold=-1):
        super(SVD_Conv2d, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}

        matrix = old_weight_data.reshape(old_weight_data.shape[0], -1)
        

        if adaptive_threshold<1 and adaptive_threshold>0:
            U, S, V = tl.truncated_svd(matrix.to('cpu').numpy(), max(int(min(matrix.shape)), 1)) 
            max_val = S[0]
            new_rank =  S.size - np.searchsorted(np.flip(S), max_val*adaptive_threshold, 'right')
        elif adaptive_threshold == 0:
            new_rank = max(int(min(matrix.shape)), 1)   
        else:
            new_rank = max(int(min(matrix.shape)*rank_frac), 1)  # rank frac is only used if adaptive threshold is outside bounds
        U, S, V = tl.truncated_svd(matrix.to('cpu').numpy(), new_rank) 
        U=torch.from_numpy(U.copy()).to(device)
        S=torch.from_numpy(S.copy()).to(device)
        V=torch.from_numpy(V.copy()).to(device)
        rank = S.shape[0]

        self.U_data = U.reshape(old_weight_data.shape[0], rank, 1, 1)
        self.vector_S = nn.Parameter(torch.empty((1, rank, 1, 1), **factory_kwargs))
        self.vector_S.data[0,:,0,0] = S
        self.V_data = V.reshape(rank, old_weight_data.shape[1],old_weight_data.shape[2],old_weight_data.shape[3])

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

    def forward(self, x):
        x = torch.nn.functional.conv2d(x, self.V_data, None , self.stride, self.padding, self.dilation, self.groups)
        x = x.mul(self.vector_S)
        x = torch.nn.functional.conv2d(x, self.U_data, self.bias, (1, 1), 0, (1, 1), 1) 
        return x


def get_svd_layer(module, rank_frac = 1, adaptive_threshold = -1):
    new_module = SVD_Conv2d(module.weight.data, module.in_channels, module.out_channels, module.kernel_size, module.stride, module.padding, module.dilation, module.groups, bias=module.bias, padding_mode=module.padding_mode, device = module.weight.device, dtype = module.weight.dtype, rank_frac = rank_frac, adaptive_threshold = adaptive_threshold)
    return new_module
