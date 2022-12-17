import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import models.cosine_transform as ct


class PREF(nn.Module):
    def __init__(self, res, ch):
        """
        INPUTS
            res: resolution
            ch: channel
        """
        super(PREF, self).__init__()
        reduced_res = np.ceil(np.log2(res)+1).astype('int')
        self.res = res
        self.ch = ch
        self.reduced_res = reduced_res

        self.phasor = nn.ParameterList([
            # nn.Parameter(0.*torch.randn((1, reduced_res[0]*ch, res[1], res[2]),
            nn.Parameter(0.*torch.randn((1, reduced_res[0]*ch, res[1], res[2]),
                                     dtype=torch.float32),
                               requires_grad=True),
            nn.Parameter(0.*torch.randn((1, reduced_res[1]*ch, res[0], res[2]),
                                     dtype=torch.float32),
                               requires_grad=True),
            nn.Parameter(0.*torch.randn((1, reduced_res[2]*ch, res[0], res[1]),
                                     dtype=torch.float32),
                               requires_grad=True)])

    def forward(self, inputs):
        inputs = inputs.reshape(1, 1, *inputs.shape) # [B, 3] to [1, 1, B, 3]
        Pu = self.phasor[0]
        Pv = self.phasor[1]
        Pw = self.phasor[2]

        Pu = F.grid_sample(Pu, inputs[..., (1, 2)], mode='bilinear',
                           align_corners=True)
        Pu = Pu.transpose(1, 3).reshape(-1, self.ch, self.reduced_res[0])
        Pv = F.grid_sample(Pv, inputs[..., (0, 2)], mode='bilinear',
                           align_corners=True)
        Pv = Pv.transpose(1, 3).reshape(-1, self.ch, self.reduced_res[1])
        Pw = F.grid_sample(Pw, inputs[..., (0, 1)], mode='bilinear',
                           align_corners=True)
        Pw = Pw.transpose(1, 3).reshape(-1, self.ch, self.reduced_res[2])

        Pu = self.numerical_integration(Pu, inputs[0, 0, ..., 0])
        Pv = self.numerical_integration(Pv, inputs[0, 0, ..., 1])
        Pw = self.numerical_integration(Pw, inputs[0, 0, ..., 2])

        outputs = Pu + Pv + Pw
        return outputs

    def numerical_integration(self, inputs, coords):
        # assume coords in [-1, 1]
        N = self.reduced_res[0] # inputs.size(-1)
        coords = (coords + 1) / 2 * ((2**(N-1)) - 1)

        '''
        out = torch.cos(torch.pi * (coords.unsqueeze(-1) + 0.5)
                        * (2 ** torch.arange(N-1).to(coords.device)) / (2**N))
        out = 2 * torch.einsum('...C,...SC->...S', out, inputs[..., 1:])
        return out + inputs[..., 0]
        '''
        out = torch.cos(torch.pi * (coords.unsqueeze(-1) + 0.5)
                        * (2 ** torch.arange(N).to(coords.device)-0.5) / (2**N))
        out = 2 * torch.einsum('...C,...SC->...S', out, inputs)
        return out

    def compute_tv(self):
        weight = (2 ** torch.arange(self.reduced_res[0]).to(self.phasor[0].device) - 1).repeat(self.ch).reshape(-1, 1, 1)
        return (self.phasor[0]*weight).square().mean() \
             + (self.phasor[1]*weight).square().mean() \
             + (self.phasor[2]*weight).square().mean()


class PREFFFT(nn.Module):
    def __init__(self, res, ch):
        """
        INPUTS
            res: resolution
            ch: channel
        """
        super(PREFFFT, self).__init__()
        # reduced_res = (np.ceil(np.log2(res)) + 1).astype('int')
        reduced_res = (np.ceil(np.log2(res)) + 0).astype('int')
        self.res = res
        self.ch = ch
        self.reduced_res = reduced_res

        self.phasor = nn.ParameterList([
            nn.Parameter(0.001*torch.randn((1, 2*reduced_res[0]*ch, res[1], res[2]),
                                     dtype=torch.float32),
                               requires_grad=True),
            nn.Parameter(0.001*torch.randn((1, 2*reduced_res[1]*ch, res[0], res[2]),
                                     dtype=torch.float32),
                               requires_grad=True),
            nn.Parameter(0.001*torch.randn((1, 2*reduced_res[2]*ch, res[0], res[1]),
                                     dtype=torch.float32),
                               requires_grad=True)])

    def forward(self, inputs):
        inputs = inputs.reshape(1, 1, *inputs.shape) # [B, 3] to [1, 1, B, 3]
        Pu = self.phasor[0]
        Pv = self.phasor[1]
        Pw = self.phasor[2]

        Pu = F.grid_sample(Pu, inputs[..., (1, 2)], mode='bilinear',
                           align_corners=True)
        Pu = Pu.transpose(1, 3).reshape(-1, 2*self.ch, self.reduced_res[0])
        Pv = F.grid_sample(Pv, inputs[..., (0, 2)], mode='bilinear',
                           align_corners=True)
        Pv = Pv.transpose(1, 3).reshape(-1, 2*self.ch, self.reduced_res[1])
        Pw = F.grid_sample(Pw, inputs[..., (0, 1)], mode='bilinear',
                           align_corners=True)
        Pw = Pw.transpose(1, 3).reshape(-1, 2*self.ch, self.reduced_res[2])

        Pu = self.numerical_integration(Pu, inputs[0, 0, ..., 0])
        Pv = self.numerical_integration(Pv, inputs[0, 0, ..., 1])
        Pw = self.numerical_integration(Pw, inputs[0, 0, ..., 2])

        outputs = Pu + Pv + Pw
        return outputs

    def numerical_integration(self, inputs, coords):
        # assume coords in [-1, 1]
        N = inputs.size(-1)
        '''
        coords = (coords + 1) / 2 * ((2**(N-1)) - 1)

        out = torch.cos(torch.pi * (coords.unsqueeze(-1) + 0.5)
                        * (2 ** torch.arange(N-1).to(coords.device)) / (2**N))
        out = 2 * torch.einsum('...C,...SC->...S', out, inputs[..., 1:])
        return out + inputs[..., 0]
        '''
        # inputs: [B, C, D]
        inputs = torch.stack(torch.split(inputs, self.ch, dim=1), -1)
        inputs = torch.view_as_complex(inputs)
        coords = (coords + 1) / 2 * (2**N - 1)
        coef = torch.cat([torch.zeros((1,)), 2**torch.arange(N-1)]).to(inputs.device)
        out = torch.exp(2j* torch.pi * coords.unsqueeze(-1) * coef / (2**N))
        out = torch.einsum('...C,...SC->...S', out, inputs)
        return out.real

