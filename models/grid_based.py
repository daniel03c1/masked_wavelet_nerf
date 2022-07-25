import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import models.cosine_transform as ct


class FreqGrid(nn.Module):
    def __init__(self, resolution: int, n_chan: int, n_freq=None,
                 freq_resolution=None):
        # assume 3 axes have the same resolution
        super().__init__()
        self.resolution = resolution
        self.n_chan = n_chan
        if freq_resolution is None:
            freq_resolution = resolution
        self.freq_resolution = freq_resolution

        if n_freq is None:
            n_freq = int(np.ceil(np.log2(freq_resolution)))
        self.n_freq = n_freq

        self.freqs = nn.Parameter(torch.linspace(0., 1, self.n_freq),
                                  requires_grad=False)

        self.grid = nn.Parameter(nn.Parameter(
            torch.zeros(3, n_chan*self.n_freq, resolution, resolution),
            requires_grad=True))

    def forward(self, coords):
        # [B, 3] to [1, B, 1, 3]
        coords = coords.reshape(1, -1, 1, coords.shape[-1])

        # coefs: [3, 1, C, B]
        grid = self.grid
        coefs = F.grid_sample(grid, torch.cat([coords[..., (1, 2)],
                                               coords[..., (0, 2)],
                                               coords[..., (0, 1)]], 0),
                              mode='bilinear',
                              padding_mode='zeros', align_corners=True)
        coefs = coefs.squeeze(-1).permute(2, 1, 0) # [B, C*F, 3]
        coefs = coefs.reshape(coefs.size(0), self.n_chan, -1, 3) # [B, C, F, 3]

        # numerical integration
        coords = coords.squeeze(0) # [B, 1, 3]

        '''
        # POS ENCODING
        outputs = torch.stack(
            [torch.cos(torch.pi * coords * self.get_freqs().unsqueeze(-1)),
             torch.sin(torch.pi * coords * self.get_freqs().unsqueeze(-1))], 1)
        outputs = 2 * (coefs * outputs.repeat(1, self.n_chan//2, 1, 1))
        '''

        coords = (coords + 1) / 2 * (self.resolution - 1)
        outputs = torch.cos(torch.pi / self.resolution * coords
                            * self.get_freqs().unsqueeze(-1))
        outputs = 2 * (coefs * outputs.unsqueeze(-3)) # [B, C, F, 3]
        return outputs.reshape(outputs.shape[0], -1)

    def compute_tv(self):
        weight = self.get_freqs().repeat(self.n_chan).reshape(-1, 1, 1)
        return (self.grid * weight).square().mean()

    def get_freqs(self):
        return -1 + 2**(self.freqs.clamp(min=0, max=1)
                        * np.log2(self.freq_resolution))


class VQ(nn.Module):
    def __init__(self, resolution: int, n_chan: int, n_freq=None,
                 freq_resolution=None, bitwidth=4, grid_num=1, channel_wise=True):
        # assume 3 axes have the same resolution
        super().__init__()
        self.resolution = resolution
        self.n_chan = n_chan
        if freq_resolution is None:
            freq_resolution = resolution
        self.freq_resolution = freq_resolution

        if n_freq is None:
            n_freq = int(np.ceil(np.log2(freq_resolution)))
        self.n_freq = n_freq

        self.freqs = nn.Parameter(torch.linspace(0., 1, self.n_freq),
                                  requires_grad=False)

        # Assume that each channel has its own codebook

        self.bitwidth = bitwidth
        self.channel_wise = channel_wise
        if channel_wise:
            self.codebook = nn.Parameter(torch.normal(0, 0.1, size=(3, 2**bitwidth, self.n_chan, self.n_freq)), requires_grad=True)
            self.indices = nn.Parameter(torch.zeros(3, self.n_chan, resolution * resolution, 2**bitwidth), requires_grad=True) 
        else :
            self.codebook = nn.Parameter(torch.normal(0, 0.1, size=(3, 2**bitwidth, self.n_chan * self.n_freq)), requires_grad=True)
            self.indices = nn.Parameter(torch.zeros(3, resolution * resolution, 2**bitwidth), requires_grad=True)


    def forward(self, coords):
        # [B, 3] to [1, B, 1, 3]
        coords = coords.reshape(1, -1, 1, coords.shape[-1]) 

        # coefs: [3, 1, C, B]
        grid = self.get_grid()
        coefs = F.grid_sample(grid, torch.cat([coords[..., (1, 2)],
                                               coords[..., (0, 2)],
                                               coords[..., (0, 1)]], 0),
                              mode='bilinear',
                              padding_mode='zeros', align_corners=True)
        coefs = coefs.squeeze(-1).permute(2, 1, 0) # [B, C*F, 3]
        coefs = coefs.reshape(coefs.size(0), self.n_chan, -1, 3) # [B, C, F, 3]

        # numerical integration
        coords = coords.squeeze(0) # [B, 1, 3]

        '''
        # POS ENCODING
        outputs = torch.stack(
            [torch.cos(torch.pi * coords * self.get_freqs().unsqueeze(-1)),
             torch.sin(torch.pi * coords * self.get_freqs().unsqueeze(-1))], 1)

        outputs = 2 * (coefs * outputs.repeat(1, self.n_chan//2, 1, 1))
        '''

        coords = (coords + 1) / 2 * (self.resolution - 1)
        outputs = torch.cos(torch.pi / self.resolution * coords
                            * self.get_freqs().unsqueeze(-1))
        outputs = 2 * (coefs * outputs.unsqueeze(-3)) # [B, C, F, 3]
        return outputs.reshape(outputs.shape[0], -1)

    def compute_tv(self):
        weight = self.get_freqs().repeat(self.n_chan).reshape(-1, 1, 1)
        return (self.get_grid() * weight).square().mean()

    def get_freqs(self):
        return -1 + 2**(self.freqs.clamp(min=0, max=1)
                        * np.log2(self.freq_resolution))

    def get_grid(self):
        if self.channel_wise:
            softened_mat = torch.softmax(self.indices, dim=3)
            grid_chan = []
            for i in range(self.n_chan):
                grid_chan.append(softened_mat[:,i,...] @ self.codebook[...,i,:])
            grid = torch.stack(grid_chan, dim=1)
            
        else:
            softened_mat = torch.softmax(self.indices, dim=2)
            grid = softened_mat @ self.codebook
        return grid.view(3, -1, self.resolution, self.resolution)


class PREF(nn.Module):
    def __init__(self, res, ch):
        """
        INPUTS
            res: resolution
            ch: channel
        """
        super().__init__()
        if not hasattr(res, 'len'):
            res = np.array([res, res, res])
        reduced_res = (np.ceil(np.log2(res))).astype('int')
        self.res = res
        self.ch = ch
        self.reduced_res = reduced_res
        self.mask = 128
 
        self.phasor = nn.ParameterList([
        nn.Parameter(torch.zeros((1, 2*reduced_res[0]*ch, res[1], res[2]),
                                 dtype=torch.float32), requires_grad=True),
        nn.Parameter(torch.zeros((1, 2*reduced_res[1]*ch, res[0], res[2]),
                                 dtype=torch.float32), requires_grad=True),
        nn.Parameter(torch.zeros((1, 2*reduced_res[2]*ch, res[0], res[1]),
                                 dtype=torch.float32), requires_grad=True)]) 

        '''
        self.phasor = nn.ParameterList([
            nn.Parameter(torch.zeros((1, ch*reduced_res[0], res[1], res[2], 2),
                                     dtype=torch.float32),
                               requires_grad=True),
            nn.Parameter(torch.zeros((1, ch*reduced_res[1], res[0], res[2], 2),
                                     dtype=torch.float32),
                               requires_grad=True),
            nn.Parameter(torch.zeros((1, ch*reduced_res[2], res[0], res[1], 2),
                                     dtype=torch.float32),
                               requires_grad=True)])
        '''

    def forward(self, inputs):
        inputs = inputs.reshape(1, 1, *inputs.shape) # [B, 3] to [1, 1, B, 3]
        Pu, Pv, Pw = self.phasor

        # mask = torch.zeros((self.res[0], self.res[1], 1)).to(Pu)
        # mask[:self.mask, :self.mask] += 1
        # Pu = Pu * mask
        # Pv = Pv * mask
        # Pw = Pw * mask

        '''
        Pu = torch.fft.ifft2(torch.view_as_complex(Pu))
        Pu = torch.view_as_real(Pu).permute(0, 1, 4, 2, 3).reshape(1, -1, self.res[0], self.res[1])
        Pv = torch.fft.ifft2(torch.view_as_complex(Pv))
        Pv = torch.view_as_real(Pv).permute(0, 1, 4, 2, 3).reshape(1, -1, self.res[0], self.res[1])
        Pw = torch.fft.ifft2(torch.view_as_complex(Pw))
        Pw = torch.view_as_real(Pw).permute(0, 1, 4, 2, 3).reshape(1, -1, self.res[0], self.res[1])
        '''

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

        # inputs: [B, C, D]
        inputs = torch.stack(torch.split(inputs, self.ch, dim=1), -1)
        inputs = torch.view_as_complex(inputs)
        coords = (coords + 1) / 2 * (2**N - 1)
        coef = (2**torch.arange(N) - 1).to(inputs)
        out = torch.exp(2j* torch.pi * coords.unsqueeze(-1) * coef / (2**N))
        out = torch.einsum('...C,...SC->...S', out, inputs)
        return out.real

    def compute_tv(self):
        '''
        freqs0 = 2 ** torch.arange(self.reduced_res[0]).to(self.phasor[0]) - 1
        freqs1 = torch.arange(self.res[0]).to(freqs0)
        weight = torch.stack(torch.meshgrid(freqs0, freqs1, freqs1), -1)
        weight = 2 * torch.pi * weight.square().sum(-1, keepdims=True).repeat(self.ch, 1, 1, 2)
        return (self.phasor[0].square() * weight).mean() \
             + (self.phasor[1].square() * weight).mean() \
             + (self.phasor[2].square() * weight).mean()
        '''
        weight = (2**torch.arange(self.reduced_res[0])-1).to(self.phasor[0].device).repeat(self.ch).reshape(-1, 1, 1, 1)
        return (self.phasor[0]*weight).square().mean() \
             + (self.phasor[1]*weight).square().mean() \
             + (self.phasor[2]*weight).square().mean()

