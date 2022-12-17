import torch
from pytorch_wavelets import DWTInverse, DWTForward, DWT1DInverse, DWT1DForward


# utils
def split2d(inputs, level=4):
    assert inputs.size(-1) % 2**level == 0
    assert inputs.size(-2) % 2**level == 0

    res0, res1 = inputs.shape[-2:]

    yl = inputs[..., :res0//(2**level), :res1//(2**level)]
    yh = [
        torch.stack([inputs[..., :res0//(2**(i+1)),
                                 res1//(2**(i+1)):res1//(2**i)],
                     inputs[..., res0//(2**(i+1)):res0//(2**i),
                                 :res1//(2**(i+1))],
                     inputs[..., res0//(2**(i+1)):res0//(2**i),
                                 res1//(2**(i+1)):res1//(2**i)]], 2)/(level-i+1)
        for i in range(level)
    ]

    return yl, yh


def split1d(inputs, level=4):
    assert inputs.size(-1) % 2**level == 0

    res = inputs.shape[-1]

    yl = inputs[..., :res//(2**level)]
    yh = [inputs[..., res//(2**(i+1)):res//(2**i)] # / (level-i+1)
          for i in range(level)]

    return yl, yh


# inverse and forward
def inverse(inputs, level=4, trans_func='bior4.4'):
    if trans_func == 'cosine':
        return idctn(inputs, (-2, -1))
    return DWTInverse(wave=trans_func, mode='periodization').to(inputs.device)\
            (split2d(inputs, level))


def forward(inputs, level=4, trans_func='bior4.4'):
    assert inputs.size(-1) % 2**level == 0
    assert inputs.size(-2) % 2**level == 0

    if trans_func == 'cosine':
        return dctn(inputs, (-2, -1))

    yl, yh = DWTForward(wave=trans_func, J=level,
                        mode='periodization').to(inputs.device)(inputs)
    outs = yl

    for i in range(level):
        cf = yh[-i-1] * (i+ 2)
        outs = torch.cat([torch.cat([outs, cf[..., 0, :, :]], -1),
                          torch.cat([cf[..., 1, :, :], cf[..., 2, :, :]], -1)],
                         -2)
    return outs


def inverse1d(inputs, level=4):
    return DWT1DInverse(wave='bior4.4', mode='periodization').to(inputs.device)\
            (split1d(inputs, level))


def forward1d(inputs, level=4):
    assert inputs.size(-1) % 2**level == 0

    yl, yh = DWT1DForward(wave='bior4.4', J=level,
                          mode='periodization').to(inputs.device)(inputs)
    outs = yl

    for i in range(level):
        cf = yh[-i-1] # * (i+2)
        outs = torch.cat([outs, cf], -1)
    return outs


def dct(coefs, coords=None):
    '''
    coefs: [..., C] # C: n_coefs
    coords: [..., S] # S: n_samples
    '''
    if coords is None:
        coords = torch.ones_like(coefs) \
               * torch.arange(coefs.size(-1)).to(coefs.device) # \
    cos = torch.cos(torch.pi * (coords.unsqueeze(-1) + 0.5) / coefs.size(-1)
                    * (torch.arange(coefs.size(-1)).to(coefs.device) + 0.5))
    return torch.einsum('...C,...SC->...S', coefs*(2/coefs.size(-1))**0.5, cos)


def dctn(coefs, axes=None):
    if axes is None:
        axes = tuple(range(len(coefs.shape)))
    out = coefs
    for ax in axes:
        out = out.transpose(-1, ax)
        out = dct(out)
        out = out.transpose(-1, ax)
    return out


def idctn(coefs, axes=None, n_out=None, **kwargs):
    if axes is None:
        axes = tuple(range(len(coefs.shape)))

    if n_out is None or isinstance(n_out, int):
        n_out = [n_out] * len(axes)

    out = coefs
    for ax, n_o in zip(axes, n_out):
        out = out.transpose(-1, ax)
        out = idct(out, n_o, **kwargs)
        out = out.transpose(-1, ax)
    return out


def idct(coefs, n_out=None):
    N = coefs.size(-1)
    if n_out is None:
        n_out = N
    # TYPE IV
    out = torch.cos(torch.pi * (torch.arange(N).to(coefs.device) + 0.5) / N
                    * (torch.linspace(0, N-1, n_out).unsqueeze(-1).to(coefs.device) + 0.5))
    return torch.einsum('...C,...SC->...S', coefs*(2/N)**0.5, out)


if __name__ == '__main__':
    a = torch.randn(3, 5, 64, 64).cuda() * 10
    print(a.shape, inverse(a).shape)
    print((a - forward(inverse(a))).abs().max())
    print((a - inverse(forward(a))).abs().max())

    a = torch.randn(3, 32, 64).cuda() * 5
    print(a.shape, inverse1d(a).shape)
    print((a - forward1d(inverse1d(a))).abs().std())
    print((a - inverse1d(forward1d(a))).abs().std())

