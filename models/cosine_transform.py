import torch


def dct(coefs, coords=None):
    '''
    coefs: [..., C] # C: n_coefs
    coords: [..., S] # S: n_samples
    '''
    if coords is None:
        coords = torch.ones_like(coefs) \
               * torch.arange(coefs.size(-1)).to(coefs.device) # \
               # / coefs.size(-1)
    # cos = torch.cos(torch.pi * coords.unsqueeze(-1)
    cos = torch.cos(torch.pi * (coords.unsqueeze(-1) + 0.5) / coefs.size(-1)
                    * (torch.arange(coefs.size(-1)).to(coefs.device) + 0.5))
    # cos = torch.cos(torch.pi * (coords.unsqueeze(-1) + 0.) / coefs.size(-1)
    #                 * (torch.arange(coefs.size(-1)).to(coefs.device) + 0.))
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
    '''
    # TYPE II
    out = torch.cos(torch.pi * (torch.arange(N).unsqueeze(-1) + 0.5)
                    * torch.arange(1, N) / N)
    out = 2 * torch.einsum('...C,...SC->...S', coefs[..., 1:], out)
    return out + coefs[..., :1]
    '''
    # TYPE IV
    out = torch.cos(torch.pi * (torch.arange(N).to(coefs.device) + 0.5) / N
                    * (torch.linspace(0, N-1, n_out).unsqueeze(-1).to(coefs.device) + 0.5))
    # CCT version
    # out = torch.cos(torch.pi / N * (torch.arange(N).to(coefs.device))
    #                 * (torch.linspace(0, N-1, n_out).unsqueeze(-1).to(coefs.device)))
    # return 2 * torch.einsum('...C,...SC->...S', coefs, out)
    return torch.einsum('...C,...SC->...S', coefs*(2/N)**0.5, out)


if __name__ == '__main__':
    from scipy.fftpack import dct as org_dct
    from scipy.fftpack import dctn as org_dctn
    from scipy.fftpack import idct as org_idct
    from scipy.fftpack import idctn as org_idctn

    arr = torch.randn((1, 8, 240, 250)) * 10
    print((arr - dctn(idctn(arr, (-2, -1)), (-2, -1))).abs().max())
    print((arr - idctn(dctn(arr, (-2, -1)), (-2, -1))).abs().max())
    print((arr - dctn(idctn(arr, (-2,)), (-2,))).abs().max())
    print((arr - idctn(dctn(arr, (-2,)), (-2,))).abs().max())
    print((arr - idctn(dctn(arr, (-2,)), (-2,))).abs().max())
    print((org_idctn(arr.numpy(), 4, axes=(-2, -1), norm='ortho')
          - idctn(arr, (-2, -1)).numpy()).max())
    '''
    arr = torch.randn((3, 8))

    print(arr) # org_idct(arr.numpy(), 4))
    print(dct(idct(arr)))
    print(idct(dct(arr)))
    print(idct(arr).numpy())
    print(org_idctn(arr.numpy(), 4, axes=(-2, -1), norm='ortho') - idctn(arr).numpy())

    print(arr)
    print(org_dct(arr.numpy()))
    print(org_dct(arr.numpy()) - dct(arr, torch.arange(8) / 8).numpy())
    print()
    print(org_dct(org_dct(arr.numpy()), type=3))
    print(org_dct(org_dct(arr.numpy()), type=3)
          - idct(dct(arr, torch.arange(8) / 8)).numpy())
    '''

    # print(idct(dct(arr, torch.arange(16) / 16)) / torch.sqrt(torch.tensor(16)))
    # print(idct(dct(arr)))
    # print(org_dct(arr.numpy()) - dct(arr).numpy())

    # ndarr = torch.randn((3, 2, 4, 5))
    # axes = (3, ) # (1, 2, 3)
    # print(org_dctn(ndarr.numpy(), axes=axes) - dctn(ndarr, axes=axes).numpy())

