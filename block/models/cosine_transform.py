import torch


def dct(coefs, coords=None):
    '''
    coefs: [..., C] # C: n_coefs
    coords: [..., S] # S: n_samples
    '''
    if coords is None:
        coords = torch.ones_like(coefs) \
               * torch.arange(coefs.size(-1)).to(coefs.device) \
               / coefs.size(-1)
    cos = torch.cos(torch.pi * coords.unsqueeze(-1)
                    * (torch.arange(coefs.size(-1)).to(coefs.device) + 0.5))
    return 2 * torch.einsum('...C,...SC->...S', coefs, cos)


def dctn(coefs, axes=None):
    if axes is None:
        axes = tuple(range(len(coefs.shape)))
    out = coefs
    for ax in axes:
        out = out.transpose(-1, ax)
        out = dct(out)
        out = out.transpose(-1, ax)
    return out


def idctn(coefs, axes=None, **kwargs):
    if axes is None:
        axes = tuple(range(len(coefs.shape)))
    out = coefs
    for ax in axes:
        out = out.transpose(-1, ax)
        out = idct(out, **kwargs)
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
    out = torch.cos(torch.pi / N * (torch.arange(N).to(coefs.device) + 0.5)
                    * (torch.linspace(0, N-1, n_out).unsqueeze(-1).to(coefs.device) + 0.5))
    # CCT version
    # out = torch.cos(torch.pi / N * (torch.arange(N).to(coefs.device))
    #                 * (torch.linspace(0, N-1, n_out).unsqueeze(-1).to(coefs.device)))


    # print(out.shape, coefs.shape, N, n_out)
    # print(torch.linspace(0, N-1, n_out).unsqueeze(-1).to(coefs.device).shape)
    # print(torch.linspace(0, N-1, n_out).unsqueeze(-1).to(coefs.device).flatten())
    # print(torch.einsum('...C,...SC->...S', coefs, out).shape)
    return 2 * torch.einsum('...C,...SC->...S', coefs, out)


if __name__ == '__main__':
    from scipy.fftpack import dct as org_dct
    from scipy.fftpack import dctn as org_dctn
    from scipy.fftpack import idct as org_idct
    from scipy.fftpack import idctn as org_idctn

    arr = torch.randn((2,3,4,5,6))

    # org_dctn(arr.numpy(), type=4, axes=(1,2)).flatten()
    a = org_idctn(arr.numpy(), type=4, axes=(2,4)).flatten()
    b = idctn(arr, axes=(2,4)).flatten()
    print(max(abs(a-b.numpy())))
    # print(idct(arr, 15).numpy())
    # print(org_idct(arr.numpy(), 4) - idct(arr).numpy())

    '''
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

