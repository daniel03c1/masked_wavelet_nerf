import torch
from pytorch_wavelets import DWTInverse, DWTForward


def inverse(inputs, level=4):
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

    return DWTInverse(wave='bior4.4',
                      mode='periodization').to(inputs.device)((yl, yh))


def forward(inputs, level=4):
    assert inputs.size(-1) % 2**level == 0
    assert inputs.size(-2) % 2**level == 0

    yl, yh = DWTForward(wave='bior4.4', J=level,
                        mode='periodization').to(inputs.device)(inputs)
    outs = yl

    for i in range(level):
        cf = yh[-i-1] * (i+2)
        outs = torch.cat([torch.cat([outs, cf[..., 0, :, :]], -1),
                          torch.cat([cf[..., 1, :, :], cf[..., 2, :, :]], -1)],
                         -2)
    return outs


if __name__ == '__main__':
    a = torch.randn(3, 5, 64, 80).cuda() * 10
    print(a.shape, inverse(a).shape)
    print((a - forward(inverse(a))).abs().max())
    print((a - inverse(forward(a))).abs().max())

