import torch.nn as nn


class DensityNet(nn.Module):
    def __init__(self, net: nn.Sequential):
        super().__init__()
        self.net = net

    def forward(self, *args, **kwargs):
        return self.net(*args, **kwargs)

    def compute_tv(self):
        # compute total variation
        return sum([m.compute_tv() if hasattr(m, 'compute_tv') else 0
                    for m in self.net.modules()])


class AppearanceNet(nn.Module):
    def __init__(self, first_net: nn.Sequential, second_net: nn.Sequential):
        """
        first_net: a network that preprocess coordinates to latent features
        second_net: a network that uses both latent features, coordinates,
                    and viewdirs
        """
        super().__init__()
        self.first_net = first_net
        self.second_net = second_net

    def forward(self, coords, viewdirs):
        latent_features = self.first_net(coords)
        return self.second_net(coords, viewdirs, latent_features)

    def compute_tv(self):
        # compute total variation
        tv = 0
        for net in [self.first_net, self.second_net]:
            if isinstance(net, nn.Sequential):
                tv += sum([m.compute_tv() if hasattr(m, 'compute_tv') else 0
                           for m in net.modules()])
            else: # assume it's nn.Module
                if hasattr(net, 'compute_tv'):
                    tv += net.compute_tv()
        return tv

