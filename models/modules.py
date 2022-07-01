import torch


def get_module(shadingMode, in_dim, pos_pe, view_pe, fea_pe, featureC):
    if shadingMode == 'MLP_PE':
        return MLPRender_PE(in_dim, view_pe, pos_pe, featureC)
    elif shadingMode == 'MLP_Fea':
        return MLPRender_Fea(in_dim, view_pe, fea_pe, featureC)
    elif shadingMode == 'MLP':
        return MLPRender(in_dim, view_pe, featureC)
    elif shadingMode == 'SH':
        return SHRender
    elif shadingMode == 'RGB':
        assert in_dim == 3
        return RGBRender
    else:
        raise ValueError(f"Unrecognized shading module: {shadingMode}")


# utils
def positional_encoding(positions, freqs):
    freq_bands = (2**torch.arange(freqs).float()).to(positions.device) # (F,)
    pts = (positions[..., None] * freq_bands).reshape(
        positions.shape[:-1] + (freqs * positions.shape[-1], )) # (..., DF)
    pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
    return pts


# modules
class MLPRender_Fea(torch.nn.Module):
    def __init__(self, inChanel, viewpe=6, feape=6, featureC=128):
        super(MLPRender_Fea, self).__init__()
        '''
        inChannel: the dimension size of incomming features
        viewpe: the degree of positional encoding (view direction)
        feape: the degree of positional encoding (features)
        featureC: hidden dimension size
        '''
        self.in_mlpC = (2 * viewpe + 1) * 3 + (2 * feape + 1) * inChanel
        self.viewpe = viewpe
        self.feape = feape

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.in_mlpC, featureC),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(featureC, featureC),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(featureC, 3))
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features):
        indata = [features, viewdirs]
        if self.feape > 0:
            indata += [positional_encoding(features, self.feape)]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        return torch.sigmoid(rgb)


class MLPRender_PE(torch.nn.Module):
    def __init__(self,inChanel, viewpe=6, pospe=6, featureC=128):
        super(MLPRender_PE, self).__init__()

        self.in_mlpC = (3+2*viewpe*3)+ (3+2*pospe*3)  + inChanel #
        self.viewpe = viewpe
        self.pospe = pospe
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC,3)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features):
        indata = [features, viewdirs]
        if self.pospe > 0:
            indata += [positional_encoding(pts, self.pospe)]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb


class MLPRender(torch.nn.Module):
    def __init__(self,inChanel, viewpe=6, featureC=128):
        super(MLPRender, self).__init__()

        self.in_mlpC = (3+2*viewpe*3) + inChanel
        self.viewpe = viewpe
        
        layer1 = torch.nn.Linear(self.in_mlpC, featureC)
        layer2 = torch.nn.Linear(featureC, featureC)
        layer3 = torch.nn.Linear(featureC,3)

        self.mlp = torch.nn.Sequential(layer1, torch.nn.ReLU(inplace=True), layer2, torch.nn.ReLU(inplace=True), layer3)
        torch.nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features):
        indata = [features, viewdirs]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb


def SHRender(xyz_sampled, viewdirs, features):
    sh_mult = eval_sh_bases(2, viewdirs)[:, None]
    rgb_sh = features.view(-1, 3, sh_mult.shape[-1])
    rgb = torch.relu(torch.sum(sh_mult * rgb_sh, dim=-1) + 0.5)
    return rgb


def RGBRender(xyz_sampled, viewdirs, features):
    return features

