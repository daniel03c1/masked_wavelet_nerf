import torch
import torch.nn as nn


def get_module(shadingMode, in_dim, pos_pe, view_pe, fea_pe, hidden_dim):
    if shadingMode == 'MLP_PE':
        return MLP(in_dim, include_pos=True, include_view=True,
                   pos_n_freq=pos_pe, view_n_freq=view_pe,
                   hidden_dim=hidden_dim)
    elif shadingMode == 'MLP_Fea':
        return MLP(in_dim, include_view=True, feat_n_freq=fea_pe,
                   view_n_freq=view_pe, hidden_dim=hidden_dim)
    elif shadingMode == 'MLP':
        return MLP(in_dim, include_view=True, view_n_freq=view_pe,
                   hidden_dim=hidden_dim)
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
    return torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)


# modules
class MLP(nn.Module):
    def __init__(self, feat_dim,
                 include_feat=True, include_pos=False, include_view=False,
                 feat_n_freq=0, pos_n_freq=0, view_n_freq=0, hidden_dim=128):
        super().__init__()

        self.include_feat = include_feat
        self.include_pos = include_pos
        self.include_view = include_view
        self.feat_n_freq = feat_n_freq
        self.pos_n_freq = pos_n_freq
        self.view_n_freq = view_n_freq

        in_size = feat_dim * (include_feat + 2 * feat_n_freq) \
                + 3 * (include_pos + 2 * pos_n_freq) \
                + 3 * (include_view + 2 * view_n_freq)

        self.mlp = nn.Sequential(
            nn.Linear(in_size, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3),
            nn.Sigmoid())
        nn.init.constant_(self.mlp[-2].bias, 0)

    def forward(self, pts, viewdirs, features):
        inputs = []
        if self.include_feat:
            inputs.append(features)
        if self.include_pos:
            inputs.append(pts)
        if self.include_view:
            inputs.append(viewdirs)

        if self.feat_n_freq > 0:
            inputs.append(positional_encoding(features, self.feat_n_freq))
        if self.pos_n_freq > 0:
            inputs.append(positional_encoding(pts, self.pos_n_freq))
        if self.view_n_freq > 0:
            inputs.append(positional_encoding(viewdirs, self.view_n_freq))

        return self.mlp(torch.cat(inputs, -1))


def SHRender(xyz_sampled, viewdirs, features):
    sh_mult = eval_sh_bases(2, viewdirs)[:, None]
    rgb_sh = features.view(-1, 3, sh_mult.shape[-1])
    return torch.relu(torch.sum(sh_mult * rgb_sh, dim=-1) + 0.5)


def RGBRender(xyz_sampled, viewdirs, features):
    return features

