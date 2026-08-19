import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Sinc_conv(nn.Module):
    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, out_channels, kernel_size, sample_rate=16000, in_channels=1, min_low_hz=50, min_band_hz=50):
        super(Sinc_conv, self).__init__()

        if in_channels != 1:
            raise ValueError("Sinc_conv only supports in_channels=1")

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        if kernel_size % 2 == 0:
            self.kernel_size = kernel_size + 1

        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        low_hz = 30
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)

        mel = np.linspace(self.to_mel(low_hz), self.to_mel(high_hz), self.out_channels + 1)
        hz = self.to_hz(mel)

        self.low_hz_ = torch.Tensor(hz[:-1]).view(-1, 1)
        self.band_hz_ = torch.Tensor(np.diff(hz)).view(-1, 1)

        n_lin = torch.linspace(0, (self.kernel_size / 2) - 1, int(self.kernel_size / 2))
        self.window_ = 0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / self.kernel_size)

        n_ = (self.kernel_size - 1) / 2.0
        self.n_ = 2 * math.pi * torch.arange(-n_, n_ + 1).view(1, -1) / self.sample_rate

    def forward(self, waveforms):
        low_hz = self.low_hz_.to(waveforms.device)
        band_hz = self.band_hz_.to(waveforms.device)
        window = self.window_.to(waveforms.device)
        n = self.n_.to(waveforms.device)

        low = self.min_low_hz + torch.abs(low_hz)
        high = torch.clamp(low + self.min_band_hz + torch.abs(band_hz), self.min_low_hz, self.sample_rate / 2)
        band = (high - low)[:, 0]

        f_times_t_low = torch.matmul(low, n)
        f_times_t_high = torch.matmul(high, n)

        band_pass_left = (torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) / (n / 2)
        band_pass_center = 2 * band.view(-1, 1)
        band_pass_right = (torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) / (n / 2)

        band_pass = torch.cat(
            [band_pass_left[:, : int(self.kernel_size / 2)], band_pass_center, band_pass_right[:, int(self.kernel_size / 2) + 1 :]],
            dim=1,
        )
        band_pass = band_pass / (2 * band[:, None])

        full_window = torch.cat([window, torch.ones(1, device=waveforms.device), window.flip(0)])
        filters = band_pass * full_window

        filters = filters.view(self.out_channels, 1, self.kernel_size)
        return F.conv1d(waveforms, filters, stride=10, padding=0, dilation=1, groups=1)


class Residual_block(nn.Module):
    def __init__(self, nb_filts, first=False):
        super(Residual_block, self).__init__()
        self.first = first
        if not self.first:
            self.bn1 = nn.BatchNorm2d(num_features=nb_filts[0])
        self.conv1 = nn.Conv2d(in_channels=nb_filts[0], out_channels=nb_filts[1], kernel_size=(2, 3), padding=(1, 1))
        self.selu = nn.SELU(inplace=True)
        self.bn2 = nn.BatchNorm2d(num_features=nb_filts[1])
        self.conv2 = nn.Conv2d(in_channels=nb_filts[1], out_channels=nb_filts[1], kernel_size=(2, 3), padding=(0, 1))

        if nb_filts[0] != nb_filts[1]:
            self.conv_downsample = nn.Conv2d(in_channels=nb_filts[0], out_channels=nb_filts[1], kernel_size=(1, 3), padding=(0, 1))
        else:
            self.conv_downsample = None
        self.mp = nn.MaxPool2d((1, 2))

    def forward(self, x):
        identity = x
        if not self.first:
            out = self.bn1(x)
            out = self.selu(out)
        else:
            out = x

        out = self.conv1(out)
        out = self.bn2(out)
        out = self.selu(out)
        out = self.conv2(out)

        if self.conv_downsample is not None:
            identity = self.conv_downsample(identity)

        out += identity
        out = self.mp(out)
        return out


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super(GraphAttentionLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_weight = nn.Parameter(torch.Tensor(out_dim, 1))
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.selu = nn.SELU(inplace=True)

    def forward(self, x):
        e = torch.tanh(self.att_proj(x))
        e = torch.matmul(e, self.att_weight)
        alpha = F.softmax(e, dim=1)
        x_att = torch.sum(x * alpha, dim=1, keepdim=True)
        h_att = self.proj_with_att(x_att)
        h_no_att = self.proj_without_att(x)
        h = h_no_att + h_att
        h = h.transpose(1, 2)
        h = self.bn(h)
        h = h.transpose(1, 2)
        return self.selu(h)


class HtrgGraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super(HtrgGraphAttentionLayer, self).__init__()
        self.proj_type1 = nn.Linear(in_dim[0], in_dim[0])
        self.proj_type2 = nn.Linear(in_dim[1], in_dim[1])
        self.att_proj = nn.Linear(in_dim[0], out_dim)
        self.att_projM = nn.Linear(in_dim[0], out_dim)
        self.att_weight11 = nn.Parameter(torch.Tensor(out_dim, 1))
        self.att_weight22 = nn.Parameter(torch.Tensor(out_dim, 1))
        self.att_weight12 = nn.Parameter(torch.Tensor(out_dim, 1))
        self.att_weightM = nn.Parameter(torch.Tensor(out_dim, 1))

        self.proj_with_att = nn.Linear(in_dim[0], out_dim)
        self.proj_without_att = nn.Linear(in_dim[0], out_dim)
        self.proj_with_attM = nn.Linear(in_dim[0], out_dim)
        self.proj_without_attM = nn.Linear(in_dim[0], out_dim)

        self.bn = nn.BatchNorm1d(out_dim)
        self.selu = nn.SELU(inplace=True)

    def forward(self, x1, x2, master):
        h1 = self.proj_type1(x1)
        h2 = self.proj_type2(x2)
        e11 = torch.matmul(torch.tanh(self.att_proj(h1)), self.att_weight11)
        e22 = torch.matmul(torch.tanh(self.att_proj(h2)), self.att_weight22)
        e12 = torch.matmul(torch.tanh(self.att_proj(h1)), self.att_weight12)
        eM = torch.matmul(torch.tanh(self.att_projM(master)), self.att_weightM)

        alpha1 = F.softmax(e11, dim=1)
        alpha2 = F.softmax(e22, dim=1)

        x1_att = torch.sum(h1 * alpha1, dim=1, keepdim=True)
        x2_att = torch.sum(h2 * alpha2, dim=1, keepdim=True)

        h1_out = self.proj_without_att(h1) + self.proj_with_att(x1_att)
        h2_out = self.proj_without_att(h2) + self.proj_with_att(x2_att)
        m_out = self.proj_without_attM(master) + self.proj_with_attM(x1_att + x2_att)

        h_cat = torch.cat([h1_out, h2_out, m_out], dim=1)
        h_cat = h_cat.transpose(1, 2)
        h_cat = self.bn(h_cat)
        h_cat = h_cat.transpose(1, 2)
        h_cat = self.selu(h_cat)
        return h_cat[:, : h1.size(1), :], h_cat[:, h1.size(1) : h1.size(1) + h2.size(1), :], h_cat[:, -1:, :]


class GraphPool(nn.Module):
    def __init__(self, in_dim):
        super(GraphPool, self).__init__()
        self.proj = nn.Linear(in_dim, 1)

    def forward(self, x):
        alpha = F.softmax(self.proj(x), dim=1)
        out = torch.sum(x * alpha, dim=1)
        return out


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        filts = config["filts"]
        self.first_conv = Sinc_conv(out_channels=filts[0], kernel_size=1024, in_channels=1)
        self.first_bn = nn.BatchNorm2d(1)

        self.encoder = nn.ModuleList([
            nn.ModuleList([Residual_block(filts[1], first=True)]),
            nn.ModuleList([Residual_block(filts[2])]),
            nn.ModuleList([Residual_block(filts[3])]),
            nn.ModuleList([Residual_block(filts[4])]),
            nn.ModuleList([Residual_block(filts[4])]),
            nn.ModuleList([Residual_block(filts[4])]),
        ])

        gat_dims = config["gat_dims"]
        self.pos_S = nn.Parameter(torch.randn(1, 23, gat_dims[0]))
        self.master1 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.master2 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))

        self.GAT_layer_S = GraphAttentionLayer(gat_dims[0], gat_dims[0])
        self.GAT_layer_T = GraphAttentionLayer(gat_dims[0], gat_dims[0])

        self.HtrgGAT_layer_ST11 = HtrgGraphAttentionLayer([gat_dims[0], gat_dims[0]], gat_dims[1])
        self.HtrgGAT_layer_ST12 = HtrgGraphAttentionLayer([gat_dims[1], gat_dims[1]], gat_dims[1])
        self.HtrgGAT_layer_ST21 = HtrgGraphAttentionLayer([gat_dims[0], gat_dims[0]], gat_dims[1])
        self.HtrgGAT_layer_ST22 = HtrgGraphAttentionLayer([gat_dims[1], gat_dims[1]], gat_dims[1])

        self.pool_S = GraphPool(gat_dims[0])
        self.pool_T = GraphPool(gat_dims[0])
        self.pool_hS1 = GraphPool(gat_dims[1])
        self.pool_hT1 = GraphPool(gat_dims[1])
        self.pool_hS2 = GraphPool(gat_dims[1])
        self.pool_hT2 = GraphPool(gat_dims[1])

        self.out_layer = nn.Linear(5 * gat_dims[1], 2)

    def forward(self, x):
        # x shape: (batch, num_samples)
        x = x.unsqueeze(1)
        x = self.first_conv(x)
        x = x.unsqueeze(1)
        x = F.max_pool2d(torch.abs(x), (3, 3))
        x = self.first_bn(x)
        x = F.selu(x)

        for block in self.encoder:
            x = block[0](x)

        b, c, f, t = x.size()
        x_S = x.max(dim=-1)[0].permute(0, 2, 1)
        x_T = x.max(dim=-2)[0].permute(0, 2, 1)

        x_S = x_S + self.pos_S
        x_S = self.GAT_layer_S(x_S)
        x_T = self.GAT_layer_T(x_T)

        x_S1, x_T1, m1 = self.HtrgGAT_layer_ST11(x_S, x_T, self.master1)
        x_S1, x_T1, m1 = self.HtrgGAT_layer_ST12(x_S1, x_T1, m1)

        x_S2, x_T2, m2 = self.HtrgGAT_layer_ST21(x_S, x_T, self.master2)
        x_S2, x_T2, m2 = self.HtrgGAT_layer_ST22(x_S2, x_T2, m2)

        p_S = self.pool_S(x_S)
        p_T = self.pool_T(x_T)
        p_hS1 = self.pool_hS1(x_S1)
        p_hT1 = self.pool_hT1(x_T1)
        p_hS2 = self.pool_hS2(x_S2)
        p_hT2 = self.pool_hT2(x_T2)

        out_concat = torch.cat([p_hS1, p_hT1, p_hS2, p_hT2, m1.squeeze(1)], dim=1)
        output = self.out_layer(out_concat)
        return out_concat, output
