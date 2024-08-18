import math
import pdb

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


def import_class(name):
    components = name.split(".")
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2.0 / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode="fan_out")
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        if hasattr(m, "weight"):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if (
            hasattr(m, "bias")
            and m.bias is not None
            and isinstance(m.bias, torch.Tensor)
        ):
            nn.init.constant_(m.bias, 0)
    elif classname.find("BatchNorm") != -1:
        if hasattr(m, "weight") and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            m.bias.data.fill_(0)


class TemporalConv(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, dilation=1
    ):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1),
        )

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class MultiScale_TemporalConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        dilations=[1, 2, 3, 4],
        residual=True,
        residual_kernel_size=1,
    ):

        super().__init__()
        assert (
            out_channels % (len(dilations) + 2) == 0
        ), "# out channels should be multiples of # branches"

        # Multiple branches of temporal convolution
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size] * len(dilations)
        # Temporal Convolution branches
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels, branch_channels, kernel_size=1, padding=0
                    ),
                    nn.BatchNorm2d(branch_channels),
                    nn.ReLU(inplace=True),
                    TemporalConv(
                        branch_channels,
                        branch_channels,
                        kernel_size=ks,
                        stride=stride,
                        dilation=dilation,
                    ),
                )
                for ks, dilation in zip(kernel_size, dilations)
            ]
        )

        # Additional Max & 1x1 branch
        self.branches.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels, branch_channels, kernel_size=1, padding=0
                ),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(
                    kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)
                ),
                nn.BatchNorm2d(branch_channels),  # 为什么还要加bn
            )
        )

        self.branches.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    padding=0,
                    stride=(stride, 1),
                ),
                nn.BatchNorm2d(branch_channels),
            )
        )

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(
                in_channels,
                out_channels,
                kernel_size=residual_kernel_size,
                stride=stride,
            )

        # initialize
        self.apply(weights_init)

    def forward(self, x):
        # Input dim: (N,C,T,V)
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        out += res
        return out


class CTRGC(nn.Module):
    def __init__(
        self, in_channels, out_channels, rel_reduction=8, mid_reduction=1
    ):
        super(CTRGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels in [3, 6, 9]:
            self.rel_channels = 8
            self.mid_channels = 16
        else:
            self.rel_channels = in_channels // rel_reduction
            self.mid_channels = in_channels // mid_reduction
        self.conv1 = nn.Conv2d(
            self.in_channels, self.rel_channels, kernel_size=1
        )
        self.conv2 = nn.Conv2d(
            self.in_channels, self.rel_channels, kernel_size=1
        )
        self.conv3 = nn.Conv2d(
            self.in_channels, self.out_channels, kernel_size=1
        )
        self.conv4 = nn.Conv2d(
            self.rel_channels, self.out_channels, kernel_size=1
        )
        self.tanh = nn.Tanh()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x, A=None, alpha=1):
        x1, x2, x3 = (
            self.conv1(x).mean(-2),
            self.conv2(x).mean(-2),
            self.conv3(x),
        )
        x1 = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))
        x1 = self.conv4(x1) * alpha + (
            A.unsqueeze(0).unsqueeze(0) if A is not None else 0
        )  # N,C,V,V
        x1 = torch.einsum("ncuv,nctv->nctu", x1, x3)
        return x1


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class unit_gcn(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        A,
        coff_embedding=4,
        adaptive=True,
        residual=True,
    ):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.adaptive = adaptive
        self.num_subset = A.shape[0]
        self.convs = nn.ModuleList()
        for i in range(self.num_subset):
            self.convs.append(CTRGC(in_channels, out_channels))

        if residual:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels),
                )
            else:
                self.down = lambda x: x
        else:
            self.down = lambda x: 0
        if self.adaptive:
            self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        else:
            self.A = Variable(
                torch.from_numpy(A.astype(np.float32)), requires_grad=False
            )
        self.alpha = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

    def forward(self, x):
        y = None
        if self.adaptive:
            A = self.PA
        else:
            A = self.A.cuda(x.get_device())
        for i in range(self.num_subset):
            z = self.convs[i](x, A[i], self.alpha)
            y = z + y if y is not None else z
        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)

        return y



class SpatialTemporalAttention(nn.Module):
    def __init__(self, in_channels: int, r: int):
        super(SpatialTemporalAttention, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = in_channels // r

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, self.inter_channels, 1),
            nn.BatchNorm2d(self.inter_channels),
            nn.Hardswish(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.inter_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(self.inter_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid(),
        )

    def forward(self, query, value):
        N, C, T, V = query.size()
        spatial_pool = self.conv1(query.mean(-1))  # N, C/r, T
        temporal_pool = self.conv1(query.mean(-2))  # N, C/r, V

        spatial_pool = self.conv2(spatial_pool).unsqueeze(-1)  # N, C, T, 1
        temporal_pool = self.conv3(temporal_pool).unsqueeze(-2)  # N, C, 1, V

        att_map = torch.matmul(spatial_pool, temporal_pool)

        return att_map * value


class AngularMotionUnit(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        inter_ratio: int = 2,
        r: int = 2,
        residual: bool = True,
    ):
        super(AngularMotionUnit, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = in_channels * inter_ratio
        self.out_channels = out_channels

        if residual and in_channels == out_channels:
            self.res = lambda x: x
        elif residual and in_channels != out_channels:
            self.res = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.res = lambda x: 0

        self.att = SpatialTemporalAttention(
            self.out_channels,
            r,
        )

        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, self.inter_channels, 1),
            nn.BatchNorm2d(self.inter_channels),
            nn.ReLU(),
            nn.Conv2d(self.inter_channels, self.out_channels, 1),
        )
        self.norm = nn.BatchNorm2d(self.out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = self.res(x)
        x = self.mlp(x)
        x = self.att(x, x)
        return self.relu(self.norm(res + x))


class TCN_GCN_unit(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        A,
        stride=1,
        residual=True,
        adaptive=True,
        kernel_size=5,
        dilations=[1, 2],
    ):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A, adaptive=adaptive)
        self.tcn1 = MultiScale_TemporalConv(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilations=dilations,
            residual=False,
        )
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0
            self.residual_att = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
            self.residual_att = lambda x: x

        else:
            self.residual = unit_tcn(
                in_channels, out_channels, kernel_size=1, stride=stride
            )
            self.residual_att = unit_tcn(
                in_channels, out_channels, kernel_size=1, stride=stride
            )
        self.att = SpatialTemporalAttention(out_channels, 2)

    def forward(self, x, am=None):
        y = self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))
        if self.use_am:
            res = self.residual_att(y)
            y = self.relu(self.att(y, am) + res)
        return y


class Model(nn.Module):
    def __init__(
        self,
        use_am,
        num_class=60,
        num_point=25,
        num_person=2,
        graph=None,
        graph_args=dict(),
        in_channels=3,
        am_channels=3,
        drop_out=0,
        adaptive=True,
    ):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A  # 3,25,25

        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)
        base_channel = 64
        self.use_am = use_am
        if use_am:
            self.am_bn = nn.BatchNorm1d(num_person * am_channels * num_point)
            self.am_l1 = AngularMotionUnit(am_channels, base_channel)
            self.am_l2 = AngularMotionUnit(base_channel, base_channel)
            self.am_l3 = AngularMotionUnit(base_channel, base_channel)
            self.am_l4 = AngularMotionUnit(base_channel, base_channel)
            self.am_l5 = AngularMotionUnit(base_channel, base_channel * 2)
            self.am_l6 = AngularMotionUnit(base_channel * 2, base_channel * 2)
            self.am_l7 = AngularMotionUnit(base_channel * 2, base_channel * 2)
            self.am_l8 = AngularMotionUnit(base_channel * 2, base_channel * 4)
            self.am_l9 = AngularMotionUnit(base_channel * 4, base_channel * 4)
            self.am_l10 = AngularMotionUnit(base_channel * 4, base_channel * 4)

        self.l1 = TCN_GCN_unit(
            in_channels, base_channel, A, residual=False, adaptive=adaptive
        )
        self.l2 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l3 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l4 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l5 = TCN_GCN_unit(
            base_channel, base_channel * 2, A, stride=2, adaptive=adaptive
        )
        self.l6 = TCN_GCN_unit(
            base_channel * 2, base_channel * 2, A, adaptive=adaptive
        )
        self.l7 = TCN_GCN_unit(
            base_channel * 2, base_channel * 2, A, adaptive=adaptive
        )
        self.l8 = TCN_GCN_unit(
            base_channel * 2, base_channel * 4, A, stride=2, adaptive=adaptive
        )
        self.l9 = TCN_GCN_unit(
            base_channel * 4, base_channel * 4, A, adaptive=adaptive
        )
        self.l10 = TCN_GCN_unit(
            base_channel * 4, base_channel * 4, A, adaptive=adaptive
        )

        self.fc = nn.Linear(base_channel * 4, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2.0 / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x, am=None):
        if len(x.shape) == 3:
            N, T, VC = x.shape
            x = (
                x.view(N, T, self.num_point, -1)
                .permute(0, 3, 1, 2)
                .contiguous()
                .unsqueeze(-1)
            )
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = (
            x.view(N, M, V, C, T)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
            .view(N * M, C, T, V)
        )
        if self.use_am and am is not None:
            N, C, T, V, M = am.size()
            print(am.size())
            am = am.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
            am = self.data_bn(am)
            am = (
                am.view(N, M, V, C, T)
                .permute(0, 1, 3, 4, 2)
                .contiguous()
                .view(N * M, C, T, V)
            )

        if self.use_am:
            am = self.am_l1(am)
        x = self.l1(x, am)
        if self.use_am:
            am = self.am_l2(am)
        x = self.l2(x, am)
        if self.use_am:
            am = self.am_l3(am)
        x = self.l3(x, am)
        if self.use_am:
            am = self.am_l4(am)
        x = self.l4(x, am)
        if self.use_am:
            am = self.am_l5(am)
        x = self.l6(x, am)
        if self.use_am:
            am = self.am_l7(am)
        x = self.l7(x, am)
        if self.use_am:
            am = self.am_l8(am)
        x = self.l8(x, am)
        if self.use_am:
            am = self.am_l9(am)
        x = self.l9(x, am)
        if self.use_am:
            am = self.am_l10(am)
        x = self.l10(x, am)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x = self.drop_out(x)

        return self.fc(x)
