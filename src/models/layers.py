import torch
from torch import nn




class TCN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        branch_cf=[(3, 1), (3, 2), (3, 3), (3, 4), ("max", 3), ("1x1")],
        stride: int = 1,
        dropout_rate: float = 0,
        act_layer=nn.ReLU,
    ):
        """
        Temporal Module 
        :param in_channels: input channels size
        :param out_channels: output channels size
        :param branch_cf: configs of branches
        :param stride: stride of block
        :param dropout_rate: dropout rate of TCN
        :param act_layer: activation function to use
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.num_branches = len(branch_cf)
        self.dropout = (
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        )

        self.act = act_layer() if act_layer is not None else nn.Identity()

        self.branch_channels = out_channels // self.num_branches
        self.rem_branch_channels = out_channels - self.branch_channels * (
            self.num_branches - 1
        )

        branches = []
        for i, cf in enumerate(branch_cf):
            branch_c = (
                self.rem_branch_channels if i == 0 else self.branch_channels
            )
            if cf == "1x1":
                branches.append(
                    nn.Conv2d(
                        in_channels,
                        branch_c,
                        kernel_size=1,
                        stride=(stride, 1),
                    )
                )
            elif cf[0] == "max":
                padding = cf[1] // 2
                branches.append(
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels,
                            branch_c,
                            kernel_size=1,
                        ),
                        nn.BatchNorm2d(branch_c),
                        self.act,
                        self.dropout,
                        nn.MaxPool2d(
                            kernel_size=(cf[1], 1),
                            stride=(stride, 1),
                            padding=(padding, 0),
                        ),
                    )
                )
            else:
                padding = (cf[0] + (cf[0] - 1) * (cf[1] - 1) - 1) // 2
                branches.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, branch_c, kernel_size=1),
                        nn.BatchNorm2d(branch_c),
                        self.act,
                        self.dropout,
                        nn.Conv2d(
                            branch_c,
                            branch_c,
                            kernel_size=(cf[0], 1),
                            stride=(stride, 1),
                            dilation=(cf[1], 1),
                            padding=(padding, 0),
                        ),
                    )
                )
        self.branches = nn.ModuleList(branches)

    def forward(self, x):
        N, C, T, V = x.size()
        branch_outs = []
        for branch in self.branches:
            branch_outs.append(branch(x))

        y = torch.cat(branch_outs, dim=1)
        return y


# Our code is based on CTR-GCN repo: https://github.com/Uason-Chen/CTR-GCN
# with little changes to adapt to our experiments and data processing
class CTRGCN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        rel_reduction: int = 8,
        first_block: bool = False,
    ):
        """
        CTRGCN 
        :param in_channels: input channels size
        :param out_channels: output channels size
        :param rel_reduction: reduction ratio for rel channels
        :param first_block: whether this is the first block
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if first_block:
            self.rel_channels = 8
        else:
            self.rel_channels = in_channels // rel_reduction

        self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, 1)
        self.conv2 = nn.Conv2d(self.in_channels, self.rel_channels, 1)
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, 1)
        self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, 1)
        self.tanh = nn.Tanh()

    def forward(self, x, A=None, alpha=1):
        N, C, T, V = x.size()
        x1 = self.conv1(x).mean(-2)
        x2 = self.conv2(x).mean(-2)
        x3 = self.conv3(x)

        y = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))
        adj = self.conv4(y) * alpha + (
            A.unsqueeze(0).unsqueeze(0) if A is not None else 0
        )
        z = torch.einsum("ncuv,nctv->nctu", adj, x3).contiguous()

        return z


# One GCN unit
class GCN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        A: torch.Tensor,
        dropout_rate: float = 0,
        act_layer=nn.ReLU,
        adaptive=True,
        first_block=False,
    ):
        """
        GCN (spatial module)
        :param in_channels: input channels size
        :param out_channels: output channels size
        :param A: adjacent matrices
        :param dropout_rate: dropout rate of block
        :param act_layer: which activation function to use
        :param adaptive: whether to use adaptive adjacent matrices
        :param first_block: whether this is the first block
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.dropout = (
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        )
        self.act = act_layer() if act_layer is not None else nn.Identity()

        self.register_buffer("A", A.clone())
        self.adaptive = adaptive
        if self.adaptive:
            self.PA = nn.Parameter(A.clone())

        self.num_subsets = self.A.size(0)
        self.convs = nn.ModuleList()
        for _ in range(self.num_subsets):
            self.convs.append(
                CTRGCN(in_channels, out_channels, first_block=first_block)
            )

        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        N, C, T, V = x.size()
        if self.adaptive:
            A = self.PA
        else:
            A = self.A
        y = None
        for s in range(self.num_subsets):
            z = self.convs[s](x, A[s], self.alpha)
            if y is not None:
                y = y + z
            else:
                y = z

        return y


# One block in our network
class Block(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        A,
        stride: int = 1,
        dropout_rate: float = 0,
        residual: bool = True,
        act_layer=nn.ReLU,
        first_block: bool = False,
        adaptive: bool = False,
    ):
        """
        One block in our networks
        :param in_channels: input channels size
        :param out_channels: output channels size
        :param A: adjacent matrices
        :param stride: stride of block
        :param dropout_rate: dropout rate of block
        :param residual: whether to use residual connection
        :param act_layer: which activation layer to use
        :param first_block: whether this is the first block
        :param adaptive: whether to use adaptive adjacent matrices
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Spatial Module (GCN)
        self.gcn = GCN(
            in_channels,
            out_channels,
            A=A,
            dropout_rate=dropout_rate,
            first_block=first_block,
            act_layer=act_layer,
            adaptive=adaptive,
        )
        if not residual:
            self.g_res = lambda x: 0
        elif in_channels == out_channels:
            self.g_res = lambda x: x
        else:
            self.g_res = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
            )
        self.g_norm = nn.BatchNorm2d(out_channels)

        # Temporal Module (MS-TCN)
        tcn_branch_cf = [(5, 1), (5, 2), ("max", 3), ("1x1")]
        self.tcn = TCN(
            out_channels,
            out_channels,
            branch_cf=tcn_branch_cf,
            stride=stride,
            dropout_rate=dropout_rate,
            act_layer=act_layer,
        )
        if not residual:
            self.res = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.res = lambda x: x
        else:
            self.res = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        self.t_norm = nn.BatchNorm2d(out_channels)
        self.act = act_layer()
        self.dropout = (
            nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        )

    def forward(self, x):
        g_res = self.g_res(x)
        res = self.res(x)

        x = self.gcn(x)
        x = self.act(self.g_norm(x) + g_res)
        x = self.dropout(x)

        x = self.tcn(x)
        x = self.act(self.t_norm(x) + res)
        x = self.dropout(x)

        return x
