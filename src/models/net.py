import torch
from torch import nn

from src.models.layers import Block
from src.graph.ntu_graph import Graph


class STGCN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int = 60,
        act_layer=nn.ReLU,
        dropout_rate: float = 0,
        adaptive: bool = True,
        num_people: int = 2,
    ):
        """
        :param in_channels: input channels size
        :param num_classes: number of classes
        :param act_layer: which activation layer to use
        :param dropout_rate: dropout rate of block
        :param adaptive: whether to use adaptive adjacent matrices
        """
        super().__init__()
        self.graph = Graph()
        self.data_norm = nn.BatchNorm1d(
            in_channels * self.graph.get_A().shape[0] * num_people
        )
        layer_cfs = [
            (in_channels, 64, 1),
            (64, 64, 1),
            (64, 64, 1),
            (64, 64, 1),
            (64, 128, 2),
            (128, 128, 1),
            (128, 128, 1),
            (128, 256, 2),
            (256, 256, 1),
            (256, 256, 1),
        ]
        self.blocks = nn.ModuleList()
        A = torch.from_numpy(self.graph.get_spatial_A())
        for i, cf in enumerate(layer_cfs):
            self.blocks.append(
                Block(
                    cf[0],
                    cf[1],
                    A,
                    stride=cf[2],
                    dropout_rate=dropout_rate,
                    residual=(i > 0),
                    act_layer=act_layer,
                    first_block=(i == 0),
                    adaptive=adaptive,
                )
            )
        self.avg_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.head = nn.Linear(layer_cfs[-1][1], num_classes)

    def forward(self, x):
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_norm(x)
        x = (
            x.view(N, M, V, C, T)
            .permute(0, 1, 3, 4, 2)
            .contiguous()
            .view(N * M, C, T, V)
        )

        for block in self.blocks:
            x = block(x)

        NM, C, T, V = x.size()
        x = x.view(N, M, C, T, V)
        x = x.view(N, M, C, T * V)
        x = x.mean(3).mean(1)
        x = self.head(x)
        return x
