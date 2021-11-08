import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNet(nn.Module):
    def __init__(self, args, num_classes=40):
        super(PointNet, self).__init__()
        self.args = args
        self.block1 = self._addBlock("block1", "conv", 3, 64, kernel_size=1, bias=False)
        self.block2 = self._addBlock(
            "block2", "conv", 64, 64, kernel_size=1, bias=False
        )
        self.block3 = self._addBlock(
            "block3", "conv", 64, 64, kernel_size=1, bias=False
        )
        self.block4 = self._addBlock(
            "block4", "conv", 64, 128, kernel_size=1, bias=False
        )
        self.block5 = self._addBlock(
            "block5", "conv", 128, args.emb_dims, kernel_size=1, bias=False
        )
        self.block6 = self._addBlock(
            "block6", "linear", args.emb_dims, 512, bias=False, relu=False, dropout=True
        )
        self.block7 = self._addBlock("block7", "linear", 512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)  # Conv1d
        x = self.block2(x)  # Conv1d
        x = self.block3(x)  # Conv1d
        x = self.block4(x)  # Conv1d
        x = self.block5(x)  # Conv1d
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        x = self.block6(x)  # Linear
        x = self.block7(x)  # Linear
        return x

    def _addBlock(
        self,
        name: str,
        type: str,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        bias: bool = False,
        bn: bool = True,
        relu: bool = True,
        dropout: bool = False,
    ):
        block = nn.Sequential()
        if type == "linear":
            block.add_module(
                f"{name}_lin", nn.Linear(in_channels, out_channels, bias=bias)
            )
        elif type == "conv":
            block.add_module(
                f"{name}_conv",
                nn.Conv1d(
                    in_channels, out_channels, kernel_size=kernel_size, bias=bias
                ),
            )
        if bn:
            block.add_module(f"{name}_bn", nn.BatchNorm1d(out_channels))
        if relu:
            block.add_module(f"{name}_relu", nn.ReLU(inplace=True))
        if dropout:
            block.add_module(f"{name}_dropout", nn.Dropout())

        return block
