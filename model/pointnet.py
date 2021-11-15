import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def addBlock(
    name: str,
    type: str,
    in_channels: int,
    out_channels: int,
    kernel_size: int = 1,
    bias: bool = False,
    bn: bool = True,
    relu: bool = True,
    dropout: bool = False,
):
    block = nn.Sequential()
    if type == "linear":
        block.add_module(f"{name}_lin", nn.Linear(in_channels, out_channels, bias=bias))
    elif type == "conv":
        block.add_module(
            f"{name}_conv", nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, bias=bias),
        )
    if dropout:
        block.add_module(f"{name}_dropout", nn.Dropout(p=0.3))
    if bn:
        block.add_module(f"{name}_bn", nn.BatchNorm1d(out_channels))
    if relu:
        block.add_module(f"{name}_relu", nn.ReLU(inplace=True))

    return block


def feature_transform_regularizer(trans: torch.Tensor) -> float:
    batch_size = trans.size()[0]
    d = trans.size()[1]
    identity_matrix = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        identity_matrix = identity_matrix.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - identity_matrix, dim=(1, 2)))
    return loss


class TNet(nn.Module):
    def __init__(self, k: int = 3):
        super(TNet, self).__init__()
        self.k = k
        self.conv_block1 = addBlock("conv_block1", "conv", self.k, 64, kernel_size=1, bias=False)
        self.conv_block2 = addBlock("conv_block2", "conv", 64, 128, kernel_size=1, bias=False)
        self.conv_block3 = addBlock("conv_block3", "conv", 128, 1024, kernel_size=1, bias=False)

        self.linear_block1 = addBlock("linear_block1", "linear", 1024, 512)
        self.linear_block2 = addBlock("linear_block2", "linear", 512, 256)
        self.linear_block3 = addBlock("linear_block3", "linear", 256, self.k * self.k, bn=False, relu=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size()[0]
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        x, _ = torch.max(x, 2, keepdim=True)
        x = x.view(-1, 1024)
        x = self.linear_block1(x)
        x = self.linear_block2(x)
        x = self.linear_block3(x)

        # iden = torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32)).view(1,9).repeat(batch_size,1)
        # iden = torch.from_numpy(np.eye(self.k).flatten().astype(np.float32)).view(1,self.k*self.k).repeat(batch_size,1)
        identity_matrix = (
            torch.from_numpy(np.eye(self.k).flatten().astype(np.float32)).view(1, self.k * self.k).repeat(batch_size, 1)
        )

        if x.is_cuda:
            identity_matrix = identity_matrix.cuda()
        x = x + identity_matrix
        x = x.view(-1, self.k, self.k)
        return x


class PointNet_feature(nn.Module):
    def __init__(self, k: int = 3, global_feature: bool = True, feature_transform: bool = False):
        super(PointNet_feature, self).__init__()
        self.global_feature = global_feature
        self.feature_transform = feature_transform

        self.k = k
        self.tnet_3 = TNet(k=self.k)
        self.conv_block1 = addBlock("conv_block1", "conv", self.k, 64, kernel_size=1)
        self.conv_block2 = addBlock("conv_block2", "conv", 64, 128, 1)
        self.conv_block3 = addBlock("conv_block3", "conv", 128, 1024, 1, relu=False)
        if self.feature_transform:
            self.tnet_64 = TNet(k=64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_points = x.size()[2]
        trans = self.tnet_3(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)

        x = self.conv_block1(x)
        if self.feature_transform:
            trans_feat = self.tnet_64(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x, _ = torch.max(x, 2, keepdim=True)
        x = x.view(-1, 1024)
        if self.global_feature:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024).repeat(1, 1, num_points)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


class PointNet_Classification(nn.Module):
    def __init__(self, args, num_classes: int = 40):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self.feature_transform = self.args.feature_transform
        self.feature = PointNet_feature(global_feature=True, feature_transform=self.feature_transform)

        self.linear_block1 = addBlock("linear_block1", "linear", 1024, 512)
        self.linear_block2 = addBlock("linear_block2", "linear", 512, 256, dropout=True)
        self.linear_block3 = addBlock("linear_block3", "linear", 256, self.num_classes, bn=False, relu=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, trans, trans_feat = self.feature(x)
        x = self.linear_block1(x)
        x = self.linear_block2(x)
        x = self.linear_block3(x)
        return F.log_softmax(x, dim=1), trans, trans_feat
