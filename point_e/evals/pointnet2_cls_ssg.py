"""
Based on: https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/eb64fe0b4c24055559cea26299cb485dcb43d8dd/models/pointnet2_cls_ssg.py

MIT License

Copyright (c) 2019 benny

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch.nn as nn
import torch.nn.functional as F

from .pointnet2_utils import PointNetSetAbstraction


class get_model(nn.Module):
    def __init__(self, num_class, normal_channel=True, width_mult=1):
        super(get_model, self).__init__()
        self.width_mult = width_mult
        in_channel = 6 if normal_channel else 3
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstraction(
            npoint=512,
            radius=0.2,
            nsample=32,
            in_channel=in_channel,
            mlp=[64 * width_mult, 64 * width_mult, 128 * width_mult],
            group_all=False,
        )
        self.sa2 = PointNetSetAbstraction(
            npoint=128,
            radius=0.4,
            nsample=64,
            in_channel=128 * width_mult + 3,
            mlp=[128 * width_mult, 128 * width_mult, 256 * width_mult],
            group_all=False,
        )
        self.sa3 = PointNetSetAbstraction(
            npoint=None,
            radius=None,
            nsample=None,
            in_channel=256 * width_mult + 3,
            mlp=[256 * width_mult, 512 * width_mult, 1024 * width_mult],
            group_all=True,
        )
        self.fc1 = nn.Linear(1024 * width_mult, 512 * width_mult)
        self.bn1 = nn.BatchNorm1d(512 * width_mult)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512 * width_mult, 256 * width_mult)
        self.bn2 = nn.BatchNorm1d(256 * width_mult)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256 * width_mult, num_class)

    def forward(self, xyz, features=False):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024 * self.width_mult)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        result_features = self.bn2(self.fc2(x))
        x = self.drop2(F.relu(result_features))
        x = self.fc3(x)
        x = F.log_softmax(x, -1)

        if features:
            return x, l3_points, result_features
        else:
            return x, l3_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        total_loss = F.nll_loss(pred, target)

        return total_loss
