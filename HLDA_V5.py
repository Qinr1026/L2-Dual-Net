##% HLDA的变体5

import torch
import torch.nn as nn
import math

from attention_module.HLDA_L2Pool import HLDA_L2Pool
from attention_module.HLDA_GSpatial import GlobalAttentionModule

class LocalAttentionModule_V5(nn.Module):
    def __init__(self, num_segments):
        super(LocalAttentionModule_V5, self).__init__()

        self.num_segments = num_segments

        self.conv = nn.Conv2d(self.num_segments, self.num_segments, kernel_size=3, stride=1, padding=1,
                              groups=1)  # Fix the groups parameter
        # self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 将其转换为[1, W, H]
        l2_norm = torch.norm(x, p=2, dim=1, keepdim=True)

        # 沿着W方向分割为8个大小为[1, W/8, H]的子张量
        split_tensors = torch.chunk(l2_norm, self.num_segments, dim=2)

        local_concat = torch.cat(split_tensors, dim=1)  # 沿通道拼接

        output = self.sigmoid((self.conv(local_concat)))  # 卷积跨通道交互

        # 还原
        restored_tensors = output.chunk(self.num_segments, dim=1)
        restored_original_tensor = torch.cat(restored_tensors, dim=2).view(l2_norm.shape)

        return restored_original_tensor

class HLDA_DualSpatial_V5(nn.Module):
    def __init__(self, num_segments):
        super(HLDA_DualSpatial_V5, self).__init__()

        self.local_att = LocalAttentionModule_V5(num_segments)
        self.global_att = GlobalAttentionModule()

    def forward(self, x):
        local_att = self.local_att(x)
        global_att = self.global_att(x)
        combined_att = local_att + global_att
        return combined_att

class HLDA_V5(nn.Module):
    def __init__(self, in_channels, num_segments):
        super(HLDA_V5, self).__init__()

        self.HLDA_L2Pool = HLDA_L2Pool(in_channels)
        self.HLDA_DualSpatial_V2 = HLDA_DualSpatial_V5(num_segments)

    def forward(self, x):
        HLDA_L2Pool = self.HLDA_L2Pool(x)
        x1 = x * HLDA_L2Pool
        HLDA_DualSpatial = self.HLDA_DualSpatial_V2(x1)
        x2 = x1 * HLDA_DualSpatial
        return x2

# input_feature = torch.randn(2, 64, 32, 32)
# hlda = HLDA_V5(in_channels=64, num_segments=4)
# output_feature = hlda(input_feature)
# print(output_feature.shape)