#%% HLDA

import torch
import torch.nn as nn

from attention_module.HLDA_DualSpatial import HLDA_DualSpatial
from attention_module.HLDA_L2Pool_V2 import HLDA_L2Pool

class HLDA(nn.Module):
    def __init__(self, in_channels, num_segments):
        super(HLDA, self).__init__()

        self.HLDA_DualSpatial = HLDA_DualSpatial(num_segments)
        self.HLDA_L2Pool = HLDA_L2Pool(in_channels)

    def forward(self, x):
        HLDA_L2Pool = self.HLDA_L2Pool(x)
        x1 = x * HLDA_L2Pool   # 先通道注意力
        HLDA_DualSpatial = self.HLDA_DualSpatial(x1)
        x2 = x1 * HLDA_DualSpatial   # 后空间注意力

        # HLDA_DualSpatial = self.HLDA_DualSpatial(x)
        # x1 = x * HLDA_DualSpatial  # 先空间注意力
        # HLDA_L2Pool = self.HLDA_L2Pool(x1)
        # x2 = x1 * HLDA_L2Pool  # 先空间注意力

        # HLDA_L2Pool = self.HLDA_L2Pool(x)
        # x_1 = x * HLDA_L2Pool  # 先通道注意力
        # HLDA_DualSpatial = self.HLDA_DualSpatial(x)
        # x_2 = x * HLDA_DualSpatial  # 后空间注意力
        # x2 = x_1 + x_2   # 并行计算

        return x2
#
# input_feature = torch.randn(1, 64, 256, 256)
# hlda = HLDA(in_channels=64, num_segments=4)
# output_feature = hlda(input_feature)
# print(output_feature.shape)
#
# # 打印模型的参数量
# HLDA_DualSpatial = HLDA_DualSpatial(num_segments=16)
# HLDA_L2Pool = HLDA_L2Pool(in_channels=64)
# HLDA_DualSpatial_params = sum(p.numel() for p in HLDA_DualSpatial.parameters())
# HLDA_L2Pool_params = sum(p.numel() for p in HLDA_L2Pool.parameters())
# total_params = sum(p.numel() for p in hlda.parameters())
# # print(f"HLDA_L2Pool, Total parameters: {HLDA_L2Pool_params}")
# # print(f"HLDA_DualSpatial, Total parameters: {HLDA_DualSpatial_params}")
# print(f"HLDA, Total parameters: {total_params}")
