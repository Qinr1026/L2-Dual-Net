##% HLDA的变体4

import torch
import torch.nn as nn

from attention_module.HLDA_L2Pool import HLDA_L2Pool
from attention_module.HLDA_GSpatial import GlobalAttentionModule

class LocalAttentionModule_V4(nn.Module):
    def __init__(self):
        #### num_segments必须是2的整倍数!!!
        super(LocalAttentionModule_V4, self).__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, groups=1)
        # self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        l2_norm = torch.norm(x, p=2, dim=1, keepdim=True)  # torch.Size([1, 1, 32, 32])
        output = self.sigmoid((self.conv(l2_norm)))
        # # 还原
        # restored_tensors = output.chunk(self.num_segments, dim=1)
        # restored_original_tensor = torch.cat(restored_tensors, dim=2).view(l2_norm.shape)
        return output


class HLDA_DualSpatial_V4(nn.Module):
    def __init__(self):
        super(HLDA_DualSpatial_V4, self).__init__()

        self.local_att = LocalAttentionModule_V4()
        self.global_att = GlobalAttentionModule()

    def forward(self, x):
        local_att = self.local_att(x)
        global_att = self.global_att(x)
        combined_att = local_att + global_att
        return combined_att

class HLDA_V4(nn.Module):
    def __init__(self, in_channels):
        super(HLDA_V4, self).__init__()

        self.HLDA_L2Pool = HLDA_L2Pool(in_channels)
        self.HLDA_DualSpatial_V4 = HLDA_DualSpatial_V4()

    def forward(self, x):
        HLDA_L2Pool = self.HLDA_L2Pool(x)
        x1 = x * HLDA_L2Pool
        HLDA_DualSpatial = self.HLDA_DualSpatial_V4(x1)
        x2 = x1 * HLDA_DualSpatial
        return x2

# input_feature = torch.randn(2, 64, 256, 256)
# hlda = HLDA_V4(in_channels=64)
# output_feature = hlda(input_feature)
# print(output_feature.shape)

