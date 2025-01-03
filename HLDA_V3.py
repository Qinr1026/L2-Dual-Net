##% HLDA的变体3

import torch
import torch.nn as nn
import math

from attention_module.HLDA_L2Pool import HLDA_L2Pool
from attention_module.HLDA_GSpatial import GlobalAttentionModule


class LocalAttentionModule_V3(nn.Module):
    def __init__(self, num_segments):
        #### num_segments必须是2的整倍数!!!
        super(LocalAttentionModule_V3, self).__init__()

        self.num_segments = num_segments
        # 这里将num_segments开方处理就可以得到行和列上的个数
        self.size_segments = int(math.sqrt(num_segments))

        # self.split_size = in_channels // num_segments
        self.conv = nn.Conv2d(self.num_segments, self.num_segments, kernel_size=3, stride=1, padding=1, groups=1)  # Fix the groups parameter
        # self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        l2_norm = torch.norm(x, p=2, dim=1, keepdim=True)  # torch.Size([1, 1, 32, 32])
        # 在本质上，W_small_size = H_small_size
        W_small_size = int(l2_norm.size(2) // self.size_segments)
        # H_small_size = int(l2_norm.size(3) // self.size_segments)
        splitted_features = []
        for i in range(self.size_segments):
            for j in range(self.size_segments):
                feature = l2_norm[:, :, i * W_small_size:(i + 1) * W_small_size, j * W_small_size:(j + 1) * W_small_size]
                splitted_features.append(feature)
        # # 输出每个小特征向量的大小
        # for i, feature in enumerate(splitted_features):
        #     print(f"Feature {i + 1} size: {feature.size()}")

        local_concat = torch.cat(splitted_features, dim=1) # 沿通道拼接
        output = self.sigmoid((self.conv(local_concat))) # 卷积跨通道交互


        # 还原
        restored_tensors = output.chunk(self.num_segments, dim=1)
        restored_original_tensor = torch.cat(restored_tensors, dim=2).view(l2_norm.shape)
        return restored_original_tensor + l2_norm

class HLDA_DualSpatial_V2(nn.Module):
    def __init__(self, num_segments):
        super(HLDA_DualSpatial_V2, self).__init__()

        self.local_att = LocalAttentionModule_V3(num_segments)
        self.global_att = GlobalAttentionModule()

    def forward(self, x):
        local_att = self.local_att(x)
        global_att = self.global_att(x)
        combined_att = local_att + global_att
        return combined_att

class HLDA_V3(nn.Module):
    def __init__(self, in_channels, num_segments):
        super(HLDA_V3, self).__init__()

        self.HLDA_L2Pool = HLDA_L2Pool(in_channels)
        self.HLDA_DualSpatial_V2 = HLDA_DualSpatial_V2(num_segments)

    def forward(self, x):
        HLDA_L2Pool = self.HLDA_L2Pool(x)
        x1 = x * HLDA_L2Pool
        HLDA_DualSpatial = self.HLDA_DualSpatial_V2(x1)
        x2 = x1 * HLDA_DualSpatial
        return x2

# input_feature = torch.randn(2, 64, 256, 256)
# hlda = HLDA_V3(in_channels=64, num_segments=4)
# output_feature = hlda(input_feature)
# print(output_feature.shape)
