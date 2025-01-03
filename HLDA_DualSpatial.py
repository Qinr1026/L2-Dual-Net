#%% DualSpatial 注意力
# 这个模块只是得到了最终的注意力权重 没有与输入向量进行相乘！

import torch
import torch.nn as nn
import math
# from attention_module.HLDA_GSpatial import GlobalAttentionModule
# from attention_module.HLDA_LSpatial import LocalAttentionModule

class HLDA_DualSpatial(nn.Module):
    def __init__(self, num_segments):
        super(HLDA_DualSpatial, self).__init__()

        self.local_att = LocalAttentionModule(num_segments)
        self.global_att = GlobalAttentionModule()

    def forward(self, x):
        local_att = self.local_att(x)
        global_att = self.global_att(x)
        combined_att = local_att + global_att

        return combined_att

class LocalAttentionModule(nn.Module):
    def __init__(self, num_segments):
        #### num_segments必须是2的整倍数!!!
        super(LocalAttentionModule, self).__init__()

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

        local_concat = torch.cat(splitted_features, dim=1) # 沿通道拼接
        output = self.sigmoid((self.conv(local_concat))) # 卷积跨通道交互

        # 还原
        restored_tensors = output.chunk(self.num_segments, dim=1)
        restored_original_tensor = torch.cat(restored_tensors, dim=2).view(l2_norm.shape)
        return restored_original_tensor

class GlobalAttentionModule(nn.Module):
    def __init__(self):
        super(GlobalAttentionModule, self).__init__()

        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Sigmoid()
        )

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        mean_out = torch.mean(x, dim=1, keepdim=True)
        cat_out = torch.cat((max_out, mean_out), dim=1)
        out = self.conv(cat_out)
        return out







# input_feature = torch.randn(1, 64, 32, 32)
# cam = HLDA_DualSpatial(num_segments=4)
# output_feature = cam(input_feature)
# # print(output_feature.shape)
#
# # 打印模型的参数量
# total_params = sum(p.numel() for p in cam.parameters())
# print(f"HLDA_DualSpatial, Total parameters: {total_params}")
