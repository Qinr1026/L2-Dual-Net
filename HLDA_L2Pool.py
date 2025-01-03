#%% 构建 Hybrid HLDA_L2Pool attention module
# 这个模块只是得到了最终的注意力权重 没有与输入向量进行相乘！


import torch
import torch.nn as nn

class HLDA_L2Pool(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(HLDA_L2Pool, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 平均池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 最大池化

        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )    # MLP

    def forward(self, x):
        avg_pool = self.avg_pool(x)
        max_pool = self.max_pool(x)

        avg_pool = avg_pool.view(avg_pool.size(0), -1)
        max_pool = max_pool.view(max_pool.size(0), -1)

        # 计算L2范数
        l2_norm = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        l2_norm = l2_norm.view(l2_norm.size(0), -1)

        avg_out = self.fc(avg_pool)
        max_out = self.fc(max_pool)
        l2_out = self.fc(l2_norm)

        # channel_attention = torch.sigmoid(avg_out + max_out).unsqueeze(2).unsqueeze(3)
        channel_attention = torch.sigmoid(avg_out + max_out + l2_out).unsqueeze(2).unsqueeze(3)
        return channel_attention

# # 示例用法
# # # 输入特征大小为 [batch_size, in_channels, width, height]
# input_feature = torch.randn(1, 64, 256, 256)
# #
# # # 创建并应用ChannelAttentionModule
# cam = HLDA_L2Pool(in_channels=64)
# output_feature = cam(input_feature)
# print(output_feature.shape)
# #
# # # 打印模型的参数量
# total_params = sum(p.numel() for p in cam.parameters())
# print(f"HLDA_L2Pool, Total parameters: {total_params}")

