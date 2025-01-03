#%% 空间局部注意力
import torch
import torch.nn as nn
import math

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

# input_feature = torch.randn(1, 64, 32, 32)
# cam = LocalAttentionModule(num_segments=4)
# output_feature = cam(input_feature)
# # print(output_feature.shape)
#
# # 打印模型的参数量
# total_params = sum(p.numel() for p in cam.parameters())
# print(f"LocalAttentionModule, Total parameters: {total_params}")
