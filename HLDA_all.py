#%% 这个脚本将所有的HLDA模块算法进行汇总

import torch
import torch.nn as nn
import math
from thop import profile, clever_format

#%% device 设置
print(torch.cuda.is_available())
# 指定使用GPU的块数
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#%% 带有 L2-NORM 的通道注意力模块
class HLDA_L2Pool(nn.Module):
    def __init__(self, in_channels, gamma=2, b=1):
        super(HLDA_L2Pool, self).__init__()
        self.in_channels = in_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 平均池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 最大池化
        kernel_size = int(abs((math.log(self.in_channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg_pool = self.avg_pool(x)
        max_pool = self.max_pool(x)
        # 计算L2范数
        l2_norm = torch.norm(x, p=2, dim=(2, 3), keepdim=True)

        # 变为 [1, 1, channels] 方便进行1d卷积
        avg_out = self.conv(avg_pool.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        max_out = self.conv(max_pool.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        l2_out = self.conv(l2_norm.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # print(avg_out.shape)

        # channel_attention = torch.sigmoid(avg_out + max_out).unsqueeze(2).unsqueeze(3)
        channel_attention = self.sigmoid(avg_out + max_out + l2_out)
        # channel_attention = self.sigmoid(avg_out + max_out )
        return channel_attention

#%% 带有局部空间交互的空间注意力模块
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

#%% 具有最大池化和平均池化的空间注意力模块
class GlobalAttentionModule(nn.Module):
    def __init__(self, kernel_size = 3):
        super(GlobalAttentionModule, self).__init__()

        assert kernel_size in (3, 7), "Kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=padding)
        )

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        mean_out = torch.mean(x, dim=1, keepdim=True)
        cat_out = torch.cat((max_out, mean_out), dim=1)

        out = self.sigmoid(self.conv(cat_out))
        return out

#%%  带有子空间和全局空间的空间注意力模块
class HLDA_DualSpatial(nn.Module):
    def __init__(self, num_segments):
        super(HLDA_DualSpatial, self).__init__()

        self.local_att = LocalAttentionModule(num_segments)
        self.global_att = GlobalAttentionModule()

    def forward(self, x):
        local_att = self.local_att(x)
        global_att = self.global_att(x)
        combined_att = local_att + global_att
        combined_att = global_att

        return combined_att

#%% HLDA 注意力模块
class HLDA(nn.Module):
    def __init__(self, in_channels, num_segments, type):
        super(HLDA, self).__init__()

        self.HLDA_DualSpatial = HLDA_DualSpatial(num_segments)
        self.HLDA_L2Pool = HLDA_L2Pool(in_channels)

        self.type = type

    def forward(self, x):
        if self.type == 'channel_first':
            HLDA_L2Pool = self.HLDA_L2Pool(x)
            x1 = x * HLDA_L2Pool   # 先通道注意力
            HLDA_DualSpatial = self.HLDA_DualSpatial(x1)
            x2 = x1 * HLDA_DualSpatial   # 后空间注意力

        elif self.type == 'spatial_first':
            HLDA_DualSpatial = self.HLDA_DualSpatial(x)
            x1 = x * HLDA_DualSpatial  # 先空间注意力
            HLDA_L2Pool = self.HLDA_L2Pool(x1)
            x2 = x1 * HLDA_L2Pool  # 先空间注意力

        elif self.type == 'parallel':
            HLDA_L2Pool = self.HLDA_L2Pool(x)
            x_1 = x * HLDA_L2Pool  # 通道注意力并行计算
            HLDA_DualSpatial = self.HLDA_DualSpatial(x)
            x_2 = x * HLDA_DualSpatial  # 空间注意力并行计算
            x2 = x_1 + x_2  # 并行计算

        else:
            raise ValueError("Invalid type. Supported types: 'channel_first' or 'spatial_first' or 'parallel'.")

        return x2

#%% 打印 HLDA 注意力模块的参数量和 FLOPS
batch = 16
in_channels = 64
size_W = 256
size_H = 256

num_segments = 16
type = 'spatial_first'

input_tensor = torch.randn(batch, in_channels, size_W, size_H).to(device)
model = HLDA(in_channels, num_segments, type).to(device)

print('================================')
try:
    # 估计模型的 FLOPs
    macs, params = profile(model, inputs=(input_tensor.to(device),))
    macs, params = clever_format([macs, params], "%.3f")
    print(type)
    print(macs, params)
    # 打印估计的 FLOPs
    #     print("Estimated FLOPs:", macs)
    #
    #     # 将 FLOPs 格式化为易读的字符串
    #     macs, _ = clever_format([macs], "%.3f")
    #     print("Formatted FLOPs:", macs)
except Exception as e:
    print(f"Error: {e}")
print('================================')