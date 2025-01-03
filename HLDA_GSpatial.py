#%% 空间全局注意力
import torch
import torch.nn as nn

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
# cam = GlobalAttentionModule()
# output_feature = cam(input_feature)
# # print(output_feature.shape)
#
# # 打印模型的参数量
# total_params = sum(p.numel() for p in cam.parameters())
# print(f"GlobalAttentionModule, Total parameters: {total_params}")
