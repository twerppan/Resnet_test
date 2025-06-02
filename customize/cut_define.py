import torch
import torch.nn as nn
import torch.nn.functional as F

# 原始网络定义（如你所给）
class CustomNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 第一阶段
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()
        self.pool  = nn.MaxPool2d(2)
        # 残差块
        self.conv3 = nn.Conv2d(16, 16, 3, padding=1)
        self.bn3   = nn.BatchNorm2d(16)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(16, 16, 3, padding=1)
        self.bn4   = nn.BatchNorm2d(16)
        # 并行分支
        self.br1_conv = nn.Conv2d(16, 8, 1)
        self.br1_bn   = nn.BatchNorm2d(8)
        self.br2_conv = nn.Conv2d(16, 8, 1)
        self.br2_bn   = nn.BatchNorm2d(8)
        # 全局池化 + 全连接
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc          = nn.Linear(16, 10)

    def forward(self, x):
        # ---- up to residual add ----
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.pool(x)
        identity = x

        out = self.relu3(self.bn3(self.conv3(x)))
        out = self.bn4(self.conv4(out))
        x = out + identity           # ← 这里是切点：先相加，不做激活
        # ---- rest of network ----
        x = F.relu(x)                # ← 这里是第二部分的第一步
        b1 = self.br1_bn(self.br1_conv(x))
        b2 = self.br2_bn(self.br2_conv(x))
        x  = torch.cat([b1, b2], dim=1)
        x  = self.global_pool(x)
        x  = torch.flatten(x, 1)
        x  = self.fc(x)
        return x

# ============================
# 子网络 Part1：输出 residual sum
# ============================
class Part1(nn.Module):
    def __init__(self, base: CustomNet):
        super().__init__()
        # 直接引用原网络的层，权重共享
        self.conv1 = base.conv1
        self.bn1   = base.bn1
        self.relu1 = base.relu1
        self.conv2 = base.conv2
        self.bn2   = base.bn2
        self.relu2 = base.relu2
        self.pool  = base.pool
        self.conv3 = base.conv3
        self.bn3   = base.bn3
        self.relu3 = base.relu3
        self.conv4 = base.conv4
        self.bn4   = base.bn4

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.pool(x)
        identity = x
        out = self.relu3(self.bn3(self.conv3(x)))
        out = self.bn4(self.conv4(out))
        # 切点：不做 ReLU，直接返回“残差相加”的结果
        return out + identity         # 这个 tensor 将成为 Part2 的输入

# ============================
# 子网络 Part2：从 ReLU 开始到输出
# ============================
class Part2(nn.Module):
    def __init__(self, base: CustomNet):
        super().__init__()
        # 继续引用原网络的并行分支和 classifier
        self.br1_conv   = base.br1_conv
        self.br1_bn     = base.br1_bn
        self.br2_conv   = base.br2_conv
        self.br2_bn     = base.br2_bn
        self.global_pool = base.global_pool
        self.fc          = base.fc

    def forward(self, x):
        x = F.relu(x)                  # 将 Part1 的输出激活
        b1 = self.br1_bn(self.br1_conv(x))
        b2 = self.br2_bn(self.br2_conv(x))
        x  = torch.cat([b1, b2], dim=1)
        x  = self.global_pool(x)
        x  = torch.flatten(x, 1)
        x  = self.fc(x)
        return x

# ============================
# 如何使用
# ============================
# 1) 先实例化原模型，加载或训练好权重
model = CustomNet()
# model.load_state_dict(torch.load(...))

# 2) 构造两个子模块，权重共享
part1 = Part1(model)
part2 = Part2(model)

# 3) 在推理时，先跑 Part1，再跑 Part2
input = torch.randn(1,3,224,224)
z     = part1(input)    # 得到 residual sum
out   = part2(z)        # 得到最终分类结果
print(out.shape)        # torch.Size([1,10])
