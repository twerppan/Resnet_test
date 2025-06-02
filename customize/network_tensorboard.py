import torch
import torch.nn as nn
import torch.profiler as profiler
from torch.utils.tensorboard import SummaryWriter

import torch.nn.functional as F
from tabulate import tabulate

class CustomNet(nn.Module):
    def __init__(self):
        super().__init__()
        # ====== 第一阶段: 卷积 + BN + ReLU ======
        # 输入通道 3，输出通道 16，卷积核大小 3，padding 保持尺寸不变
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        # 批归一化层，跟 conv1 输出通道一致
        self.bn1 = nn.BatchNorm2d(16)
        # 激活函数 ReLU
        self.relu1 = nn.ReLU()

        # 第二个卷积模块，保持通道数不变
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()
        # 池化层，将特征图宽高压缩一半
        self.pool = nn.MaxPool2d(2)

        # ====== 残差块: Conv-BN-ReLU -> Conv-BN + 跳跃连接 ======
        # 第一个残差分支的卷积
        self.conv3 = nn.Conv2d(16, 16, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(16)
        self.relu3 = nn.ReLU()
        # 第二个残差分支的卷积
        self.conv4 = nn.Conv2d(16, 16, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(16)

        # ====== 并行分支: 两条 1x1 卷积分支，再拼接 ======
        # 分支 1: 将通道数压减到 8
        self.br1_conv = nn.Conv2d(16, 8, 1)
        self.br1_bn = nn.BatchNorm2d(8)
        # 分支 2: 同样将通道数压减到 8
        self.br2_conv = nn.Conv2d(16, 8, 1)
        self.br2_bn = nn.BatchNorm2d(8)

        # ====== 全局池化 + 全连接分类器 ======
        # 自适应平均池化到 1x1
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        # 最终线性层: 输入维度 16（两个 8 通道拼接），输出 10 类
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        # ====== 前向传播: 第一阶段 ======
        # 卷积 -> BN -> ReLU
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        # 池化降采样
        x = self.pool(x)

        # ====== 前向传播: 残差块 ======
        # 保存跳跃连接的输入
        identity = x
        # 主分支: Conv -> BN -> ReLU
        out = self.relu3(self.bn3(self.conv3(x)))
        # 主分支: Conv -> BN
        out = self.bn4(self.conv4(out))
        # 跳跃连接后再激活
        x = nn.functional.relu(out + identity)

        # ====== 前向传播: 并行分支 ======
        # 分支 1
        b1 = self.br1_bn(self.br1_conv(x))
        # 分支 2
        b2 = self.br2_bn(self.br2_conv(x))
        # 按通道维度拼接，两分支各 8 通道，共 16 通道
        x = torch.cat([b1, b2], dim=1)

        # ====== 前向传播: 分类头 ======
        x = self.global_pool(x)              # 自适应池化
        x = torch.flatten(x, 1)             # 展平为向量
        x = self.fc(x)                      # 全连接分类
        return x

# 定义一个包含卷积、池化、全连接等层的神经网络
class ComplexResNet(nn.Module):
    def __init__(self, input_channels=3, output_classes=10):
        super(ComplexResNet, self).__init__()

        # 卷积层 1
        self.conv1 = nn.Conv2d(
            in_channels=input_channels,  # 输入通道数
            out_channels=16,  # 输出通道数
            kernel_size=3,  # 卷积核大小
            stride=1,  # 步长
            padding=1  # 填充大小
        )
        self.bn1 = nn.BatchNorm2d(16)  # 批归一化
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 卷积层 2
        self.conv2 = nn.Conv2d(
            in_channels=16,  # 输入通道数
            out_channels=32,  # 输出通道数
            kernel_size=3,  # 卷积核大小
            stride=1,  # 步长
            padding=1  # 填充大小
        )
        self.bn2 = nn.BatchNorm2d(32)  # 批归一化

        # 卷积层 3
        self.conv3 = nn.Conv2d(
            in_channels=32,  # 输入通道数
            out_channels=64,  # 输出通道数
            kernel_size=3,  # 卷积核大小
            stride=1,  # 步长
            padding=1  # 填充大小
        )
        self.bn3 = nn.BatchNorm2d(64)  # 批归一化

        # 残差连接
        self.residual1 = nn.Conv2d(16, 32, kernel_size=1, stride=1)  # 修改了步长
        self.residual2 = nn.Conv2d(32, 64, kernel_size=1, stride=1)  # 修改了步长

        # 全连接层
        self.fc1 = nn.Linear(64 * 4 * 4, 128)  # 输入特征数和输出特征数
        self.fc2 = nn.Linear(128, output_classes)

        # 激活函数
        self.relu = nn.ReLU()

    def forward(self, x):
        # 卷积层 1 + 批归一化 + 激活 + 池化
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        # 卷积层 2 + 批归一化 + 激活 + 残差连接 + 池化
        residual = self.residual1(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = x + residual  # 残差连接
        x = self.pool(x)

        # 卷积层 3 + 批归一化 + 激活 + 残差连接 + 池化
        residual = self.residual2(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = x + residual  # 残差连接
        x = self.pool(x)

        # 展平
        x = x.view(x.size(0), -1)

        # 全连接层 1 + 激活
        x = self.relu(self.fc1(x))

        # 全连接层 2（输出层，不使用激活函数）
        x = self.fc2(x)

        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomNet()
# 创建网络实例
# model = ComplexResNet(input_channels=3, output_classes=10)
model = model.to(device)
model.eval()  # 切换到评估模式
writer = SummaryWriter(log_dir='./log/logCustomizeNet2')
# 定义输入数据
input_data = torch.randn(1, 3, 32, 32).to(device)  # 1张3通道的224x224图像
with torch.no_grad():
    writer.add_graph(model, input_data)
writer.close()
# 配置并使用 Profiler
with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],  # 分析 CPU 和 CUDA 活动
    schedule=torch.profiler.schedule(
        wait=1,  # 前1步不采样
        warmup=1,  # 第2步作为热身，不计入结果
        active=3,  # 采集后面3步的性能数据
        repeat=2),  # 重复2轮
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/logCustomizeNet2'),  # 保存日志以供 TensorBoard 可视化
    record_shapes=True,  # 记录输入张量的形状
    profile_memory=True,  # 分析内存分配
    with_flops=True,
    with_stack=True  # 记录操作的调用堆栈信息
) as prof:

    for step in range(10):
        with torch.no_grad():
            output = model(input_data)
        prof.step()  # 更新 profiler 的步骤

# 打印 Profiler 生成的报告
print(prof.key_averages(group_by_input_shape=True).table(sort_by='cpu_time_total'))