import torch.nn.functional as F
import netron
from tabulate import tabulate


# 定义一个包含卷积、池化、全连接等层的神经网络
import torch
from torch import nn
from torch.fx import symbolic_trace
import networkx as nx

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
class CustomNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Initial conv-bn-relu sequence
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()
        self.pool = nn.MaxPool2d(2)

        # Residual block
        self.conv3 = nn.Conv2d(16, 16, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(16)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(16, 16, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(16)

        # Parallel branches
        self.br1_conv = nn.Conv2d(16, 8, 1)
        self.br1_bn = nn.BatchNorm2d(8)
        self.br2_conv = nn.Conv2d(16, 8, 1)
        self.br2_bn = nn.BatchNorm2d(8)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        # Stage 1
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.pool(x)

        # Residual connection
        identity = x
        out = self.relu3(self.bn3(self.conv3(x)))
        out = self.bn4(self.conv4(out))
        x = nn.functional.relu(out + identity)

        # Parallel branches
        b1 = self.br1_bn(self.br1_conv(x))
        b2 = self.br2_bn(self.br2_conv(x))
        x = torch.cat([b1, b2], dim=1)

        # Final classifier
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def get_layer_info(model, input_shape):
    layer_info = []
    dummy_input = torch.randn(1, *input_shape)
    hooks = []

    # 定义一个类来管理层信息
    class LayerInfo:
        def __init__(self):
            self.index = 1  # 用于标记层的原始顺序

        def hook(self, module, input, output):
            # 计算输出数据量
            if len(output.shape) == 4:  # 卷积层或池化层
                data_volume = output.shape[1] * output.shape[2] * output.shape[3]
            else:  # 全连接层
                data_volume = output.shape[1]

            layer_info.append({
                'original_index': self.index,
                'name': module.__class__.__name__,
                'output_shape': output.shape,
                'data_volume': data_volume
            })

            self.index += 1  # 每处理一层，索引加1

    layer_manager = LayerInfo()

    # 注册 hook
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.MaxPool2d, nn.BatchNorm2d)):
            hooks.append(module.register_forward_hook(layer_manager.hook))

    model(dummy_input)
    for hook in hooks:
        hook.remove()

    return layer_info
# 创建网络实例
model = ComplexResNet(input_channels=3, output_classes=10)

# model = CustomNet()
input = torch.ones((1, 3, 32, 32))

torch.onnx.export(model, input, f='CustomizeNet.onnx')  # 导出 .onnx 文件
netron.start('CustomizeNet.onnx')  # 展示结构图

layer_info = get_layer_info(model, (3, 32, 32))

# 按输出数据量升序排序
sorted_layers = sorted(layer_info, key=lambda x: x['data_volume'])

# 打印表格
table = []
for idx, layer in enumerate(sorted_layers):
    table.append([
        idx + 1,
        layer['original_index'],
        layer['name'],
        f"{layer['output_shape']}",
        f"{layer['data_volume']}"
    ])

headers = ["Rank", "Original Index", "Layer Type", "Output Shape", "Data Volume"]
print(tabulate(table, headers=headers, tablefmt="grid"))