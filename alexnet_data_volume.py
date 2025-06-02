import torch
import torch.nn as nn
from tabulate import tabulate
from torchvision.models import alexnet


# class AlexNet(nn.Module):
#     def __init__(self, num_classes=1000):
#         super(AlexNet, self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#             nn.Conv2d(64, 192, kernel_size=5, padding=2),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#             nn.Conv2d(192, 384, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(384, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#         )
#         self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
#         self.classifier = nn.Sequential(
#             nn.Dropout(),
#             nn.Linear(256 * 6 * 6, 4096),
#             nn.ReLU(inplace=True),
#             nn.Dropout(),
#             nn.Linear(4096, 4096),
#             nn.ReLU(inplace=True),
#             nn.Linear(4096, num_classes),
#         )
#
#     def forward(self, x):
#         x = self.features(x)
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.classifier(x)
#         return x


# 创建模型实例
model = alexnet()


# 获取每层的输入输出张量形状和数据量大小
def get_layer_info(model, input_shape):
    layer_info = []
    dummy_input = torch.randn(1, *input_shape)
    hooks = []
    layer_index = 1  # 用于标记层的原始顺序

    def hook(module, input, output):
        nonlocal layer_index  # 使用nonlocal关键字来修改外层函数中的变量

        # 计算输出数据量
        if len(output.shape) == 4:  # 卷积层或池化层
            data_volume = output.shape[1] * output.shape[2] * output.shape[3]
        else:  # 全连接层
            data_volume = output.shape[1]

        layer_info.append({
            'original_index': layer_index,
            'name': module.__class__.__name__,
            'output_shape': output.shape,
            'data_volume': data_volume
        })

        layer_index += 1  # 每处理一层，索引加1

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.MaxPool2d, nn.ReLU, nn.AdaptiveAvgPool2d)):
            hooks.append(module.register_forward_hook(hook))

    model(dummy_input)
    for hook in hooks:
        hook.remove()

    return layer_info


layer_info = get_layer_info(model, (3, 224, 224))

# 按输出数据量升序排序
# sorted_layers = sorted(layer_info, key=lambda x: x['data_volume'])

# 打印表格
table = []
for idx, layer in enumerate(layer_info):
    table.append([
        idx + 1,
        layer['original_index'],
        layer['name'],
        f"{layer['output_shape']}",
        f"{layer['data_volume']}"
    ])

headers = ["Rank", "Original Index", "Layer Type", "Output Shape", "Data Volume"]
print(tabulate(table, headers=headers, tablefmt="grid"))