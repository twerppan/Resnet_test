import torch
import torch.nn as nn
from torchvision.models import resnet50

# 定义ResNet50的三个部分（与之前相同）
class ResNet50_Part1(nn.Module):
    def __init__(self):
        super(ResNet50_Part1, self).__init__()
        self.model = resnet50(pretrained=True)
        self.part1 = nn.Sequential(
            self.model.conv1,
            self.model.bn1,
            self.model.relu,
            self.model.maxpool,
            self.model.layer1
        )

    def forward(self, x):
        return self.part1(x)

class ResNet50_Part2(nn.Module):
    def __init__(self):
        super(ResNet50_Part2, self).__init__()
        self.model = resnet50(pretrained=True)
        self.part2 = nn.Sequential(
            self.model.layer2,
            self.model.layer3
        )

    def forward(self, x):
        return self.part2(x)

class ResNet50_Part3(nn.Module):
    def __init__(self):
        super(ResNet50_Part3, self).__init__()
        self.model = resnet50(pretrained=True)
        self.part3 = nn.Sequential(
            self.model.layer4,
            self.model.avgpool,
            nn.Flatten(),
            self.model.fc
        )

    def forward(self, x):
        return self.part3(x)

# 测试函数
def test_model_parts():
    # 创建模型实例并加载保存的权重
    model_part1 = ResNet50_Part1()
    model_part1.load_state_dict(torch.load('resnet50_part1.pth'))
    model_part1.eval()

    model_part2 = ResNet50_Part2()
    model_part2.load_state_dict(torch.load('resnet50_part2.pth'))
    model_part2.eval()

    model_part3 = ResNet50_Part3()
    model_part3.load_state_dict(torch.load('resnet50_part3.pth'))
    model_part3.eval()

    # 准备测试数据
    # 假设输入是3通道的224x224图像
    test_input = torch.randn(1, 3, 224, 224)  # (batch_size, channels, height, width)

    # 测试Part1
    with torch.no_grad():
        output_part1 = model_part1(test_input)
        print("Part1 output shape:", output_part1.shape)

    # 测试Part2
    with torch.no_grad():
        output_part2 = model_part2(output_part1)
        print("Part2 output shape:", output_part2.shape)

    # 测试Part3
    with torch.no_grad():
        output_part3 = model_part3(output_part2)
        print("Part3 output shape:", output_part3.shape)
        print("Final output:", output_part3)

    print("测试完成，所有部分正常工作！")

# 运行测试
test_model_parts()