import torch
import torch.nn as nn
from torchvision.models import resnet50

# 定义ResNet50的三个部分
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

# 保存模型的三个部分
model_part1 = ResNet50_Part1()
torch.save(model_part1.state_dict(), 'resnet50_part1.pth')

model_part2 = ResNet50_Part2()
torch.save(model_part2.state_dict(), 'resnet50_part2.pth')

model_part3 = ResNet50_Part3()
torch.save(model_part3.state_dict(), 'resnet50_part3.pth')