import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision import transforms
from PIL import Image

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
def test_model_with_image(image_path):
    # 创建模型实例并加载保存的权重
    model_part1 = ResNet50_Part1()
    model_part1.load_state_dict(torch.load('resnet50_part1.pth', weights_only=True))
    model_part1.eval()

    model_part2 = ResNet50_Part2()
    model_part2.load_state_dict(torch.load('resnet50_part2.pth', weights_only=True))
    model_part2.eval()

    model_part3 = ResNet50_Part3()
    model_part3.load_state_dict(torch.load('resnet50_part3.pth', weights_only=True))
    model_part3.eval()

    # 图像预处理
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 加载并预处理图像
    image = Image.open(image_path)
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # 创建一个批量维度

    # 测试Part1
    with torch.no_grad():
        output_part1 = model_part1(input_batch)
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

    # 获取预测结果
    probabilities = torch.nn.functional.softmax(output_part3[0], dim=0)
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    LABELS_PATH = 'imagenet_classes.txt'
    with open(LABELS_PATH) as f:
        labels = [line.strip() for line in f.readlines()]
    print("\nTop 5 predictions:")
    for i in range(5):
        predicted_label = labels[top5_catid[i]]
        print(f'Top {i + 1} 类别: {predicted_label}, 置信度: {top5_prob[i].item():.4f}')

    print("测试完成，所有部分正常工作！")

# 运行测试
test_model_with_image('dog.png')


