import os
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

# 配置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载预训练的AlexNet模型
alexnet = models.alexnet(pretrained=True).to(device)
alexnet.eval()

# 图像预处理Transform
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# 加载并预处理图像
img_path = os.path.join('dog.png')       # 图像文件路径
img = Image.open(img_path).convert('RGB')
input_tensor = preprocess(img)            # 预处理后形状 [3, 224, 224]
input_tensor = input_tensor.unsqueeze(0).to(device)
# 切割为四个相等子张量
x, _, H, W = input_tensor.shape
h_half, w_half = H // 2, W // 2
patches = [
    input_tensor[:, :, :h_half + 8, :w_half + 8],  # 左上
    input_tensor[:, :, :h_half + 8, w_half - 16:],  # 右上
    input_tensor[:, :, h_half - 16:, :w_half + 8],  # 左下
    input_tensor[:, :, h_half - 16:, w_half - 16:],  # 右下
]
print(patches[0].shape)
print(patches[1].shape)
print(patches[2].shape)
print(patches[3].shape)
# 提取features部分直到AdaptiveAvgPool2d之前
feats = []
with torch.no_grad():
    for patch in patches:
        for i in range(6):  # 应用feature[3], feature[4], feature[5], feature[6]
            patch = alexnet.features[i](patch)
        feats.append(patch)
print(feats[0].shape)
print(feats[1].shape)
print(feats[2].shape)
print(feats[3].shape)
# 合并四个子特征图
# 假设所有feat shape相同
b, c, h_f, w_f = feats[0].shape
# 创建空Tensor用于拼接
full_feat = torch.zeros((b, c, h_f * 2+1, w_f * 2+1), device=device)
# 按位置拼接
full_feat[:, :, :h_f, :w_f] = feats[0]   # 左上
full_feat[:, :, :h_f, w_f:] = feats[1]   # 右上
full_feat[:, :, h_f:, :w_f] = feats[2]   # 左下
full_feat[:, :, h_f:, w_f:] = feats[3]   # 右下

# 通过AdaptiveAvgPool2d和classifier
with torch.no_grad():
    for i in range(6, 13):  # 应用feature[7]到feature[12]
        full_feat = alexnet.features[i](full_feat)
    pooled = alexnet.avgpool(full_feat)   # -> [1, 256, 6, 6]
    flatten = torch.flatten(pooled, 1)    # -> [1, 256*6*6]
    outputs = alexnet.classifier(flatten)


probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

# 获取Top 5预测类别及置信度
top5_prob, top5_catid = torch.topk(probabilities, 5)

# 加载imagenet_classes.txt文件
# 这里假设你已经下载了ImageNet的标签文件imagenet_classes.txt
LABELS_PATH = 'imagenet_classes.txt'

with open(LABELS_PATH) as f:
    labels = [line.strip() for line in f.readlines()]

# 输出Top 5类别和对应的置信度
for i in range(5):
    predicted_label = labels[top5_catid[i]]
    print(f'Top {i+1} 类别: {predicted_label}, 置信度: {top5_prob[i].item():.4f}')

