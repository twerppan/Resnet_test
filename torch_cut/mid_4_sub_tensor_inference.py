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
img_path = os.path.join('dog.png')  # 图像文件路径
img = Image.open(img_path).convert('RGB')
input_tensor = preprocess(img)      # 预处理后形状 [3, 224, 224]
input_tensor = input_tensor.unsqueeze(0).to(device)


with torch.no_grad():
    x = input_tensor
    for i in range(6):  # 应用feature[0], feature[1], feature[2]
        x = alexnet.features[i](x)
    feat_after_feature2 = x

# 切割张量为四份
b, c, h, w = feat_after_feature2.shape
h_half, w_half = h // 2, w // 2
patches = [
    feat_after_feature2[:, :, :h_half+2, :w_half+2],    # 左上
    feat_after_feature2[:, :, :h_half+2, w_half-2:],    # 右上
    feat_after_feature2[:, :, h_half-2:, :w_half+2],    # 左下
    feat_after_feature2[:, :, h_half-2:, w_half-2:],    # 右下
]
# print(patches[0].shape)
# print(patches[1].shape)
# print(patches[2].shape)
# print(patches[3].shape)
# 对四份子张量分别应用feature[3]到feature[6]
processed_patches = []
with torch.no_grad():
    for patch in patches:
        for i in range(6, 13):  # 应用feature[3], feature[4], feature[5], feature[6]
            patch = alexnet.features[i](patch)
        processed_patches.append(patch)

# print(processed_patches[0].shape)
# print(processed_patches[1].shape)
# print(processed_patches[2].shape)
# print(processed_patches[3].shape)
# 拼接处理后的子张量
b, c, h_f, w_f = processed_patches[0].shape
full_feat = torch.zeros((b, c, h_f * 2+1, w_f * 2+1), device=device)
full_feat[:, :, :h_f, :w_f] = processed_patches[0]  # 左上
full_feat[:, :, :h_f, w_f:] = processed_patches[1]  # 右上
full_feat[:, :, h_f:, :w_f] = processed_patches[2]  # 左下
full_feat[:, :, h_f:, w_f:] = processed_patches[3]  # 右下
# print(full_feat.shape)
# 通过AdaptiveAvgPool2d和classifier
with torch.no_grad():
    # for i in range(7, 13):  #
    #     full_feat = alexnet.features[i](full_feat)
    pooled = alexnet.avgpool(full_feat)
    flatten = torch.flatten(pooled, 1)
    outputs = alexnet.classifier(flatten)

# 获取预测结果
probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
top5_prob, top5_catid = torch.topk(probabilities, 5)

# 加载imagenet_classes.txt文件
LABELS_PATH = 'imagenet_classes.txt'
with open(LABELS_PATH) as f:
    labels = [line.strip() for line in f.readlines()]

# 输出Top 5类别和对应的置信度
for i in range(5):
    predicted_label = labels[top5_catid[i]]
    print(f'Top {i+1} 类别: {predicted_label}, 置信度: {top5_prob[i].item():.4f}')