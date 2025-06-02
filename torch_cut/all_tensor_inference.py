import torch
from torchvision import models, transforms
from PIL import Image

# 加载预训练的AlexNet模型
model = models.alexnet(pretrained=True)
model.eval()

# 定义预处理步骤，包含尺寸调整、归一化等
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载图片
img_path = 'dog.png'
img = Image.open(img_path)

# 对图片进行预处理
input_tensor = preprocess(img)
input_batch = input_tensor.unsqueeze(0)  # 为批量大小增加一个维度

# 检查是否有可用的GPU，如果有，使用GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
input_batch = input_batch.to(device)

# 进行预测
with torch.no_grad():
    output = model(input_batch)

# 获取分类结果的概率
probabilities = torch.nn.functional.softmax(output[0], dim=0)

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
