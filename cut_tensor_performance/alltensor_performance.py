import os
from PIL import Image
import torch
import torchvision
from torchvision import transforms
import torch.nn.functional as F

# 设置文件路径
images_dir = r"D:\BaiduNetdiskDownload\ILSVRC2012_img_val"
predict_txt = r"predict.txt"

# 数据预处理
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载预训练的AlexNet模型
model = torchvision.models.alexnet(pretrained=True)
model.eval()  # 设置模型为评估模式

# 确保使用GPU（如果有可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 遍历目录中的所有图片文件
image_files = [f for f in os.listdir(images_dir) if f.endswith(".JPEG")]
total_images = len(image_files)

# 初始化预测文件
with open(predict_txt, 'w') as pf:
    pass  # 创建文件

# 进行预测
with open(predict_txt, 'a') as pf:
    for img_file in image_files:
        img_path = os.path.join(images_dir, img_file)
        img = Image.open(img_path).convert('RGB')

        # 预处理图像
        input_tensor = preprocess(img)
        input_batch = input_tensor.unsqueeze(0).to(device)

        # 不计算梯度进行推理
        with torch.no_grad():
            output = model(input_batch)

        # 获取预测类别
        probabilities = F.softmax(output[0], dim=0)
        predicted_class = torch.argmax(probabilities).item()

        # 写入预测结果
        img_name = img_file.split('.')[0]
        pf.write(f"{img_name}: {predicted_class}\n")

        # 打印到控制台
        print(f"图像: {img_name}, 预测类别: {predicted_class}")

print("所有图像的预测结果已写入到 predict.txt")