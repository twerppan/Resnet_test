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

# 定义保存预测结果的文件路径
predict_txt = r"head0_6_cut_predict_16pad.txt"

# 初始化预测文件
with open(predict_txt, 'w') as f:
    pass  # 创建文件

# 遍历目录中的所有图片文件
images_dir = r"D:\BaiduNetdiskDownload\ILSVRC2012_img_val"
image_files = [f for f in os.listdir(images_dir) if f.endswith(".JPEG")]
total_images = len(image_files)

for img_file in image_files:
    img_path = os.path.join(images_dir, img_file)
    img = Image.open(img_path).convert('RGB')
    input_tensor = preprocess(img)      # 预处理后形状 [3, 224, 224]
    input_tensor = input_tensor.unsqueeze(0).to(device)


    # 切割张量为四份
    b, c, h, w = input_tensor.shape
    h_half, w_half = h // 2, w // 2
    patches = [
        input_tensor[:, :, :h_half + 16, :w_half + 16],  # 左上
        input_tensor[:, :, :h_half + 16, w_half - 16:],  # 右上
        input_tensor[:, :, h_half - 16:, :w_half + 16],  # 左下
        input_tensor[:, :, h_half - 16:, w_half - 16:],  # 右下
    ]

    # 对四份子张量分别应用feature[3]到feature[6]
    processed_patches = []
    with torch.no_grad():
        for patch in patches:
            for i in range(6):  # 应用feature[3], feature[4], feature[5], feature[6]
                patch = alexnet.features[i](patch)
            # feat = alexnet.features(x)  # 输出形状 [1, 256, h_f, w_f]
            processed_patches.append(patch)

    # 拼接处理后的子张量
    b, c, h_f, w_f = processed_patches[0].shape
    full_feat = torch.zeros((b, c, h_f * 2, w_f * 2), device=device)
    full_feat[:, :, :h_f, :w_f] = processed_patches[0]  # 左上
    full_feat[:, :, :h_f, w_f:] = processed_patches[1]  # 右上
    full_feat[:, :, h_f:, :w_f] = processed_patches[2]  # 左下
    full_feat[:, :, h_f:, w_f:] = processed_patches[3]  # 右下

    # 通过AdaptiveAvgPool2d和classifier
    with torch.no_grad():
        for i in range(6, 13):  # 应用feature[7]到feature[12]
            full_feat = alexnet.features[i](full_feat)
        pooled = alexnet.avgpool(full_feat)
        flatten = torch.flatten(pooled, 1)
        outputs = alexnet.classifier(flatten)

    # 获取预测结果
    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    predicted_class = torch.argmax(probabilities).item()

    # 写入预测结果到文件并打印到控制台
    img_name = img_file.split('.')[0]
    with open(predict_txt, 'a') as f:
        f.write(f"{img_name}: {predicted_class}\n")
    print(f"图像: {img_name}, 预测类别: {predicted_class}")

print("所有图像的预测结果已写入到 head_cut_predict.txt")