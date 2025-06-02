import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from PIL import Image
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义ResNet50模型
class ResNet50Manual(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet50Manual, self).__init__()
        self.num_classes = num_classes

        # 前置层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Layer1
        self.layer1_block1_conv1 = nn.Conv2d(64, 64, kernel_size=1, bias=False)
        self.layer1_block1_bn1 = nn.BatchNorm2d(64)
        self.layer1_block1_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1_block1_bn2 = nn.BatchNorm2d(64)
        self.layer1_block1_conv3 = nn.Conv2d(64, 256, kernel_size=1, bias=False)
        self.layer1_block1_bn3 = nn.BatchNorm2d(256)
        self.layer1_block1_downsample = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256)
        )

        self.layer1_block2_conv1 = nn.Conv2d(256, 64, kernel_size=1, bias=False)
        self.layer1_block2_bn1 = nn.BatchNorm2d(64)
        self.layer1_block2_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1_block2_bn2 = nn.BatchNorm2d(64)
        self.layer1_block2_conv3 = nn.Conv2d(64, 256, kernel_size=1, bias=False)
        self.layer1_block2_bn3 = nn.BatchNorm2d(256)

        self.layer1_block3_conv1 = nn.Conv2d(256, 64, kernel_size=1, bias=False)
        self.layer1_block3_bn1 = nn.BatchNorm2d(64)
        self.layer1_block3_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1_block3_bn2 = nn.BatchNorm2d(64)
        self.layer1_block3_conv3 = nn.Conv2d(64, 256, kernel_size=1, bias=False)
        self.layer1_block3_bn3 = nn.BatchNorm2d(256)

        # Layer2
        self.layer2_block1_conv1 = nn.Conv2d(256, 128, kernel_size=1, bias=False)
        self.layer2_block1_bn1 = nn.BatchNorm2d(128)
        self.layer2_block1_conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.layer2_block1_bn2 = nn.BatchNorm2d(128)
        self.layer2_block1_conv3 = nn.Conv2d(128, 512, kernel_size=1, bias=False)
        self.layer2_block1_bn3 = nn.BatchNorm2d(512)
        self.layer2_block1_downsample = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(512)
        )

        self.layer2_block2_conv1 = nn.Conv2d(512, 128, kernel_size=1, bias=False)
        self.layer2_block2_bn1 = nn.BatchNorm2d(128)
        self.layer2_block2_conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer2_block2_bn2 = nn.BatchNorm2d(128)
        self.layer2_block2_conv3 = nn.Conv2d(128, 512, kernel_size=1, bias=False)
        self.layer2_block2_bn3 = nn.BatchNorm2d(512)

        self.layer2_block3_conv1 = nn.Conv2d(512, 128, kernel_size=1, bias=False)
        self.layer2_block3_bn1 = nn.BatchNorm2d(128)
        self.layer2_block3_conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer2_block3_bn2 = nn.BatchNorm2d(128)
        self.layer2_block3_conv3 = nn.Conv2d(128, 512, kernel_size=1, bias=False)
        self.layer2_block3_bn3 = nn.BatchNorm2d(512)

        self.layer2_block4_conv1 = nn.Conv2d(512, 128, kernel_size=1, bias=False)
        self.layer2_block4_bn1 = nn.BatchNorm2d(128)
        self.layer2_block4_conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer2_block4_bn2 = nn.BatchNorm2d(128)
        self.layer2_block4_conv3 = nn.Conv2d(128, 512, kernel_size=1, bias=False)
        self.layer2_block4_bn3 = nn.BatchNorm2d(512)

        # Layer3
        self.layer3_block1_conv1 = nn.Conv2d(512, 256, kernel_size=1, bias=False)
        self.layer3_block1_bn1 = nn.BatchNorm2d(256)
        self.layer3_block1_conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False)
        self.layer3_block1_bn2 = nn.BatchNorm2d(256)
        self.layer3_block1_conv3 = nn.Conv2d(256, 1024, kernel_size=1, bias=False)
        self.layer3_block1_bn3 = nn.BatchNorm2d(1024)
        self.layer3_block1_downsample = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(1024)
        )

        self.layer3_block2_conv1 = nn.Conv2d(1024, 256, kernel_size=1, bias=False)
        self.layer3_block2_bn1 = nn.BatchNorm2d(256)
        self.layer3_block2_conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer3_block2_bn2 = nn.BatchNorm2d(256)
        self.layer3_block2_conv3 = nn.Conv2d(256, 1024, kernel_size=1, bias=False)
        self.layer3_block2_bn3 = nn.BatchNorm2d(1024)

        self.layer3_block3_conv1 = nn.Conv2d(1024, 256, kernel_size=1, bias=False)
        self.layer3_block3_bn1 = nn.BatchNorm2d(256)
        self.layer3_block3_conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer3_block3_bn2 = nn.BatchNorm2d(256)
        self.layer3_block3_conv3 = nn.Conv2d(256, 1024, kernel_size=1, bias=False)
        self.layer3_block3_bn3 = nn.BatchNorm2d(1024)

        self.layer3_block4_conv1 = nn.Conv2d(1024, 256, kernel_size=1, bias=False)
        self.layer3_block4_bn1 = nn.BatchNorm2d(256)
        self.layer3_block4_conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer3_block4_bn2 = nn.BatchNorm2d(256)
        self.layer3_block4_conv3 = nn.Conv2d(256, 1024, kernel_size=1, bias=False)
        self.layer3_block4_bn3 = nn.BatchNorm2d(1024)

        self.layer3_block5_conv1 = nn.Conv2d(1024, 256, kernel_size=1, bias=False)
        self.layer3_block5_bn1 = nn.BatchNorm2d(256)
        self.layer3_block5_conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer3_block5_bn2 = nn.BatchNorm2d(256)
        self.layer3_block5_conv3 = nn.Conv2d(256, 1024, kernel_size=1, bias=False)
        self.layer3_block5_bn3 = nn.BatchNorm2d(1024)

        self.layer3_block6_conv1 = nn.Conv2d(1024, 256, kernel_size=1, bias=False)
        self.layer3_block6_bn1 = nn.BatchNorm2d(256)
        self.layer3_block6_conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer3_block6_bn2 = nn.BatchNorm2d(256)
        self.layer3_block6_conv3 = nn.Conv2d(256, 1024, kernel_size=1, bias=False)
        self.layer3_block6_bn3 = nn.BatchNorm2d(1024)

        # Layer4
        self.layer4_block1_conv1 = nn.Conv2d(1024, 512, kernel_size=1, bias=False)
        self.layer4_block1_bn1 = nn.BatchNorm2d(512)
        self.layer4_block1_conv2 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.layer4_block1_bn2 = nn.BatchNorm2d(512)
        self.layer4_block1_conv3 = nn.Conv2d(512, 2048, kernel_size=1, bias=False)
        self.layer4_block1_bn3 = nn.BatchNorm2d(2048)
        self.layer4_block1_downsample = nn.Sequential(
            nn.Conv2d(1024, 2048, kernel_size=1, stride=2, bias=False),
            nn.BatchNorm2d(2048)
        )

        self.layer4_block2_conv1 = nn.Conv2d(2048, 512, kernel_size=1, bias=False)
        self.layer4_block2_bn1 = nn.BatchNorm2d(512)
        self.layer4_block2_conv2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer4_block2_bn2 = nn.BatchNorm2d(512)
        self.layer4_block2_conv3 = nn.Conv2d(512, 2048, kernel_size=1, bias=False)
        self.layer4_block2_bn3 = nn.BatchNorm2d(2048)

        self.layer4_block3_conv1 = nn.Conv2d(2048, 512, kernel_size=1, bias=False)
        self.layer4_block3_bn1 = nn.BatchNorm2d(512)
        self.layer4_block3_conv2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer4_block3_bn2 = nn.BatchNorm2d(512)
        self.layer4_block3_conv3 = nn.Conv2d(512, 2048, kernel_size=1, bias=False)
        self.layer4_block3_bn3 = nn.BatchNorm2d(2048)

        # 分类层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        # 前置层
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Layer1
        # Block1
        identity = x
        out = self.layer1_block1_conv1(x)
        out = self.layer1_block1_bn1(out)
        out = self.relu(out)
        out = self.layer1_block1_conv2(out)
        out = self.layer1_block1_bn2(out)
        out = self.relu(out)
        out = self.layer1_block1_conv3(out)
        out = self.layer1_block1_bn3(out)
        identity = self.layer1_block1_downsample(x)
        out += identity
        x = self.relu(out)

        # Block2
        identity = x
        out = self.layer1_block2_conv1(x)
        out = self.layer1_block2_bn1(out)
        out = self.relu(out)
        out = self.layer1_block2_conv2(out)
        out = self.layer1_block2_bn2(out)
        out = self.relu(out)
        out = self.layer1_block2_conv3(out)
        out = self.layer1_block2_bn3(out)
        out += identity
        x = self.relu(out)

        # Block3
        identity = x
        out = self.layer1_block3_conv1(x)
        out = self.layer1_block3_bn1(out)
        out = self.relu(out)
        out = self.layer1_block3_conv2(out)
        out = self.layer1_block3_bn2(out)
        out = self.relu(out)
        out = self.layer1_block3_conv3(out)
        out = self.layer1_block3_bn3(out)
        out += identity
        x = self.relu(out)

        # Layer2
        # Block1
        identity = x
        out = self.layer2_block1_conv1(x)
        out = self.layer2_block1_bn1(out)
        out = self.relu(out)
        out = self.layer2_block1_conv2(out)
        out = self.layer2_block1_bn2(out)
        out = self.relu(out)
        out = self.layer2_block1_conv3(out)
        out = self.layer2_block1_bn3(out)
        identity = self.layer2_block1_downsample(x)
        out += identity
        x = self.relu(out)

        # Block2
        identity = x
        out = self.layer2_block2_conv1(x)
        out = self.layer2_block2_bn1(out)
        out = self.relu(out)
        out = self.layer2_block2_conv2(out)
        out = self.layer2_block2_bn2(out)
        out = self.relu(out)
        out = self.layer2_block2_conv3(out)
        out = self.layer2_block2_bn3(out)
        out += identity
        x = self.relu(out)

        # Block3
        identity = x
        out = self.layer2_block3_conv1(x)
        out = self.layer2_block3_bn1(out)
        out = self.relu(out)
        out = self.layer2_block3_conv2(out)
        out = self.layer2_block3_bn2(out)
        out = self.relu(out)
        out = self.layer2_block3_conv3(out)
        out = self.layer2_block3_bn3(out)
        out += identity
        x = self.relu(out)

        # Block4
        identity = x
        out = self.layer2_block4_conv1(x)
        out = self.layer2_block4_bn1(out)
        out = self.relu(out)
        out = self.layer2_block4_conv2(out)
        out = self.layer2_block4_bn2(out)
        out = self.relu(out)
        out = self.layer2_block4_conv3(out)
        out = self.layer2_block4_bn3(out)
        out += identity
        x = self.relu(out)

        # Layer3
        # Block1
        identity = x
        out = self.layer3_block1_conv1(x)
        out = self.layer3_block1_bn1(out)
        out = self.relu(out)
        out = self.layer3_block1_conv2(out)
        out = self.layer3_block1_bn2(out)
        out = self.relu(out)
        out = self.layer3_block1_conv3(out)
        out = self.layer3_block1_bn3(out)
        identity = self.layer3_block1_downsample(x)
        out += identity
        x = self.relu(out)

        # Block2
        identity = x
        out = self.layer3_block2_conv1(x)
        out = self.layer3_block2_bn1(out)
        out = self.relu(out)
        out = self.layer3_block2_conv2(out)
        out = self.layer3_block2_bn2(out)
        out = self.relu(out)
        out = self.layer3_block2_conv3(out)
        out = self.layer3_block2_bn3(out)
        out += identity
        x = self.relu(out)

        # Block3
        identity = x
        out = self.layer3_block3_conv1(x)
        out = self.layer3_block3_bn1(out)
        out = self.relu(out)
        out = self.layer3_block3_conv2(out)
        out = self.layer3_block3_bn2(out)
        out = self.relu(out)
        out = self.layer3_block3_conv3(out)
        out = self.layer3_block3_bn3(out)
        out += identity
        x = self.relu(out)

        # Block4
        identity = x
        out = self.layer3_block4_conv1(x)
        out = self.layer3_block4_bn1(out)
        out = self.relu(out)
        out = self.layer3_block4_conv2(out)
        out = self.layer3_block4_bn2(out)
        out = self.relu(out)
        out = self.layer3_block4_conv3(out)
        out = self.layer3_block4_bn3(out)
        out += identity
        x = self.relu(out)

        # Block5
        identity = x
        out = self.layer3_block5_conv1(x)
        out = self.layer3_block5_bn1(out)
        out = self.relu(out)
        out = self.layer3_block5_conv2(out)
        out = self.layer3_block5_bn2(out)
        out = self.relu(out)
        out = self.layer3_block5_conv3(out)
        out = self.layer3_block5_bn3(out)
        out += identity
        x = self.relu(out)

        # Block6
        identity = x
        out = self.layer3_block6_conv1(x)
        out = self.layer3_block6_bn1(out)
        out = self.relu(out)
        out = self.layer3_block6_conv2(out)
        out = self.layer3_block6_bn2(out)
        out = self.relu(out)
        out = self.layer3_block6_conv3(out)
        out = self.layer3_block6_bn3(out)
        out += identity
        x = self.relu(out)

        # Layer4
        # Block1
        identity = x
        out = self.layer4_block1_conv1(x)
        out = self.layer4_block1_bn1(out)
        out = self.relu(out)
        out = self.layer4_block1_conv2(out)
        out = self.layer4_block1_bn2(out)
        out = self.relu(out)
        out = self.layer4_block1_conv3(out)
        out = self.layer4_block1_bn3(out)
        identity = self.layer4_block1_downsample(x)
        out += identity
        x = self.relu(out)

        # Block2
        identity = x
        out = self.layer4_block2_conv1(x)
        out = self.layer4_block2_bn1(out)
        out = self.relu(out)
        out = self.layer4_block2_conv2(out)
        out = self.layer4_block2_bn2(out)
        out = self.relu(out)
        out = self.layer4_block2_conv3(out)
        out = self.layer4_block2_bn3(out)
        out += identity
        x = self.relu(out)

        # Block3
        identity = x
        out = self.layer4_block3_conv1(x)
        out = self.layer4_block3_bn1(out)
        out = self.relu(out)
        out = self.layer4_block3_conv2(out)
        out = self.layer4_block3_bn2(out)
        out = self.relu(out)
        out = self.layer4_block3_conv3(out)
        out = self.layer4_block3_bn3(out)
        out += identity
        x = self.relu(out)

        # 分类层
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

# 定义处理函数
def process_case(model, input_tensor, cut_layer_name, paste_layer_name):
    # 将模型和输入张量都移动到同一个设备上
    model = model.to(device)
    input_tensor = input_tensor.to(device)

    # 获取模型的所有模块及其名称
    module_names = []
    for name, module in model.named_modules():
        module_names.append(name)

    print(f'模型的所有模块名称: {module_names}')

    # 找到切割点和拼接点的索引
    try:
        cut_index = module_names.index(cut_layer_name)
        paste_index = module_names.index(paste_layer_name)
    except ValueError:
        print(f"Error: Layer {cut_layer_name} or {paste_layer_name} not found")
        return None

    # 获取切割点到拼接点之间的层
    layers_between = module_names[cut_index+1:paste_index+1]

    print(f'layers_between: {layers_between}')

    # 创建一个字典来存储层
    layer_dict = {}
    for name in layers_between:
        layer = model
        attrs = name.split('.')
        for attr in attrs:
            if hasattr(layer, attr):
                layer = getattr(layer, attr)
            else:
                break
        else:  # 如果循环正常结束（没有break）
            if layer is not None:
                layer_dict[name] = layer.to(device)

    # 检查layer_dict是否包含所有需要的层
    for name in layers_between:
        if name not in layer_dict:
            print(f"Warning: Layer {name} not found in model")
            return None

    # 前向传播到切割点
    x = input_tensor
    for name in module_names[:cut_index+1]:
        layer = model
        attrs = name.split('.')
        for attr in attrs:
            if hasattr(layer, attr):
                layer = getattr(layer, attr)
            else:
                break
        else:  # 如果循环正常结束（没有break）
            if layer is not None:
                x = layer(x)
            else:
                print(f"Error during forward: Layer {name} not found")
                return None
    print(x.shape)
    # 切割张量为四份
    b, c, h, w = x.shape
    h_half, w_half = h // 2, w // 2
    patches = [
        x[:, :, :h_half, :w_half],    # 左上
        x[:, :, :h_half, w_half:],    # 右上
        x[:, :, h_half:, :w_half],    # 左下
        x[:, :, h_half:, w_half:]     # 右下
    ]
    for x in patches:
        print(x.shape)
    # 对每个子张量进行处理
    processed_patches = []
    with torch.no_grad():
        for patch in patches:

            out = patch
            for name in layers_between:
                layer = layer_dict[name]
                out = layer(out)
                # 检查输出张量的形状
                print(f"Layer {name} output shape: {out.shape}")

            processed_patches.append(out)
    for x in processed_patches:
        print(x.shape)
    # 拼接处理后的子张量
    b, c, h_f, w_f = processed_patches[0].shape
    full_feat = torch.zeros((b, c, h_f * 2, w_f * 2), device=device)
    full_feat[:, :, :h_f, :w_f] = processed_patches[0]  # 左上
    full_feat[:, :, :h_f, w_f:] = processed_patches[1]  # 右上
    full_feat[:, :, h_f:, :w_f] = processed_patches[2]  # 左下
    full_feat[:, :, h_f:, w_f:] = processed_patches[3]  # 右下

    # 继续前向传播到模型输出
    for name in module_names[paste_index+1:]:
        layer = model
        attrs = name.split('.')
        for attr in attrs:
            if hasattr(layer, attr):
                layer = getattr(layer, attr)
            else:
                break
        else:  # 如果循环正常结束（没有break）
            if layer is not None:
                full_feat = layer(full_feat)
                # 检查输出张量的形状
                print(f"Layer {name} output shape: {full_feat.shape}")

    return full_feat

# 测试代码
if __name__ == "__main__":
    model = ResNet50Manual()
    model.eval()

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
    input_tensor = input_tensor.unsqueeze(0)

    # 定义测试案例
    cases = [
        {'cut_layer': 'layer1_block1_conv3', 'paste_layer': 'layer2_block2_conv3'},
        # 更多案例...
    ]

    for case in cases:
        output = process_case(model, input_tensor, case['cut_layer'], case['paste_layer'])
        if isinstance(output, torch.Tensor):
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            top5_prob, top5_catid = torch.topk(probabilities, 5)
            print(f"Case: cut_layer={case['cut_layer']}, paste_layer={case['paste_layer']}")
            for i in range(5):
                print(f"Class: {top5_catid[i]}, Probability: {top5_prob[i].item():.4f}")
        else:
            print(f"Case: cut_layer={case['cut_layer']}, paste_layer={case['paste_layer']} failed")

