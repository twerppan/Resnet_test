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
        # Layer1/block1
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
        # Layer1/block2
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
        # Layer1 # Block1
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    input_tensor = input_tensor.to(device)

    # 使用 FX 跟踪模型
    traced_model = torch.fx.symbolic_trace(model)
    print(f'traced_model: {traced_model}')
    graph = traced_model.graph
    print(f'graph:\n{graph}')
    graph.lint()  # 验证图的合法性

    # 构建层字典（处理所有节点类型）
    layer_dict = {}
    valid_layers = []
    print(f'traced_model:{traced_model}')
    for node in graph.nodes:
        try:
            if node.op == 'call_module':
                module = traced_model.get_submodule(node.target)
                layer_dict[node.name] = module.to(device)
                valid_layers.append(node.name)
            elif node.op == 'call_function' or node.op == 'call_method':
                layer_dict[node.name] = (node.op, node.target)
                valid_layers.append(node.name)
        except AttributeError:
            continue
    print(f'layer_dict:{layer_dict}')
    print(f'valid_layers:{valid_layers}')

    # 前向传播到切割点
    x = input_tensor
    for node in graph.nodes:
        try:
            if node.name == cut_layer_name:
                break  # 到达切割点
            if node.op == 'call_module':
                module = traced_model.get_submodule(node.target)
                x = module(x)
            elif node.op == 'call_function':
                args = [traced_model.getattr(arg) if isinstance(arg, str) else arg for arg in node.args]
                kwargs = {k: (traced_model.getattr(v) if isinstance(v, str) else v) for k, v in node.kwargs.items()}
                x = node.target(*args, **kwargs)
            elif node.op == 'call_method':
                obj = x if node.args[0] == 'self' else traced_model.getattr(node.args[0])
                method = getattr(obj, node.target)
                args = list(node.args[1:])
                args = [traced_model.getattr(arg) if isinstance(arg, str) else arg for arg in args]
                x = method(*args)
        except Exception as e:
            print(f"错误处理节点 {node.name}: {e}")
            continue

    print(f"切割点输出形状: {x.shape}")

    # 切割张量为四份
    b, c, h, w = x.shape
    h_half, w_half = h // 2, w // 2
    patches = [
        x[:, :, :h_half, :w_half],
        x[:, :, :h_half, w_half:],
        x[:, :, h_half:, :w_half],
        x[:, :, h_half:, w_half:]
    ]
    # for patch in patches:
    #     print(patch.shape)
    # 对每个子张量进行处理（基于拓扑结构）
    processed_patches = []
    with torch.no_grad():
        for patch in patches:
            out = patch  # 主路径的输入
            current_block_input = patch  # 当前块的输入张量
            for node in graph.nodes:
                if node.name == x:
                    continue
                # print(node.name,cut_layer_name,paste_layer_name)
                # print(node.name<cut_layer_name)
                # print(node.name>paste_layer_name)
                if node.name < cut_layer_name or node.name > paste_layer_name:
                    continue  # 跳过非目标区域
                print(f'{node.name}前张亮形状:{out.shape}')
                try:
                    if node.op == 'call_module':
                        module = layer_dict[node.name]
                        if node.name.endswith('downsample'):
                            # 动态获取当前块的输入 x
                            input_node = next(arg for arg in node.args if isinstance(arg, Node))
                            current_block_input = input_node
                            out = module(current_block_input)

                        else:
                            out = module(out)
                    elif node.op == 'call_function' or node.op == 'call_method':
                        op_type, target = layer_dict[node.name]
                        if op_type == 'call_function':
                            args = [out] + [traced_model.getattr(arg) if isinstance(arg, str) else arg for arg in node.args[1:]]
                            out = target(*args)
                        elif op_type == 'call_method':
                            obj = out if node.args[0] == 'self' else traced_model.getattr(node.args[0])
                            method = getattr(obj, target)
                            args = list(node.args[1:])
                            args = [traced_model.getattr(arg) if isinstance(arg, str) else arg for arg in args]
                            out = method(*args)
                except RuntimeError as e:
                    print(f"[ERROR] Layer {node.name} failed with input shape {out.shape}: {e}")
                    raise
                print(f'{node.name}后张亮形状:{out.shape}')
            processed_patches.append(out)
    for x in processed_patches:
        print(f"拼接点前每个子重量输出形状: {x.shape}")

    # 拼接处理后的子张量
    b, c, h_f, w_f = processed_patches[0].shape
    full_feat = torch.zeros((b, c, h_f * 2, w_f * 2), device=device)
    full_feat[:, :, :h_f, :w_f] = processed_patches[0]
    full_feat[:, :, :h_f, w_f:] = processed_patches[1]
    full_feat[:, :, h_f:, :w_f] = processed_patches[2]
    full_feat[:, :, h_f:, w_f:] = processed_patches[3]

    # 继续前向传播到模型输出
    for node in graph.nodes:
        if node.name > paste_layer_name:
            break  # 到达拼接点
        try:
            if node.op == 'call_module':
                module = traced_model.get_submodule(node.target)
                full_feat = module(full_feat)
            elif node.op == 'call_function':
                args = [full_feat] + [traced_model.getattr(arg) if isinstance(arg, str) else arg for arg in node.args[1:]]
                full_feat = node.target(*args)
            elif node.op == 'call_method':
                obj = full_feat if node.args[0] == 'self' else traced_model.getattr(node.args[0])
                method = getattr(obj, node.target)
                args = list(node.args[1:])
                args = [traced_model.getattr(arg) if isinstance(arg, str) else arg for arg in args]
                full_feat = method(*args)
        except Exception as e:
            print(f"错误处理节点 {node.name}: {e}")
            continue

    return full_feat# 测试代码
if __name__ == "__main__":
    model = ResNet50Manual()

    model.eval()
    # print(model)
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
        {'cut_layer': 'maxpool', 'paste_layer': 'layer1_block3_bn3'},
        # {'cut_layer': 'layer1_block3_conv3', 'paste_layer': 'layer3_block1_conv3'},
        # 更多案例...
    ]

    for case in cases:
        output = process_case(model, input_tensor, case['cut_layer'], case['paste_layer'])
        print(output.shape)
        if isinstance(output, torch.Tensor):
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            top5_prob, top5_catid = torch.topk(probabilities, 5)
            print(f"Case: cut_layer={case['cut_layer']}, paste_layer={case['paste_layer']}")
            for i in range(5):
                print(f"Class: {top5_catid[i]}, Probability: {top5_prob[i].item():.4f}")
        else:
            print(f"Case: cut_layer={case['cut_layer']}, paste_layer={case['paste_layer']} failed")