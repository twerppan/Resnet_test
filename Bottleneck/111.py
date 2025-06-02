import torch
import torch.nn as nn
from torch.fx import symbolic_trace, GraphModule
from torchvision import transforms,models
from PIL import Image
import os
import torchvision.models as models
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
class CustomNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 第一阶段
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()
        self.pool  = nn.MaxPool2d(2)
        # 残差块
        self.conv3 = nn.Conv2d(16, 16, 3, padding=1)
        self.bn3   = nn.BatchNorm2d(16)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(16, 16, 3, padding=1)
        self.bn4   = nn.BatchNorm2d(16)
        self.relu4 = nn.ReLU()
        # 并行分支
        self.br1_conv = nn.Conv2d(16, 8, 1)
        self.br1_bn   = nn.BatchNorm2d(8)
        self.br2_conv = nn.Conv2d(16, 8, 1)
        self.br2_bn   = nn.BatchNorm2d(8)
        # 全局池化 + 全连接
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc          = nn.Linear(16, 10)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.pool(x)
        identity = x
        out = self.relu3(self.bn3(self.conv3(x)))
        out = self.bn4(self.conv4(out))
        x = out + identity           # ← 切点：add op
        x = self.relu4(x)
        b1 = self.br1_bn(self.br1_conv(x))
        b2 = self.br2_bn(self.br2_conv(x))
        x  = torch.cat([b1, b2], dim=1)
        x  = self.global_pool(x)
        x  = torch.flatten(x, 1)
        x  = self.fc(x)
        return x


def split_model(model, start_cut, end_cut):
    traced = symbolic_trace(model)

    # 查找起始和结束节点（增加容错处理）
    nodes = list(traced.graph.nodes)
    # for i in nodes:
    #     print(i.target)
    #     print(i.name)
    start_node = next(n for n in nodes if n.name == start_cut)
    end_node = next(n for n in nodes if n.name == end_cut)

    # 构建前向部分
    front_graph = torch.fx.Graph()
    front_remap = {}
    front_input = None

    # 复制输入节点和前置节点
    for node in nodes:
        if node.op == 'placeholder':
            front_input = front_graph.placeholder(node.name)
            front_remap[node] = front_input
        if node == start_node:
            break
        if node.op != 'placeholder':
            new_node = front_graph.node_copy(node, lambda n: front_remap[n])
            front_remap[node] = new_node

    # 前向部分输出为start_node的输入
    front_graph.output(front_remap[start_node.args[0]])
    front_module = GraphModule(traced, front_graph)

    # 构建中间部分
    mid_graph = torch.fx.Graph()
    mid_input = mid_graph.placeholder('mid_input')
    mid_remap = {start_node.args[0]: mid_input}

    current_node = start_node
    while current_node != end_node.next:
        new_node = mid_graph.node_copy(current_node, lambda n: mid_remap.get(n, mid_input))
        mid_remap[current_node] = new_node
        current_node = current_node.next

    mid_graph.output(mid_remap[end_node])
    mid_module = GraphModule(traced, mid_graph)

    # 构建后向部分（关键修正部分）
    tail_graph = torch.fx.Graph()
    tail_input = tail_graph.placeholder('tail_input')
    tail_remap = {end_node: tail_input}

    current_node = end_node.next
    output_val = None

    # 只复制有效节点，跳过output节点
    while current_node is not None:
        if current_node.op == 'output':
            # 捕获原始输出值
            output_val = current_node.args[0]
            break
        new_node = tail_graph.node_copy(
            current_node,
            lambda n: tail_remap.get(n, tail_input)
        )
        tail_remap[current_node] = new_node
        current_node = current_node.next

    # 添加新的输出节点
    if output_val is not None:
        tail_graph.output(tail_remap[output_val])
    else:
        # 处理没有后续层的情况
        tail_graph.output(tail_input)

    tail_module = GraphModule(traced, tail_graph)

    return front_module, mid_module, tail_module


def create_split_fn(model, start_layer, end_layer, num_splits=4):
    front, mid, tail = split_model(model, start_layer, end_layer)

    def forward_fn(x):
        # 前向传播
        front_out = front(x)
        print(f'前向传播后shape：{front_out.shape}')
        # 空间维度分割为4份
        _, _, h, w = front_out.shape
        h_chunk = h // 2
        w_chunk = w // 2

        chunks = [
            front_out[:, :, :h_chunk, :w_chunk],
            front_out[:, :, :h_chunk, w_chunk:],
            front_out[:, :, h_chunk:, :w_chunk],
            front_out[:, :, h_chunk:, w_chunk:],
        ]
        for chunk in chunks:
            print(f'mid前每个子张量shape：{chunk.shape}')
        # 并行处理中间部分
        mid_outs = [mid(chunk) for chunk in chunks]
        for mid_out in mid_outs:
            print(f'mid后每个字张量shape{mid_out.shape}')
        # 重新拼接张量
        top = torch.cat([mid_outs[0], mid_outs[1]], dim=3)
        bottom = torch.cat([mid_outs[2], mid_outs[3]], dim=3)
        concat_out = torch.cat([top, bottom], dim=2)
        print(f'拼接后的shape：{concat_out.shape}')
        # 后向传播
        print(f'最后shape：{tail(concat_out).shape}')
        return tail(concat_out)

    return forward_fn


if __name__ == '__main__':

    # 使用示例
    # model = CustomNet()


    # model = ResNet50Manual()
    # weights_path = 'resnet50_weights.pth'
    # model.load_state_dict(torch.load(weights_path))
    # print("Model's state dict:")
    # for param_tensor in model.state_dict():
    #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    model = models.resnet50(pretrained=True)
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
    img_path = os.path.join('cat.png')  # 图像文件路径
    img = Image.open(img_path).convert('RGB')
    input_tensor = preprocess(img)  # 预处理后形状 [3, 224, 224]
    input_tensor = input_tensor.unsqueeze(0)
    # 测试推理
    cases = [
        # {'cut_layer': 'maxpool', 'paste_layer': 'layer1.1.relu'},
        # {'cut_layer': 'maxpool', 'paste_layer': 'layer1.0.downsample.1'},

        {'cut_layer': 'maxpool', 'paste_layer': 'layer1_0_downsample_1'},
        # 更多案例...
    ]
    LABELS_PATH = 'imagenet_classes.txt'

    with open(LABELS_PATH) as f:
        labels = [line.strip() for line in f.readlines()]
    for case in cases:
        split_fn = create_split_fn(model, case['cut_layer'], case['paste_layer'])

        # split_fn = create_split_fn(model, 'maxpool', 'layer1_block3_bn3')

        output = split_fn(input_tensor)
        print(output.shape)  # 应输出torch.Size([2, 10])
        if isinstance(output, torch.Tensor):
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            top5_prob, top5_catid = torch.topk(probabilities, 5)
            print(f"Case: cut_layer={case['cut_layer']}, paste_layer={case['paste_layer']}")
            for i in range(5):
                predicted_label = labels[top5_catid[i]]
                print(f'Top {i + 1} 类别: {predicted_label}, 置信度: {top5_prob[i].item():.4f}')
        else:
            print(f"Case: cut_layer={case['cut_layer']}, paste_layer={case['paste_layer']} failed")