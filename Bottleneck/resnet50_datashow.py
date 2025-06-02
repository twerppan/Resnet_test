import netron
import time
import torch
from torch import nn
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp
import networkx as nx
import pandas as pd
import operator
import torch.nn.functional as F
import matplotlib.pyplot as plt
import itertools
from collections import defaultdict, deque

class ResNet50Manual(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet50Manual, self).__init__()

        # 前置层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Layer1
        # Block1
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

        # Block2
        self.layer1_block2_conv1 = nn.Conv2d(256, 64, kernel_size=1, bias=False)
        self.layer1_block2_bn1 = nn.BatchNorm2d(64)
        self.layer1_block2_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1_block2_bn2 = nn.BatchNorm2d(64)
        self.layer1_block2_conv3 = nn.Conv2d(64, 256, kernel_size=1, bias=False)
        self.layer1_block2_bn3 = nn.BatchNorm2d(256)

        # Block3
        self.layer1_block3_conv1 = nn.Conv2d(256, 64, kernel_size=1, bias=False)
        self.layer1_block3_bn1 = nn.BatchNorm2d(64)
        self.layer1_block3_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1_block3_bn2 = nn.BatchNorm2d(64)
        self.layer1_block3_conv3 = nn.Conv2d(64, 256, kernel_size=1, bias=False)
        self.layer1_block3_bn3 = nn.BatchNorm2d(256)

        # Layer2
        # Block1
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

        # Block2
        self.layer2_block2_conv1 = nn.Conv2d(512, 128, kernel_size=1, bias=False)
        self.layer2_block2_bn1 = nn.BatchNorm2d(128)
        self.layer2_block2_conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer2_block2_bn2 = nn.BatchNorm2d(128)
        self.layer2_block2_conv3 = nn.Conv2d(128, 512, kernel_size=1, bias=False)
        self.layer2_block2_bn3 = nn.BatchNorm2d(512)

        # Block3
        self.layer2_block3_conv1 = nn.Conv2d(512, 128, kernel_size=1, bias=False)
        self.layer2_block3_bn1 = nn.BatchNorm2d(128)
        self.layer2_block3_conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer2_block3_bn2 = nn.BatchNorm2d(128)
        self.layer2_block3_conv3 = nn.Conv2d(128, 512, kernel_size=1, bias=False)
        self.layer2_block3_bn3 = nn.BatchNorm2d(512)

        # Block4
        self.layer2_block4_conv1 = nn.Conv2d(512, 128, kernel_size=1, bias=False)
        self.layer2_block4_bn1 = nn.BatchNorm2d(128)
        self.layer2_block4_conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer2_block4_bn2 = nn.BatchNorm2d(128)
        self.layer2_block4_conv3 = nn.Conv2d(128, 512, kernel_size=1, bias=False)
        self.layer2_block4_bn3 = nn.BatchNorm2d(512)

        # Layer3
        # Block1
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

        # Block2
        self.layer3_block2_conv1 = nn.Conv2d(1024, 256, kernel_size=1, bias=False)
        self.layer3_block2_bn1 = nn.BatchNorm2d(256)
        self.layer3_block2_conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer3_block2_bn2 = nn.BatchNorm2d(256)
        self.layer3_block2_conv3 = nn.Conv2d(256, 1024, kernel_size=1, bias=False)
        self.layer3_block2_bn3 = nn.BatchNorm2d(1024)

        # Block3
        self.layer3_block3_conv1 = nn.Conv2d(1024, 256, kernel_size=1, bias=False)
        self.layer3_block3_bn1 = nn.BatchNorm2d(256)
        self.layer3_block3_conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer3_block3_bn2 = nn.BatchNorm2d(256)
        self.layer3_block3_conv3 = nn.Conv2d(256, 1024, kernel_size=1, bias=False)
        self.layer3_block3_bn3 = nn.BatchNorm2d(1024)

        # Block4
        self.layer3_block4_conv1 = nn.Conv2d(1024, 256, kernel_size=1, bias=False)
        self.layer3_block4_bn1 = nn.BatchNorm2d(256)
        self.layer3_block4_conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer3_block4_bn2 = nn.BatchNorm2d(256)
        self.layer3_block4_conv3 = nn.Conv2d(256, 1024, kernel_size=1, bias=False)
        self.layer3_block4_bn3 = nn.BatchNorm2d(1024)

        # Block5
        self.layer3_block5_conv1 = nn.Conv2d(1024, 256, kernel_size=1, bias=False)
        self.layer3_block5_bn1 = nn.BatchNorm2d(256)
        self.layer3_block5_conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer3_block5_bn2 = nn.BatchNorm2d(256)
        self.layer3_block5_conv3 = nn.Conv2d(256, 1024, kernel_size=1, bias=False)
        self.layer3_block5_bn3 = nn.BatchNorm2d(1024)

        # Block6
        self.layer3_block6_conv1 = nn.Conv2d(1024, 256, kernel_size=1, bias=False)
        self.layer3_block6_bn1 = nn.BatchNorm2d(256)
        self.layer3_block6_conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer3_block6_bn2 = nn.BatchNorm2d(256)
        self.layer3_block6_conv3 = nn.Conv2d(256, 1024, kernel_size=1, bias=False)
        self.layer3_block6_bn3 = nn.BatchNorm2d(1024)

        # Layer4
        # Block1
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

        # Block2
        self.layer4_block2_conv1 = nn.Conv2d(2048, 512, kernel_size=1, bias=False)
        self.layer4_block2_bn1 = nn.BatchNorm2d(512)
        self.layer4_block2_conv2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer4_block2_bn2 = nn.BatchNorm2d(512)
        self.layer4_block2_conv3 = nn.Conv2d(512, 2048, kernel_size=1, bias=False)
        self.layer4_block2_bn3 = nn.BatchNorm2d(2048)

        # Block3
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

def find_critical_edges_dag(edges):
    """
    自动识别源节点和汇节点，并返回DAG中的关键割边
    :param edges: 边列表，格式如 [(u1, v1), (u2, v2), ...]
    :return: 关键割边列表
    """
    # ------------------- 自动识别源节点和汇节点 -------------------
    # 统计入度和出度
    in_degree = defaultdict(int)
    out_degree = defaultdict(int)
    nodes = set()
    for u, v in edges:
        out_degree[u] += 1
        in_degree[v] += 1
        nodes.add(u)
        nodes.add(v)

    # 确定唯一的源节点（入度为0）和汇节点（出度为0）
    sources = [n for n in nodes if in_degree[n] == 0]
    sinks = [n for n in nodes if out_degree[n] == 0]

    # 检查源/汇节点唯一性
    if len(sources) != 1 or len(sinks) != 1:
        raise ValueError("图必须包含且仅包含一个源节点和一个汇节点")
    source, sink = sources[0], sinks[0]

    # ------------------- 核心算法（与原逻辑一致） -------------------
    # 构建邻接表和逆邻接表
    adj = defaultdict(list)
    reverse_adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        reverse_adj[v].append(u)

    # 拓扑排序（从源节点开始）
    topo_order = []
    queue = deque([source])
    temp_in_degree = defaultdict(int)
    for v in in_degree:  # 深拷贝入度
        temp_in_degree[v] = in_degree[v]

    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in adj[u]:
            temp_in_degree[v] -= 1
            if temp_in_degree[v] == 0:
                queue.append(v)

    # 计算源到各节点的路径数
    src_paths = defaultdict(int)
    src_paths[source] = 1
    for u in topo_order:
        for v in adj[u]:
            src_paths[v] += src_paths[u]

    # 计算各节点到汇的路径数
    dest_paths = defaultdict(int)
    dest_paths[sink] = 1
    reverse_topo = reversed(topo_order)
    for v in reverse_topo:
        for u in reverse_adj[v]:
            dest_paths[u] += dest_paths[v]

    # 判断关键割边
    total = src_paths[sink]
    critical_edges = []
    for u, v in edges:
        if src_paths[u] * dest_paths[v] == total:
            critical_edges.append((u, v))

    return critical_edges

def build_graph_and_find_cuts(fx_net: torch.fx.GraphModule):
    G = nx.DiGraph()
    for n in fx_net.graph.nodes:
        G.add_node(n.name)
    for n in fx_net.graph.nodes:
        for inp in n.all_input_nodes:
            G.add_edge(inp.name, n.name)
    # src = next(n.name for n in fx_net.graph.nodes if n.op=='placeholder')
    # dst = next(n.name for n in fx_net.graph.nodes if n.op=='output')
    # pos = nx.circular_layout(G)  # 使用 spring 布局
    # nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=300, edge_color='k', linewidths=1, font_size=10)
    # plt.title("Directed Graph")
    # plt.show()
    # cuts = []
    # for u,v in list(G.edges()):
    #     G2 = G.copy()
    #     G2.remove_edge(u,v)
    #     if not nx.has_path(G2, src, dst):
    #         cuts.append((u,v))
    edges = list(G.edges())
    return find_critical_edges_dag(edges)
    # return cuts

def flops_module(mod, out_shape):
    elems = int(torch.tensor(out_shape).prod())
    if isinstance(mod, nn.Conv2d):
        Cin, Cout = mod.in_channels, mod.out_channels
        Kh, Kw = mod.kernel_size
        H, W = out_shape[2], out_shape[3]
        g = mod.groups
        return 2 * (Cin//g)*Kh*Kw * Cout * H * W
    if isinstance(mod, nn.Linear):
        return 2 * mod.in_features * mod.out_features
    if isinstance(mod, (nn.BatchNorm2d, nn.BatchNorm1d)):
        return 2 * elems
    if isinstance(mod, (nn.ReLU, nn.MaxPool2d, nn.AdaptiveAvgPool2d)):
        return elems
    return 0

def flops_function(fn, out_shape):
    elems = int(torch.tensor(out_shape).prod())
    if fn in (F.relu,):
        return elems
    if fn in (operator.add,):
        return elems
    # flatten、cat等可以补充
    return 0

def profile_and_tabulate(model: nn.Module, input_shape=(1,3,224,224)):
    # 1. trace + shape prop
    fx_net = symbolic_trace(model)
    ShapeProp(fx_net).propagate(torch.randn(input_shape))
    # start_time = time.time()
    # 2. 找割边
    cuts = build_graph_and_find_cuts(fx_net)
    # print(cuts)
    # end_time = time.time()
    # execution_time = end_time - start_time
    # print(f"算法执行时间: {execution_time} 秒")

    # 3. 遍历节点，积累 FLOPs
    # total = 0
    rows = []
    nodes = list(fx_net.graph.nodes)
    for idx, node in enumerate(nodes):
        # 3.1 取 shape
        tm = node.meta.get('tensor_meta')
        shape = tuple(tm.shape) if tm is not None else None
        # 3.2 计算本节点 FLOPs
        if node.op == 'call_module':
            mod = fx_net.get_submodule(node.target)
            f = flops_module(mod, shape) if shape else 0
        elif node.op == 'call_function':
            f = flops_function(node.target, shape) if shape else 0
        else:
            f = 0
        # total += f

        # 3.3 记录到表里
        next_edge = (node.name, nodes[idx+1].name) if idx+1 < len(nodes) else None
        rows.append({
            'idx': idx,
            'node': node.name,
            'shape': shape,
            'size': int(torch.tensor(shape).prod()) if shape else None,
            'flops': f,
            'edge_to_next': next_edge,
        })

    # 4. 标记切割点，计算 prefix/suffix
    pref = 0
    for r in rows:
        # pref += r['flops']
        is_cut = r['edge_to_next'] in cuts
        r.update({
            'is_cut': is_cut,
            # 'prefix_MFLOPs': pref/1e6,
            # 'suffix_MFLOPs': ((total - pref)/1e6) if is_cut else None
        })

    # 5. 输出 DataFrame
    df = pd.DataFrame(rows)
    # df.rename(columns={'flops':'flops (MFLOPs)'}, inplace=True)
    # df['flops (MFLOPs)'] = df['flops (MFLOPs)'] / 1e6
    # df = df[['idx','node','shape','size','flops (MFLOPs)','is_cut','prefix_MFLOPs','suffix_MFLOPs']]
    # print(df.to_markdown(index=False))
    df.rename(columns={'flops':'flops (MFLOPs)'}, inplace=True)
    df['flops (MFLOPs)'] = df['flops (MFLOPs)'] / 1e6
    # 这里把 edge_to_next 一起带出来
    df = df[[
        'idx','node','shape','size','flops (MFLOPs)',
        'is_cut','edge_to_next'
    ]]
    return df

# 测试代码
if __name__ == "__main__":
    model = ResNet50Manual()
    netron.start('ResNet50Manual.onnx')
    # netron.start('ResNet50Manual.onnx')
    # df = profile_and_tabulate(model, input_shape=(1, 3, 224, 224))
    # print(df.to_markdown(index=False))

    input_tensor = torch.randn(1, 3, 224, 224)
    print(f'输入张量形状：{input_tensor.shape}')
    output = model(input_tensor)
    print(f'ResNet50Manual最终输出张量shape：{output.shape}')  # 应输出 torch.Size([1, 1000])