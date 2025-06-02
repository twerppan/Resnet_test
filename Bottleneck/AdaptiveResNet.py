
from torchvision.models import resnet50
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

class AdaptiveResNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        # 加载预训练ResNet主干
        resnet = resnet50(pretrained=True)

        # 移除原始的全连接层和全局平均池化
        self.backbone = nn.Sequential(*list(resnet.children()))[:-2]

        # 添加自适应平均池化层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 新的全连接层（适配目标类别数）
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        # 提取特征（允许任意输入尺寸）
        features = self.backbone(x)  # 输出形状: [B, 2048, H, W]

        # 自适应池化到1x1
        pooled = self.avgpool(features)  # 输出形状: [B, 2048, 1, 1]

        # 展平并分类
        flattened = torch.flatten(pooled, 1)
        output = self.fc(flattened)
        return output
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
if __name__ == "__main__":

    model = AdaptiveResNet()
    df = profile_and_tabulate(model, input_shape=(1, 3, 960, 960))
    print(df.to_markdown(index=False))

    device_resources = [1500.0, 1800.0, 1600.0]