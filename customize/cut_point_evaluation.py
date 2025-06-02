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
from torchvision.models import AlexNet,resnet18


class CustomNet(nn.Module):
    def __init__(self):
        super().__init__()
        # ====== 第一阶段: 卷积 + BN + ReLU ======
        # 输入通道 3，输出通道 16，卷积核大小 3，padding 保持尺寸不变
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        # 批归一化层，跟 conv1 输出通道一致
        self.bn1 = nn.BatchNorm2d(16)
        # 激活函数 ReLU
        self.relu1 = nn.ReLU()

        # 第二个卷积模块，保持通道数不变
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()
        # 池化层，将特征图宽高压缩一半
        self.pool = nn.MaxPool2d(2)

        # ====== 残差块: Conv-BN-ReLU -> Conv-BN + 跳跃连接 ======
        # 第一个残差分支的卷积
        self.conv3 = nn.Conv2d(16, 16, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(16)
        self.relu3 = nn.ReLU()
        # 第二个残差分支的卷积
        self.conv4 = nn.Conv2d(16, 16, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(16)

        # ====== 并行分支: 两条 1x1 卷积分支，再拼接 ======
        # 分支 1: 将通道数压减到 8
        self.br1_conv = nn.Conv2d(16, 8, 1)
        self.br1_bn = nn.BatchNorm2d(8)
        # 分支 2: 同样将通道数压减到 8
        self.br2_conv = nn.Conv2d(16, 8, 1)
        self.br2_bn = nn.BatchNorm2d(8)

        # ====== 全局池化 + 全连接分类器 ======
        # 自适应平均池化到 1x1
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        # 最终线性层: 输入维度 16（两个 8 通道拼接），输出 10 类
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        # ====== 前向传播: 第一阶段 ======
        # 卷积 -> BN -> ReLU
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        # 池化降采样
        x = self.pool(x)

        # ====== 前向传播: 残差块 ======
        # 保存跳跃连接的输入
        identity = x
        # 主分支: Conv -> BN -> ReLU
        out = self.relu3(self.bn3(self.conv3(x)))
        # 主分支: Conv -> BN
        out = self.bn4(self.conv4(out))
        # 跳跃连接后再激活
        x = nn.functional.relu(out + identity)

        # ====== 前向传播: 并行分支 ======
        # 分支 1
        b1 = self.br1_bn(self.br1_conv(x))
        # 分支 2
        b2 = self.br2_bn(self.br2_conv(x))
        # 按通道维度拼接，两分支各 8 通道，共 16 通道
        x = torch.cat([b1, b2], dim=1)

        # ====== 前向传播: 分类头 ======
        x = self.global_pool(x)              # 自适应池化
        x = torch.flatten(x, 1)             # 展平为向量
        x = self.fc(x)                      # 全连接分类
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

# def select_model_cuts(df: pd.DataFrame, device_resources):
#     """
#     在 DataFrame 中选择 N-1 个切割点，将模型分为 N 段。
#     - df: 包含列 ['idx', 'edge_to_next', 'is_cut', 'prefix_MFLOPs'] 的表，
#           且按 idx 升序排列、prefix_MFLOPs 单位为 MFLOPs。
#     - device_resources: 长度为 N 的列表，每个元素是对应设备剩余计算能力（MFLOPs）。
#
#     返回:
#       best_edges: 选出的切割边列表 [(u1,v1), …]，长度 N-1；
#       best_segments: 每段对应的 FLOPs 列表，长度 N；
#       best_utils: 每段对应的“利用率”列表，长度 N。
#     """
#     # 1. 准备候选切割位置：取所有 is_cut==True 的行
#     candidates = df[df['is_cut']].copy().reset_index(drop=True)
#     # 切割点的分割位置用 idx 表示“在 idx 处切”
#     cut_indices = candidates['idx'].tolist()
#
#     N = len(device_resources)
#     assert N >= 1, "设备数 N 必须 >= 1"
#     if N == 1:
#         # 不切割，整条链都跑在一个设备上
#         total = df['prefix_MFLOPs'].iloc[-1]
#         if total > device_resources[0]:
#             raise ValueError("单设备能力不足以运行整个模型")
#         return [], [total], [total / device_resources[0]]
#
#     best_max_util = float('inf')
#     best_choice = None
#
#     total = df['prefix_MFLOPs'].iloc[-1]
#     # 2. 在所有候选切割位置中枚举选 N-1 个
#     for comb in itertools.combinations(cut_indices, N - 1):
#         # comb 是一个递增的 idx 序列
#         # 计算每段的 FLOPs
#         seg_flops = []
#         prev = 0.0
#         for cut in comb:
#             # prefix_MFLOPs 表示从 start 到该 idx（含该行）的 FLOPs
#             fl = df.loc[df['idx'] == cut, 'prefix_MFLOPs'].iloc[0] - prev
#             seg_flops.append(fl)
#             prev += fl
#         # 最后一段
#         seg_flops.append(total - prev)
#
#         # 检查是否都不超过设备能力
#         ok = all(fl <= cap for fl, cap in zip(seg_flops, device_resources))
#         if not ok:
#             continue
#
#         # 计算各段利用率，目标是最小化最大利用率
#         utils = [fl / cap for fl, cap in zip(seg_flops, device_resources)]
#         max_util = max(utils)
#         if max_util < best_max_util:
#             best_max_util = max_util
#             best_choice = (comb, seg_flops, utils)
#
#     if best_choice is None:
#         raise ValueError("在给定资源限制下无法找到可行的切割方案")
#
#     comb, best_segments, best_utils = best_choice
#     # 将 idx 映射回实际的边 (u,v)
#     # df 中的 edge_to_next 列正好存了 (u,v)
#     idx_to_edge = dict(zip(df['idx'], df['edge_to_next']))
#     best_edges = [idx_to_edge[i] for i in comb]
#
#     return best_edges, best_segments, best_utils
if __name__ == "__main__":
    # model = AlexNet()
    model = CustomNet()
    df = profile_and_tabulate(model, input_shape=(1, 3, 224, 224))
    print(df.to_markdown(index=False))

    device_resources = [1500.0, 1800.0, 1600.0]
    # edges, seg_flops, utils = select_model_cuts(df, device_resources)
    # print("选中的切割边：", edges)
    # print("各段 FLOPs：", seg_flops)
    # print("各段利用率：", utils)