import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import pandas as pd
from collections import defaultdict, deque
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp
from torchvision.models import alexnet

def find_critical_edges_dag(edges):
    in_degree = defaultdict(int)
    out_degree = defaultdict(int)
    nodes = set()
    for u, v in edges:
        out_degree[u] += 1
        in_degree[v] += 1
        nodes.update([u, v])
    sources = [n for n in nodes if in_degree[n] == 0]
    sinks = [n for n in nodes if out_degree[n] == 0]
    if len(sources) != 1 or len(sinks) != 1:
        raise ValueError("图必须只含一个源节点和一个汇节点")
    source, sink = sources[0], sinks[0]
    adj, reverse_adj = defaultdict(list), defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        reverse_adj[v].append(u)
    topo, q = [], deque([source])
    tmp_in = in_degree.copy()
    while q:
        u = q.popleft(); topo.append(u)
        for v in adj[u]:
            tmp_in[v] -= 1
            if tmp_in[v] == 0: q.append(v)
    src_paths = defaultdict(int); src_paths[source] = 1
    for u in topo:
        for v in adj[u]: src_paths[v] += src_paths[u]
    dest_paths = defaultdict(int); dest_paths[sink] = 1
    for v in reversed(topo):
        for u in reverse_adj[v]: dest_paths[u] += dest_paths[v]
    total = src_paths[sink]
    return [(u, v) for u, v in edges if src_paths[u] * dest_paths[v] == total]


def build_graph_and_find_cuts(fx_net):
    G = nx.DiGraph()
    for n in fx_net.graph.nodes:
        G.add_node(n.name)
    for n in fx_net.graph.nodes:
        for inp in n.all_input_nodes:
            G.add_edge(inp.name, n.name)
    return find_critical_edges_dag(list(G.edges()))


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
    if fn in (F.relu,): return elems
    if fn in (torch.add,): return elems
    return 0


def profile_and_tabulate(model, input_shape=(1, 3, 224, 224)):
    fx_net = symbolic_trace(model)              #转换为静态计算图
    input_tensor = torch.randn(input_shape)
    ShapeProp(fx_net).propagate(input_tensor)           #前向传播记录各节点输出维度

    # 收集权重矩阵和偏置的内存需求
    weight_memory = {}
    bias_memory = {}
    for name, module in fx_net.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.BatchNorm1d, nn.BatchNorm2d)):      #此暂时只考虑卷积、线性、Bn的权重参数，transformer、Embedding后续在考虑
            # 权重矩阵的内存需求（字节）
            weight_memory[name] = module.weight.numel() * 4  # 假设为float32位4字节
            # 偏置的内存需求（字节）
            if module.bias is not None:
                bias_memory[name] = module.bias.numel() * 4
            else:
                bias_memory[name] = 0

    cuts = build_graph_and_find_cuts(fx_net)        #关键边初步筛选
    rows, nodes = [], list(fx_net.graph.nodes)

    for idx, node in enumerate(nodes):
        tm = node.meta.get('tensor_meta')
        shape = tuple(tm.shape) if tm else None# 获取形状信息
        if node.op == 'call_module':
            mod = fx_net.get_submodule(node.target)
            f = flops_module(mod, shape) if shape else 0#根据节点类型计算浮点计算量
            weight_mem = weight_memory.get(node.target, 0)# 从预存字典获取内存需求
            bias_mem = bias_memory.get(node.target, 0)
        elif node.op == 'call_function':
            f = flops_function(node.target, shape) if shape else 0
            weight_mem = 0
            bias_mem = 0
        else:
            f = 0
            weight_mem = 0
            bias_mem = 0

        next_edge = (node.name, nodes[idx + 1].name) if idx + 1 < len(nodes) else None
        # 激活值内存需求（假设为float32）
        activation_mem = int(torch.tensor(shape).prod()) * 4 if shape else 0

        rows.append({
            'idx': idx,
            'node': node.name,
            'shape': shape,
            'size': activation_mem,  # 激活值内存需求
            'flops': f,
            'edge_to_next': next_edge,
            'is_cut': next_edge in cuts,
            'weight_memory': weight_mem,  # 权重内存需求
            'bias_memory': bias_mem  # 偏置内存需求
        })

    df = pd.DataFrame(rows)
    df['flops (MFLOPs)'] = df['flops'] / 1e6
    # 总内存需求（激活值 + 权重 + 偏置）
    df['total_memory'] = df['size'] + df['weight_memory'] + df['bias_memory']
    return df[
        ['idx', 'node', 'shape', 'size', 'weight_memory', 'bias_memory', 'total_memory', 'flops (MFLOPs)', 'is_cut',
         'edge_to_next']]


def find_bottleneck_blocks(df, window=3, threshold=1.5):
    """识别连续瓶颈段并生成独立区块"""
    flops = df['flops (MFLOPs)'].values
    blocks = []
    current_block = None

    for i in range(len(flops)):
        # 检测峰值点
        if flops[i] > threshold * flops.mean():
            if current_block is None:
                current_block = [i, i + 1]
            else:
                current_block[1] = i + 1
        else:
            if current_block is not None:
                blocks.append(tuple(current_block))
                current_block = None

    # 合并相邻区块
    merged = []
    for block in sorted(blocks):
        if not merged or block[0] > merged[-1][1]:
            merged.append(list(block))
        else:
            merged[-1][1] = max(merged[-1][1], block[1])

    return [tuple(b) for b in merged]
# ----------------- 新增函数：瓶颈段二分切割 -----------------
def split_bottlenecks(bottleneck_blocks):
    """将每个瓶颈段均匀分割为两个子段"""
    split_blocks = []
    for s, e in bottleneck_blocks:
        mid = s + (e - s) // 2
        split_blocks.extend([(s, mid), (mid, e)])
    return split_blocks


# ----------------- 修改后的候选方案生成 -----------------
def generate_hybrid_arms(df, bottleneck_blocks, devices):
    """生成混合切割方案候选"""
    # Step1: 分割瓶颈段
    split_bn = split_bottlenecks(bottleneck_blocks)

    # Step2: 获取非瓶颈段索引
    all_indices = set(range(len(df)))
    bn_indices = set()
    for s, e in bottleneck_blocks:
        bn_indices.update(range(s, e))
    non_bn = sorted(all_indices - bn_indices)

    # Step3: 生成非瓶颈段切割方案
    nonbn_arms = []
    for k in range(1, len(non_bn)):
        for cuts in itertools.combinations(range(1, len(non_bn)), k):
            pts = [0] + list(cuts) + [len(non_bn)]
            nonbn_partitions = [(non_bn[pts[i]], non_bn[pts[i + 1] - 1] + 1)
                                for i in range(len(pts) - 1)]
            nonbn_arms.append(nonbn_partitions)

    # Step4: 组合所有切割方案
    full_arms = []
    for nb_part in nonbn_arms:
        # 合并非瓶颈段与瓶颈子段
        full_part = sorted(nb_part + split_bn, key=lambda x: x[0])
        # 生成设备分配
        bn_dev_pairs = list(itertools.permutations(devices, 2))
        for bn_assign in bn_dev_pairs:
            # 分配瓶颈子段设备
            dev_map = {}
            for i, (s, e) in enumerate(split_bn):
                dev_map[(s, e)] = bn_assign[i % 2]  # 交替分配

            # 分配非瓶颈段设备
            remaining_devs = [d for d in devices if d not in bn_assign]
            for nb_assign in itertools.permutations(remaining_devs, len(nb_part)):
                assign = {**dev_map, **dict(zip(nb_part, nb_assign))}
                full_arms.append((full_part, assign))

    return full_arms


# ----------------- 修改后的指标计算 -----------------
def compute_hybrid_metrics(bottlenecks,df, partitions, assignment):
    """计算混合切割方案的综合指标"""
    total_time = 0
    comm_time = 0

    # 遍历所有分区
    for (s, e), dev in assignment.items():
        # 计算执行时间
        flops = df.loc[s:e - 1, 'flops (MFLOPs)'].sum() * 1e6
        total_time += flops / DEVICE_FLOPS[dev]

        # 计算通信开销
        if (s, e) in split_bottlenecks(bottlenecks):  # 张量子段
            # 假设每个张量子段需要同步梯度，通信量为权重大小
            weight_size = df.loc[s:e - 1, 'weight_memory'].sum()
            comm_time += weight_size * 8 / BANDWIDTH_BPS  # 转为比特

    # 添加流水线通信
    sorted_parts = sorted(partitions, key=lambda x: x[0])
    for i in range(len(sorted_parts) - 1):
        prev_end = sorted_parts[i][1]
        curr_start = sorted_parts[i + 1][0]
        if prev_end != curr_start:  # 非连续段需要传输数据
            data_size = df.loc[prev_end - 1, 'size']
            comm_time += data_size * 8 / BANDWIDTH_BPS

    return total_time + comm_time


# ----------------- 新版多臂赌博机 -----------------
class HybridMAB:
    def __init__(self, arms):
        self.arms = arms
        self.values = {str(a): 0 for a in arms}

    def select(self, epsilon=0.1):
        if torch.rand(1).item() < epsilon:
            return self.arms[torch.randint(len(self.arms), (1,)).item()]
        else:
            return min(self.arms, key=lambda a: self.values[str(a)])

    def update(self, arm, metric):
        self.values[str(arm)] += 1 / (metric + 1e-6)  # 指标越小奖励越高


# ----------------- 修改后的主函数 -----------------
def main(model, rounds=500):
    # 模型分析
    df = profile_and_tabulate(model)

    # 识别瓶颈段
    bottlenecks = find_bottleneck_blocks(df)
    # 生成候选方案
    all_arms = generate_hybrid_arms(df, bottlenecks, list(DEVICE_FLOPS.keys()))

    # 初始化MAB
    mab = HybridMAB(all_arms)

    # 训练循环
    for _ in range(rounds):
        arm = mab.select()
        cost = compute_hybrid_metrics(bottlenecks,df, *arm)
        mab.update(arm, cost)

    # 输出最佳方案
    best = min(all_arms, key=lambda a: mab.values[str(a)])
    print("最佳切割方案:")
    for seg in sorted(best[0], key=lambda x: x[0]):
        s, e = seg
        print(f"段 [{s}-{e}] → 设备 {best[1][seg]}")

    return best


# ----------------- 配置参数 -----------------
DEVICE_FLOPS = {
    'gpu0': 12e12,  # 12 TFLOPS
    'gpu1': 10e12,
    'gpu2': 9e12,
    'cpu': 0.5e12
}
BANDWIDTH_BPS = 100e9  # 100Gbps

# ----------------- 示例执行 -----------------
if __name__ == "__main__":
    class CustomModel(nn.Module):
        def __init__(self):
            super().__init__()
            # 定义示例模型层


    model = CustomModel()
    best_partition = main(model)