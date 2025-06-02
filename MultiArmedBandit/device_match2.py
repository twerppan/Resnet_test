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

# ----- 前面给出的函数 -----
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

# ----- 设备计算能力 (FLOPS 每秒) -----
DEVICE_FLOPS = {
    'pc_cpu': 200e9,
    # 'pc_gpu': 2100e9,
    'rpi5_1': 32e9,
    'rpi5_2': 32e9,
    'jetson': 472e9
}
BANDWIDTH_BPS = 1e6
MEM_LIMIT = 4 * 1024**3


def compute_metrics(df, partitions, assignment):        #根据df性能列表、切割分区以及分配设备得到最终的执行、传输时间与内存占用
    total_exec_time = total_transfer_time = 0.0
    max_mem_ratio = 0.0

    # 计算每个设备的内存使用情况
    device_memory_usage = defaultdict(int)

    for part_idx, (start, end) in enumerate(partitions):
        device = assignment[part_idx]
        # 计算当前分区所有节点的总内存需求（激活值 + 权重 + 偏置）
        partition_memory = df.loc[(df['idx'] >= start) & (df['idx'] < end), 'total_memory'].sum()

        # 累加到设备的内存使用
        device_memory_usage[device] += partition_memory

        # 计算执行时间
        flops = df.loc[(df['idx'] >= start) & (df['idx'] < end), 'flops (MFLOPs)'].sum() * 1e6
        exec_time = flops / DEVICE_FLOPS[device]
        total_exec_time += exec_time

        # 如果不是最后一个分区，计算传输时间
        if part_idx != len(partitions) - 1:
            # 分区结束节点的输出数据大小（仅激活值）
            size_elems = df.loc[end - 1, 'size']
            bytes_ = size_elems  # 激活值内存需求已经是字节单位
            transfer = (bytes_ * 8) / BANDWIDTH_BPS  # 转换为比特然后除以比特每秒
            total_transfer_time += transfer

    # 计算每个设备的最大内存使用率
    for device, memory_used in device_memory_usage.items():
        mem_ratio = memory_used / MEM_LIMIT
        if mem_ratio > max_mem_ratio:
            max_mem_ratio = mem_ratio

    return total_exec_time, total_transfer_time, max_mem_ratio


class EpsilonGreedy:
    def __init__(self, arms, epsilon=0.1):
        self.arms = [arm for arm in arms if self._is_valid(arm)]
        self.epsilon = epsilon
        # 使用可哈希的 arms：将 partitions 列表转换为元组
        self.counts = {a: 0 for a in self.arms}# 策略被选择的次数
        self.values = {a: 0.0 for a in self.arms}# 策略的平均奖励值

    def _is_valid(self, arm):
        assignment = arm[1]
        return len(set(assignment)) == len(assignment)  # 确保设备分配中没有重复设备，之间出现分三段，1、3分给同一个设备的状况

    def select(self):
        if torch.rand(1).item() < self.epsilon:
            # 随机选择一个有效的策略
            return self.arms[torch.randint(len(self.arms), (1,)).item()]
        else:
            # 选择当前最佳策略
            return max(self.arms, key=lambda a: self.values[a])

    def update(self, arm, reward):
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n


def main(model, input_shape=(1, 3, 224, 224), rounds=10000):
    df = profile_and_tabulate(model, input_shape)
    # 过滤掉起始节点作为切割点，避免产生长度为0的分区
    print(df.to_markdown(index=False))
    cut_idxs = [i for i in df.index[df['is_cut']] if i > 0]
    arms = []
    device_list = list(DEVICE_FLOPS.keys())

    for k in range(len(DEVICE_FLOPS)):      #遍历切割点数k=0到3，而不蛋蛋只是找3个切割点，这里最好定义为设别数量N
        for cuts in itertools.combinations(cut_idxs, k):
            pts = [0] + list(cuts) + [len(df)]# 构建分区列表
            partitions = tuple(zip(pts[:-1], pts[1:]))  # 转成可哈希元组
            # 生成设备分配，确保每个设备只能被选择一次
            for assignment in itertools.permutations(device_list, len(partitions)):
                arms.append((partitions, assignment))

    mab = EpsilonGreedy(arms)# 初始化算法
    for _ in range(rounds):
        arm = mab.select()
        exec_t, trans_t, max_mem_ratio = compute_metrics(df, arm[0], arm[1])
        # 奖励基于最大内存占比，以及执行时间。最大内存占比越小，执行时间越短奖励越高
        reward = - (exec_t + trans_t) + (1 / (max_mem_ratio + 1e-6))  # 添加一个小的常数以避免除以零
        mab.update(arm, reward)

    best = max(arms, key=lambda a: mab.values[a])
    best_exec_t, best_trans_t, best_mem_r = compute_metrics(df, best[0], best[1])
    print("最佳方案:", best)
    print(
        f"执行时间: {best_exec_t:.4f}, 传输时间: {best_trans_t:.4f}, 最大内存占比: {best_mem_r:.4f}")
    return best

if __name__ == '__main__':
    model = alexnet(pretrained=False)
    best_config = main(model)
    print("完成模型切割部署评估。")