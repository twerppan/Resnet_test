import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import pandas as pd
from collections import defaultdict, deque
from torch.fx import symbolic_trace
from torch.fx.passes.shape_prop import ShapeProp
from torchvision.models import resnet50
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

DEVICE_FLOPS = {
    'pc_cpu': 500e9,
    'rpi5_1': 320e9,
    'rpi5_2': 320e9,
    'rpi5_3': 320e9,
    'jetson': 472e9
}
BANDWIDTH_BPS = 1e6
MEM_LIMIT = 4 * 1024**3
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


# ----------------- 新增函数：带通信开销的瓶颈段处理 -----------------
def find_bottleneck_segments(df, window=4, thresh_ratio=3):
    """识别连续高计算量层段并合并为单一逻辑层"""
    flops = df['flops (MFLOPs)'].values
    avg = flops.mean()
    segs = []

    # Step 1: 滑动窗口检测
    for i in range(len(flops) - window + 1):
        s = sum(flops[i:i + window])
        if s > avg * thresh_ratio:
            segs.append((i, i + window))

    # Step 2: 合并重叠段
    merged = []
    for st, ed in sorted(segs):
        if not merged or st > merged[-1][1]:
            merged.append([st, ed])
        else:
            merged[-1][1] = max(merged[-1][1], ed)

    # Step 3: 生成逻辑层映射表
    logical_layers = []
    prev = 0
    for st, ed in merged:
        if prev < st:
            logical_layers.extend(range(prev, st))
        logical_layers.append((st, ed))  # 瓶颈段视为单个逻辑层
        prev = ed
    if prev < len(flops):
        logical_layers.extend(range(prev, len(flops)))

    return merged, logical_layers


# ----------------- 修改后的compute_metrics -----------------
def compute_metrics(df, partitions, assignment, bottleneck_segments):
    """计算包含张量并行通信的开销"""
    total_exec_time = total_transfer_time = 0.0
    max_mem_ratio = 0.0
    device_memory_usage = defaultdict(int)

    for part_idx, (start, end) in enumerate(partitions):
        device = assignment[part_idx]
        # 判断当前分区是否为张量并行段
        is_tensor = any((s <= start and end <= e) for (s, e) in bottleneck_segments)

        # 内存计算
        partition_memory = df.loc[(df['idx'] >= start) & (df['idx'] < end), 'total_memory'].sum()
        device_memory_usage[device] += partition_memory

        # 执行时间（张量并行需分摊计算）
        flops = df.loc[(df['idx'] >= start) & (df['idx'] < end), 'flops (MFLOPs)'].sum() * 1e6
        if is_tensor:
            # 假设张量并行使用2个设备
            exec_time = flops / (2 * DEVICE_FLOPS[device])
        else:
            exec_time = flops / DEVICE_FLOPS[device]
        total_exec_time += exec_time

        # 通信时间
        if part_idx != len(partitions) - 1:
            size_elems = df.loc[end - 1, 'size']
            bytes_ = size_elems
            if is_tensor:
                # 张量并行段需要额外同步梯度，通信量为权重大小
                weight_bytes = df.loc[start:end - 1, 'weight_memory'].sum()
                transfer = (bytes_ + weight_bytes) * 8 / BANDWIDTH_BPS
            else:
                transfer = bytes_ * 8 / BANDWIDTH_BPS
            total_transfer_time += transfer

    # 内存使用率计算
    for dev, mem in device_memory_usage.items():
        mem_ratio = mem / MEM_LIMIT
        max_mem_ratio = max(max_mem_ratio, mem_ratio)

    return total_exec_time, total_transfer_time, max_mem_ratio


# ----------------- 修改后的main函数 -----------------
def main(model, input_shape=(1, 3, 224, 224), rounds=500):
    df = profile_and_tabulate(model, input_shape)

    # Step 1: 识别瓶颈段并构建逻辑层序列
    bottleneck_segments, logical_layers = find_bottleneck_segments(df)
    # print("逻辑层结构解析:")
    # for i, layer in enumerate(logical_layers):
    #     if isinstance(layer, tuple):
    #         print(f"逻辑层{i}: 瓶颈段 {layer[0]}-{layer[1] - 1}层")
    #     else:
    #         print(f"逻辑层{i}: 单层 {layer}")
    print(f'瓶颈层：{bottleneck_segments}')
    print(len(logical_layers))
    # Step 2: 生成逻辑层边界切割点
    logical_cut_points = []
    for i in range(1, len(logical_layers)):# 确保切割点在逻辑层之间,也就是瓶颈块只会再累加分配到别的设备，而不会被切分
        prev_end = logical_layers[i - 1][1] if isinstance(logical_layers[i - 1], tuple) else logical_layers[i - 1] + 1
        curr_start = logical_layers[i][0] if isinstance(logical_layers[i], tuple) else logical_layers[i]
        if curr_start == prev_end:
            logical_cut_points.append(prev_end)

    # Step 3: 生成候选切割方案
    arms = []
    device_list = list(DEVICE_FLOPS.keys())

    # 遍历所有可能的切割点数
    for k in range(1, min(len(logical_cut_points) + 1, len(device_list))):#最大切割点数量是设备数量与切割点数量的较小值
        for cuts in itertools.combinations(logical_cut_points, k):
            # 将逻辑切割点转换为物理层索引
            physical_cuts = []
            for cut in sorted(cuts):
                # 找到切割点对应的物理层结束位置
                for layer in logical_layers:
                    if isinstance(layer, tuple):
                        if cut == layer[1]:
                            physical_cuts.append(layer[1])
                            break
                    else:
                        if cut == layer + 1:
                            physical_cuts.append(layer + 1)
                            break

            # 构建物理层分区
            pts = [0] + physical_cuts + [len(df)]
            partitions = [(s, e) for s, e in zip(pts[:-1], pts[1:])]

            # 生成有效设备分配
            for assignment in generate_valid_assignments(partitions, logical_layers, device_list):
                arms.append((tuple(partitions), assignment))
    # Step 4: 多臂赌博机优化
    mab = EpsilonGreedy(arms, epsilon=0.10)
    for _ in range(rounds):
        arm = mab.select()
        exec_t, trans_t, mem_r = compute_metrics(df, arm[0], arm[1], bottleneck_segments)
        reward = 1 / (exec_t + trans_t + 1e-6) * (1 - mem_r)  # 综合奖励函数
        mab.update(arm, reward)

    # 输出结果
    best = max(arms, key=lambda a: mab.values[a])
    print("\n=== 最优切割方案 ===")
    print_partition_details(df, best[0], best[1], logical_layers)
    return best


def generate_valid_assignments(partitions, logical_layers, devices):
    """生成满足约束的设备分配方案"""
    # 映射分区（里面是元组）到逻辑层类型
    partition_types = []
    for s, e in partitions:
        # 查找包含的逻辑层
        layer_type = []
        current = s
        while current < e:
            for layer in logical_layers:#遍历逻辑层每一段
                if isinstance(layer, tuple): # 瓶颈段
                    if layer[0] <= current < layer[1]:
                        layer_type.append('tensor')
                        current = layer[1]
                        break
                else: # 普通段
                    if layer == current:
                        layer_type.append('pipeline')
                        current += 1
                        break
        # 确定分区类型（混合类型需升级为张量并行）
        partition_types.append('tensor' if 'tensor' in layer_type else 'pipeline')

    # 生成设备分配
    tensor_count = sum(1 for t in partition_types if t == 'tensor')
    pipeline_count = len(partition_types) - tensor_count

    # 设备分配必须满足：张量段数×2 + 流水线段数 ≤ 总设备数
    if 2 * tensor_count + pipeline_count > len(devices):    #这里暂定张量切割为两部分、后面可以改
        return []

    # 生成所有可能的设备组合
    assigned = set()
    for assignment in itertools.permutations(devices, len(partition_types)):
        valid = True
        used = set()
        for dev, ptype in zip(assignment, partition_types):
            if ptype == 'tensor':
                # 张量段需要两个不同设备（此处简化处理，实际需要配对）
                if dev in used:
                    valid = False
                    break
                used.add(dev)
        if valid:
            yield assignment


def print_partition_details(df, partitions, assignment, logical_layers):
    """可视化打印分区详情"""
    for (start, end), dev in zip(partitions, assignment):
        # 查找对应的逻辑层
        layers = []
        current = start
        while current < end:
            for layer in logical_layers:
                if isinstance(layer, tuple):
                    if layer[0] <= current < layer[1]:
                        layers.append(f"瓶颈段({layer[0]}-{layer[1] - 1})")
                        current = layer[1]
                        break
                else:
                    if current == layer:
                        layers.append(f"单层{current}")
                        current += 1
                        break

        # 计算分区指标
        flops = df.loc[start:end - 1, 'flops (MFLOPs)'].sum()
        mem = df.loc[start:end - 1, 'total_memory'].sum() / (1024 ** 3)  # 转换为GB

        print(f"分区 {start}-{end - 1} (设备 {dev}):")
        print(f"  包含逻辑层: {', '.join(layers)}")
        print(f"  计算量: {flops:.1f} MFLOPs, 内存占用: {mem:.2f}GB")
        print("-" * 50)
if __name__ == '__main__':

    model = AdaptiveResNet()
    best_partitions, best_assignment = main(model, rounds=500)
    print(best_assignment)
    print(best_partitions)
    # print("张量并行段设备:", [d for p,d in zip(best_partitions, best_assignment) if any(isinstance(l,tuple) for l in p])