import time
from torchvision.models import resnet50,ResNet50_Weights
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import pandas as pd
from collections import defaultdict, deque
from torch.fx.passes.shape_prop import ShapeProp
import os
from torchvision import transforms
from PIL import Image
from torch.fx import symbolic_trace, GraphModule
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import concurrent.futures
import multiprocessing
from functools import partial
# --- 步骤 1.1：关键路径判断算法识别关键路径。 & 将不可分割节点合并为逻辑块 ---
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

#---浮点数计算模块---
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

#---浮点数计算方法---
def flops_function(fn, out_shape):
    elems = int(torch.tensor(out_shape).prod())
    if fn in (F.relu,): return elems
    if fn in (torch.add,): return elems
    return 0

#---收集每一层内存消耗、计算量消耗、张量形状、是否可以切割，形成df表存储数据---
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


#---构建得到关键边后的逻辑块---
def build_logic_blocks(fx_net):
    """构建逻辑块：将连通且不可在关键路径处剪切的节点合并为块  返回逻辑块列表，每个块为节点名列表"""
    # 构建 FX 图的 DAG
    G = nx.DiGraph()
    for n in fx_net.graph.nodes:
        G.add_node(n.name)
    for n in fx_net.graph.nodes:
        for inp in n.all_input_nodes:
            G.add_edge(inp.name, n.name)
    # 求关键割边
    cuts = find_critical_edges_dag(list(G.edges()))
    # 将连续节点按是否为割边进行分块
    nodes = list(fx_net.graph.nodes)
    blocks = []
    current = []
    for idx, node in enumerate(nodes):
        current.append(node.name)
        # 下一个边
        next_edge = (node.name, nodes[idx+1].name) if idx+1 < len(nodes) else None
        # 如果该边是割边或到末尾，则当前块结束
        if next_edge in cuts or next_edge is None:
            blocks.append(current)
            current = []
    return blocks

    # --- 步骤 1.2：滑动窗口合并检测瓶颈块 ---

def find_bottleneck_segments(df, blocks, window=3, thresh_ratio=1.3):
    """
        merged: 合并后的瓶颈段(节点索引范围)
        merged_topk: 按总计算量排序的前K个瓶颈段
        logical_layers: 逻辑层映射表
    """
    # 步骤1: 计算每个逻辑块的总计算量和节点索引范围
    block_flops = []
    block_indices = []  # 存储每个逻辑块的起始和结束节点索引
    block_nodes = []  # 存储每个逻辑块的节点名列表

    # 创建节点名到df行的映射
    node_to_row = {row['node']: row for _, row in df.iterrows()}

    for block in blocks:
        # 计算逻辑块的总FLOPs
        total_flops = 0
        start_idx = float('inf')
        end_idx = -1
        node_list = []

        for node_name in block:
            row = node_to_row.get(node_name)
            if row is not None:
                total_flops += row['flops (MFLOPs)']
                node_list.append(node_name)
                start_idx = min(start_idx, row['idx'])
                end_idx = max(end_idx, row['idx'])

        if start_idx <= end_idx:  # 确保有效的索引范围
            block_flops.append(total_flops)
            block_indices.append((start_idx, end_idx))
            block_nodes.append(node_list)

    if not block_flops:
        return [], [], []

    # 步骤2: 滑动窗口检测瓶颈段
    avg_flops = sum(block_flops) / len(block_flops)
    segments = []

    # 在逻辑块序列上滑动窗口
    for i in range(len(block_flops) - window + 1):
        window_flops = sum(block_flops[i:i + window])

        # 检查是否超过阈值
        if window_flops > window * avg_flops * thresh_ratio:
            start_idx = block_indices[i][0]  # 窗口第一个块的起始节点索引
            end_idx = block_indices[i + window - 1][1]  # 窗口最后一个块的结束节点索引
            segments.append((start_idx, end_idx, window_flops))

    # 步骤3: 合并重叠段
    if not segments:
        return [], [], []

    # 按起始索引排序
    segments.sort(key=lambda x: x[0])
    merged = []

    current_start, current_end, current_flops = segments[0]
    for seg in segments[1:]:
        start, end, flops = seg
        # 检查是否有重叠
        if start <= current_end:
            # 扩展当前段
            current_end = max(current_end, end)
            current_flops += flops
        else:
            # 保存当前段并开始新段
            merged.append((current_start, current_end))
            current_start, current_end, current_flops = start, end, flops

    merged.append((current_start, current_end))

    # 步骤4: 按总计算量排序取TopK
    merged_topk = []
    for seg in merged:
        start, end = seg
        # 计算该段的总FLOPs
        total_flops = df[(df['idx'] >= start) & (df['idx'] <= end)]['flops (MFLOPs)'].sum()
        merged_topk.append((start, end, total_flops))

    # 按总计算量降序排序
    merged_topk.sort(key=lambda x: x[2], reverse=True)
    # 取前10个瓶颈段
    merged_topk = merged_topk[:5]

    logical_layers = [block.copy() for block in block_nodes]

    # 标记哪些块需要合并
    merge_flags = [False] * len(block_nodes)

    # 对于每个瓶颈段，标记需要合并的块
    for seg_start, seg_end in merged:
        for i, (block_start, block_end) in enumerate(block_indices):
            # 检查块是否完全在瓶颈段内
            if block_start >= seg_start and block_end <= seg_end:
                merge_flags[i] = True

    # 合并连续的标记块
    merged_groups = []
    current_group = []

    for i, flag in enumerate(merge_flags):
        if flag:
            current_group.append(i)
        else:
            if current_group:
                merged_groups.append(current_group)
                current_group = []

    if current_group:
        merged_groups.append(current_group)

    # 从后向前处理合并组，避免索引变化问题
    for group in sorted(merged_groups, key=lambda x: x[0], reverse=True):
        # 创建合并后的节点列表
        merged_nodes = []
        for block_idx in group:
            merged_nodes.extend(logical_layers[block_idx])

        # 替换第一个块为合并后的节点列表
        logical_layers[group[0]] = merged_nodes

        # 删除其他被合并的块
        for block_idx in group[1:]:
            logical_layers[block_idx] = None

    # 过滤掉被删除的块
    logical_layers = [block for block in logical_layers if block is not None]

    return merged, merged_topk, logical_layers

#   根据瓶颈段信息得到索引序号对应的层名
def case_select(df, merged_topk):

    cases = []
    # 遍历每个瓶颈段
    for seg in merged_topk:
        start_idx, end_idx, total_flops = seg

        # 获取起始节点信息
        start_row = df[df['idx'] == start_idx]
        if not start_row.empty:
            cut_layer = start_row.iloc[0]['node']
        else:
            cut_layer = None

        # 获取结束节点信息
        end_row = df[df['idx'] == end_idx]
        if not end_row.empty:
            paste_layer = end_row.iloc[0]['node']
        else:
            paste_layer = None

        # 如果找到了有效的切割点，添加到配置列表
        if cut_layer and paste_layer:
            cases.append({
                'cut_layer': cut_layer,
                'paste_layer': paste_layer,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'total_flops': total_flops
            })

    return cases
#根据case的边重新划分模型，构建瓶颈块的前、后三部分
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

def create_split_fn_4_grid(model, start_layer, end_layer, num_splits=4):
    front, mid, tail = split_model(model, start_layer, end_layer)

    def forward_fn(x):
        # 前向传播
        front_out = front(x)

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
        # 并行处理中间部分
        mid_outs = [mid(chunk) for chunk in chunks]

        # 重新拼接张量
        top = torch.cat([mid_outs[0], mid_outs[1]], dim=3)
        bottom = torch.cat([mid_outs[2], mid_outs[3]], dim=3)
        concat_out = torch.cat([top, bottom], dim=2)

        return tail(concat_out)

    return forward_fn
def create_split_fn_4_weight(model, start_layer, end_layer, num_splits=4):
    front, mid, tail = split_model(model, start_layer, end_layer)

    front, mid, tail = split_model(model, start_layer, end_layer)

    def forward_fn(x):
        front_out = front(x)
        _, _, h, w = front_out.shape
        chunk_width = w // 4  # 计算每份宽度

        # 沿宽度切割为4份
        chunks = [
            front_out[:, :, :, 0:chunk_width],
            front_out[:, :, :, chunk_width:2 * chunk_width],
            front_out[:, :, :, 2 * chunk_width:3 * chunk_width],
            front_out[:, :, :, 3 * chunk_width:]
        ]

        # 并行处理中间层
        mid_outs = [mid(chunk) for chunk in chunks]

        # 沿宽度维度拼接结果
        concat_out = torch.cat(mid_outs, dim=3)
        return tail(concat_out)

    return forward_fn


def create_split_fn_4_height(model, start_layer, end_layer, num_splits=4):
    front, mid, tail = split_model(model, start_layer, end_layer)

    def forward_fn(x):
        front_out = front(x)
        _, _, h, w = front_out.shape
        chunk_height = h // 4  # 计算每份高度

        # 沿高度切割为4份
        chunks = [
            front_out[:, :, 0:chunk_height, :],
            front_out[:, :, chunk_height:2 * chunk_height, :],
            front_out[:, :, 2 * chunk_height:3 * chunk_height, :],
            front_out[:, :, 3 * chunk_height:, :]
        ]

        # 并行处理中间层
        mid_outs = [mid(chunk) for chunk in chunks]

        # 沿高度维度拼接结果
        concat_out = torch.cat(mid_outs, dim=2)
        return tail(concat_out)

    return forward_fn

def create_split_fn_2_weight(model, start_layer, end_layer):
    front, mid, tail = split_model(model, start_layer, end_layer)

    def forward_fn(x):
        # 前向传播
        front_out = front(x)

        # 沿宽度分割为两份
        _, _, h, w = front_out.shape
        w_split = w // 2  # 计算分割点

        # 切割左右两部分
        left_chunk = front_out[:, :, :, :w_split]
        right_chunk = front_out[:, :, :, w_split:]

        # 并行处理中间部分
        mid_left = mid(left_chunk)
        mid_right = mid(right_chunk)

        # 沿宽度重新拼接
        concat_out = torch.cat([mid_left, mid_right], dim=3)

        return tail(concat_out)

    return forward_fn


def create_split_fn_2_height(model, start_layer, end_layer):
    front, mid, tail = split_model(model, start_layer, end_layer)

    def forward_fn(x):
        # 前向传播
        front_out = front(x)

        # 沿高度分割为两份
        _, _, h, w = front_out.shape
        h_split = h // 2  # 计算高度分割点

        # 切割上下两部分
        top_chunk = front_out[:, :, :h_split, :]
        bottom_chunk = front_out[:, :, h_split:, :]

        # 并行处理中间部分
        mid_top = mid(top_chunk)
        mid_bottom = mid(bottom_chunk)

        # 沿高度重新拼接
        concat_out = torch.cat([mid_top, mid_bottom], dim=2)

        return tail(concat_out)

    return forward_fn
def create_split_fn_3_weight(model, start_layer, end_layer, num_splits=3):
    front, mid, tail = split_model(model, start_layer, end_layer)

    def forward_fn(x):
        front_out = front(x)
        _, _, h, w = front_out.shape
        chunk_width = w // 3  # 计算每份宽度

        # 沿宽度切割为4份
        chunks = [
            front_out[:, :, :, 0:chunk_width],
            front_out[:, :, :, chunk_width:2 * chunk_width],
            front_out[:, :, :, 2 * chunk_width:],
        ]

        # 并行处理中间层
        mid_outs = [mid(chunk) for chunk in chunks]

        # 沿宽度维度拼接结果
        concat_out = torch.cat(mid_outs, dim=3)
        return tail(concat_out)

    return forward_fn

def create_split_fn_3_height(model, start_layer, end_layer, num_splits=3):
    front, mid, tail = split_model(model, start_layer, end_layer)
    def forward_fn(x):
        front_out = front(x)
        _, _, h, w = front_out.shape
        chunk_height = h // 3  # 计算每份高度

        # 沿高度切割为4份
        chunks = [
            front_out[:, :, 0:chunk_height, :],
            front_out[:, :, chunk_height:2 * chunk_height, :],
            front_out[:, :, 2 * chunk_height:, :],
        ]

        # 并行处理中间层
        mid_outs = [mid(chunk) for chunk in chunks]

        # 沿高度维度拼接结果
        concat_out = torch.cat(mid_outs, dim=2)
        return tail(concat_out)

    return forward_fn

# 定义 ImageDataset 类加载测试数据集
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_files, images_dir, preprocess):
        self.image_files = image_files
        self.images_dir = images_dir
        self.preprocess = preprocess

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        img = Image.open(img_path).convert('RGB')
        input_tensor = self.preprocess(img)
        return input_tensor, self.image_files[idx]

#精度损失评估类
class MultiModelEvaluator:
    def __init__(self, cases, images_dir, ground_truth_path, batch_size=64):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cases = cases
        self.images_dir = images_dir
        self.ground_truth_path = ground_truth_path
        self.batch_size = batch_size

        # 加载原始模型
        self.base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to(self.device)
        self.base_model.eval()

        # 定义十种种切割方法
        self.split_methods = {
            'split_2_height': create_split_fn_2_height,
            'split_2_weight': create_split_fn_2_weight,
            'split_3_height': create_split_fn_3_height,
            'split_3_weight': create_split_fn_3_weight,
            'split_4_height': create_split_fn_4_height,
            'split_4_weight': create_split_fn_4_weight,
            'split_4_grid': create_split_fn_4_grid
        }

        # 创建所有切割模型
        self.models = {}
        for case in cases:
            case_key = f"{case['cut_layer']}--{case['paste_layer']}"
            self.models[case_key] = {}

            # 为每种切割方法创建模型
            for method_name, split_fn in self.split_methods.items():
                model = self.create_cut_model(case, split_fn)
                self.models[case_key][method_name] = {
                    'model': model,
                    'predictions': []
                }

        # 图像预处理
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # 加载图像文件列表
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith(".JPEG")]
        self.dataset = ImageDataset(self.image_files, images_dir, self.preprocess)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        # 加载真实标签
        self.ground_truth = self.load_ground_truth()

    def create_cut_model(self, case, split_fn):
        """创建切割模型（使用指定的切割函数）"""
        # 使用基础模型的权重引用，避免复制
        model = resnet50(weights=None).to(self.device)

        # 共享基础权重
        for param_src, param_dest in zip(self.base_model.parameters(), model.parameters()):
            param_dest.data = param_src.data
            param_dest.requires_grad = False

        # 应用指定的切割函数
        forward_fn = split_fn(model, case['cut_layer'], case['paste_layer'])
        return forward_fn

    def load_ground_truth(self):
        """加载真实标签"""
        ground_truth = {}
        with open(self.ground_truth_path, 'r') as f:
            for line in f:
                parts = line.strip().split(': ')
                if len(parts) >= 2:
                    img_name = parts[0]
                    true_class = parts[1]
                    ground_truth[img_name] = true_class
        return ground_truth

    def evaluate(self):
        """执行评估（包含七种切割方式）"""
        start_time = time.time()

        # 使用大batch一次处理所有模型
        with torch.no_grad():
            for inputs, img_names in tqdm(self.dataloader, desc="处理批次"):
                inputs = inputs.to(self.device, non_blocking=True)

                # 对每个切割方案执行推理
                for case_key, methods in self.models.items():
                    for method_name, method_info in methods.items():
                        outputs = method_info['model'](inputs)
                        probabilities = torch.nn.functional.softmax(outputs, dim=1)
                        predicted_classes = torch.argmax(probabilities, dim=1).cpu().numpy()

                        # 存储预测结果
                        for img_name, pred_class in zip(img_names, predicted_classes):
                            img_base = os.path.splitext(img_name)[0]
                            method_info['predictions'].append((img_base, str(pred_class)))

        # 计算准确率并保存结果
        for case in self.cases:
            case_key = f"{case['cut_layer']}--{case['paste_layer']}"
            case_methods = self.models.get(case_key, {})

            # 初始化准确率字典
            case['accuracies'] = {}
            best_accuracy = 0
            best_method = None

            for method_name, method_info in case_methods.items():
                predict_txt = f"test5h/CutPredict{case['cut_layer']}--{case['paste_layer']}_{method_name}_5h.txt"

                # 写入预测文件
                with open(predict_txt, 'w') as f:
                    for img_base, pred_class in method_info['predictions']:
                        f.write(f"{img_base}: {pred_class}\n")

                # 计算准确率
                correct = 0
                total = 0
                for img_base, pred_class in method_info['predictions']:
                    true_class = self.ground_truth.get(img_base)
                    if true_class and true_class == pred_class:
                        correct += 1
                    total += 1

                accuracy = correct / total if total > 0 else 0
                case['accuracies'][method_name] = accuracy

                # 更新最佳方法
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_method = method_name

                print(f"切割方案 {case_key} 方法 {method_name} 准确率: {accuracy:.4f} ({correct}/{total})")

            # 记录最佳准确率和方法
            case['best_accuracy'] = best_accuracy
            case['best_method'] = best_method
            print(f"切割方案 {case_key} 最佳方法: {best_method} 准确率: {best_accuracy:.4f}")

        elapsed = time.time() - start_time
        print(f"评估完成! 总耗时: {elapsed:.2f}秒")
        return self.cases

#重计算每个块的峰值内存、flops等数据
def preprocess_blocks(logic_blocks, df, input_size=0):
    """预处理逻辑块信息（支持分支结构）"""
    # 步骤1: 构建块级DAG
    block_edges = []
    block_to_index = {}
    node_to_block = {}

    # 建立节点到块的映射
    for block_idx, block in enumerate(logic_blocks):
        block_to_index[tuple(block)] = block_idx
        for node in block:
            node_to_block[node] = block_idx

    # 构建块间连接关系
    for i in range(len(logic_blocks)):
        current_block = logic_blocks[i]
        last_node = current_block[-1]

        # 查找当前块最后一个节点的所有后继节点
        last_node_row = df[df['node'] == last_node].iloc[0]
        if last_node_row['edge_to_next'] is None:
            continue

        successor_node = last_node_row['edge_to_next'][1]
        if successor_node in node_to_block:
            successor_block_idx = node_to_block[successor_node]
            if successor_block_idx != i:  # 避免自环
                block_edges.append((i, successor_block_idx))

    # 步骤2: 计算每个块的内部峰值和输出大小
    block_peaks = []
    block_output_sizes = []

    for block in logic_blocks:
        block_df = df[df['node'].isin(block)].sort_values('idx')

        if block_df.empty:
            block_peaks.append(0)
            block_output_sizes.append(0)
            continue

        # 计算块内峰值激活（考虑输入-输出叠加）
        intra_peak = 0
        prev_output = input_size if block == logic_blocks[0] else 0

        for _, row in block_df.iterrows():
            # 节点计算时瞬时内存 = 输入激活 + 输出激活
            instant_activation = prev_output + row['size']
            if instant_activation > intra_peak:
                intra_peak = instant_activation
            prev_output = row['size']  # 当前输出成为下一节点的输入

        block_peaks.append(intra_peak)
        block_output_sizes.append(block_df.iloc[-1]['size'])

    # 步骤3: 处理分支结构
    block_info = []

    for block_idx, block in enumerate(logic_blocks):
        block_df = df[df['node'].isin(block)]

        # 查找当前块的所有直接后继块
        successors = [v for u, v in block_edges if u == block_idx]

        if len(successors) > 1:  # 存在分支
            # 计算每个分支的峰值内存
            branch_peaks = [block_peaks[succ_idx] for succ_idx in successors]

            # 并行峰值 = 当前块输出 + 所有分支峰值之和
            parallel_peak = block_output_sizes[block_idx] + sum(branch_peaks)

            # 最终峰值 = max(块内峰值, 并行峰值)
            peak_activation = max(block_peaks[block_idx], parallel_peak)
        else:
            peak_activation = block_peaks[block_idx]  # 无分支使用块内峰值

        # 计算块的其他信息
        total_flops = block_df['flops (MFLOPs)'].sum() * 1e6
        weight_mem = block_df['weight_memory'].sum()
        bias_mem = block_df['bias_memory'].sum()
        output_size = block_output_sizes[block_idx]

        block_info.append({
            'flops': total_flops,
            'weight_mem': weight_mem,
            'bias_mem': bias_mem,
            'peak_activation': peak_activation,
            'output_size': output_size,
            'total_memory': weight_mem + bias_mem + peak_activation
        })

    return block_info


def generate_and_select_cut_schemes(logic_blocks, big_block_index, block_number, num_devices, top_k=50):
    """
    生成切割方案并根据均衡性筛选优质方案
    :param logic_blocks: 逻辑块信息
    :param big_block_index: 超大块起始索引
    :param block_number: 超大块分割份数
    :param num_devices: 设备数量
    :param top_k: 保留的优质方案数量
    :return: 优质切割方案列表
    """
    num_blocks = len(logic_blocks)
    candidate_schemes = []

    # 超大块内部必须的切割点 (将超大块分为block_number份)
    big_block_cuts = []
    if block_number > 1:
        # 在超大块内部生成block_number-1个切割点
        big_block_cuts = list(np.linspace(
            big_block_index + 1,
            big_block_index + len(logic_blocks[big_block_index]['ops']) - 1,
            block_number,
            dtype=int
        ))[:-1]  # 去掉最后一个点（已经是边界）

    # 生成非超大块区域的候选切割点 (设备数量 - block_number - 1)
    non_big_cut_count = num_devices - block_number - 1
    candidate_positions = [
        i for i in range(1, num_blocks)
        if i < big_block_index or i >= big_block_index + len(logic_blocks[big_block_index]['ops'])
    ]

    # 随机生成切割方案
    for _ in range(1000):  # 生成1000个候选方案
        if non_big_cut_count > 0 and candidate_positions:
            non_big_cuts = sorted(random.sample(candidate_positions, non_big_cut_count))
        else:
            non_big_cuts = []

        full_cuts = sorted(non_big_cuts + big_block_cuts)
        candidate_schemes.append(full_cuts)

    # 计算每个切割方案的两个均衡分数
    scored_schemes = []
    for cuts in candidate_schemes:
        compute_score, comm_score = evaluate_cut_scheme(logic_blocks, cuts, big_block_index, block_number)
        # 使用调和平均作为综合分数
        combined_score = 2 * compute_score * comm_score / (compute_score + comm_score + 1e-9)
        scored_schemes.append((combined_score, cuts, compute_score, comm_score))

    # 选择分数最高的top_k个方案
    scored_schemes.sort(key=lambda x: x[0], reverse=True)
    return [scheme[1] for scheme in scored_schemes[:top_k]]


def evaluate_cut_scheme(logic_blocks, cut_points, big_block_index, block_number):
    """
    评估切割方案的两个均衡分数
    :return: (计算量均衡分数, 通信量均衡分数)
    """
    segments = split_into_segments(logic_blocks, cut_points, big_block_index, block_number)

    # 1. 计算量均衡分数 (各段计算量的标准差倒数)
    compute_loads = [seg['total_flops'] for seg in segments]
    compute_std = np.std(compute_loads)
    compute_score = 1 / (compute_std + 1e-9)

    # 2. 通信量均衡分数 (各切割点通信量的标准差倒数)
    comm_loads = []
    for i in range(len(segments) - 1):
        comm_loads.append(segments[i]['output_size'])  # 段间通信量

    comm_std = np.std(comm_loads)
    comm_score = 1 / (comm_std + 1e-9)

    return compute_score, comm_score


def split_into_segments(logic_blocks, cut_points, big_block_index, block_number):
    """
    根据切割点划分逻辑块为多个段落
    """
    segments = []
    start = 0
    all_cuts = sorted(cut_points)

    for cut in all_cuts:
        segment_blocks = logic_blocks[start:cut]
        segments.append(calculate_segment_stats(segment_blocks))
        start = cut

    # 添加最后一个段落
    last_segment = logic_blocks[start:]
    segments.append(calculate_segment_stats(last_segment))

    # 处理超大块的特殊分割
    for i, seg in enumerate(segments):
        if big_block_index in [block['index'] for block in seg['blocks']]:
            # 超大块需要分割为多个子段落
            big_segments = split_big_block(seg, big_block_index, block_number)
            segments = segments[:i] + big_segments + segments[i + 1:]
            break

    return segments


def calculate_segment_stats(blocks):
    """计算段落的统计信息"""
    total_flops = sum(block['flops'] for block in blocks)
    total_memory = max(block['memory'] for block in blocks)  # 峰值内存
    output_size = blocks[-1]['output_size'] if blocks else 0

    return {
        'blocks': blocks,
        'total_flops': total_flops,
        'total_memory': total_memory,
        'output_size': output_size
    }


def split_big_block(segment, big_block_index, block_number):
    """分割超大块为多个子段落"""
    big_block = None
    other_blocks = []

    # 分离超大块和其他块
    for block in segment['blocks']:
        if block['index'] == big_block_index:
            big_block = block
        else:
            other_blocks.append(block)

    if not big_block:
        return [segment]

    # 分割超大块内部的算子
    ops = big_block['ops']
    op_chunks = np.array_split(ops, block_number)

    big_segments = []
    for chunk in op_chunks:
        # 创建子块
        sub_block = {
            'index': big_block_index,
            'ops': chunk,
            'flops': sum(op['flops'] for op in chunk),
            'memory': max(op['memory'] for op in chunk),
            'output_size': chunk[-1]['output_size'] if chunk else 0
        }
        # 每个子段落只包含超大块的一个子块
        big_segments.append({
            'blocks': [sub_block],
            'total_flops': sub_block['flops'],
            'total_memory': sub_block['memory'],
            'output_size': sub_block['output_size']
        })

    return big_segments


def optimize_device_assignment(segments, device_flops, mem_limits, bandwidth_matrix):
    """
    优化段落到设备的分配（考虑异构带宽）
    :param segments: 段落列表（长度等于设备数）
    :param device_flops: 设备计算能力 {device_id: FLOPS}
    :param mem_limits: 设备内存限制 {device_id: bytes}
    :param bandwidth_matrix: 设备间带宽矩阵 [i][j] = 从设备i到j的带宽 (bps)
    :return: 帕累托最优解列表，每个解为(分配方案, 目标值)
    """
    num_devices = len(device_flops)
    device_ids = list(device_flops.keys())

    # NSGA-II 参数
    POP_SIZE = 100
    MAX_GEN = 200
    CX_RATE = 0.85
    MUT_RATE = 0.15

    # 初始化种群 (分配方案)
    population = []
    for _ in range(POP_SIZE):
        # 随机排列段落
        perm = list(range(len(segments)))
        random.shuffle(perm)
        population.append(perm)

    # 存储帕累托前沿
    pareto_front = []

    for gen in range(MAX_GEN):
        # 评估种群
        fitness = [evaluate_assignment(perm, segments, device_flops, mem_limits, bandwidth_matrix, device_ids)
                   for perm in population]

        # 非支配排序
        fronts = non_dominated_sorting(fitness)

        # 更新帕累托前沿 (保留第一前沿)
        current_front = [population[i] for i in fronts[0]]
        pareto_front = update_pareto_front(pareto_front, current_front, fitness, fronts[0])

        # 选择父代 (基于前沿等级)
        parents = []
        for rank, front in enumerate(fronts):
            if len(parents) + len(front) > POP_SIZE // 2:
                # 按拥挤度选择部分
                crowding = calculate_crowding([fitness[i] for i in front])
                selected = select_by_crowding(front, crowding, POP_SIZE // 2 - len(parents))
                parents.extend([population[i] for i in selected])
                break
            else:
                parents.extend([population[i] for i in front])

        # 生成子代
        offspring = []
        while len(offspring) < POP_SIZE - len(parents):
            p1, p2 = random.sample(parents, 2)

            # 交叉
            if random.random() < CX_RATE:
                c1, c2 = order_crossover(p1, p2)
            else:
                c1, c2 = p1[:], p2[:]

            # 变异
            if random.random() < MUT_RATE:
                c1 = swap_mutation(c1)
            if random.random() < MUT_RATE:
                c2 = swap_mutation(c2)

            offspring.append(c1)
            if len(offspring) < POP_SIZE - len(parents):
                offspring.append(c2)

        # 新一代种群 = 父代 + 子代
        population = parents + offspring

    # 返回帕累托最优解（分配方案和对应的目标值）
    return [(assign, evaluate_assignment(assign, segments, device_flops, mem_limits, bandwidth_matrix, device_ids))
            for assign in pareto_front]


def evaluate_assignment(assignment, segments, device_flops, mem_limits, bandwidth_matrix, device_ids):
    """
    评估分配方案的四个目标值
    :return: (总延迟, 总通信量, 内存不均衡度, 计算负载不均衡度)
    """
    num_devices = len(device_ids)

    # 1. 计算每个设备的计算时间
    compute_times = []
    memory_usages = []
    for i, device_id in enumerate(device_ids):
        seg_idx = assignment[i]
        seg = segments[seg_idx]
        compute_time = seg['total_flops'] / device_flops[device_id]
        compute_times.append(compute_time)
        memory_usages.append(seg['total_memory'] / mem_limits[device_id])

    # 2. 计算通信时间
    comm_times = []
    comm_volumes = []
    for i in range(num_devices - 1):
        src_seg = segments[assignment[i]]
        dst_seg = segments[assignment[i + 1]]

        # 查找源设备和目标设备ID
        src_device_idx = assignment.index(assignment[i])
        dst_device_idx = assignment.index(assignment[i + 1])
        src_device = device_ids[src_device_idx]
        dst_device = device_ids[dst_device_idx]

        # 获取设备间带宽
        bandwidth = bandwidth_matrix[src_device][dst_device]
        comm_volume = src_seg['output_size']
        comm_time = comm_volume / bandwidth if bandwidth > 0 else float('inf')

        comm_times.append(comm_time)
        comm_volumes.append(comm_volume)

    # 3. 计算总延迟 (关键路径)
    # 使用动态规划计算流水线延迟
    delays = [compute_times[0]]
    for i in range(1, num_devices):
        comm_delay = comm_times[i - 1] if i - 1 < len(comm_times) else 0
        start_time = max(delays[i - 1], delays[i - 1] + comm_delay)
        delays.append(start_time + compute_times[i])
    total_delay = delays[-1]

    # 4. 总通信量
    total_comm = sum(comm_volumes)

    # 5. 内存不均衡度 (标准差)
    mem_imbalance = np.std(memory_usages)

    # 6. 计算负载不均衡度 (标准差)
    compute_imbalance = np.std(compute_times)

    return (total_delay, total_comm, mem_imbalance, compute_imbalance)


def non_dominated_sorting(fitness_values):
    """快速非支配排序"""
    num_solutions = len(fitness_values)
    dominates = [[] for _ in range(num_solutions)]
    dominated_by = [0] * num_solutions
    fronts = [[]]

    # 计算支配关系
    for i in range(num_solutions):
        for j in range(num_solutions):
            if i == j:
                continue
            if dominates_solution(fitness_values[i], fitness_values[j]):
                dominates[i].append(j)
            elif dominates_solution(fitness_values[j], fitness_values[i]):
                dominated_by[i] += 1

        if dominated_by[i] == 0:
            fronts[0].append(i)

    # 分层其他前沿
    current_front = 0
    while fronts[current_front]:
        next_front = []
        for i in fronts[current_front]:
            for j in dominates[i]:
                dominated_by[j] -= 1
                if dominated_by[j] == 0:
                    next_front.append(j)

        if next_front:
            fronts.append(next_front)
        current_front += 1

    return fronts


def dominates_solution(a, b):
    """检查解a是否支配解b (所有目标最小化)"""
    # a在所有目标上不劣于b，且至少一个目标严格更好
    not_worse = all(a_i <= b_i for a_i, b_i in zip(a, b))
    better = any(a_i < b_i for a_i, b_i in zip(a, b))
    return not_worse and better


def calculate_crowding(fitness_values):
    """计算拥挤度距离"""
    num_solutions = len(fitness_values)
    if num_solutions == 0:
        return []

    num_objectives = len(fitness_values[0])
    crowding = [0.0] * num_solutions

    for obj_idx in range(num_objectives):
        # 按当前目标值排序
        sorted_indices = sorted(range(num_solutions), key=lambda i: fitness_values[i][obj_idx])

        # 边界解有无限拥挤度
        crowding[sorted_indices[0]] = float('inf')
        crowding[sorted_indices[-1]] = float('inf')

        if len(sorted_indices) <= 2:
            continue

        # 归一化目标值范围
        min_val = fitness_values[sorted_indices[0]][obj_idx]
        max_val = fitness_values[sorted_indices[-1]][obj_idx]
        value_range = max_val - min_val
        if value_range < 1e-9:
            continue

        # 计算内部解的拥挤度
        for i in range(1, len(sorted_indices) - 1):
            idx = sorted_indices[i]
            next_val = fitness_values[sorted_indices[i + 1]][obj_idx]
            prev_val = fitness_values[sorted_indices[i - 1]][obj_idx]
            crowding[idx] += (next_val - prev_val) / value_range

    return crowding


def update_pareto_front(current_front, candidates, fitness_values, candidate_indices):
    """更新帕累托前沿"""
    new_front = current_front.copy()
    candidate_fitness = [fitness_values[i] for i in candidate_indices]

    # 添加非支配候选解
    for i, cand in enumerate(candidates):
        cand_fit = candidate_fitness[i]

        # 检查是否被当前前沿支配
        dominated = False
        to_remove = []

        for j, sol in enumerate(new_front):
            sol_fit = evaluate_assignment(sol)  # 需要实际评估函数

            if dominates_solution(sol_fit, cand_fit):
                dominated = True
                break
            elif dominates_solution(cand_fit, sol_fit):
                to_remove.append(j)

        if not dominated:
            # 移除被新解支配的旧解
            for idx in sorted(to_remove, reverse=True):
                new_front.pop(idx)
            new_front.append(cand)

    return new_front


def order_crossover(p1, p2):
    """顺序交叉 (OX)"""
    size = len(p1)
    a, b = sorted(random.sample(range(size), 2))

    # 创建子代
    c1 = [-1] * size
    c2 = [-1] * size

    # 复制中间段
    c1[a:b] = p1[a:b]
    c2[a:b] = p2[a:b]

    # 填充剩余位置
    fill_crossover(c1, p2, b, a)
    fill_crossover(c2, p1, b, a)

    return c1, c2


def fill_crossover(child, parent, start, end):
    """填充交叉后的剩余位置"""
    size = len(child)
    current = start
    ptr = start

    while -1 in child:
        if ptr >= size:
            ptr = 0

        if parent[ptr] not in child:
            child[current] = parent[ptr]
            current = (current + 1) % size

        ptr = (ptr + 1) % size

        # 防止无限循环
        if ptr == start:
            break


def swap_mutation(perm):
    """交换变异"""
    a, b = random.sample(range(len(perm)), 2)
    perm[a], perm[b] = perm[b], perm[a]
    return perm


def find_global_optimum(cut_schemes, logic_blocks, device_flops, big_block_index,mem_limits, bandwidth_matrix):
    """
    全局优化：从优质切割方案中寻找最佳切割点+设备分配
    :return: (最佳切割点, 最佳设备分配, 目标值)
    """
    all_pareto_solutions = []

    # 对每个切割方案优化设备分配
    for i, cuts in enumerate(cut_schemes):
        print(f"优化切割方案 {i + 1}/{len(cut_schemes)}: 切割点 {cuts}")
        segments = split_into_segments(logic_blocks, cuts, big_block_index, block_number)
        pareto_solutions = optimize_device_assignment(
            segments, device_flops, mem_limits, bandwidth_matrix
        )

        # 存储切割方案信息
        for assign, objectives in pareto_solutions:
            all_pareto_solutions.append({
                'cut_scheme': cuts,
                'assignment': assign,
                'objectives': objectives
            })

    # 非支配排序所有解
    fitness_values = [sol['objectives'] for sol in all_pareto_solutions]
    fronts = non_dominated_sorting(fitness_values)

    # 返回帕累托前沿的所有解
    pareto_front = [all_pareto_solutions[i] for i in fronts[0]]
    return pareto_front
# def backbone_optimize(logic_blocks, big_block_index, block_number,df, device_flops, mem_limit, bandwidth_bps):
#     # 设置2个固定切割点：
#     # 1. 在超大块前（原index_front位置）
#     fixed_cut1 = big_block_index
#     # 2. 在超大块内部（分割两部分）
#     fixed_cut2 = big_block_index + 1
#     # 3. 在超大块后（原index_front+1位置）
#     fixed_cuts = [fixed_cut1, fixed_cut2]
#     block_info = preprocess_blocks(logic_blocks, df)
#     num_blocks = len(block_info)
#     num_devices = len(device_flops)
#     num_backbone_method = 20    #一次生成20个优质方案
#     # 计算切割点数量（如果是切割为4份子张量，寻找num_devices-6,如果切割为3份则num_devices-5，依次num_devices-4）
#     num_cuts = num_devices - block_number - 2
#     population = backbone_method_select(num_blocks, num_cuts, fixed_cuts,num_backbone_method)
#
#     return population, block_info, fixed_cuts
#
# def backbone_method_select(num_blocks,num_cuts,fixed_cuts,num_backbone_method):
#     population = 1
#     return population
#

# 主流程示例
if __name__ == "__main__":
    # 初始化模型并进行 shape propagation
    model = resnet50()
    fx = symbolic_trace(model)
    ShapeProp(fx).propagate(torch.randn(1, 3, 224, 224))

    # 1.1 构建逻辑块
    blocks = build_logic_blocks(fx)
    print("关键边后的逻辑块blocks:", blocks)
    df = profile_and_tabulate(model)
    # 1.2 滑动窗口合并
    merged,merged_topk,logical_layers = find_bottleneck_segments(df, blocks,window=2,thresh_ratio=1.8)
    print("合并后的瓶颈块merged:", merged)
    print("数据量降序排序的瓶颈块merged_topk:",merged_topk)
    print("包含合并瓶颈块以及前面逻辑块的完整层logical_layers:", logical_layers)

    cases = case_select(df, merged_topk)
    print(f'验证的瓶颈层案例cases：{cases}')

    # start_time = time.time()
    # cases = evaluate_precision_decay(cases)
    # print(f'预测精度后的cases{cases}')
    # # # print(f'预测精度后的cases{evaluate_precision_decay(cases)}')
    # end_time = time.time()
    # print(f'调用精度函数耗时{end_time - start_time}s')

    images_dir = r"../5h"
    ground_truth_path = r"ground_truth_label_5h.txt"

    # 创建评估器
    evaluator = MultiModelEvaluator(
        cases=cases,
        images_dir=images_dir,
        ground_truth_path=ground_truth_path,
        batch_size=32  # 更大的批处理大小
    )

    # 执行评估
    results = evaluator.evaluate()


    #
    # accuracy_list = [75.68,74.90,74.91]
    # for i in range(3):
    #     cases[i]['accuracy']=accuracy_list[i]

    best_case = max(results, key=lambda x: x['best_accuracy'])
    print(f"最佳切割方案: {best_case['cut_layer']}--{best_case['paste_layer']}")
    print(f"最佳切割方法: {best_case['best_method']}")

    #帕累托边界
    index_front = 0
    block_number = 0
    for i in range(len(logical_layers)):
        if logical_layers[i][0]==best_case['cut_layer']:
            index_front = i
    index_back = index_front+1
    block_number = 0
    if best_case['best_method'] in ['split_4_height','split_4_weight','split_4_grid']:
        block_number = 4
    elif best_case['best_method'] in ['split_3_height','split_3_weight']:
        block_number = 3
    else:
        block_number = 2
    # front_block = logical_layers[:index_front]
    # back_block = logical_layers[index_back:]
    num_devices = 7
    device_flops = {
        'rpi5_1': 320e9,
        'rpi5_2': 320e9,
        'rpi5_3': 320e9,
        'jetson1': 472e9,
        'jetson2': 472e9,
        'jetson3': 472e9,
        'pc_cpu': 500e9,
    }
    mem_limits = {
        'rpi5_1': 16e9,
        'rpi5_2': 16e9,
        'rpi5_3': 16e9,
        'jetson1': 32e9,
        'jetson2': 32e9,
        'jetson3': 32e9,
        'pc_cpu': 64e9,
    }

    # 3. 带宽矩阵 (设备i到设备j的带宽)
    bandwidth_matrix = {
        'rpi5_1': {'rpi5_1': 0,'rpi5_2': 102.1, 'rpi5_3': 94.5, 'jetson1': 55.6, 'jetson2': 57.8,'jetson3': 55.5,'pc_cpu': 66.7},
        'rpi5_2': {'rpi5_1': 102.1,'rpi5_2': 0, 'rpi5_3': 103.9, 'jetson1': 49.3, 'jetson2': 49.9,'jetson3': 59.9,'pc_cpu': 87.5},
        'rpi5_3': {'rpi5_1': 94.5,'rpi5_2': 103.9, 'rpi5_3': 0, 'jetson1': 54.9, 'jetson2': 41.3, 'jetson3': 50.1, 'pc_cpu': 85.5},
        'jetson1': {'rpi5_1': 55.6,'rpi5_2': 49.3, 'rpi5_3': 54.9, 'jetson1': 0, 'jetson2': 207.5, 'jetson3': 214.1, 'pc_cpu': 170.0},
        'jetson2': {'rpi5_1': 57.8,'rpi5_2': 49.9, 'rpi5_3': 41.3, 'jetson1': 207.5, 'jetson2': 0, 'jetson3': 207.7, 'pc_cpu': 178.5},
        'jetson3': {'rpi5_1': 55.5,'rpi5_2': 59.9, 'rpi5_3': 50.1, 'jetson1': 214.1, 'jetson2': 207.7, 'jetson3': 0, 'pc_cpu': 152.3},
        'pc_cpu': {'rpi5_1': 66.7,'rpi5_2': 87.5, 'rpi5_3': 85.5, 'jetson1': 170.0, 'jetson2': 178.5, 'jetson3': 152.3, 'pc_cpu': 0},
    }

    # 第一阶段：生成并筛选优质切割方案
    top_cut_schemes = generate_and_select_cut_schemes(
        logical_layers, index_front, block_number, num_devices, top_k=50
    )

    # 第二阶段：全局优化
    pareto_optimal = find_global_optimum(
        top_cut_schemes, logical_layers, device_flops, mem_limits, bandwidth_matrix
    )

    print(f"找到 {len(pareto_optimal)} 个帕累托最优解")
    for i, sol in enumerate(pareto_optimal):
        print(f"方案 {i + 1}:")
        print(f"  切割点: {sol['cut_scheme']}")
        print(f"  设备分配: {sol['assignment']}")
        print(f"  目标值 (延迟, 通信量, 内存不均衡, 计算不均衡): {sol['objectives']}")