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


def find_bottleneck_segments(df, blocks, thresh_ratio=1.8):
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

    # 步骤2: 使用滑动窗口检测瓶颈块
    n_blocks = len(block_flops)
    avg_flops = sum(block_flops) / n_blocks
    segments = []  # 存储检测到的瓶颈段
    left = 0  # 窗口左指针
    current_sum = 0  # 当前窗口内FLOPs总和

    # 使用右指针遍历所有逻辑块
    for right in range(n_blocks):
        current_sum += block_flops[right]
        window_size = right - left + 1
        current_avg = current_sum / window_size

        # 当窗口平均值小于等于阈值时，记录为瓶颈段
        if current_avg <= thresh_ratio * avg_flops:
            # 记录当前窗口（排除新加入的右边界元素）
            if window_size > 1:  # 至少2个块才能形成有效段
                start_idx = block_indices[left][0]
                end_idx = block_indices[right - 1][1]
                segments.append((start_idx, end_idx))

            # 重置窗口到下一个位置
            left = right + 1
            current_sum = 0
        # 否则继续扩展窗口（自动发生在循环中）

    # 检查剩余未处理块
    if left < n_blocks:
        # 剩余块作为一个整体检查
        window_size = n_blocks - left
        current_avg = sum(block_flops[left:]) / window_size
        if current_avg <= thresh_ratio * avg_flops:
            start_idx = block_indices[left][0]
            end_idx = block_indices[-1][1]
            segments.append((start_idx, end_idx))
    # 步骤4: 按总计算量排序取TopK
    merged_topk = []
    for seg in segments:
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
    for seg_start, seg_end in segments:
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

    return segments, merged_topk, logical_layers


def case_select(df, merged_topk):

    #   根据瓶颈段信息生成切割点配置
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

        # print(f'前向传播后shape：{front_out.shape}')

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

        # for chunk in chunks:
        #     print(f'mid前每个子张量shape：{chunk.shape}')

        # 并行处理中间部分
        mid_outs = [mid(chunk) for chunk in chunks]

        # for mid_out in mid_outs:
        #     print(f'mid后每个字张量shape{mid_out.shape}')

        # 重新拼接张量
        top = torch.cat([mid_outs[0], mid_outs[1]], dim=3)
        bottom = torch.cat([mid_outs[2], mid_outs[3]], dim=3)
        concat_out = torch.cat([top, bottom], dim=2)

        # print(f'拼接后的shape：{concat_out.shape}')
        # # 后向传播
        # print(f'最后shape：{tail(concat_out).shape}')

        return tail(concat_out)

    return forward_fn
# 定义 ImageDataset 类
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

        # 创建所有切割模型
        self.models = {}
        for case in cases:
            key = f"{case['cut_layer']}--{case['paste_layer']}"
            model = self.create_cut_model(case)
            self.models[key] = {
                'model': model,
                'case': case,
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

    def create_cut_model(self, case):
        """创建切割模型（不复制基础权重）"""
        # 使用基础模型的权重引用，避免复制
        model = resnet50(weights=None).to(self.device)

        # 共享基础权重
        for param_src, param_dest in zip(self.base_model.parameters(), model.parameters()):
            param_dest.data = param_src.data
            param_dest.requires_grad = False

        # 应用切割
        split_fn = create_split_fn(model, case['cut_layer'], case['paste_layer'])
        return split_fn

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
        """执行评估"""
        start_time = time.time()

        # 使用大batch一次处理所有模型
        with torch.no_grad():
            for inputs, img_names in tqdm(self.dataloader, desc="处理批次"):
                inputs = inputs.to(self.device, non_blocking=True)

                # 对每个模型执行推理
                for key, model_info in self.models.items():
                    outputs = model_info['model'](inputs)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    predicted_classes = torch.argmax(probabilities, dim=1).cpu().numpy()

                    # 存储预测结果
                    for img_name, pred_class in zip(img_names, predicted_classes):
                        img_base = os.path.splitext(img_name)[0]
                        model_info['predictions'].append((img_base, str(pred_class)))

        # 计算准确率并保存结果
        for key, model_info in self.models.items():
            case = model_info['case']
            predict_txt = f"test5h/CutPredict{case['cut_layer']} -- {case['paste_layer']}_5h.txt"

            # 写入预测文件
            with open(predict_txt, 'w') as f:
                for img_base, pred_class in model_info['predictions']:
                    f.write(f"{img_base}: {pred_class}\n")

            # 计算准确率
            correct = 0
            total = 0
            for img_base, pred_class in model_info['predictions']:
                true_class = self.ground_truth.get(img_base)
                if true_class and true_class == pred_class:
                    correct += 1
                total += 1

            accuracy = correct / total if total > 0 else 0
            case['accuracy'] = accuracy
            print(f"切割方案 {key} 准确率: {accuracy:.4f} ({correct}/{total})")

        elapsed = time.time() - start_time
        print(f"评估完成! 总耗时: {elapsed:.2f}秒")
        return self.cases

def preprocess_blocks(logic_blocks, df):
    """预处理逻辑块信息"""
    block_info = []
    for block in logic_blocks:
        # 获取当前块的所有节点数据
        block_df = df[df['node'].isin(block)]

        # 计算块的总计算量 (FLOPs)
        total_flops = block_df['flops (MFLOPs)'].sum() * 1e6

        # 计算块的权重和偏置内存 (字节)
        weight_mem = block_df['weight_memory'].sum()
        bias_mem = block_df['bias_memory'].sum()

        # 计算块的峰值激活内存 (字节)
        peak_activation = block_df['size'].max()

        # 块的输出大小 (最后一个节点的输出)
        output_size = block_df.iloc[-1]['size']

        block_info.append({
            'flops': total_flops,
            'weight_mem': weight_mem,
            'bias_mem': bias_mem,
            'peak_activation': peak_activation,
            'output_size': output_size,
            'total_memory': weight_mem + bias_mem + peak_activation
        })
    return block_info

# --- 步骤 1.4：NSGA-II 多目标优化寻找切割方案的帕累托前沿 ---


def nsga2_optimize(logic_blocks, big_block_index, df, device_flops, mem_limit, bandwidth_bps,
                   pop_size=50, ngen=100, threshold=0.01, crossover_rate=0.8, mutation_rate=0.1):
    """
    使用 NSGA-II 算法优化模型划分方案
    切割点数量固定为设备数量减一
    """
    # big_block = logic_blocks[big_block_index]

    # 设置2个固定切割点：
    # 1. 在超大块前（原index_front位置）
    fixed_cut1 = big_block_index
    # 2. 在超大块内部（分割两部分）
    fixed_cut2 = big_block_index + 1
    # 3. 在超大块后（原index_front+1位置）

    fixed_cuts = [fixed_cut1, fixed_cut2]

    # 1. 预处理逻辑块信息
    block_info = preprocess_blocks(logic_blocks, df)

    split_ratio = 0.5  # 等分
    big_block_part1 = {
        'name': "bottleneck_part1",
        'flops': block_info[big_block_index]['flops'] * split_ratio,
        'total_memory': block_info[big_block_index]['total_memory'] * split_ratio,
        'output_size': block_info[big_block_index]['output_size'] * split_ratio
    }
    big_block_part2 = {
        'name': "bottleneck_part1",
        'flops': block_info[big_block_index]['flops'] * (1 - split_ratio),
        'total_memory': block_info[big_block_index]['total_memory'] * (1 - split_ratio),
        'output_size': block_info[big_block_index]['output_size'] * (1 - split_ratio)
    }

    # # 创建新的逻辑块列表（包含拆分后的超大块）
    new_logic_blocks = (
            logic_blocks[:big_block_index] +
            [big_block_part1, big_block_part2] +
            logic_blocks[big_block_index + 1:]
    )
    num_blocks = len(block_info)
    num_devices = len(device_flops)

    # 计算切割点数量（设备数量减一）
    num_cuts = num_devices - 4

    # 2. 初始化种群
    population = initialize_population(num_blocks, num_cuts, pop_size,fixed_cuts)
    # 3. NSGA-II 主循环
    prev_front = None
    for gen in range(ngen):
        # print(f'population:{population}')

        # 评估种群
        fitness = []
        for ind in population:
            # try:
            fit = evaluate_individual(ind, big_block_index,block_info, device_flops, mem_limit, bandwidth_bps)
            fitness.append(fit)
            # except Exception as e:
            #     print(f"评估个体时出错11: {e}")
            #     fitness.append((float('inf'), float('inf'), float('inf'), float('inf')))

        # 非支配排序
        fronts, ranks = non_dominated_sorting(fitness)

        # 如果没有前沿，跳过这一代
        if not fronts or not any(fronts):
            print(f"第 {gen} 代: 没有找到前沿")
            population = initialize_population(num_blocks, num_cuts, pop_size,fixed_cuts)
            continue

        # 计算拥挤度
        crowding = calculate_crowding(fitness, fronts)
        # 选择父代 (锦标赛选择)
        parents = tournament_selection(population, fitness, ranks, crowding, pop_size)
        # 生成子代
        offspring = []
        while len(offspring) < pop_size:
            # 选择两个父代
            if len(parents) < 2:
                parents.extend(initialize_population(num_blocks, num_cuts, max(2, pop_size - len(parents)),fixed_cuts))

            p1, p2 = random.sample(parents, 2)
            # 交叉
            if random.random() < crossover_rate:
                try:
                    c1, c2 = crossover(p1, p2, num_blocks, num_cuts,fixed_cuts)
                    # print(f'crossoverc1:{c1},c2:{c2}')

                except Exception as e:
                    print(f"交叉时出错: {e}")
                    c1, c2 = p1[:], p2[:]
            else:
                c1, c2 = p1[:], p2[:]

            # 变异
            if random.random() < mutation_rate:
                c1 = mutate(c1, num_blocks, num_cuts,fixed_cuts)
                # print(f'mutate(c1):{c1}')
            if random.random() < mutation_rate:
                c2 = mutate(c2, num_blocks, num_cuts,fixed_cuts)
                # print(f'mutate(c2):{c2}')

            # 如果子代数量超过种群大小，只取所需数量
            if len(offspring) + 2 <= pop_size:
                offspring.extend([c1, c2])
            else:
                offspring.append(c1)

        # 保留精英
        try:
            population = elitism(population, offspring, fitness, pop_size)
        except Exception as e:
            print(f"精英保留时出错: {e}")
            population = parents[:pop_size]

        # 检查终止条件 (帕累托前沿变化)
        if gen > 0 and fronts and prev_front and check_convergence(fronts[0], prev_front, threshold):
            print(f"在 {gen} 代收敛")
            break

        if fronts and fronts[0]:
            prev_front = fronts[0]

        # 打印进度
        if gen % 10 == 0:
            min_delay = min(f[0] for f in fitness) if fitness and any(f[0] != float('inf') for f in fitness) else float(
                'inf')
            # 检查种群中切割点是否重复
            duplicate_count = sum(1 for ind in population if len(set(ind)) != len(ind))
            print(
                f"第 {gen} 代: 前沿大小 {len(fronts[0]) if fronts and fronts[0] else 0}, 最小延迟 {min_delay:.2f}, 重复个体数 {duplicate_count}")

    # 4. 返回帕累托最优解和块信息
    return population, block_info, fixed_cuts, new_logic_blocks
def initialize_population(num_blocks, num_cuts, pop_size, fixed_cuts):
    """
    初始化种群（避开固定切割点）
    """
    population = []
    for _ in range(pop_size):
        if num_cuts == 0:
            individual = []
        else:
            # 生成候选位置（排除固定切割点）
            candidate_positions = [i for i in range(1, num_blocks) if i not in fixed_cuts]  # 修改点
            # 确保切割点不重复且避开固定点
            cut_points = sorted(random.sample(candidate_positions, num_cuts))  # 修改点
            individual = cut_points

        population.append(individual)
    return population

def evaluate_individual2(ind, big_block_index, block_info, device_flops, mem_limit, bandwidth_bps):
    """
    评估个体的适应度
    """
    cut_points = ind.copy( )
    num_blocks = len(block_info)
    num_devices = len(device_flops)
    device_list = list(device_flops.keys())
    cut_points.append(big_block_index)
    cut_points.append(big_block_index+1)

    all_cuts = sorted(list(set(cut_points)))  # 修改点：加入固定切割点

    # print(f'all_cuts:{all_cuts}')
    # 1. 根据切割点划分段落
    segments = []
    start = 0
    for cut in all_cuts:
        segments.append(list(range(start, cut)))
        start = cut
    segments.append(list(range(start, num_blocks)))
    # print(f'segments:{segments}')
    #
    # num_segments = len(segments)
    #
    # # 段落数量应该等于设备数量
    # if num_segments != num_devices:
    #     return (float('inf'), float('inf'), float('inf'), float('inf'))

    # 2. 计算每个段落的资源需求
    segment_stats = []
    for seg in segments:
        # 确保 seg 是整数索引列表
        if not isinstance(seg, list) or not all(isinstance(i, int) for i in seg):
            return (float('inf'),) * 4
        if seg[0] == big_block_index:
            total_flops = sum(block_info[i]['flops'] for i in seg)/2
            total_memory = sum(block_info[i]['total_memory'] for i in seg)/2
            if seg:  # 确保段落不为空
                last_block_idx = seg[-1]
                output_size = block_info[last_block_idx]['output_size']/2
            else:
                output_size = 0
            segment_stats.append({
                'name':f'{seg[0]}_{seg[-1]}_1',
                'flops': total_flops,
                'memory': total_memory,
                'output_size': output_size
            })
            segment_stats.append({
                'name':f'{seg[0]}_{seg[-1]}_2',
                'flops': total_flops,
                'memory': total_memory,
                'output_size': output_size
            })
        else:
            total_flops = sum(block_info[i]['flops'] for i in seg)
            total_memory = sum(block_info[i]['total_memory'] for i in seg)

            # 获取最后一块的输出大小
            if seg:  # 确保段落不为空
                last_block_idx = seg[-1]
                output_size = block_info[last_block_idx]['output_size']
            else:
                output_size = 0

            segment_stats.append({
                'name': f'{seg[0]}_{seg[-1]}',
                'flops': total_flops,
                'memory': total_memory,
                'output_size': output_size
            })
    segment_stats_sorted = sorted(segment_stats, key=lambda x: x['flops'])
    # 3. 将段落分配给设备（按顺序）
    device_stats = []
    for i, stats in enumerate(segment_stats_sorted):
        device = device_list[i]
        flops = device_flops[device]
        compute_time = stats['flops'] / flops if flops > 0 else float('inf')
        device_stats.append({
            'name': stats['name'],
            'compute_time': compute_time,
            'memory': stats['memory'],
            'output_size': stats['output_size'],
            'flops':stats['flops'],
            'device': device
        })

    # 4. 计算目标函数
    # 目标1: 总延迟 (关键路径时间)
    if not device_stats:
        return (float('inf'),) * 4

    delays = [device_stats[0]['compute_time']]
    comm_times = []

    # 计算通信时间
    for i in range(len(device_stats) - 1):
        comm_size = device_stats[i]['output_size']
        comm_time = comm_size / bandwidth_bps
        comm_times.append(comm_time)

    # 计算流水线延迟
    for i in range(1, len(device_stats)):
        prev_delay = delays[i - 1]
        comm_delay = comm_times[i - 1] if i - 1 < len(comm_times) else 0
        start_time = max(prev_delay, prev_delay + comm_delay)
        delays.append(start_time + device_stats[i]['compute_time'])

    total_delay = delays[-1] if delays else 0

    # 目标2: 总通信量 (字节)
    total_comm = sum(device_stats[i]['output_size'] for i in range(len(device_stats) - 1))

    # 目标3: 内存超限量 (字节)
    mem_over = 0
    for stats in device_stats:
        if stats['memory'] > mem_limit:
            mem_over += (stats['memory'] - mem_limit)
    # print(f'device_stats:{device_stats}')
    # 目标4: 负载均衡 (计算时间方差)
    compute_times = [stats['compute_time'] for stats in device_stats]
    if len(compute_times) > 1:
        compute_var = np.std(compute_times)
    else:
        compute_var = 0

    return (total_delay, total_comm, mem_over, compute_var,device_stats)



def evaluate_individual(ind, big_block_index, block_info, device_flops, mem_limit, bandwidth_bps):
    """
    评估个体的适应度
    """
    cut_points = ind.copy( )
    num_blocks = len(block_info)
    num_devices = len(device_flops)
    device_list = list(device_flops.keys())
    cut_points.append(big_block_index)
    cut_points.append(big_block_index+1)

    all_cuts = sorted(list(set(cut_points)))  # 修改点：加入固定切割点

    # print(f'all_cuts:{all_cuts}')
    # 1. 根据切割点划分段落
    segments = []
    start = 0
    for cut in all_cuts:
        segments.append(list(range(start, cut)))
        start = cut
    segments.append(list(range(start, num_blocks)))
    # print(f'segments:{segments}')
    #
    # num_segments = len(segments)
    #
    # # 段落数量应该等于设备数量
    # if num_segments != num_devices:
    #     return (float('inf'), float('inf'), float('inf'), float('inf'))

    # 2. 计算每个段落的资源需求
    segment_stats = []
    for seg in segments:
        # 确保 seg 是整数索引列表
        if not isinstance(seg, list) or not all(isinstance(i, int) for i in seg):
            return (float('inf'),) * 4
        if seg[0] == big_block_index:
            total_flops = sum(block_info[i]['flops'] for i in seg)/2
            total_memory = sum(block_info[i]['total_memory'] for i in seg)/2
            if seg:  # 确保段落不为空
                last_block_idx = seg[-1]
                output_size = block_info[last_block_idx]['output_size']/2
            else:
                output_size = 0
            segment_stats.append({
                'name':f'{seg[0]}_{seg[-1]}_1',
                'flops': total_flops,
                'memory': total_memory,
                'output_size': output_size
            })
            segment_stats.append({
                'name':f'{seg[0]}_{seg[-1]}_2',
                'flops': total_flops,
                'memory': total_memory,
                'output_size': output_size
            })
        else:
            total_flops = sum(block_info[i]['flops'] for i in seg)
            total_memory = sum(block_info[i]['total_memory'] for i in seg)

            # 获取最后一块的输出大小
            if seg:  # 确保段落不为空
                last_block_idx = seg[-1]
                output_size = block_info[last_block_idx]['output_size']
            else:
                output_size = 0

            segment_stats.append({
                'name': f'{seg[0]}_{seg[-1]}',
                'flops': total_flops,
                'memory': total_memory,
                'output_size': output_size
            })
    segment_stats_sorted = sorted(segment_stats, key=lambda x: x['flops'])
    # 3. 将段落分配给设备（按顺序）
    device_stats = []
    for i, stats in enumerate(segment_stats_sorted):
        device = device_list[i]
        flops = device_flops[device]
        compute_time = stats['flops'] / flops if flops > 0 else float('inf')
        device_stats.append({
            'compute_time': compute_time,
            'memory': stats['memory'],
            'output_size': stats['output_size'],
            'device': device
        })

    # 4. 计算目标函数
    # 目标1: 总延迟 (关键路径时间)
    if not device_stats:
        return (float('inf'),) * 4

    delays = [device_stats[0]['compute_time']]
    comm_times = []

    # 计算通信时间
    for i in range(len(device_stats) - 1):
        comm_size = device_stats[i]['output_size']
        comm_time = comm_size / bandwidth_bps
        comm_times.append(comm_time)

    # 计算流水线延迟
    for i in range(1, len(device_stats)):
        prev_delay = delays[i - 1]
        comm_delay = comm_times[i - 1] if i - 1 < len(comm_times) else 0
        start_time = max(prev_delay, prev_delay + comm_delay)
        delays.append(start_time + device_stats[i]['compute_time'])

    total_delay = delays[-1] if delays else 0

    # 目标2: 总通信量 (字节)
    total_comm = sum(device_stats[i]['output_size'] for i in range(len(device_stats) - 1))

    # 目标3: 内存超限量 (字节)
    mem_over = 0
    for stats in device_stats:
        if stats['memory'] > mem_limit:
            mem_over += (stats['memory'] - mem_limit)
    # print(f'device_stats:{device_stats}')
    # 目标4: 负载均衡 (计算时间方差)
    compute_times = [stats['compute_time'] for stats in device_stats]
    if len(compute_times) > 1:
        compute_var = np.std(compute_times)
    else:
        compute_var = 0

    return (total_delay, total_comm, mem_over, compute_var)

    # except Exception as e:
    #     print(f"评估个体时出错22: {e}")
    #     return (float('inf'), float('inf'), float('inf'), float('inf'))
def crossover(p1, p2, num_blocks, num_cuts,fixed_cuts):
    """交叉操作 - 确保切割点不重复"""
    if num_cuts == 0:
        return [], []

    # 确保切割点排序
    p1 = sorted(p1)
    p2 = sorted(p2)

    # 选择交叉点
    cross_point = random.randint(1, num_cuts - 1)

    # 创建新个体
    child1 = p1[:cross_point] + p2[cross_point:]
    child2 = p2[:cross_point] + p1[cross_point:]

    # 确保切割点不重复
    child1 = sorted(set(child1))
    child2 = sorted(set(child2))

    # 如果去重后切割点数量不足，补充随机点
    def complete_individual(ind, num_cuts, num_blocks):
        if len(ind) < num_cuts:
            # 获取所有可能的切割点
            all_positions = [p for p in range(1, num_blocks) if p not in fixed_cuts]
            # 移除已存在的点
            available = [p for p in all_positions if p not in ind]
            # 随机选择补充点
            if available:
                additional = random.sample(available, num_cuts - len(ind))
                ind.extend(additional)
                ind.sort()
        return ind

    child1 = complete_individual(child1, num_cuts, num_blocks)
    child2 = complete_individual(child2, num_cuts, num_blocks)

    return child1, child2


def mutate(individual, num_blocks, num_cuts,fixed_cuts):
    """变异操作 - 确保切割点不重复"""
    if num_cuts == 0:
        return individual

    # 创建新个体的副本
    new_individual = individual[:]

    # 随机选择一个切割点进行变异
    idx = random.randint(0, num_cuts-1)

    # 生成新的切割点位置，确保不重复且在有效范围内
    possible_positions = [i for i in range(1, num_blocks)
                         if i not in new_individual and i not in fixed_cuts]
    if possible_positions:
        new_cut = random.choice(possible_positions)
        new_individual[idx] = new_cut
        new_individual.sort()
    # new_individual = sorted(set(new_individual))
    # def complete_individual(ind, num_cuts, num_blocks):
    #     if len(ind) < num_cuts+2:
    #         # 获取所有可能的切割点
    #         all_positions = [p for p in range(1, num_blocks) if p not in fixed_cuts]
    #         # 移除已存在的点
    #         available = [p for p in all_positions if p not in ind]
    #         # 随机选择补充点
    #         if available:
    #             additional = random.sample(available, num_cuts - len(ind))
    #             ind.extend(additional)
    #             ind.sort()
    #     return ind
    return new_individual
    # return complete_individual(new_individual,num_cuts, num_blocks)
def non_dominated_sorting(fitness):
    """非支配排序"""
    pop_size = len(fitness)
    if pop_size == 0:
        return [], []

    # 支配关系矩阵
    dominates = [[False] * pop_size for _ in range(pop_size)]
    # 被支配计数
    dominated_count = [0] * pop_size
    # 支配解集合
    dominates_set = [[] for _ in range(pop_size)]
    # 前沿列表
    fronts = [[]]

    # 计算支配关系
    for i in range(pop_size):
        for j in range(pop_size):
            if i == j:
                continue
            if dominates_fitness(fitness[i], fitness[j]):
                dominates[i][j] = True
                dominates_set[i].append(j)
            elif dominates_fitness(fitness[j], fitness[i]):
                dominated_count[i] += 1

        # 第一前沿 (非被支配解)
        if dominated_count[i] == 0:
            fronts[0].append(i)

    # 如果没有找到任何前沿，直接返回空结果
    if not fronts[0]:
        return [], [0] * pop_size

    # 分层其他前沿
    current_front = 0
    while current_front < len(fronts) and fronts[current_front]:
        next_front = []
        for i in fronts[current_front]:
            for j in dominates_set[i]:
                dominated_count[j] -= 1
                if dominated_count[j] == 0:
                    next_front.append(j)

        if next_front:
            fronts.append(next_front)
        current_front += 1

    # 为每个个体分配前沿等级
    ranks = [0] * pop_size
    for rank, front in enumerate(fronts):
        for i in front:
            ranks[i] = rank

    return fronts, ranks


def dominates_fitness(f1, f2):
    """判断f1是否支配f2 (所有目标最小化)"""
    # f1在所有目标上不劣于f2，且至少一个目标更好
    not_worse = all(x <= y for x, y in zip(f1, f2))
    better = any(x < y for x, y in zip(f1, f2))
    return not_worse and better


def calculate_crowding(fitness, fronts):
    """计算拥挤度距离"""
    num_objectives = len(fitness[0]) if fitness else 0
    crowding = [0] * len(fitness) if fitness else []

    if not fitness or not fronts:
        return crowding

    for front in fronts:
        if not front:
            continue

        # 对每个目标计算拥挤度
        for obj_idx in range(num_objectives):
            # 按当前目标排序
            try:
                sorted_front = sorted(front, key=lambda i: fitness[i][obj_idx])
            except:
                continue

            # 跳过空前沿
            if not sorted_front:
                continue

            min_val = fitness[sorted_front[0]][obj_idx]
            max_val = fitness[sorted_front[-1]][obj_idx]

            # 边界点有无限拥挤度
            if len(sorted_front) > 1:
                crowding[sorted_front[0]] = float('inf')
                crowding[sorted_front[-1]] = float('inf')
            elif len(sorted_front) == 1:
                crowding[sorted_front[0]] = float('inf')

            # 计算内部点的拥挤度
            if max_val - min_val < 1e-6 or len(sorted_front) <= 2:
                continue

            for i in range(1, len(sorted_front) - 1):
                idx = sorted_front[i]
                next_val = fitness[sorted_front[i + 1]][obj_idx]
                prev_val = fitness[sorted_front[i - 1]][obj_idx]
                crowding[idx] += (next_val - prev_val) / (max_val - min_val)

    return crowding


def tournament_selection(population, fitness, ranks, crowding, num_parents):
    """锦标赛选择父代"""
    parents = []
    if not population or not fitness:
        return initialize_population(len(population[0]) if population else [])

    for _ in range(num_parents):
        # 随机选择4个候选
        candidates = random.sample(range(len(population)), min(4, len(population)))
        # 按前沿等级和拥挤度排序
        try:
            candidates.sort(key=lambda i: (ranks[i], -crowding[i]))
        except:
            parents.append(population[random.choice(candidates)])
            continue

        parents.append(population[candidates[0]])
    return parents


def elitism(parents, offspring, parent_fitness, pop_size):
    """精英保留策略"""
    combined = parents + offspring
    # 评估所有子代
    offspring_fitness = []
    for ind in offspring:
        try:
            # 这里需要传入实际参数，但当前上下文没有
            # 由于这是一个简化版，我们假设子代已经被评估
            fit = (0, 0, 0, 0)  # 占位符
            offspring_fitness.append(fit)
        except:
            offspring_fitness.append((float('inf'), float('inf'), float('inf'), float('inf')))

    combined_fitness = parent_fitness + offspring_fitness

    # 非支配排序
    fronts, ranks = non_dominated_sorting(combined_fitness)
    if not fronts:
        return parents[:pop_size]

    # 计算拥挤度
    crowding = calculate_crowding(combined_fitness, fronts)

    # 选择前pop_size个个体
    indices = list(range(len(combined)))
    try:
        indices.sort(key=lambda i: (ranks[i], -crowding[i]))
    except:
        indices = list(range(len(combined)))

    return [combined[i] for i in indices[:pop_size]]


def check_convergence(current_front, prev_front, threshold):
    """检查帕累托前沿是否收敛"""
    if not prev_front or not current_front:
        return False

    # 计算前沿变化率 (使用目标空间距离)
    min_distances = []
    for i in current_front:
        min_dist = min(np.linalg.norm(np.array(i) - np.array(j)) for j in prev_front)
        min_distances.append(min_dist)

    avg_change = sum(min_distances) / len(min_distances)
    return avg_change < threshold


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
    merged,merged_topk,logical_layers = find_bottleneck_segments(df, blocks,thresh_ratio=1.8)
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

    max_case = max(cases, key=lambda x: x['accuracy'])
    print(max_case)

    #帕累托边界
    index_front = 0

    for i in range(len(logical_layers)):
        if logical_layers[i][0]==max_case['cut_layer']:
            index_front = i
    index_back = index_front+1

    # front_block = logical_layers[:index_front]
    # back_block = logical_layers[index_back:]
    device_flops = {
        'rpi5_1': 320e9,
        'rpi5_2': 320e9,
        'rpi5_3': 320e9,
        'jetson1': 472e9,
        'jetson2': 472e9,
        'jetson3': 472e9,
        'pc_cpu': 500e9,
    }
    mem_limit = 4 * 1024 ** 2  # 4GB
    bandwidth_bps = 4e6  # 1MB/s

    # 5. 运行优化
    pareto_solutions, block_info, fixed_cuts, new_logic_blocks = nsga2_optimize(
        logic_blocks=logical_layers,
        big_block_index=index_front,
        df=df,
        device_flops=device_flops,
        mem_limit=mem_limit,
        bandwidth_bps=bandwidth_bps,
        pop_size=50,
        ngen=100
    )
    # print(pareto_solutions)
    if pareto_solutions:
        print(f"找到 {len(pareto_solutions)} 个帕累托最优解")

        # 按总延迟排序
        pareto_solutions.sort(
            key=lambda ind: evaluate_individual(ind, index_front, block_info, device_flops, mem_limit, bandwidth_bps)[0]
        )

        # 输出前3个最优解
        for i, solution in enumerate(pareto_solutions[:3]):
            # 合并所有切割点

            delay, comm, mem_over, balance,device_stats = evaluate_individual2(
                solution, index_front, block_info, device_flops, mem_limit, bandwidth_bps
            )
            solution.append(index_front)
            solution.append(index_front+1)

            print(f"\n方案 {i + 1}:")
            print(f"总延迟: {delay:.4f}秒")
            print(f"总通信量: {comm / 1e6:.2f}MB")
            print(f"内存超限: {mem_over / 1e6:.2f}MB")
            print(f"负载均衡方差: {balance:.4f}")

            # 显示所有切割点
            print(f"所有切割点位置: {sorted(solution)}")
            print(f"固定切割点位置: {fixed_cuts}")
            for sta in device_stats:
                print(f"{logical_layers[int(sta['name'].split('_')[0]):int(sta['name'].split('_')[1])+1]}被分配在{sta['device']}上")
            print(f'切割段落数据：{device_stats}')

    else:
        print("未找到可行解")