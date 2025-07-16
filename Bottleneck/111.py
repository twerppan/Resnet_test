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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from collections import defaultdict
import signal
from scipy.optimize import linear_sum_assignment
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
            'block_idx': block_idx,
            'start_node': block[0],          # 新增：块起始节点
            'end_node': block[-1],           # 新增：块结束节点
            'flops': total_flops,
            'weight_mem': weight_mem,
            'bias_mem': bias_mem,
            'peak_activation': peak_activation,
            'output_size': output_size,
            'total_memory': weight_mem + bias_mem + peak_activation
        })

    return block_info


# --- 步骤 1.4：NSGA-II 多目标优化寻找切割方案的帕累托前沿 --
def nsga2_optimize(logic_blocks, big_block_index, block_number,df, device_flops, mem_limit, device_bandwidth_dict,
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
    print(f'block_info:{block_info}')


    num_blocks = len(block_info)
    num_devices = len(device_flops)

    # 计算切割点数量（如果是切割为4份子张量，寻找num_devices-6,如果切割为3份则num_devices-5，依次num_devices-4）
    num_cuts = num_devices - block_number-2

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
            fit = evaluate_individual(ind, big_block_index,block_number,block_info, device_flops, mem_limit, device_bandwidth_dict)
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
                f"第 {gen} 代: 前沿大小 {len(fronts[0]) if fronts and fronts[0] else 0}, 最小延迟 {min_delay:.8f}, 重复个体数 {duplicate_count}")

    # 4. 返回帕累托最优解和块信息
    return population, block_info, fixed_cuts


def initialize_population(num_blocks, num_cuts, pop_size, fixed_cuts):
    population = []
    for _ in range(pop_size):
        if num_cuts == 0:
            individual = []
        else:
            candidate_positions = [i for i in range(1, num_blocks) if i not in fixed_cuts]

            # 关键修复：确保不请求超过可用位置的数量
            actual_cuts = min(num_cuts, len(candidate_positions))
            if actual_cuts > 0:
                cut_points = sorted(random.sample(candidate_positions, actual_cuts))
            else:
                cut_points = []
            individual = cut_points
        population.append(individual)
    return population

def evaluate_individual2(ind, big_block_index, block_number, block_info, device_flops, mem_limit,device_bandwidth_dict):
    """
    评估个体的适应度（考虑异构设备负载和带宽）

    参数新增：
    device_load_dict: 设备名->当前负载(0-1)
    device_bandwidth_dict: (from_device, to_device)->带宽(Mbps)
    """
    # 设置超时
    max_time = 5
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(max_time)
    try:
        device_load_dict = {
            'rpi5_1': 0.182,
            'rpi5_2': 0.235,
            'rpi5_3': 0.203,
            'jetson1': 0.217,
            'jetson2': 0.246,
            'jetson3': 0.3,
            'pc_cpu': 0.483,
        }
        # print(f'种群切割点ind：{ind}')

        # 1. 处理切割点
        cut_points = ind.copy()
        num_blocks = len(block_info)
        device_list = list(device_flops.keys())
        cut_points.append(big_block_index)
        cut_points.append(big_block_index + 1)
        all_cuts = sorted(list(set(cut_points)))

        # 2. 划分段落
        segments = []
        start = 0
        for cut in all_cuts:
            segments.append(list(range(start, cut)))
            start = cut
        segments.append(list(range(start, num_blocks)))

        # print(f'根据切割点切好的段落segments：{segments}')

        # 3. 计算段落资源需求
        segment_stats = []
        segment_types = []  # 记录每个片段的类型：'normal' 或 'big_block'
        for seg in segments:
            if seg and seg[0] == big_block_index:  # 特殊处理大块
                total_flops = sum(block_info[i]['flops'] for i in seg) / block_number
                total_memory = max(block_info[i]['total_memory'] for i in seg) / block_number
                last_block_idx = seg[-1]
                output_size = block_info[last_block_idx]['output_size'] / block_number

                # 拆分为block_number个相同子段落
                for i in range(1, block_number + 1):
                    segment_stats.append({
                        'name': f'{seg[0]}_{seg[-1]}_{i}',
                        'start_node': block_info[seg[0]]['start_node'],  # 新增：块起始节点
                        'end_node': block_info[seg[-1]]['end_node'],
                        'flops': total_flops,
                        'memory': total_memory,
                        'output_size': output_size,
                        'original_index': seg[0],  # 原始块索引
                        'is_big_block': True
                    })
                    segment_types.append('big_block')
            else:
                if not seg:  # 空段落处理
                    return (float('inf'),) * 4

                total_flops = sum(block_info[i]['flops'] for i in seg)
                total_memory = sum(block_info[i]['total_memory'] for i in seg)
                last_block_idx = seg[-1]
                output_size = block_info[last_block_idx]['output_size']

                segment_stats.append({
                    'name': f'{seg[0]}_{seg[-1]}',
                    'start_node': block_info[seg[0]]['start_node'],  # 新增：块起始节点
                    'end_node': block_info[seg[-1]]['end_node'],
                    'flops': total_flops,
                    'memory': total_memory,
                    'output_size': output_size,
                    'is_big_block': False
                })
                segment_types.append('normal')
        # print(f'每个段落的具体数据segment_stats：{segment_stats}')

        # 4. 计算每个段落在每个设备上的计算时间（考虑负载）
        compute_times = np.zeros((len(segment_stats), len(device_list)))
        LARGE_VALUE = 1e15
        for seg_idx, seg in enumerate(segment_stats):
            for dev_idx, dev in enumerate(device_list):
                effective_flops = device_flops[dev] * (1 - device_load_dict[dev])
                if effective_flops <= 0:
                    compute_times[seg_idx, dev_idx] = LARGE_VALUE
                else:
                    compute_times[seg_idx, dev_idx] = seg['flops'] / effective_flops
        row_ind, col_ind = linear_sum_assignment(compute_times)
        # 8. 构建分配结果
        assignment = {}
        for i in range(len(row_ind)):
            seg_idx = row_ind[i]
            dev_idx = col_ind[i]
            assignment[seg_idx] = device_list[dev_idx]

        # 9. 构建device_stats
        device_stats = []
        for seg_idx, seg in enumerate(segment_stats):
            dev = assignment[seg_idx]
            effective_flops = device_flops[dev] * (1 - device_load_dict[dev])
            compute_time = seg['flops'] / effective_flops if effective_flops > 0 else LARGE_VALUE

            device_stats.append({
                'segment':seg['name'],
                'start_node': seg['start_node'],  # 新增：块起始节点
                'end_node': seg['end_node'],
                'compute_time': compute_time,
                'memory': seg['memory'],
                'output_size': seg['output_size'],
                'device': dev,
                'seg_idx': seg_idx,
                'is_big_block': 'original_index' in seg,
                'original_index': seg.get('original_index', -1)
            })

        device_available = {dev: 0 for dev in device_list}
        segment_end_time = [0] * len(segment_stats)
        big_block_groups = defaultdict(list)  # 分组存储大块切片

        # 分组大块切片
        for i, seg in enumerate(device_stats):
            if seg['is_big_block']:
                big_block_groups[seg['original_index']].append(i)

        # 按顺序处理每个片段
        for seg_idx in range(len(segment_stats)):
            seg = device_stats[seg_idx]
            dev = seg['device']

            # 如果是大块的一部分，需要特殊处理
            if seg['is_big_block']:
                group = big_block_groups[seg['original_index']]

                # 只处理组内的第一个片段（组内并行处理）
                if seg_idx != group[0]:
                    continue

                # 找到前一个片段（非大块）
                prev_seg_idx = None
                for i in range(seg_idx - 1, -1, -1):
                    if not device_stats[i]['is_big_block']:
                        prev_seg_idx = i
                        break

                # 计算通信时间（从前一个片段到当前大块组）
                comm_time_to_group = 0
                if prev_seg_idx is not None:
                    prev_dev = device_stats[prev_seg_idx]['device']
                    comm_size = device_stats[prev_seg_idx]['output_size']
                    bw_mbps = device_bandwidth_dict.get((prev_dev, dev), 0)
                    if bw_mbps > 0:
                        bw_bps = (bw_mbps * 1e6) / 8
                        comm_time_to_group = comm_size / bw_bps

                # 计算大块组的开始时间
                start_time = max(
                    device_available[dev],
                    segment_end_time[prev_seg_idx] + comm_time_to_group if prev_seg_idx is not None else 0
                )

                # 处理大块组内的所有子片段（并行）
                group_end_times = []
                for sub_idx in group:
                    sub_seg = device_stats[sub_idx]
                    sub_dev = sub_seg['device']

                    # 子片段开始时间（考虑设备可用性）
                    sub_start = max(device_available[sub_dev], start_time)

                    # 子片段结束时间
                    sub_end = sub_start + sub_seg['compute_time']
                    segment_end_time[sub_idx] = sub_end
                    device_available[sub_dev] = sub_end
                    group_end_times.append(sub_end)

                # 大块组的结束时间是所有子片段中最晚的
                group_end = max(group_end_times) if group_end_times else 0
            else:
                # 普通片段处理
                # 找到前一个片段
                prev_seg_idx = None
                for i in range(seg_idx - 1, -1, -1):
                    # 跳过同组的大块子片段
                    if not device_stats[i]['is_big_block'] or i == \
                            big_block_groups.get(device_stats[i].get('original_index', -1), [-1])[0]:
                        prev_seg_idx = i
                        break

                # 计算通信时间
                comm_time = 0
                if prev_seg_idx is not None:
                    prev_dev = device_stats[prev_seg_idx]['device']
                    comm_size = device_stats[prev_seg_idx]['output_size']
                    bw_mbps = device_bandwidth_dict.get((prev_dev, dev), 0)
                    if bw_mbps > 0:
                        bw_bps = (bw_mbps * 1e6) / 8
                        comm_time = comm_size / bw_bps

                # 计算开始时间
                start_time = max(
                    device_available[dev],
                    segment_end_time[prev_seg_idx] + comm_time if prev_seg_idx is not None else 0
                )

                # 计算结束时间
                end_time = start_time + seg['compute_time']
                segment_end_time[seg_idx] = end_time
                device_available[dev] = end_time

        # 9. 计算总延迟（最后一个片段的结束时间）
        total_delay = max(segment_end_time) if segment_end_time else float('inf')

        # 10. 计算总通信量
        total_comm = 0
        for i in range(1, len(segment_stats)):
            prev_seg = device_stats[i - 1]
            curr_seg = device_stats[i]

            # 如果是大块组内的通信，不计入总通信量（内部通信）
            if prev_seg['is_big_block'] and curr_seg['is_big_block']:
                if prev_seg.get('original_index', -1) == curr_seg.get('original_index', -2):
                    continue

            # 如果是大块组后的第一个片段，只计算一次通信
            if curr_seg['is_big_block'] and i == big_block_groups[curr_seg['original_index']][0]:
                # 已经在前面的流水线计算中处理
                continue

            if prev_seg['device'] != curr_seg['device']:
                total_comm += prev_seg['output_size']

        # 11. 计算内存占用率标准差
        memory_ratios = []
        for stats in device_stats:
            # 跳过大块组内的重复片段（只计算一次）
            if stats['is_big_block'] and stats['seg_idx'] != big_block_groups.get(stats['original_index'], [-1])[0]:
                continue
            memory_ratio = stats['memory'] / mem_limit
            memory_ratios.append(memory_ratio)

        mem_over = np.std(memory_ratios) if memory_ratios else 0

        # 12. 计算负载均衡（计算时间方差）
        compute_times = []
        for stats in device_stats:
            # 跳过大块组内的重复片段（只计算一次）
            if stats['is_big_block'] and stats['seg_idx'] != big_block_groups.get(stats['original_index'], [-1])[0]:
                continue
            compute_times.append(stats['compute_time'])

        if len(compute_times) > 1:
            compute_var = np.std(compute_times)
        else:
            compute_var = 0

        return (total_delay, total_comm, mem_over, compute_var, device_stats)
    except TimeoutException:
        print(f"计算超时（>{max_time}秒），返回不可行解")
        return (float('inf'),) * 4
    except Exception as e:
        print(f"计算过程中发生异常: {str(e)}")
        return (float('inf'),) * 4
    finally:
        signal.alarm(0)  # 取消超时设置


class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("算法超时")

def evaluate_individual(ind, big_block_index, block_number, block_info, device_flops, mem_limit,device_bandwidth_dict):
    """
    评估个体的适应度（考虑异构设备负载和带宽）

    参数新增：
    device_load_dict: 设备名->当前负载(0-1)
    device_bandwidth_dict: (from_device, to_device)->带宽(Mbps)
    """
    # 设置超时
    max_time = 5
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(max_time)
    try:
        device_load_dict = {
            'rpi5_1': 0.182,
            'rpi5_2': 0.235,
            'rpi5_3': 0.203,
            'jetson1': 0.217,
            'jetson2': 0.246,
            'jetson3': 0.3,
            'pc_cpu': 0.483,
        }
        # print(f'种群切割点ind：{ind}')

        # 1. 处理切割点
        cut_points = ind.copy()
        num_blocks = len(block_info)
        device_list = list(device_flops.keys())
        cut_points.append(big_block_index)
        cut_points.append(big_block_index + 1)
        all_cuts = sorted(list(set(cut_points)))

        # 2. 划分段落
        segments = []
        start = 0
        for cut in all_cuts:
            segments.append(list(range(start, cut)))
            start = cut
        segments.append(list(range(start, num_blocks)))

        # print(f'根据切割点切好的段落segments：{segments}')

        # 3. 计算段落资源需求
        segment_stats = []
        segment_types = []  # 记录每个片段的类型：'normal' 或 'big_block'
        for seg in segments:
            if seg and seg[0] == big_block_index:  # 特殊处理大块
                total_flops = sum(block_info[i]['flops'] for i in seg) / block_number
                total_memory = max(block_info[i]['total_memory'] for i in seg) / block_number
                last_block_idx = seg[-1]
                output_size = block_info[last_block_idx]['output_size'] / block_number

                # 拆分为block_number个相同子段落
                for i in range(1, block_number + 1):
                    segment_stats.append({
                        'name': f'{seg[0]}_{seg[-1]}_{i}',
                        'flops': total_flops,
                        'memory': total_memory,
                        'output_size': output_size,
                        'original_index': seg[0],  # 原始块索引
                        'is_big_block': True
                    })
                    segment_types.append('big_block')
            else:
                if not seg:  # 空段落处理
                    return (float('inf'),) * 4

                total_flops = sum(block_info[i]['flops'] for i in seg)
                total_memory = sum(block_info[i]['total_memory'] for i in seg)
                last_block_idx = seg[-1]
                output_size = block_info[last_block_idx]['output_size']

                segment_stats.append({
                    'name': f'{seg[0]}_{seg[-1]}',
                    'flops': total_flops,
                    'memory': total_memory,
                    'output_size': output_size,
                    'is_big_block': False
                })
                segment_types.append('normal')
        # print(f'每个段落的具体数据segment_stats：{segment_stats}')

        # 4. 计算每个段落在每个设备上的计算时间（考虑负载）
        compute_times = np.zeros((len(segment_stats), len(device_list)))
        LARGE_VALUE = 1e15
        for seg_idx, seg in enumerate(segment_stats):
            for dev_idx, dev in enumerate(device_list):
                effective_flops = device_flops[dev] * (1 - device_load_dict[dev])
                if effective_flops <= 0:
                    compute_times[seg_idx, dev_idx] = LARGE_VALUE
                else:
                    compute_times[seg_idx, dev_idx] = seg['flops'] / effective_flops
        row_ind, col_ind = linear_sum_assignment(compute_times)
        # 8. 构建分配结果
        assignment = {}
        for i in range(len(row_ind)):
            seg_idx = row_ind[i]
            dev_idx = col_ind[i]
            assignment[seg_idx] = device_list[dev_idx]

        # 9. 构建device_stats
        device_stats = []
        for seg_idx, seg in enumerate(segment_stats):
            dev = assignment[seg_idx]
            effective_flops = device_flops[dev] * (1 - device_load_dict[dev])
            compute_time = seg['flops'] / effective_flops if effective_flops > 0 else LARGE_VALUE

            device_stats.append({
                'compute_time': compute_time,
                'memory': seg['memory'],
                'output_size': seg['output_size'],
                'device': dev,
                'seg_idx': seg_idx,
                'is_big_block': 'original_index' in seg,
                'original_index': seg.get('original_index', -1)
            })
        # # 5. 构建最小费用流图（考虑流水线延迟）
        # G = nx.DiGraph()
        #
        # source = "source"
        # sink = "sink"
        # G.add_node(source)
        # G.add_node(sink)
        #
        # # 添加段落节点和设备节点
        # seg_nodes = [f"seg_{i}" for i in range(len(segment_stats))]
        # dev_nodes = [f"dev_{j}" for j in range(len(device_list))]
        #
        # for node in seg_nodes + dev_nodes:
        #     G.add_node(node)
        #
        # # 添加边：源点到段落
        # for seg_node in seg_nodes:
        #     G.add_edge(source, seg_node, capacity=1, weight=0)
        #
        # # 添加边：段落到设备（考虑内存约束）
        # for seg_idx, seg_node in enumerate(seg_nodes):
        #     seg = segment_stats[seg_idx]
        #     for dev_idx, dev_node in enumerate(dev_nodes):
        #         dev = device_list[dev_idx]
        #
        #         # 检查内存约束
        #         if seg['memory'] > mem_limit:
        #             continue  # 跳过不满足内存约束的分配
        #
        #         # 获取计算时间
        #         comp_time = compute_times[seg_idx, dev_idx]
        #
        #         # 添加边
        #         G.add_edge(seg_node, dev_node, capacity=1, weight=comp_time)
        #
        # # 添加边：设备到汇点
        # for dev_node in dev_nodes:
        #     G.add_edge(dev_node, sink, capacity=1, weight=0)
        #
        # # 6. 计算最小费用流
        # try:
        #
        #     # 计算最小费用流
        #     min_cost_flow = nx.max_flow_min_cost(G, source, sink)
        #
        #     # 提取匹配结果
        #     assignment = {}
        #     for seg_node in seg_nodes:
        #         for dev_node, flow in min_cost_flow[seg_node].items():
        #             if flow > 0 and dev_node.startswith("dev_"):
        #                 seg_idx = int(seg_node.split("_")[1])
        #                 dev_idx = int(dev_node.split("_")[1])
        #                 assignment[seg_idx] = device_list[dev_idx]
        #                 break
        # except (nx.NetworkXUnfeasible, nx.NetworkXError):
        #     return (float('inf'),) * 4
        #
        # # 检查是否所有段落都分配了设备
        # if len(assignment) != len(segment_stats):
        #     return (float('inf'),) * 4
        #
        # # 7. 按位置顺序排序段落
        # device_stats = []
        # for seg_idx, seg in enumerate(segment_stats):
        #     dev = assignment[seg_idx]
        #     effective_flops = device_flops[dev] * (1 - device_load_dict[dev])
        #     compute_time = seg['flops'] / effective_flops if effective_flops > 0 else float('inf')
        #
        #     device_stats.append({
        #         'compute_time': compute_time,
        #         'memory': seg['memory'],
        #         'output_size': seg['output_size'],
        #         'device': dev,
        #         'seg_idx': seg_idx,
        #         'is_big_block': seg['is_big_block'],
        #         'original_index': seg.get('original_index', seg_idx)  # 用于大块分组
        #     })

        device_available = {dev: 0 for dev in device_list}
        segment_end_time = [0] * len(segment_stats)
        big_block_groups = defaultdict(list)  # 分组存储大块切片

        # 分组大块切片
        for i, seg in enumerate(device_stats):
            if seg['is_big_block']:
                big_block_groups[seg['original_index']].append(i)

        # 按顺序处理每个片段
        for seg_idx in range(len(segment_stats)):
            seg = device_stats[seg_idx]
            dev = seg['device']

            # 如果是大块的一部分，需要特殊处理
            if seg['is_big_block']:
                group = big_block_groups[seg['original_index']]

                # 只处理组内的第一个片段（组内并行处理）
                if seg_idx != group[0]:
                    continue

                # 找到前一个片段（非大块）
                prev_seg_idx = None
                for i in range(seg_idx - 1, -1, -1):
                    if not device_stats[i]['is_big_block']:
                        prev_seg_idx = i
                        break

                # 计算通信时间（从前一个片段到当前大块组）
                comm_time_to_group = 0
                if prev_seg_idx is not None:
                    prev_dev = device_stats[prev_seg_idx]['device']
                    comm_size = device_stats[prev_seg_idx]['output_size']
                    bw_mbps = device_bandwidth_dict.get((prev_dev, dev), 0)
                    if bw_mbps > 0:
                        bw_bps = (bw_mbps * 1e6) / 8
                        comm_time_to_group = comm_size / bw_bps

                # 计算大块组的开始时间
                start_time = max(
                    device_available[dev],
                    segment_end_time[prev_seg_idx] + comm_time_to_group if prev_seg_idx is not None else 0
                )

                # 处理大块组内的所有子片段（并行）
                group_end_times = []
                for sub_idx in group:
                    sub_seg = device_stats[sub_idx]
                    sub_dev = sub_seg['device']

                    # 子片段开始时间（考虑设备可用性）
                    sub_start = max(device_available[sub_dev], start_time)

                    # 子片段结束时间
                    sub_end = sub_start + sub_seg['compute_time']
                    segment_end_time[sub_idx] = sub_end
                    device_available[sub_dev] = sub_end
                    group_end_times.append(sub_end)

                # 大块组的结束时间是所有子片段中最晚的
                group_end = max(group_end_times) if group_end_times else 0
            else:
                # 普通片段处理
                # 找到前一个片段
                prev_seg_idx = None
                for i in range(seg_idx - 1, -1, -1):
                    # 跳过同组的大块子片段
                    if not device_stats[i]['is_big_block'] or i == \
                            big_block_groups.get(device_stats[i].get('original_index', -1), [-1])[0]:
                        prev_seg_idx = i
                        break

                # 计算通信时间
                comm_time = 0
                if prev_seg_idx is not None:
                    prev_dev = device_stats[prev_seg_idx]['device']
                    comm_size = device_stats[prev_seg_idx]['output_size']
                    bw_mbps = device_bandwidth_dict.get((prev_dev, dev), 0)
                    if bw_mbps > 0:
                        bw_bps = (bw_mbps * 1e6) / 8
                        comm_time = comm_size / bw_bps

                # 计算开始时间
                start_time = max(
                    device_available[dev],
                    segment_end_time[prev_seg_idx] + comm_time if prev_seg_idx is not None else 0
                )

                # 计算结束时间
                end_time = start_time + seg['compute_time']
                segment_end_time[seg_idx] = end_time
                device_available[dev] = end_time

        # 9. 计算总延迟（最后一个片段的结束时间）
        total_delay = max(segment_end_time) if segment_end_time else float('inf')

        # 10. 计算总通信量
        total_comm = 0
        for i in range(1, len(segment_stats)):
            prev_seg = device_stats[i - 1]
            curr_seg = device_stats[i]

            # 如果是大块组内的通信，不计入总通信量（内部通信）
            if prev_seg['is_big_block'] and curr_seg['is_big_block']:
                if prev_seg.get('original_index', -1) == curr_seg.get('original_index', -2):
                    continue

            # 如果是大块组后的第一个片段，只计算一次通信
            if curr_seg['is_big_block'] and i == big_block_groups[curr_seg['original_index']][0]:
                # 已经在前面的流水线计算中处理
                continue

            if prev_seg['device'] != curr_seg['device']:
                total_comm += prev_seg['output_size']

        # 11. 计算内存占用率标准差
        memory_ratios = []
        for stats in device_stats:
            # 跳过大块组内的重复片段（只计算一次）
            if stats['is_big_block'] and stats['seg_idx'] != big_block_groups.get(stats['original_index'], [-1])[0]:
                continue
            memory_ratio = stats['memory'] / mem_limit
            memory_ratios.append(memory_ratio)

        mem_over = np.std(memory_ratios) if memory_ratios else 0

        # 12. 计算负载均衡（计算时间方差）
        compute_times = []
        for stats in device_stats:
            # 跳过大块组内的重复片段（只计算一次）
            if stats['is_big_block'] and stats['seg_idx'] != big_block_groups.get(stats['original_index'], [-1])[0]:
                continue
            compute_times.append(stats['compute_time'])

        if len(compute_times) > 1:
            compute_var = np.std(compute_times)
        else:
            compute_var = 0

        return (total_delay, total_comm, mem_over, compute_var)
    except TimeoutException:
        print(f"计算超时（>{max_time}秒），返回不可行解")
        return (float('inf'),) * 4
    except Exception as e:
        print(f"计算过程中发生异常: {str(e)}")
        return (float('inf'),) * 4
    finally:
        signal.alarm(0)  # 取消超时设置
def crossover(p1, p2, num_blocks, num_cuts,fixed_cuts):
    """交叉操作 - 确保切割点不重复"""
    if num_cuts <= 1:
        return p1[:], p2[:]

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


def plot_pareto_front(pareto_solutions, block_info,big_block_index, device_flops, mem_limit, device_bandwidth_dict):
    """
    Plot Pareto front with 3D scatter plot and parallel coordinates plot
    """
    if not pareto_solutions:
        print("No Pareto solutions found")
        return

    # Re-evaluate all Pareto solutions
    fitness_values = []
    all_device_stats = []

    for solution in pareto_solutions:
        result = evaluate_individual2(
            solution,
            big_block_index,
            block_number,
            block_info,
            device_flops,
            mem_limit,
            device_bandwidth_dict
        )

        if result[0] < float('inf'):
            fitness_values.append(result[:4])  # First four objectives
            all_device_stats.append(result[4])  # Device statistics

    if not fitness_values:
        print("All Pareto solutions are invalid")
        return

    # Convert to numpy array
    fitness_array = np.array(fitness_values)
    objectives = ['Total Delay', 'Total Communication', 'Memory Std Dev', 'Compute Time Std Dev']

    # Create figure
    fig = plt.figure(figsize=(18, 12))

    # 3D Scatter Plot
    ax1 = fig.add_subplot(231, projection='3d')
    sc = ax1.scatter(
        fitness_array[:, 0],
        fitness_array[:, 1],
        fitness_array[:, 2],
        c=fitness_array[:, 3],
        cmap='viridis',
        s=50
    )
    ax1.set_xlabel(objectives[0])
    ax1.set_ylabel(objectives[1])
    ax1.set_zlabel(objectives[2])
    ax1.set_title('Pareto Front (3D View)')
    fig.colorbar(sc, ax=ax1, label=objectives[3])

    # 2D Projections
    ax2 = fig.add_subplot(232)
    ax2.scatter(fitness_array[:, 0], fitness_array[:, 1], c=fitness_array[:, 3], cmap='viridis')
    ax2.set_xlabel(objectives[0])
    ax2.set_ylabel(objectives[1])
    ax2.set_title(f'{objectives[0]} vs {objectives[1]}')

    ax3 = fig.add_subplot(233)
    ax3.scatter(fitness_array[:, 0], fitness_array[:, 2], c=fitness_array[:, 3], cmap='viridis')
    ax3.set_xlabel(objectives[0])
    ax3.set_ylabel(objectives[2])
    ax3.set_title(f'{objectives[0]} vs {objectives[2]}')

    ax4 = fig.add_subplot(234)
    ax4.scatter(fitness_array[:, 1], fitness_array[:, 2], c=fitness_array[:, 3], cmap='viridis')
    ax4.set_xlabel(objectives[1])
    ax4.set_ylabel(objectives[2])
    ax4.set_title(f'{objectives[1]} vs {objectives[2]}')

    # Parallel Coordinates Plot
    ax5 = fig.add_subplot(235)
    # Avoid division by zero
    ranges = fitness_array.max(axis=0) - fitness_array.min(axis=0)
    ranges[ranges == 0] = 1  # Prevent division by zero
    normalized = (fitness_array - fitness_array.min(axis=0)) / ranges

    for i in range(normalized.shape[0]):
        ax5.plot(normalized[i], 'o-', alpha=0.5, linewidth=1)

    ax5.set_xticks(range(len(objectives)))
    ax5.set_xticklabels(objectives, rotation=45)
    ax5.set_title('Parallel Coordinates Plot')
    ax5.set_ylabel('Normalized Value')
    ax5.grid(True, linestyle='--', alpha=0.6)

    # Device Load Distribution
    ax6 = fig.add_subplot(236)

    if all_device_stats:
        # Select the best solution (minimal total delay)
        best_idx = np.argmin(fitness_array[:, 0])
        device_stats = all_device_stats[best_idx]

        devices = [stat['device'] for stat in device_stats]
        compute_times = [stat['compute_time'] for stat in device_stats]
        memory_usage = [stat['memory'] for stat in device_stats]

        width = 0.35
        x = np.arange(len(devices))

        ax6.bar(x - width / 2, compute_times, width, label='Compute Time (s)')
        ax6.bar(x + width / 2, memory_usage, width, label='Memory Usage (MB)')

        ax6.set_xlabel('Devices')
        ax6.set_ylabel('Resource Usage')
        ax6.set_title('Device Resource Usage (Best Solution)')
        ax6.set_xticks(x)
        ax6.set_xticklabels(devices)
        ax6.legend()
        ax6.grid(True, axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig('pareto/pareto_front.png', dpi=300, bbox_inches='tight')

    # Print best solution details
    best_solution = pareto_solutions[best_idx]
    best_fitness = fitness_array[best_idx]

    print("\n" + "=" * 80)
    print("Best Solution Details (Minimal Total Delay):")
    print(f"Cut Points: {best_solution}")
    print(f"Fixed Cut Points: {fixed_cuts}")
    print(f"Total Delay: {best_fitness[0]:.6f} seconds")
    print(f"Total Communication: {best_fitness[1] / 1e6:.2f} MB")
    print(f"Memory Std Deviation: {best_fitness[2]:.4f}")
    print(f"Compute Time Std Deviation: {best_fitness[3]:.6f}")

    print("\nDevice Assignment Details:")
    for stat in device_stats:
        print(f"Device {stat['device']}: "
              f"Compute Time = {stat['compute_time']:.6f}s, "
              f"Memory Usage = {stat['memory'] / 1e6:.2f} MB, "
              f"Output Size = {stat['output_size'] / 1e6:.2f} MB")

    print("=" * 80)

if __name__ == "__main__":
    # 初始化模型并进行 shape propagation
    model = resnet50()
    fx = symbolic_trace(model)
    ShapeProp(fx).propagate(torch.randn(1, 3, 224, 224))

    # 1.1 构建逻辑块
    blocks = build_logic_blocks(fx)
    print("1：寻找关键边后合并的逻辑块blocks:", blocks)
    df = profile_and_tabulate(model)
    # 1.2 滑动窗口合并
    merged,merged_topk,logical_layers = find_bottleneck_segments(df, blocks,window=2,thresh_ratio=1.8)
    print("2：找到的瓶颈块索引序号merged:", merged)
    print("3：数据量降序排的瓶颈块merged_topk:",merged_topk)
    print("4：包含瓶颈块的待处理逻辑块logical_layers:", logical_layers)

    cases = case_select(df, merged_topk)
    print(f'5；每个瓶颈块张量切割的案例信息cases：{cases}')



    # 执行评估
    results = [{'cut_layer': 'layer1_1_relu_2', 'paste_layer': 'add_4', 'start_idx': 26, 'end_idx': 57, 'total_flops': np.float64(1627.809792), 'accuracies': {'split_2_height': 0.74, 'split_2_weight': 0.74, 'split_3_height': 0.734, 'split_3_weight': 0.738, 'split_4_height': 0.728, 'split_4_weight': 0.74, 'split_4_grid': 0.744}, 'best_accuracy': 0.744, 'best_method': 'split_4_grid'}, {'cut_layer': 'layer2_2_relu_2', 'paste_layer': 'add_8', 'start_idx': 68, 'end_idx': 99, 'total_flops': np.float64(1623.1434239999999), 'accuracies': {'split_2_height': 0.73, 'split_2_weight': 0.74, 'split_3_height': 0.724, 'split_3_weight': 0.72, 'split_4_height': 0.74, 'split_4_weight': 0.72, 'split_4_grid': 0.734}, 'best_accuracy': 0.74, 'best_method': 'split_2_weight'}, {'cut_layer': 'layer3_4_relu_2', 'paste_layer': 'add_14', 'start_idx': 130, 'end_idx': 161, 'total_flops': np.float64(1620.8102399999998), 'accuracies': {'split_2_height': 0.722, 'split_2_weight': 0.734, 'split_3_height': 0.718, 'split_3_weight': 0.732, 'split_4_height': 0.706, 'split_4_weight': 0.724, 'split_4_grid': 0.718}, 'best_accuracy': 0.734, 'best_method': 'split_2_weight'}]


    print(results)

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
        block_number = 2
    elif best_case['best_method'] in ['split_3_height','split_3_weight']:
        block_number = 2
    else:
        block_number = 2
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
    mem_limit = 4 * 1024 ** 3  # 4GB
    device_bandwidth_dict = {
        ('rpi5_1', 'rpi5_2'): 102.1,
        ('rpi5_1', 'rpi5_3'): 94.5,
        ('rpi5_1', 'jetson1'): 55.6,
        ('rpi5_1', 'jetson2'): 57.8,
        ('rpi5_1', 'jetson3'): 55.5,
        ('rpi5_1', 'pc_cpu'): 66.7,
        ('rpi5_2', 'rpi5_1'): 102.1,
        ('rpi5_2', 'rpi5_3'): 103.9,
        ('rpi5_2', 'jetson1'): 49.3,
        ('rpi5_2', 'jetson2'): 49.9,
        ('rpi5_2', 'jetson3'): 59.9,
        ('rpi5_2', 'pc_cpu'): 87.5,
        ('rpi5_3', 'rpi5_1'): 94.5,
        ('rpi5_3', 'rpi5_2'): 103.9,
        ('rpi5_3', 'jetson1'): 54.9,
        ('rpi5_3', 'jetson2'): 41.3,
        ('rpi5_3', 'jetson3'): 50.1,
        ('rpi5_3', 'pc_cpu'): 85.5,
        ('jetson1', 'rpi5_1'): 55.6,
        ('jetson1', 'rpi5_2'): 49.3,
        ('jetson1', 'rpi5_3'): 54.9,
        ('jetson1', 'jetson2'): 207.5,
        ('jetson1', 'jetson3'): 214.1,
        ('jetson1', 'pc_cpu'): 170.0,
        ('jetson2', 'rpi5_1'): 57.8,
        ('jetson2', 'rpi5_2'): 49.9,
        ('jetson2', 'rpi5_3'): 41.3,
        ('jetson2', 'jetson1'): 207.5,
        ('jetson2', 'jetson3'): 207.7,
        ('jetson2', 'pc_cpu'): 178.5,
        ('jetson3', 'rpi5_1'): 55.5,
        ('jetson3', 'rpi5_2'): 59.9,
        ('jetson3', 'rpi5_3'): 50.1,
        ('jetson3', 'jetson1'): 214.1,
        ('jetson3', 'jetson2'): 207.7,
        ('jetson3', 'pc_cpu'): 152.3,
        ('pc_cpu', 'rpi5_1'): 66.7,
        ('pc_cpu', 'rpi5_2'): 87.5,
        ('pc_cpu', 'rpi5_3'): 85.5,
        ('pc_cpu', 'jetson1'): 170.0,
        ('pc_cpu', 'jetson2'): 178.5,
        ('pc_cpu', 'jetson3'): 152.3
    }

    start_time = time.time()


    # 5. 运行优化
    pareto_solutions, block_info, fixed_cuts = nsga2_optimize(
        logic_blocks=logical_layers,
        big_block_index=index_front,
        block_number=block_number,
        df=df,
        device_flops=device_flops,
        mem_limit=mem_limit,
        device_bandwidth_dict=device_bandwidth_dict,
        pop_size=50,
        ngen=50
    )
    end_time = time.time()
    plot_pareto_front(pareto_solutions, block_info, index_front,device_flops, mem_limit, device_bandwidth_dict)

    print(f'帕累托求解耗时{end_time - start_time}s')
    # print(pareto_solutions)
    if pareto_solutions:
        print(f"找到 {len(pareto_solutions)} 个帕累托最优解")

        # 按总延迟排序
        pareto_solutions.sort(
            key=lambda ind: evaluate_individual2(ind, index_front,block_number, block_info, device_flops, mem_limit, device_bandwidth_dict)[0]
        )

        # 输出前3个最优解
        for i, solution in enumerate(pareto_solutions[:2]):
            # 合并所有切割点
            delay, comm, mem_over, balance,device_stats = evaluate_individual2(
                solution, index_front, block_number, block_info, device_flops, mem_limit, device_bandwidth_dict
            )
            solution.append(index_front)
            solution.append(index_front+1)

            print(f"\n方案 {i + 1}:")
            print(f"总延迟: {delay:.8f}秒")
            print(f"总通信量: {comm / 1e6:.2f}MB")
            print(f"内存占用标准差: {mem_over:.4f}")
            print(f"负载均衡方差: {balance:.4f}")

            # 显示所有切割点
            print(f"所有切割点位置: {sorted(solution)}")
            print(f'切割段落数据{device_stats}')

        pareto_solutions.sort(
            key=lambda ind: evaluate_individual2(ind, index_front,block_number, block_info, device_flops, mem_limit, device_bandwidth_dict)[1]
        )

        # 输出前3个最优解
        for i, solution in enumerate(pareto_solutions[:2]):
            # 合并所有切割点

            delay, comm, mem_over, balance,device_stats = evaluate_individual2(
                solution, index_front, block_number, block_info, device_flops, mem_limit, device_bandwidth_dict
            )
            solution.append(index_front)
            solution.append(index_front+1)

            print(f"\n方案 {i + 1}:")
            print(f"总延迟: {delay:.8f}秒")
            print(f"总通信量: {comm / 1e6:.2f}MB")
            print(f"内存占用标准差: {mem_over:.4f}")
            print(f"负载均衡方差: {balance:.4f}")

            # 显示所有切割点
            print(f"所有切割点位置: {sorted(solution)}")
            print(f'切割段落数据：{device_stats}')

        pareto_solutions.sort(
            key=lambda ind: evaluate_individual2(ind, index_front,block_number, block_info, device_flops, mem_limit, device_bandwidth_dict)[2]
        )

        # 输出前3个最优解
        for i, solution in enumerate(pareto_solutions[:2]):
            # 合并所有切割点

            delay, comm, mem_over, balance,device_stats = evaluate_individual2(
                solution, index_front, block_number, block_info, device_flops, mem_limit, device_bandwidth_dict
            )
            solution.append(index_front)
            solution.append(index_front+1)

            print(f"\n方案 {i + 1}:")
            print(f"总延迟: {delay:.8f}秒")
            print(f"总通信量: {comm / 1e6:.2f}MB")
            print(f"内存占用标准差: {mem_over:.4f} ")
            print(f"负载均衡方差: {balance:.4f}")

            # 显示所有切割点
            print(f"所有切割点位置: {sorted(solution)}")
            print(f'切割段落数据：{device_stats}')

        pareto_solutions.sort(
            key=lambda ind: evaluate_individual2(ind, index_front,block_number, block_info, device_flops, mem_limit, device_bandwidth_dict)[3]
        )

        # 输出前3个最优解
        for i, solution in enumerate(pareto_solutions[:1]):
            # 合并所有切割点

            delay, comm, mem_over, balance,device_stats = evaluate_individual2(
                solution, index_front, block_number, block_info, device_flops, mem_limit, device_bandwidth_dict
            )
            solution.append(index_front)
            solution.append(index_front+1)

            print(f"\n方案 {i + 1}:")
            print(f"总延迟: {delay:.8f}秒")
            print(f"总通信量: {comm / 1e6:.2f}MB")
            print(f"内存占用标准差: {mem_over:.4f}")
            print(f"负载均衡方差: {balance:.4f}")

            # 显示所有切割点
            print(f"所有切割点位置: {sorted(solution)}")
            print(f'切割段落数据：{device_stats}')

    else:
        print("未找到可行解")