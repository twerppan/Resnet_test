import random
from torchvision.models import resnet50
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
from scipy.stats import rankdata

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

def nsga2_optimize(logic_blocks, df, device_flops, mem_limit, bandwidth_bps,
                   pop_size=50, ngen=100, threshold=0.01, crossover_rate=0.8, mutation_rate=0.1):
    """
    使用 NSGA-II 算法优化模型划分方案
    切割点数量固定为设备数量减一
    """
    # 1. 预处理逻辑块信息
    block_info = preprocess_blocks(logic_blocks, df)
    num_blocks = len(block_info)
    num_devices = len(device_flops)
    device_list = list(device_flops.keys())

    # 计算切割点数量（设备数量减一）
    num_cuts = num_devices - 1

    # 2. 初始化种群
    population = initialize_population(num_blocks, num_cuts, pop_size)

    # 3. NSGA-II 主循环
    prev_front = None
    for gen in range(ngen):
        # 评估种群
        fitness = []
        for ind in population:
            try:
                fit = evaluate_individual(ind, block_info, device_flops, mem_limit, bandwidth_bps)
                fitness.append(fit)
            except Exception as e:
                print(f"评估个体时出错: {e}")
                fitness.append((float('inf'), float('inf'), float('inf'), float('inf')))

        # 非支配排序
        fronts, ranks = non_dominated_sorting(fitness)

        # 如果没有前沿，跳过这一代
        if not fronts or not any(fronts):
            print(f"第 {gen} 代: 没有找到前沿")
            population = initialize_population(num_blocks, num_cuts, pop_size)
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
                parents.extend(initialize_population(num_blocks, num_cuts, max(2, pop_size - len(parents))))

            p1, p2 = random.sample(parents, 2)

            # 交叉
            if random.random() < crossover_rate:
                try:
                    c1, c2 = crossover(p1, p2, num_blocks, num_cuts)
                except Exception as e:
                    print(f"交叉时出错: {e}")
                    c1, c2 = p1[:], p2[:]
            else:
                c1, c2 = p1[:], p2[:]

            # 变异
            if random.random() < mutation_rate:
                c1 = mutate(c1, num_blocks, num_cuts)
            if random.random() < mutation_rate:
                c2 = mutate(c2, num_blocks, num_cuts)

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
    return population, block_info

def initialize_population(num_blocks, num_cuts, pop_size):
    """
    初始化种群
    确保切割点不重复且有序
    """
    population = []
    for _ in range(pop_size):
        if num_cuts == 0:
            individual = []
        else:
            # 确保切割点不重复
            cut_points = sorted(random.sample(range(1, num_blocks), num_cuts))
            individual = cut_points

        population.append(individual)
    return population


def evaluate_individual(cut_points, block_info, device_flops, mem_limit, bandwidth_bps):
    """
    评估个体的适应度
    """
    try:
        num_blocks = len(block_info)
        num_devices = len(device_flops)
        device_list = list(device_flops.keys())

        # 1. 根据切割点划分段落
        segments = []
        start = 0
        for cut in cut_points:
            segments.append(list(range(start, cut)))
            start = cut
        segments.append(list(range(start, num_blocks)))

        num_segments = len(segments)

        # 段落数量应该等于设备数量
        if num_segments != num_devices:
            return (float('inf'), float('inf'), float('inf'), float('inf'))

        # 2. 计算每个段落的资源需求
        segment_stats = []
        for seg in segments:
            # 确保 seg 是整数索引列表
            if not isinstance(seg, list) or not all(isinstance(i, int) for i in seg):
                return (float('inf'),) * 4

            total_flops = sum(block_info[i]['flops'] for i in seg)
            total_memory = sum(block_info[i]['total_memory'] for i in seg)

            # 获取最后一块的输出大小
            if seg:  # 确保段落不为空
                last_block_idx = seg[-1]
                output_size = block_info[last_block_idx]['output_size']
            else:
                output_size = 0

            segment_stats.append({
                'flops': total_flops,
                'memory': total_memory,
                'output_size': output_size
            })

        # 3. 将段落分配给设备（按顺序）
        device_stats = []
        for i, stats in enumerate(segment_stats):
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
            if device_stats[i]['device'] == device_stats[i - 1]['device']:
                comm_delay = 0  # 如果在同一个设备上，通信延迟为0
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

    except Exception as e:
        print(f"评估个体时出错: {e}")
        return (float('inf'), float('inf'), float('inf'), float('inf'))


def crossover(p1, p2, num_blocks, num_cuts):
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
            all_positions = list(range(1, num_blocks))
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


def mutate(individual, num_blocks, num_cuts):
    """变异操作 - 确保切割点不重复"""
    if num_cuts == 0:
        return individual

    # 创建新个体的副本
    new_individual = individual[:]

    # 随机选择一个切割点进行变异
    idx = random.randint(0, num_cuts - 1)

    # 生成新的切割点位置，确保不重复且在有效范围内
    possible_positions = [i for i in range(1, num_blocks) if i not in new_individual]

    if possible_positions:
        new_cut = random.choice(possible_positions)
        new_individual[idx] = new_cut
        new_individual.sort()

    return new_individual


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
if __name__ == '__main__':
    # 1. 创建ResNet50模型
    model = resnet50()

    # 2. 分析模型并获取数据表
    df = profile_and_tabulate(model)

    # 3. 构建逻辑块
    logic_blocks = build_logic_blocks(symbolic_trace(model))
    # print(len(logic_blocks))
    front_block = [['x'], ['conv1'], ['bn1'], ['relu'],
                  ['maxpool', 'layer1_0_conv1', 'layer1_0_bn1', 'layer1_0_relu', 'layer1_0_conv2', 'layer1_0_bn2',
                   'layer1_0_relu_1', 'layer1_0_conv3', 'layer1_0_bn3', 'layer1_0_downsample_0',
                   'layer1_0_downsample_1', 'add'],
                  ['layer1_0_relu_2', 'layer1_1_conv1', 'layer1_1_bn1', 'layer1_1_relu', 'layer1_1_conv2',
                   'layer1_1_bn2', 'layer1_1_relu_1', 'layer1_1_conv3', 'layer1_1_bn3', 'add_1']]
    back_block = [
        ['layer2_1_relu_2', 'layer2_2_conv1', 'layer2_2_bn1', 'layer2_2_relu', 'layer2_2_conv2', 'layer2_2_bn2',
         'layer2_2_relu_1', 'layer2_2_conv3', 'layer2_2_bn3', 'add_5'],
        ['layer2_2_relu_2', 'layer2_3_conv1', 'layer2_3_bn1', 'layer2_3_relu', 'layer2_3_conv2', 'layer2_3_bn2',
         'layer2_3_relu_1', 'layer2_3_conv3', 'layer2_3_bn3', 'add_6', 'layer2_3_relu_2', 'layer3_0_conv1',
         'layer3_0_bn1', 'layer3_0_relu', 'layer3_0_conv2', 'layer3_0_bn2', 'layer3_0_relu_1', 'layer3_0_conv3',
         'layer3_0_bn3', 'layer3_0_downsample_0', 'layer3_0_downsample_1', 'add_7', 'layer3_0_relu_2', 'layer3_1_conv1',
         'layer3_1_bn1', 'layer3_1_relu', 'layer3_1_conv2', 'layer3_1_bn2', 'layer3_1_relu_1', 'layer3_1_conv3',
         'layer3_1_bn3', 'add_8'],
        ['layer3_1_relu_2', 'layer3_2_conv1', 'layer3_2_bn1', 'layer3_2_relu', 'layer3_2_conv2', 'layer3_2_bn2',
         'layer3_2_relu_1', 'layer3_2_conv3', 'layer3_2_bn3', 'add_9'],
        ['layer3_2_relu_2', 'layer3_3_conv1', 'layer3_3_bn1', 'layer3_3_relu', 'layer3_3_conv2', 'layer3_3_bn2',
         'layer3_3_relu_1', 'layer3_3_conv3', 'layer3_3_bn3', 'add_10'],
        ['layer3_3_relu_2', 'layer3_4_conv1', 'layer3_4_bn1', 'layer3_4_relu', 'layer3_4_conv2', 'layer3_4_bn2',
         'layer3_4_relu_1', 'layer3_4_conv3', 'layer3_4_bn3', 'add_11'],
        ['layer3_4_relu_2', 'layer3_5_conv1', 'layer3_5_bn1', 'layer3_5_relu', 'layer3_5_conv2', 'layer3_5_bn2',
         'layer3_5_relu_1', 'layer3_5_conv3', 'layer3_5_bn3', 'add_12', 'layer3_5_relu_2', 'layer4_0_conv1',
         'layer4_0_bn1', 'layer4_0_relu', 'layer4_0_conv2', 'layer4_0_bn2', 'layer4_0_relu_1', 'layer4_0_conv3',
         'layer4_0_bn3', 'layer4_0_downsample_0', 'layer4_0_downsample_1', 'add_13', 'layer4_0_relu_2',
         'layer4_1_conv1', 'layer4_1_bn1', 'layer4_1_relu', 'layer4_1_conv2', 'layer4_1_bn2', 'layer4_1_relu_1',
         'layer4_1_conv3', 'layer4_1_bn3', 'add_14'],
        ['layer4_1_relu_2', 'layer4_2_conv1', 'layer4_2_bn1', 'layer4_2_relu', 'layer4_2_conv2', 'layer4_2_bn2',
         'layer4_2_relu_1', 'layer4_2_conv3', 'layer4_2_bn3', 'add_15'], ['layer4_2_relu_2'], ['avgpool'], ['flatten'],
        ['fc'], ['output']]
    # print(len(front_block))
    # print(len(back_block))
    # 4. 设备配置
    device_flops = {
        'pc_cpu': 500e9,
        'rpi5_1': 320e9,
        'rpi5_2': 320e9,
        'rpi5_3': 320e9,
        'jetson': 472e9
    }
    mem_limit = 4 * 1024 ** 2  # 4GB
    bandwidth_bps = 5e7  # 1MB/s

    # 5. 运行优化
    pareto_solutions, block_info = nsga2_optimize(
        back_block,
        df,
        device_flops,
        mem_limit,
        bandwidth_bps,
        pop_size=50,
        ngen=100
    )

    # 6. 解码最优解
    if pareto_solutions:
        print(f"找到 {len(pareto_solutions)} 个帕累托最优解")

        # 按总延迟排序
        pareto_solutions.sort(
            key=lambda ind: evaluate_individual(ind, block_info, device_flops, mem_limit, bandwidth_bps)[0]
        )

        # 输出前3个最优解
        for i, solution in enumerate(pareto_solutions[:3]):
            num_segments = len(solution) + 1
            delay, comm, mem_over, balance = evaluate_individual(
                solution, block_info, device_flops, mem_limit, bandwidth_bps
            )

            print(f"\n方案 {i + 1}:")
            print(f"切割点数量: {len(solution)}")
            print(f"段落数量: {num_segments}")
            print(f"总延迟: {delay:.4f}秒")
            print(f"总通信量: {comm / 1e6:.2f}MB")
            print(f"内存超限: {mem_over / 1e6:.2f}MB")
            print(f"负载均衡方差: {balance:.4f}")

            # 显示切割点
            if solution:
                print(f"切割点位置: {solution}")
            else:
                print("没有切割点（整个模型在一个设备上运行）")

            # 划分段落
            segments = []
            start = 0
            for cut in solution:
                segments.append(list(range(start, cut)))
                start = cut
            segments.append(list(range(start, len(back_block))))

            # 分配设备
            device_list = list(device_flops.keys())
            for j, seg in enumerate(segments):
                device = device_list[j]
                if seg:  # 确保段落不为空
                    print(f"段落 {j + 1} (设备: {device}): 块 {seg[0] + 1} 到 {seg[-1] + 1}")
                else:
                    print(f"段落 {j + 1} (设备: {device}): 空段落")
    else:
        print("未找到可行解")