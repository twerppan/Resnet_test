import time
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

# 根据瓶颈块选择张量切割后计算模型的精度函数
def evaluate_precision_decay(cases):
    # 配置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载预训练的AlexNet模型
    model = resnet50(pretrained=True).to(device)
    model.eval()

    # 图像预处理Transform
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    for case in cases:
        # 定义保存预测结果的文件路径
        predict_txt = f"test\CutPredict{case['cut_layer']} -- {case['paste_layer']}.txt"

        # 初始化预测文件
        with open(predict_txt, 'w') as f:
            pass  # 创建文件

        # 遍历目录中的所有图片文件
        images_dir = r"D:\BaiduNetdiskDownload\ILSVRC2012_img_val"
        image_files = [f for f in os.listdir(images_dir) if f.endswith(".JPEG")]
        total_images = len(image_files)
        with open(predict_txt, 'a') as f:
            for img_file in image_files:

                img_path = os.path.join(images_dir, img_file)
                img = Image.open(img_path).convert('RGB')
                input_tensor = preprocess(img)  # 预处理后形状 [3, 224, 224]
                input_tensor = input_tensor.unsqueeze(0).to(device)
                split_fn = create_split_fn(model, case['cut_layer'], case['paste_layer'])
                outputs = split_fn(input_tensor)

                # 获取预测结果
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
                predicted_class = torch.argmax(probabilities).item()

                # 写入预测结果到文件并打印到控制台
                img_name = img_file.split('.')[0]

                f.write(f"{img_name}: {predicted_class}\n")
        print("所有图像的预测结果已写入到 cut_predict.txt")

        correct = 0
        total = 0

        # 读取真实类别和预测类别文件
        with open(r"ground_truth_label.txt", 'r') as f_label, \
                open(predict_txt, 'r') as f_predict:
            # 逐行读取并比较
            for line_label, line_predict in zip(f_label, f_predict):
                # 解析真实类别
                img_name_label, true_class = line_label.strip().split(': ')
                # 解析预测类别
                img_name_predict, predicted_class = line_predict.strip().split(': ')
                # 确保比较的是同一张图片
                if img_name_label != img_name_predict:
                    print(f"图片名称不匹配：{img_name_label} vs {img_name_predict}")
                    continue
                # 判断预测是否正确
                if true_class == predicted_class:
                    correct += 1
                total += 1

        # 计算准确率
        accuracy = correct / total if total != 0 else 0
        case.append(accuracy)
        print(f"模型准确率: {accuracy:.2%} ({correct}/{total})")
    return cases

# --- 步骤 1.4：NSGA-II 多目标优化寻找切割方案的帕累托前沿 ---

def nsga2_optimize(logic_blocks, pop_size=50, ngen=100, threshold=0.01):
    num_gens = len(logic_blocks) - 1  # 切割点数量

    # 随机生成个体（切割方案）
    def random_individual():
        return [random.randint(0, 1) for _ in range(num_gens)]

    # 从二进制编码解码为实际切割位置
    def decode(ind):
        return [i + 1 for i, g in enumerate(ind) if g == 1]

    # 占位：目标函数评估，用户可替换为实际延迟/通信/内存/负载均衡计算
    def evaluate(ind):
        delay = random.random()
        comm = random.random()
        mem = random.random()
        balance = random.random()
        return (delay, comm, mem, balance)

    # 支配关系判断
    def dominates(f1, f2):
        return all(a <= b for a, b in zip(f1, f2)) and any(a < b for a, b in zip(f1, f2))

    # 初始化种群并评估
    pop = [random_individual() for _ in range(pop_size)]
    fitness = [evaluate(ind) for ind in pop]

    # NSGA-II 主循环
    for gen in range(ngen):
        # 非支配排序
        fronts = []
        S = [[] for _ in pop]
        n = [0] * len(pop)
        rank = [0] * len(pop)
        for p in range(len(pop)):
            for q in range(len(pop)):
                if dominates(fitness[p], fitness[q]):
                    S[p].append(q)
                elif dominates(fitness[q], fitness[p]):
                    n[p] += 1
            if n[p] == 0:
                rank[p] = 0
                fronts.append([p])
        # 拥挤度计算 & 选择省略，可按需补充

        # 生成子代（交叉 + 变异）
        offspring = []
        while len(offspring) < pop_size:
            a, b = random.sample(pop, 2)
            # 单点交叉
            pt = random.randint(1, num_gens - 1)
            child = a[:pt] + b[pt:]
            # 变异
            if random.random() < 0.1:
                mpos = random.randint(0, num_gens - 1)
                child[mpos] ^= 1
            offspring.append(child)
        pop = offspring
        fitness = [evaluate(ind) for ind in pop]

        # 可在此处添加终止条件判断

    # 返回最后一代的非支配解
    pareto = [pop[i] for i, v in enumerate(fitness) if n[i] == 0]
    return pareto

# 支配关系函数放置在模块外，供 NSGA-II 使用
def dominates(f1, f2):
    """
    判断 f1 是否在所有目标上不劣且至少一项更好
    """
    return all(a <= b for a, b in zip(f1, f2)) and any(a < b for a, b in zip(f1, f2))

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
    print("按数据量降序排序的瓶颈块:",merged_topk)
    print("寻找到逻辑块后的逻辑层logical_layers:", logical_layers)

    cases = case_select(df, merged_topk)
    print(f'验证的瓶颈层案例：{cases}')
    # evaluate_precision_decay(cases)
    accuracy_list = [75.68,74.90,74.91]
    for i in range(3):
        cases[i]['accuracy']=accuracy_list[i]
    max_case = max(cases, key=lambda x: x['accuracy'])
    print(max_case)

    bottleneck = (max_case['start_idx'], max_case['end_idx'])
    # # 1.4 对剩余逻辑块执行 NSGA-II 搜索帕累托最优切割方案
    # bottleneck = merged[0]
    # print("瓶颈块:", bottleneck)
    index_front = 0

    for i in range(len(logical_layers)):
        if logical_layers[i][0]==max_case['cut_layer']:
            index_front = i
    index_back = index_front+1

    print(f'index_front:{index_front}')
    front_block = logical_layers[:index_front]
    back_block = logical_layers[index_back:]
    print(f'front_block:{front_block}')
    print(f'back_block:{back_block}')
    pareto1 = nsga2_optimize(front_block)
    pareto2 = nsga2_optimize(back_block)
    print("Pareto1 最优切割方案:", pareto1)
    print("Pareto2 最优切割方案:", pareto2)
