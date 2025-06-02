"""
基于PyTorch FX和Tarjan算法的神经网络关键路径分析工具
实现原理：通过构建计算图的无向图表示，使用Tarjan算法寻找桥边（割边）
"""

import torch
from torchvision.models import resnet18  # 使用AlexNet作为示例模型
from torch.fx import symbolic_trace  # PyTorch的图追踪工具
from collections import defaultdict  # 用于构建邻接表


model = resnet18().eval()  # 实例化AlexNet并设置为评估模式
traced = symbolic_trace(model)  # 将模型转换为GraphModule对象
print(traced)

# ==================== 节点显示名称生成 ====================
def get_node_display_name(node, traced_model):
    """
    生成所有operator的名称，包括模块、函数、操作方法、占位符
    """
    # 处理模块调用节点（如Conv2d、Linear等）
    if node.op == 'call_module':
        module = traced_model.get_submodule(node.target)  # 获取对应的PyTorch模块实例
        return f"{node.target.replace('.', '_')} ({module.__class__.__name__})"  # 格式：层路径_类型
    # 处理函数调用节点（如torch.add）
    elif node.op == 'call_function':
        return f"Func:{node.target.__name__}"  # 显示函数名称
    # 处理方法调用节点（如tensor.view）
    elif node.op == 'call_method':
        return f"Method:{node.target}"  # 显示方法名称
    # 处理输入占位符节点
    elif node.op == 'placeholder':
        return f"Input:{node.name}"  # 标记为输入节点
    # 其他类型节点保持默认
    else:
        return f"Var:{node.name}"

example_input = torch.randn(1, 3, 224, 224)  # 生成符合模型输入的示例张量
shape_dict = {}  # 存储各节点输出形状的字典


class ShapeRecorder(torch.fx.Interpreter):
    """
    自定义解释器类，用于记录每个节点的输出张量形状
    继承自torch.fx.Interpreter，重写run_node方法
    """
    def run_node(self, n):
        # 执行原始节点计算
        result = super().run_node(n)

        # 特殊处理输入节点
        if n.op == 'placeholder':
            shape_dict[n] = example_input.shape  # 直接记录示例输入形状
        # 处理张量输出
        elif isinstance(result, torch.Tensor):
            shape_dict[n] = result.shape  # 记录形状
        # 处理多输出情况（如某些返回tuple的操作）
        elif isinstance(result, (list, tuple)):
            shape_dict[n] = [r.shape if isinstance(r, torch.Tensor) else None for r in result]
        # 非张量输出标记为None
        else:
            shape_dict[n] = None
        return result


# 执行形状记录
ShapeRecorder(traced).run(example_input)  # 遍历计算图并记录形状

edges = []  # 存储有向边集合（源节点 -> 目标节点）
name_map = {}  # 节点到显示名称的映射

# 遍历所有节点构建边关系
for node in traced.graph.nodes:
    # 生成当前节点的显示名称并存入映射表
    name_map[node] = get_node_display_name(node, traced)

    # 遍历当前节点的所有输入节点（前驱节点）
    for input_node in node.all_input_nodes:
        # 添加有向边（input_node -> node）
        edges.append((input_node, node))


graph = defaultdict(list)  # 邻接表表示的无向图
edge_set = set()  # 用于去重的边集合

# 构建无向图的双向连接
for u, v in edges:
    # 检查边是否已存在（包括两个方向）
    if (u, v) not in edge_set and (v, u) not in edge_set:
        # 添加双向连接（无向边）
        graph[u].append(v)
        graph[v].append(u)
        # 记录已处理边
        edge_set.add((u, v))
        edge_set.add((v, u))

# ==================== Tarjan算法实现 ====================
visited = set()  # 已访问节点集合
disc = {}  # 节点发现时间（DFS序号）
low = {}  # 通过回边能到达的最早祖先的发现时间
time = [1]  # 全局时间计数器（使用列表实现引用传递）
bridges = []  # 存储找到的桥边
parent = {}  # 节点的父节点映射


def tarjan(u):
    """
    Tarjan算法递归实现
    参数：
        u: 当前处理的节点
    """
    # 初始化发现时间和low值
    disc[u] = low[u] = time[0]
    time[0] += 1  # 时间递增
    visited.add(u)  # 标记为已访问

    # 遍历当前节点的所有邻居
    for v in graph[u]:
        if v not in visited:
            parent[v] = u  # 记录父子关系
            tarjan(v)  # 递归处理子节点

            # 回溯时更新low值（子节点的low值可能更小）
            low[u] = min(low[u], low[v])

            # 桥边判定条件：子节点无法通过回边到达当前节点或更早节点
            if low[v] > disc[u]:
                bridges.append((u, v))  # 找到桥边

        # 处理回边（已访问过且不是父节点）
        elif v != parent.get(u, None):
            # 关键步骤：使用disc[v]而非low[v]来避免错误传播
            low[u] = min(low[u], disc[v])


# 对每个未访问节点执行算法
for node in graph:
    if node not in visited:
        tarjan(node)

# ==================== 桥边结果处理 ====================
bridge_info = []  # 存储最终的桥边信息
seen_edges = set()  # 已处理边集合（去重用）

# 遍历所有找到的桥边
for u, v in bridges:
    # 检查两个可能的方向（因为无向图的边是双向存储的）
    for src, dst in [(u, v), (v, u)]:
        # 确认边存在于原始有向边集合中
        if (src, dst) in edges:
            # 去重处理
            if (src, dst) not in seen_edges:
                # 记录格式：(源节点名称, 目标节点名称, 张量形状)
                bridge_info.append((
                    name_map[src],  # 源节点显示名称
                    name_map[dst],  # 目标节点显示名称
                    shape_dict.get(src)  # 源节点的输出形状
                ))
                seen_edges.add((src, dst))
            break  # 找到有效方向后跳出循环

# ==================== 结果输出 ====================
print("关键割集边及张量形状：")
for src_name, dst_name, shape in bridge_info:
    # 使用符号增强可读性
    print(f"[关键路径] {src_name} → {dst_name}")
    print(f"└─ 传输形状：{shape}\n{'-' * 60}")

"""
算法复杂度分析：
1. 图构建阶段：O(E)，E为边的数量
2. Tarjan算法：O(V + E)，V为节点数量
3. 结果处理：O(B)，B为桥边数量
总时间复杂度：O(V + E)，适合处理大型神经网络
"""