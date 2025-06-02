from collections import defaultdict, deque


def find_critical_edges_dag(edges, source, sink):
    # 使用 defaultdict 初始化邻接表和逆邻接表，处理缺失键的情况
    adj = defaultdict(list)  # 存储节点的子节点（正向图）
    reverse_adj = defaultdict(list)  # 存储节点的父节点（反向图）
    nodes = set()  # 收集所有节点

    # 遍历所有边，构建邻接表和逆邻接表，并记录所有节点
    for u, v in edges:
        adj[u].append(v)  # 添加正向边 u→v
        reverse_adj[v].append(u)  # 添加反向边 v←u
        nodes.update([u, v])  # 将节点加入集合

    nodes = list(nodes)  # 转换为列表以便后续处理

    # --- 拓扑排序（Kahn算法）---
    in_degree = defaultdict(int)  # 记录每个节点的入度

    # 遍历所有边，计算各节点的入度
    for u in adj:
        for v in adj[u]:
            in_degree[v] += 1  # 节点v的入度+1

    # 初始化队列：仅包含源节点（确保拓扑排序从源开始）
    queue = deque([node for node in nodes if in_degree.get(node, 0) == 0 and node == source])
    topo_order = []  # 存储拓扑排序结果

    # 执行拓扑排序
    while queue:
        u = queue.popleft()  # 取出入度为0的节点
        topo_order.append(u)  # 加入拓扑序列

        # 遍历该节点的所有子节点，更新入度
        for v in adj[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:  # 若子节点入度变为0
                queue.append(v)  # 加入队列

    # --- 计算从源到各节点的路径数 ---
    src_paths = defaultdict(int)  # 存储源到各节点的路径数
    src_paths[source] = 1  # 源到自身的路径数为1

    # 按拓扑顺序动态规划计算路径数
    for u in topo_order:
        for v in adj[u]:
            src_paths[v] += src_paths[u]  # 累加父节点的路径数

    # --- 计算各节点到汇的路径数 ---
    reverse_topo_order = reversed(topo_order)  # 反转拓扑序用于反向计算

    dest_paths = defaultdict(int)  # 存储各节点到汇的路径数
    dest_paths[sink] = 1  # 汇到自身的路径数为1

    # 按逆拓扑顺序动态规划计算路径数
    for v in reverse_topo_order:
        for u in reverse_adj[v]:  # 遍历所有父节点
            dest_paths[u] += dest_paths[v]  # 累加子节点的路径数

    # --- 确定关键割边 ---
    total = src_paths.get(sink, 0)  # 总路径数（源到汇）
    critical_edges = []  # 存储关键割边

    # 遍历所有边，检查是否为关键割边
    for u, v in edges:
        # 关键判定条件：src_paths[u] * dest_paths[v] == total
        if src_paths[u] * dest_paths[v] == total:
            critical_edges.append((u, v))

    return critical_edges


# 示例测试
edges = [
    (0, 1),
    (1, 2), (1, 3),
    (2, 4), (3, 5),
    (4, 6), (5, 6),
    (6, 7),
    (7, 8), (7, 10),
    (8, 9), (9, 10),
    (10, 11)
]
source = 0
sink = 11

print(find_critical_edges_dag(edges, source, sink))  # 输出：[(0, 1), (6, 7), (10, 11)]