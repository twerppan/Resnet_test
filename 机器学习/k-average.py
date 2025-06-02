import random
import math
import copy
# 西瓜数据集4.0[密度, 含糖率]
data = [
    [0.697, 0.460], [0.774, 0.376], [0.634, 0.264], [0.608, 0.318], [0.556, 0.215],
    [0.403, 0.237], [0.481, 0.149], [0.437, 0.211], [0.666, 0.091], [0.243, 0.267],
    [0.245, 0.057], [0.343, 0.099], [0.639, 0.161], [0.657, 0.198], [0.360, 0.370],
    [0.593, 0.042], [0.719, 0.103], [0.359, 0.188], [0.339, 0.241], [0.282, 0.257],
    [0.748, 0.232], [0.714, 0.346], [0.483, 0.312], [0.478, 0.437], [0.525, 0.369],
    [0.751, 0.489], [0.532, 0.472], [0.473, 0.376], [0.725, 0.445], [0.446, 0.459]
]
#欧几里得距离函数计算两个二维点之间的距离
def distance(p1, p2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

# K-Means 聚类主函数，data表示输入数据k是聚类个数，init_centers是初始中心点列表
def kmeans(data, k, init_centers):
    # 拷贝初始中心点，避免原始数据被修改
    centers = copy.deepcopy(init_centers)
    # 迭代直到中心点不再变化（或变化很小）
    while True:
        # 初始化空的簇列表，每个簇对应一个中心
        clusters = [[] for _ in range(k)]
        # 将每个数据点分配到最近的中心点所属的簇
        for point in data:
            dists = [distance(point, c) for c in centers]  # 计算到每个中心的距离
            cluster_index = dists.index(min(dists))       # 找到最近的中心点索引
            clusters[cluster_index].append(point)         # 分配到对应簇中
        # 计算每个簇的新中心点（即所有点的平均值）
        new_centers = []
        for cluster in clusters:
            if cluster:
                # 对每个簇按维度求平均，得到新中心
                new_centers.append([sum(x) / len(x) for x in zip(*cluster)])
            else:
                # 若某个簇没有样本，避免除0错误，暂设为(0,0)
                new_centers.append([0, 0])
        # 如果新旧中心点变化很小（<1e-6），认为收敛，退出循环
        if all(distance(c1, c2) < 1e-6 for c1, c2 in zip(centers, new_centers)):
            break
        # 否则更新中心点继续下一轮
        centers = new_centers
    return clusters, centers
#尝试的三种 k 值，三组不同的初始中心点
k_values = [2, 3, 4]
init_centers_list = [ # 组1平均分布三个点，组2包含边界点与中心点，组3包含低含糖点
    [[0.697, 0.460], [0.243, 0.267], [0.525, 0.369]],
    [[0.282, 0.257], [0.719, 0.103], [0.446, 0.459]],
    [[0.245, 0.057], [0.478, 0.437], [0.639, 0.161]]
]
for k in k_values:
    print(f"K = {k}")
    for i, init_centers in enumerate(init_centers_list):
        print(f"  初始中心组 {i + 1}: {init_centers[:k]}")
        clusters, centers = kmeans(data, k, init_centers[:k])
        print(f"  最终中心: {centers}")
