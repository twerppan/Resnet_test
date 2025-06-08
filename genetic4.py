import os
import time
import cv2
import math

import joblib
import numpy as np
import random

import pandas as pd
from scipy.linalg import bandwidth

def compute_2d_histogram(image, bins=256):
    # 将图像转换为灰度图（如果是彩色图像）
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 计算二维直方图（灰度级分布）
    hist = cv2.calcHist([image], [0], None, [bins], [0, 256])
    hist = hist / hist.sum()  # 归一化
    return hist


# 计算二维熵
def compute_entropy(image, bins=256):
    hist = compute_2d_histogram(image, bins)

    # 使用熵公式 H = -sum(p * log2(p))，排除零值
    entropy = -np.sum(hist * np.log2(hist + np.finfo(float).eps))  # 加上eps避免log(0)
    return entropy

def read_frame_and_extract_features(l0,slot):
    frame_features = []
    features = []
    # 处理第一帧的特征
    prev_gray = cv2.imread(f'../data/UA-DETRAC/MVI_63552/img{l0*slot+1:05d}.jpg', cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(prev_gray, 100, 200)
    edge_count = np.count_nonzero(edges)
    entropy = compute_entropy(prev_gray)
    std = float(100)  # 设置为最大值
    frame_features.append([0, edge_count, entropy, std])  # 添加帧号为0
    features.append({"edge_count": edge_count, "entropy": entropy, "std": std})
    frame_number = 1  # 从第二帧开始
    for i in range(l0*slot+2,l0*(slot+1)+1):
        # 读取两帧图像
        path2 = f'../data/UA-DETRAC/MVI_63552/img{i:05d}.jpg'
        if os.path.exists(path2):
            next_gray = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
            lk_params = dict(winSize=(15, 15), maxLevel=2,
                             criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

            # 检测角点 (Shi-Tomasi角点检测) - 添加maxCorners参数
            max_corners = 10000  # 最大角点数
            prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=max_corners, qualityLevel=0.3, minDistance=7,
                                               blockSize=7)
            # 计算稀疏光流
            if prev_pts is not None and len(prev_pts) > 0:
                next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, prev_pts, None, **lk_params)
                good_prev_pts = prev_pts[status == 1]
                good_next_pts = next_pts[status == 1]
                flow = good_next_pts - good_prev_pts
                magnitudes = np.linalg.norm(flow, axis=1)
                std = np.std(magnitudes)
            else:
                std = float('inf')
        else:
            # 转为灰度图
            std = float('inf')
        edges1 = cv2.Canny(prev_gray, 100, 200)  # 前一帧图像中的边缘像素值
        edge_count = np.count_nonzero(edges1)
        entropy = compute_entropy(prev_gray)  # 熵
        features.append({"edge_count":edge_count, "entropy":entropy, "std":std})
        frame_features.append([frame_number, edge_count, entropy, std])
        prev_gray = next_gray
        frame_number += 1
    X_new_raw = pd.DataFrame(features, columns=['edge_count', 'entropy', 'std'])

    # X = []
    # for frame in frame_features:
    #     edge_count = frame[1]  # 第二个元素是 edge_count
    #     entropy = frame[2]  # 第三个元素是 entropy
    #     std = frame[3]  # 第四个元素是 std
    #     X.append([edge_count, entropy, std])
    #
    # # 将 X 转换为 numpy 数组，以便输入到模型中
    # X_new_raw = np.array(X)  # X 现在是一个 (30, 3) 的特征矩阵
    loaded_svr = joblib.load('../Optical Flow/F1_same_perspective_model.pkl')
    loaded_scaler = joblib.load('../Optical Flow/F1_same_perspective-scaler.pkl')

    X_new = loaded_scaler.transform(X_new_raw)  # 对新数据进行标准化
    y_new_pred = loaded_svr.predict(X_new)
    for i, features in enumerate(frame_features):

        features.append(y_new_pred[i])
    sorted_features = sorted(frame_features, key=lambda x: x[4], reverse=False)#False倒序

    return sorted_features


def Frame_Set(proportion_i_edge,proportion_i_cloud,sorted_features,f0):
    if proportion_i_cloud <= 0 or proportion_i_edge <= 0:
        return None
    frame_Set = [0] * f0
    for i in range(round(f0 * proportion_i_cloud)):

        frame_Set[sorted_features[i][0]] = 1
    for j in range(round(f0 * proportion_i_cloud),round(f0 * (proportion_i_cloud + proportion_i_edge))):
        frame_Set[sorted_features[j][0]] = 2

    return frame_Set

def Average_Accuracy(frame_Set,func_edge,func_cloud,f0):
    Accuracy_list = [0]*f0
    for i in range(f0):
        if frame_Set[i] == 1:
            Accuracy_list[i] = func_cloud
        elif frame_Set[i] == 2:
            Accuracy_list[i] = func_edge
        else:
            Accuracy_list[i] = Accuracy_list[i-1]*math.exp(-0.05)        #暂取衰减因子λ=3
    return sum(Accuracy_list) / f0

def calculate_average_bandwidth(file_path):
    # 初始化一个长度为50的数组来存储每秒的平均带宽
    time = [[] for _ in range(100)]
    average_bandwidth = [0 for _ in range(100)]
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for i in range(len(lines)):
            timestamp, bandwidth = lines[i].strip().split()
            timestamp = float(timestamp)
            bandwidth = float(bandwidth)
            if int(timestamp) < 100:
                time[int(timestamp)].append([timestamp,bandwidth])
    pre_bandwidth = 0
    for i in range(100):
        tmp_time = i
        tmp_bandwidth = pre_bandwidth
        ave_ban = 0
        if len(time[i]) == 0:
            average_bandwidth[i] = pre_bandwidth
        else:
            for j in range(len(time[i])):
                ave_ban += (time[i][j][0]-tmp_time)*tmp_bandwidth
                tmp_bandwidth = time[i][j][1]
                tmp_time = time[i][j][0]
            ave_ban += (i+1 - tmp_time) * tmp_bandwidth
            pre_bandwidth = tmp_bandwidth
            average_bandwidth[i] = ave_ban

    return average_bandwidth


# 初始化种群，随机生成pop_size个符合条件的种群
def initialize_population(pop_size):
    population = []
    for _ in range(pop_size):
        Proportion_i_Edge = random.uniform(0,1)
        Proportion_i_Cloud = random.uniform(0,1)
        # 保证Proportion_i_Edge + Proportion_i_Cloud < 1
        population.append([Proportion_i_Edge, Proportion_i_Cloud])

    return population


# 适应度函数
def fitness(Proportion_i_Edge, Proportion_i_Cloud):
    return -f(Proportion_i_Edge, Proportion_i_Cloud)  # 目标是最小化f，因此取负值来最大化适应度


# 选择父代：轮盘赌选择
def select_parents(population):
    fitness_values = [fitness(ind[0], ind[1]) for ind in population]    #对每个种群进行适应度计算
    total_fitness = sum(fitness_values)
    probabilities = [f / total_fitness for f in fitness_values]         #如果函数值越小，适应度越大，概率=适应度除以总值，越可能被选择
    parents = random.choices(population, probabilities, k=2)            #根据概率随机选择两个母体变异
    return parents


# 交叉操作
def crossover(parent1, parent2,alpha):
    # alpha = random.random()  # 随机一个交叉率，alpha是[0, 1]随机
    Proportion_i_Edge1, Proportion_i_Cloud1 = parent1
    Proportion_i_Edge2, Proportion_i_Cloud2 = parent2
    # 交叉操作：交叉Proportion_i_Edge和Proportion_i_Cloud
    child1 = [alpha * Proportion_i_Edge1 + (1 - alpha) * Proportion_i_Edge2,
              alpha * Proportion_i_Cloud1 + (1 - alpha) * Proportion_i_Cloud2]
    child2 = [(1 - alpha) * Proportion_i_Edge1 + alpha * Proportion_i_Edge2,
              (1 - alpha) * Proportion_i_Cloud1 + alpha * Proportion_i_Cloud2]
    # 修正交叉后结果，确保满足约束条件
    child1 = repair(child1)
    child2 = repair(child2)

    return child1, child2


# 修正函数，确保交叉或者变异后的决策变量满足约束
def repair(individual):
    Proportion_i_Edge, Proportion_i_Cloud = individual
    if Proportion_i_Edge <= 0: Proportion_i_Edge = random.random() * 0.5  # 比例小于0，随机生成一个小于0.5的值
    if Proportion_i_Cloud <= 0: Proportion_i_Cloud = random.random() * 0.5
    if Proportion_i_Edge + Proportion_i_Cloud >= 1:  # 保证Proportion_i_Edge + Proportion_i_Cloud < 1
        max_val = 1 - Proportion_i_Edge
        Proportion_i_Cloud = random.uniform(0, max_val)
    return [Proportion_i_Edge, Proportion_i_Cloud]



def mutate(individual, mutation_rate):  # 变异操作，有0.1的概率函数值会重新定义，防止陷入局部最优解
    Proportion_i_Edge, Proportion_i_Cloud = individual
    if random.random() < mutation_rate:
        Proportion_i_Edge = random.uniform(0, 1)
    if random.random() < mutation_rate:
        Proportion_i_Cloud = random.uniform(0, 1)
    # 修正变异后的结果，确保满足约束条件
    return repair([Proportion_i_Edge, Proportion_i_Cloud])


# 遗传算法主函数
def genetic_algorithm(pop_size, generations, mutation_rate,alpha):
    population = initialize_population(pop_size)    #初始化
    best_overall = min(population, key=lambda ind: f(ind[0], ind[1]))  # 初始化最优个体
    # with open(f"New/para/generations3.txt", "w", encoding="utf-8") as file1:
    for gen in range(generations):#迭代
        new_population = []
        for _ in range(pop_size // 2):  # 每次选择两个父代，选择的次数等于种群的一半
            parent1, parent2 = select_parents(population)
            child1, child2 = crossover(parent1, parent2,alpha)    #交叉生生成子代
            new_population.extend([child1, child2])         #得到新的子代种群

        # 对新一代进行变异
        population = [mutate(ind,mutation_rate) for ind in new_population]    #对新种群进行变异

         # 输出当前代最好的个体
        best_individual = min(population, key=lambda ind: f(ind[0], ind[1]))
        #
        # print(
        #     f"Generation {gen}: Best Proportion_i_Edge={best_individual[0]}, Best Proportion_i_Cloud={best_individual[1]}, f(Proportion_i_Edge, Proportion_i_Cloud)={f(best_individual[0], best_individual[1])}")
        # file1.write(
        #     f" Proportion_i_Edge={best_individual[0]},Proportion_i_Cloud={best_individual[1]},Best Value={f(best_individual[0], best_individual[1])}\n")
        if f(best_individual[0], best_individual[1]) < f(best_overall[0], best_overall[1]):
            best_overall = best_individual
    # 返回最后一代中最优解
    best_individual = min(population, key=lambda ind: f(ind[0], ind[1]))
    if f(best_individual[0], best_individual[1]) < f(best_overall[0], best_overall[1]):
        best_overall = best_individual
    return best_overall
def f(Proportion_i_Edge, Proportion_i_Cloud):
    # 添加约束条件，确保在合理范围内
    if Proportion_i_Edge + Proportion_i_Cloud >= 1:
        return float('inf')
    # f0 = 30  # 视频帧率
    # l_t = 1  # 每slot长度
    # img_size = 640 # 图片输入大小，分辨率为640*640
    # Color_depth = 24  # 颜色深度，用于计算每帧数据量大小
    # # 每片段数据量
    # Amount_of_data = img_size ** 2 * (Color_depth / 8) * f0 * l_t  # 每个片段的数据量总和36864000
    #
    # # 每片段送来的数据量积压和处理完的数据量积压
    # TimePerBit_of_Edge = 1.0975e-8  # YOLO-tiny推理速度提升3-5倍，暂取4倍每秒处理99943803bit
    # TimePerBit_of_Cloud = 4.39e-8  # YOLOv7推理164张图片在13s左右，每张图片大小平均220KB。每秒22735950bit
    # # UploadedData = (Proportion_i_Edge + Proportion_i_Cloud) * Amount_of_data
    # ProcessedData = l_t * (1 / TimePerBit_of_Edge + 1 / TimePerBit_of_Cloud)/5  # 2273950假设边缘节点是与5个本地设备连接
    global ProcessedData
    # 积压队列演变
    global Query_Load_Front
    # global Query_Load_Back
    # Query_Load_Back = max(Query_Load_Front + UploadedData - ProcessedData/5, 0)

    # 每片段处理的成本和预计的成本
    # CostPerBit_of_Edge = 2.78e-3  # 边缘假设使用CPU推理，功耗100瓦特，功耗表示为100*164*4/(13*220*1024*8)=0.0027885J/bit
    # CostPerBit_of_Cloud = 2.44e-3  # 使用NVIDIA 3090 GPU本功耗为350瓦特，那么功耗可以表示为350*164/(13*220*1024*8)=0.0024499J/bit
    # Cost_of_Process = Amount_of_data * (
    #         CostPerBit_of_Edge * Proportion_i_Edge + CostPerBit_of_Cloud * Proportion_i_Cloud)
    # Cost_max = 61000  # 每片段成本阈值
    # 成本队列演变
    # global Query_Cost_Front
    # global Query_Cost_Back
    # Query_Cost_Back = max(Query_Cost_Front + Cost_of_Process - Cost_max,0)

    # 每片段处理的延迟和定义的延迟上界
    global Bandwidth
    # Bandwidth = 2.7e7  # 上行带宽测速大概260-280Mbps
    # Delay_of_Tran = Amount_of_data * Proportion_i_Cloud / (1e6*Bandwidth)
    global Delay_max
    # Delay_max = 1 # 每片段延迟阈值，设置此值是考虑每帧处理时间*2左右
    global Query_Delay_Front
    # global Query_Delay_Back
    # Query_Delay_Back = max(Query_Delay_Front + Delay_of_Tran - Delay_max,0)

    # Edge是模拟YOLO-tiny拟合的函数，c3越小，函数的变化趋势越大，展示tiny的波动#Cloud通过YOLOv7拟合的F1_Frame的函数
    AccuracyOfFrame_Edge = 0.355
    AccuracyOfFrame_Cloud = 0.407
    global sorted_features
    # 定义帧去向集合
    Frame_list = Frame_Set(Proportion_i_Edge, Proportion_i_Cloud, sorted_features, f0)
    # 平均精度求解
    Accuracy = Average_Accuracy(Frame_list, AccuracyOfFrame_Edge, AccuracyOfFrame_Cloud, f0)
    global V   # V = 1e8  # 权衡函数
    return 1 * (((Amount_of_data/ProcessedData) ** 2) * (Proportion_i_Cloud + Proportion_i_Edge) ** 2 + 2 * (Amount_of_data/ProcessedData) * (Query_Load_Front - 1) * (Proportion_i_Cloud + Proportion_i_Edge)) \
            + 1* (((Amount_of_data/Delay_max) * (Proportion_i_Cloud/Bandwidth + Proportion_i_Edge/(3*Bandwidth))) ** 2 + 2 * ((Amount_of_data/Delay_max) * (Proportion_i_Cloud/Bandwidth + Proportion_i_Edge/(3*Bandwidth))) * (Query_Delay_Front - 1)) \
            + V / (Accuracy+0.001)
    # return Query_Load_Front + (Proportion_i_Cloud + Proportion_i_Edge)*Amount_of_data/ProcessedData -1 \
    #         + Query_Delay_Front + (Proportion_i_Cloud/Bandwidth + Proportion_i_Edge/(3*Bandwidth))/Delay_max -1 \
    #         + V / (Accuracy+0.001)
if __name__ == '__main__':
    # with open(f"New/V.txt", "w", encoding="utf-8") as file:
    #     for j in range(50):
            bandwidth_list = calculate_average_bandwidth('Bandwidth/real_bandwidth2.txt')
            # 运行遗传算法
            # pop_size_list = [20,30,50,70,100]
            # generations_list = [50,100,150,200,300]
            # mutation_rate_list = [0.01,0.05,0.1,0.15,0.2]
            # alpha_list = [0.5,0.6,0.7,0.8,0.9]
            V_list = [0.4,0.6,0.8,1.2,1.4,1.6,1.8]
            list0 = []
            for i in range(7):
                start_time = time.time()
                Query_Load_Back = 0
                Query_Delay_Back = 0
                # list1 = [1,1.2,1.5,1.8,2,2.5,3,4]
                V=V_list[i]
                Delay_max = 1.5
                pop_size = 50
                generations = 50
                mutation_rate = 0.1
                alpha = 0.7
                f0 = 30  # 视频帧率
                l_t = 1  # 每slot长度
                img_size = 640 # 图片输入大小，分辨率为640*640
                Color_depth = 24  # 颜色深度，用于计算每帧数据量大小
                # 每片段数据量
                Amount_of_data = img_size ** 2 * (Color_depth / 8) * f0 * l_t  # 每个片段的数据量总和36864000
                # 每片段送来的数据量积压和处理完的数据量积压
                TimePerBit_of_Edge = 1.0975e-8  # YOLO-tiny推理速度提升3-5倍，暂取4倍每秒处理99943803bit
                TimePerBit_of_Cloud = 4.39e-8  # YOLOv7推理164张图片在13s左右，每张图片大小平均220KB。每秒22735950bit
                # UploadedData = (Proportion_i_Edge + Proportion_i_Cloud) * Amount_of_data
                ProcessedData = l_t * (1 / TimePerBit_of_Edge + 1 / TimePerBit_of_Cloud)/5  # 22779043假设边缘节点是与5个本地设备连接
                with open(f"New/V2/V{V}", "w", encoding="utf-8") as file:
                # load_query = []
                # delay_query = []
                # accuracy_query = []
                    for start in range(38):
                        sorted_features = read_frame_and_extract_features(30,start)
                        Query_Load_Front = Query_Load_Back
                        Query_Delay_Front = Query_Delay_Back
                        Bandwidth = bandwidth_list[start]*5e6
                        best_solution = genetic_algorithm(pop_size=pop_size, generations=generations, mutation_rate=mutation_rate, alpha=alpha)
                        # print(Query_Load_Front)
                        # print(Query_Delay_Front)
                        # print(best_solution)

                        Query_Load_Back = max(Query_Load_Front + (best_solution[0] + best_solution[1]) * (Amount_of_data/ProcessedData) - 1, 0)
                        Query_Delay_Back = max(Query_Delay_Front + Amount_of_data*(best_solution[1]/ Bandwidth + best_solution[0]/ (3*Bandwidth))/Delay_max - 1, 0)
                        Accuracy = Average_Accuracy(Frame_Set(best_solution[0], best_solution[1], sorted_features,30), 0.355, 0.407, 30)
                        # load_query.append(Query_Load_Back)
                        # delay_query.append(Query_Delay_Back)
                        # accuracy_query.append(Accuracy)
                        # print(Query_Load_Back)
                        # print(Query_Delay_Back)
                        # print(Accuracy)
                        # print("\n")
                        file.write(f"slot {start+1}:\n")
                        file.write(f" Proportion_i_Edge={best_solution[0]},Proportion_i_Cloud={best_solution[1]},Best Value={f(best_solution[0], best_solution[1])}\n")
                        file.write(f" Query_Load_Back:{Query_Load_Back}\n")
                        file.write(f" Query_Delay_Back:{Query_Delay_Back}\n")
                        file.write(f" Accuracy:{Accuracy}\n\n")
                # file.write(f"{sum(load_query) / 38},{sum(delay_query) / 38},{sum(accuracy_query) / 38}\n")
                end_time = time.time()
                elapsed_time = end_time - start_time
                list0.append(elapsed_time)
            print(list0)
