import numpy as np
from collections import defaultdict
training_data = [
    ["青绿", "蜷缩", "浊响", "清晰", "凹陷", "硬滑", 0.697, 0.460, "是"],
    ["乌黑", "蜷缩", "沉闷", "清晰", "凹陷", "硬滑", 0.774, 0.376, "是"],
    ["乌黑", "蜷缩", "浊响", "清晰", "凹陷", "硬滑", 0.634, 0.264, "是"],
    ["青绿", "蜷缩", "沉闷", "清晰", "凹陷", "硬滑", 0.608, 0.318, "是"],
    ["浅白", "蜷缩", "浊响", "清晰", "凹陷", "硬滑", 0.556, 0.215, "是"],
    ["青绿", "稍蜷", "浊响", "清晰", "稍凹", "软粘", 0.403, 0.237, "是"],
    ["乌黑", "稍蜷", "浊响", "稍糊", "稍凹", "软粘", 0.481, 0.149, "是"],
    ["乌黑", "稍蜷", "浊响", "清晰", "稍凹", "硬滑", 0.437, 0.211, "是"],
    ["乌黑", "稍蜷", "沉闷", "稍糊", "稍凹", "硬滑", 0.666, 0.091, "否"],
    ["青绿", "硬挺", "清脆", "清晰", "平坦", "软粘", 0.243, 0.267, "否"],
    ["浅白", "硬挺", "清脆", "模糊", "平坦", "硬滑", 0.245, 0.057, "否"],
    ["浅白", "蜷缩", "浊响", "模糊", "平坦", "软粘", 0.343, 0.099, "否"],
    ["青绿", "稍蜷", "浊响", "稍糊", "凹陷", "硬滑", 0.639, 0.161, "否"],
    ["浅白", "稍蜷", "沉闷", "稍糊", "凹陷", "硬滑", 0.657, 0.198, "否"],
    ["乌黑", "稍蜷", "浊响", "清晰", "稍凹", "软粘", 0.360, 0.370, "否"],
    ["浅白", "蜷缩", "浊响", "模糊", "平坦", "硬滑", 0.593, 0.042, "否"],
    ["青绿", "蜷缩", "沉闷", "稍糊", "稍凹", "硬滑", 0.719, 0.103, "否"]
]       #定义训练集
# 测1测试样本
test_1 = ["青绿", "蜷缩", "浊响", "清晰", "凹陷", "硬滑", 0.697, 0.460]
class NaiveBayesClassifier:#定义朴素贝叶斯分类器类
    def __init__(self):
        self.class_priors = {}  #类别先验概率
        self.feature_probs = defaultdict(dict)  #特征条件概率
        self.continuous_stats = defaultdict(lambda: defaultdict(list))  #连续特征统计信息
    def train(self, data):
        X = [row[:-1] for row in data]    #特征
        y = [row[-1] for row in data]  #标签
        class_counts = defaultdict(int) #每个类别的样本数
        for label in y:
            class_counts[label] += 1
        total_samples = len(y)  #总样本数
        for label, count in class_counts.items():# 类别先验概率=样本数/总样本数+拉普拉斯修正
            self.class_priors[label] = (count + 1) / (total_samples + 2)

        num_features = len(X[0])
        for feature_idx in range(num_features):
            if feature_idx in [6, 7]:  # 密度和含糖率是连续特征
                for label in class_counts.keys():
                    values = []
                    for i in range(len(X)):
                        if y[i] == label:
                            values.append(X[i][feature_idx])
                    mean = np.mean(values)# 计算均值
                    std = np.std(values)# 计算标准差
                    self.continuous_stats[label][feature_idx] = (mean, std)
            else: # 对其余的离散特征计算条件概率
                feature_values = set(row[feature_idx] for row in X)
                for label in class_counts.keys():
                    # 拉普拉斯修正后的分母为该类别样本数+特征可能取值数
                    denominator = class_counts[label] + len(feature_values)
                    value_counts = defaultdict(int)
                    for i in range(len(X)):
                        if y[i] == label:
                            value_counts[X[i][feature_idx]] += 1
                    # 存储每个特征值的条件概率
                    for value in feature_values:# 条件概率分子=特征值在该类别下的出现次数+1
                        self.feature_probs[(feature_idx, label)][value] = (value_counts[value] + 1) / denominator
    def predict(self, sample):
        class_scores = {}#存储每个类别的得分
        class_posteriors = {}  # 存储每个类别的后验概率
        for label, prior in self.class_priors.items():
            score = np.log(prior)  # 使用对数概率防止下溢
            for feature_idx in range(len(sample)):
                value = sample[feature_idx]
                if feature_idx in [6, 7]:  # 连续特征，密度和含糖率
                    mean, std = self.continuous_stats[label][feature_idx]
                    # 使用高斯分布计算该特征值的概率密度函数值
                    exponent = np.exp(-((value - mean) ** 2) / (2 * std ** 2))
                    prob = exponent / (np.sqrt(2 * np.pi) * std)
                    score += np.log(prob)
                else:  # 离散特征
                    prob = self.feature_probs[(feature_idx, label)].get(value, 1e-10)  # 避免除零
                    score += np.log(prob)
            class_scores[label] = score
            class_posteriors[label] = np.exp(score)  # 计算后验概率（对数的指数）

        # 返回得分最高的类别（即后验概率最大）
        print("\n类别的后验概率：")
        for label, posterior in class_posteriors.items():
            print(f"{label}: {posterior:.4f}")
        return max(class_scores, key=class_scores.get)

classifier = NaiveBayesClassifier()# 创建分类器并训练
classifier.train(training_data)
prediction = classifier.predict(test_1)# 对测试样本进行预测
# 输出结果
print(f"测1样本被分类为：{prediction}")