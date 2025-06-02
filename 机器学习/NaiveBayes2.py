import pandas as pd
import numpy as np
from collections import Counter
data = [# 训练数据
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
    ["浅白", "蜷缩", "模糊", "模糊", "平坦", "软粘", 0.343, 0.099, "否"],
    ["青绿", "蜷缩", "浊响", "稍糊", "凹陷", "硬滑", 0.639, 0.161, "否"],
    ["浅白", "稍蜷", "沉闷", "稍糊", "凹陷", "硬滑", 0.657, 0.198, "否"],
    ["乌黑", "稍蜷", "浊响", "清晰", "稍凹", "软粘", 0.360, 0.370, "否"],
    ["浅白", "蜷缩", "浊响", "模糊", "平坦", "硬滑", 0.593, 0.042, "否"],
    ["青绿", "蜷缩", "沉闷", "稍糊", "稍凹", "硬滑", 0.719, 0.103, "否"],]
columns = ["色泽", "根蒂", "敲声", "纹理", "脐部", "触感", "密度", "含糖率", "好瓜"]#列名
df = pd.DataFrame(data, columns=columns)#创建DataFrame
class NaiveBayesClassifier:#定义朴素贝叶斯分类器类
    def __init__(self):
        self.prior_probs = {}#存先验概率
        self.conditional_probs = {}#存条件概率

    def fit(self, X, y):
        self.classes = np.unique(y)
        # 应用拉普拉斯修正计算先验概率=样本数/总样本数+拉普拉斯修正
        self.prior_probs = {c: (np.sum(y == c) + 1) / (len(y) + len(self.classes)) for c in self.classes}
        self.conditional_probs = {}
        for c in self.classes:
            class_subset = X[y == c]
            self.conditional_probs[c] = {}
            for col in X.columns:#遍历每个特征列
                if X[col].dtype == 'O':
                    counts = Counter(class_subset[col])#该特征在类别中的出现次数
                    total = sum(counts.values())#总次数
                    num_unique = len(X[col].unique())#该特征所有可能的取值数目
                    self.conditional_probs[c][col] = {k: (v + 1) / (total + num_unique) for k, v in counts.items()}
                else:  #连续特征使用高斯分布
                    self.conditional_probs[c][col] = {#计算均值和标准差
                        "mean": class_subset[col].mean(),
                        "std": class_subset[col].std() + 1e-6  # 防止除零
                    }
    def predict_proba(self, sample):#定义预测方法
        probs = {}
        for c in self.classes:
            prob = self.prior_probs[c]#从先验概率开始计算
            for col in sample.index:
                value = sample[col]
                if col in self.conditional_probs[c]:#离散获取条件概率
                    if isinstance(value, str):
                        prob *= self.conditional_probs[c][col].get(value, 1e-6)  # 避免零概率
                    else:  #连续均值和标准差
                        mean = self.conditional_probs[c][col]["mean"]
                        std = self.conditional_probs[c][col]["std"]
                        prob *= (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-((value - mean) ** 2) / (2 * std ** 2))
            probs[c] = prob

        total_prob = sum(probs.values())#对概率进行归一化处理
        return {k: v / total_prob for k, v in probs.items()}
# 训练模型
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
nb_classifier = NaiveBayesClassifier()
nb_classifier.fit(X, y)#创建实例后训练模型
test_sample = pd.Series(["青绿", "蜷缩", "浊响", "清晰", "凹陷", "硬滑", 0.697, 0.460], index=X.columns)# 测试样本“测1”
probs = nb_classifier.predict_proba(test_sample)
print("测1 的分类概率:", probs)
print("测1的分类类别为:", max(probs, key=lambda k: probs[k]))