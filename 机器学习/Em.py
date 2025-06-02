import numpy as np
data = np.array([-67, -48, 6, 8, 14, 16, 23, 24, 28, 29, 41, 49, 56, 60, 75])
n_samples = len(data)
n_components = 2
# 初始化参数
pi = np.array([0.5, 0.5])  # 均匀初始化混合系数
mu = np.random.choice(data, size=n_components, replace=False)# 随机选择样本点作为均值
sigma2 = np.array([np.var(data)] * n_components)# 全局方差初始化
max_iter = 100  #设置最大迭代100次
tolerance = 1e-6  # 收敛阈值

for iter in range(max_iter):
    # E步：计算后验概率γ(z nk)
    gamma = np.zeros((n_samples, n_components))
    for k in range(n_components):# 计算每个分量的概率密度
        # 根据高斯分布概率密度公式
        exponent = -0.5 * (data - mu[k]) ** 2 / sigma2[k]
        density = np.exp(exponent) / np.sqrt(2 * np.pi * sigma2[k])
        gamma[:, k] = pi[k] * density
    gamma_sum = gamma.sum(axis=1, keepdims=True)    # 归一化
    gamma /= gamma_sum  # 每个样本的γ在两个分量上归一化
    #M步：更新参数
    N_k = gamma.sum(axis=0)#计算N_k = sum_n γ(z_nk)
    pi_new = N_k / n_samples#1更新混合系数 π_k
    mu_new = np.zeros(n_components)
    for k in range(n_components):#2更新均值 μ_k
        mu_new[k] = np.sum(gamma[:, k] * data) / N_k[k]
    sigma2_new = np.zeros(n_components)
    for k in range(n_components):#3更新方差 σ_k^2
        sigma2_new[k] = np.sum(gamma[:, k] * (data - mu_new[k]) ** 2) / N_k[k]
    delta = max(    # 检查是否收敛（参数变化小于阈值）
        np.abs(pi_new - pi).max(),
        np.abs(mu_new - mu).max(),
        np.abs(sigma2_new - sigma2).max()
    )
    if delta < tolerance:
        break
    pi = pi_new.copy()    # 更新参数
    mu = mu_new.copy()
    sigma2 = sigma2_new.copy()
print(f"迭代次数: {iter + 1}")
print(f"混合系数:π₁ = {pi[0]:.4f}, π₂ = {pi[1]:.4f}")
print(f"均值:μ₁ = {mu[0]:.1f}, μ₂ = {mu[1]:.1f}")
print(f"方差:σ₁² = {sigma2[0]:.2f}, σ₂² = {sigma2[1]:.2f}")