from bayes_opt import BayesianOptimization
import numpy as np
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
def expected_improvement(mu, sigma, mu_best, xi=0.01):
    """
    计算期望改进（EI）。

    参数:
    - mu: 预测的均值 (ndarray)
    - sigma: 预测的标准差 (ndarray)
    - mu_best: 当前已知最优值 (float)
    - xi: 探索利用的平衡参数（默认为0.01，小值意味着更多利用）

    返回:
    - EI: 每个点的期望改进 (ndarray)
    """
    sigma = np.maximum(sigma*100, 1e-9)  # 避免除以0
    Z = (mu - mu_best - xi) / sigma  # 探索利用的权衡项 xi
    ei = (mu - mu_best - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
    return ei

df=pd.read_csv('combinations_data_2_gbr.csv')

#找到EGO实验中得最大值
df_100_experiment=pd.read_csv('experiment_first16_100.csv')
mu_best_ego=df_100_experiment['strength'].max()
ei_ego=[]
for i in range(len(df)):
    a=expected_improvement(df['mean_sum'][i],df['standard_deviation'][i],mu_best_ego)
    ei_ego.append(a)

#将ei_ego列添加到df中
df['ei_ego']=ei_ego

#按照 EI 值排序，并去除重复的 EI 值
df_unique_ei = df.drop_duplicates(subset=['ei_ego']).sort_values(by='ei_ego', ascending=False)
#选取前 16 个不同的 EI 值对应的行
df_top16 = df_unique_ei.head(16)
#将 df_top16 保存为 CSV 文件
df_top16.to_csv('first_top16_data.csv', index=False)

# 按照 EI 值排序
df_sorted = df.sort_values(by='ei_ego', ascending=False)
# 按 EI 值进行分组，并在每个 EI 值相同的组内随机选择一行
df_random = df_sorted.groupby('ei_ego').apply(lambda x: x.sample(n=1)).reset_index(drop=True)
# 如果选择的行数超过16，可以取前16行
df_top16_random = df_random.tail(16)

#保存df_top16_random为csv文件
df_top16_random.to_csv('random_top16_data_second.csv', index=False)

print(1)
