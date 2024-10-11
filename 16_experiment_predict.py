import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

"""
###第一次预测与实验的的比对图
#读取csv文件
df_top16_random=pd.read_csv('random_top16_data_first.csv')
df_top16_random['experiment']=[381.92,393.46,386.3,388.62,364.03,374.59,394.85,381.88,386.07,375.81,381.83,387.7,385.05,372.04,379.49,376.4]
df_temp=df_top16_random[['Cu','Mn','Mg','Fe','Si','Zn','Zr','experiment']]

#替换掉experiment表头为strength
df_temp.rename(columns={'experiment':'strength'},inplace=True)
#将df_temp中的存为csv文件
df_temp.to_csv('experiment_first16.csv',index=False)

#取df_top16_random的两列数据experiment与mean_sum做y与x画在平面上.
plt.figure(figsize=(10, 5))
plt.scatter(df_top16_random['mean_sum'],df_top16_random['experiment'], marker='s', color='red')
# 获取当前图的 x 和 y 轴的范围
x_min, x_max = plt.xlim()
y_min, y_max = plt.ylim()

# 计算对角线的起点和终点，使其适应 x 和 y 轴的范围
line_min = max(x_min, y_min)
line_max = min(x_max, y_max)

# 绘制对角线（45度），从 (line_min, line_min) 到 (line_max, line_max)
plt.plot([line_min, line_max], [line_min, line_max], linestyle='--', color='red', label='Diagonal Line')

# 添加图例和显示
plt.legend()

plt.title('Experiment vs. Mean Sum')
plt.ylabel('Experiment')
plt.xlabel('Mean Sum')
#在该图中生成一条对角线，要符合下，与y的相对大小关系
#保存图片
plt.savefig('Experiment_vs_Mean_Sum_first.png')

plt.show()
"""


#第二次迭代实验与预测的比对图
df_top16_random=pd.read_csv('random_top16_data_second.csv')
df_top16_random['experiment']=[379.08,388.15,404.22,402.97,369.18,381.21,381.78,382.3,377.1,385.36,382.46,383.68,388.53,364.33,400.97,379.38]
df_temp=df_top16_random[['Cu','Mn','Mg','Fe','Si','Zn','Zr','experiment']]

#替换掉experiment表头为strength(便于给后续与116组的数据集合并)
df_temp.rename(columns={'experiment':'strength'},inplace=True)
#将df_temp中的存为csv文件
df_temp.to_csv('experiment_second16.csv',index=False)

#取df_top16_random的两列数据experiment与mean_sum做y与x画在平面上.
plt.figure(figsize=(10, 5))
plt.scatter(df_top16_random['mean_sum'],df_top16_random['experiment'], marker='s', color='red')

# 获取当前图的 x 和 y 轴的范围
x_min, x_max = plt.xlim()
y_min, y_max = plt.ylim()

# 计算对角线的起点和终点，使其适应 x 和 y 轴的范围
line_min = max(x_min, y_min)
line_max = min(x_max, y_max)

# 绘制对角线（45度），从 (line_min, line_min) 到 (line_max, line_max)
plt.plot([line_min, line_max], [line_min, line_max], linestyle='--', color='blue', label='Diagonal Line')

# 添加图例和显示
plt.legend()

plt.title('Experiment vs. Mean Sum')
plt.ylabel('Experiment')
plt.xlabel('Mean Sum')
#在该图中生成一条对角线，要符合下，与y的相对大小关系
#保存图片
plt.savefig('Experiment_vs_Mean_Sum_second.png')

plt.show()
