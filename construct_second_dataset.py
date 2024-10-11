#在第一次数据集上添加模拟实验的第二次数据集


#读取第一次实验数据集
import pandas as pd
import numpy as np

#读取第一次实验数据集
df_100 = pd.read_csv('data_100_experiment.csv')
#去除temperature列
df_100 = df_100.drop(['temperature'], axis=1)

#读取16个实验数据集
df_16 = pd.read_csv('experiment_first16.csv')
#将df_100与df_16合并
df_116 = pd.concat([df_100, df_16], axis=0)

#将df_116存为csv文件
df_116.to_csv('experiment_first16_100.csv', index=False)
print(1)
