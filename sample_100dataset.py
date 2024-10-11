import pandas as pd
import csv
import json
"""
#此一个迭代中获取100数据集
#获取原始样本数据
df=pd.read_csv('data_510.csv')
df_100=df.sample(n=100,replace=False)

#将df_100保存为csv文件
df_100.to_csv('data_100_experiment.csv',index=False)

n_samples_per_dataset=100  #每个数据集包含的样本数量
n_datasets=100   #数据集的数量


#创建有个列表来存储100个数据集
datasets=[]

for _ in range(n_datasets):
    sampled_df=df_100.sample(n=n_samples_per_dataset,replace=True)
    datasets.append(sampled_df)

#a=datasets[1]
#将列表嵌套的dataframe数据保存为字典
dataframes_as_dicts=[df.to_dict(orient='records') for df in datasets]
# 将列表保存为 JSON 文件
with open('datasets_100.json', 'w') as f:
    json.dump(dataframes_as_dicts, f, indent=4)

"""

#第二个迭代中获取数据集
df_116=pd.read_csv('experiment_first16_100.csv')

n_samples_per_dataset=116  #每个数据集包含的样本数量
n_datasets=100   #数据集的数量


#创建有个列表来存储100个数据集
datasets=[]

for _ in range(n_datasets):
    sampled_df=df_116.sample(n=n_samples_per_dataset,replace=True)
    datasets.append(sampled_df)

#a=datasets[1]
#将列表嵌套的dataframe数据保存为字典
dataframes_as_dicts=[df.to_dict(orient='records') for df in datasets]
# 将列表保存为 JSON 文件
with open('datasets_116.json', 'w') as f:
    json.dump(dataframes_as_dicts, f, indent=4)