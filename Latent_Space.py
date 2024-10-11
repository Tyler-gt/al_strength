#cu0.2-0.4
#Mn 0.1-0.8
#mg 0.8-1.4
#fe 0.1-0.6
#si 0.4-0.8
#zn 0.1-0.3
#zr 0.1-0.3
#temputer 480-550

import numpy as np
import pandas as pd
#构建成分空间
import itertools
import numpy as np
import pandas as pd
import joblib
# 定义各成分范围和温度范围
cu_range = np.arange(3.6, 4.2 + 0.1, 0.1)  # 每 0.1 一步
mn_range = np.arange(0.1, 0.5 + 0.1, 0.1)
mg_range = np.arange(1.0, 1.4 + 0.1, 0.1)
fe_range = np.arange(0.1, 0.4 + 0.1, 0.1)
si_range = np.arange(0.2, 0.5 + 0.1, 0.1)
zn_range = np.arange(0.0, 0.3 + 0.1, 0.1)
zr_range = np.arange(0.0, 0.3 + 0.1, 0.1)
#temp_range = np.arange(500, 550 + 10, 10)  # 温度每 10 度一步

# 生成所有成分和温度的排列组合
combinations = list(itertools.product(cu_range, mn_range, mg_range, fe_range, si_range, zn_range, zr_range))

# 将排列组合转化为DataFrame
columns = ['Cu', 'Mn', 'Mg', 'Fe', 'Si', 'Zn', 'Zr']
df_combinations = pd.DataFrame(combinations, columns=columns)

#创建列表存放100次的预测值
df_list=[]

for i in range(100):
    model = joblib.load(f'models_gbr_2/gbr_model2_{i}.pkl')
    X_pre=df_combinations
    Y_pre=model.predict(X_pre)
    df_list.append(Y_pre)

#得到模型对于潜在空间每组成分的预测值的均值
sum_list=[]
a=0
for i in range(len(df_list[1])):
    for j,row in enumerate(df_list):
        a=a+row[i]

    sum_list.append(a)
    a=0
mean_sum_list=[x/100 for x in sum_list]

#计算模型对合金成分预测的误差与不确定性（标准方差）
b=0
standard_deviation=[]

for i in range(len(df_list[1])):
    for j ,row in enumerate(df_list):
        b=b=+(row[i]-mean_sum_list[i])**2
    standard_deviation.append(b)
    b=0
mean_standard_list=[(x/100)**0.5 for x in standard_deviation]

#将硬度均值与均方误差添加到合金成分组成的dataframe中
df_combinations['mean_sum']=mean_sum_list
df_combinations['standard_deviation']=mean_standard_list

#将df_combinations保存为csv文件
df_combinations.to_csv('combinations_data_2_gbr.csv', index=False)
#junzhi=[sum+item for row in df_list for item in row]
#a=df_list[1][1]
#flat_list = list(itertools.chain(*df_list))
# 查看结果
print(df_combinations)
