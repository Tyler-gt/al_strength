#画一个数据的热力图
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('../../High_entropy_alloys/xgboost/data_final.csv')
df_100=df.sample(n=100)
#去掉掉一列
df_100=df_100.drop(['temperature'],axis=1)

#利用df_100绘制热力图
sns.heatmap(df_100.corr())
plt.show()
#保存热力图


plt.savefig('heatmap.png')
a=df_100.corr()
print(df_100.corr())


