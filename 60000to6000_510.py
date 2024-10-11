import pandas as pd
df=pd.read_csv('../../High_entropy_alloys/xgboost/data_final.csv')
#按照温度为510的行提取
df_510=df[df['temperature']==510]
#将df_510保存为csv文件
df_510.to_csv('data_510.csv',index=False)
print(1)