import joblib
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
import seaborn as sns
from sklearn.metrics import mean_squared_error

#读取excel文件，并转为csv
df1 = pd.read_excel('916.xlsx')
df=pd.read_csv('experiment_first16_100.csv')

#载入模型
model1 = joblib.load('models_gbr/gbr_model_99.pkl')
model2= joblib.load('models_gbr_2/gbr_model2_70.pkl')
#输入数据
X=df.drop('strength', axis=1)
Y_exp=df['strength']
Y1_pre=model1.predict(X)
Y2_pre=model2.predict(X)



# 计算误差
errors_model1 = Y_exp - Y1_pre
errors_model2 = Y_exp - Y2_pre

# 画两次的误差 KDE 图在同一张图上
plt.figure(figsize=(8, 6))
sns.kdeplot(errors_model1, shade=True, color='blue', label='Model 1 Errors')
sns.kdeplot(errors_model2, shade=True, color='green', label='Model 2 Errors')
plt.title('Error Distribution for Two Models (KDE)')
plt.xlabel('Error')
plt.ylabel('Density')
plt.legend()
plt.show()

#计算均方误差
mse_model1 = mean_squared_error(Y_exp, Y1_pre)
mse_model2 = mean_squared_error(Y_exp, Y2_pre)
print("Model 1 MSE:", mse_model1)
print("Model 2 MSE:", mse_model2)
print(1)
