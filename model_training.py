import torch
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import json
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import joblib
import numpy as np
with open('datasets_100.json', 'r') as f:
    loaded_data=json.load(f)

#将字典列表转化为dataframe列表
datasets_list_dataframe=[pd.DataFrame(dict) for dict in loaded_data]


#创建models列表来保存模型
models_svr_rbf = []
models_svr_poly=[]
models_rfr=[]
models_gbr=[]

#创建REMS列表保存每次训练的RMSE
RMES_models_svr_rbf=[]
RMES_models_svr_poly=[]
RMES_rfr=[]
RMES_gbr=[]

"""
svr_rbf = SVR(kernel='rbf', C=1.0, epsilon=0.1)
svr_poly = SVR(kernel='poly', degree=3, C=1.0, epsilon=0.1, coef0=1)
rfr = RandomForestRegressor(n_estimators=100, random_state=42)
gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
#bpnn=MLP()
"""
for i,dataset in enumerate(datasets_list_dataframe):
    #通过重新实例化，将每次训练的模型重置为初始状态
    svr_rbf = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    svr_poly = SVR(kernel='poly', degree=3, C=1.0, epsilon=0.1, coef0=1)
    rfr = RandomForestRegressor(n_estimators=100, random_state=42)
    gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    X=dataset.drop(['strength','temperature'],axis=1)
    Y=dataset['strength']

    #将数据集划分为训练集和测试集
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    #训练模型
    #model=svr_rbf.fit(X_train,Y_train)
    model1=svr_rbf.fit(X_train,Y_train)
    model2=svr_poly.fit(X_train,Y_train)
    model3=rfr.fit(X_train,Y_train)
    model4=gbr.fit(X_train,Y_train)

    #在训练集上验证模型(RMSE)
    y_pred_model1 = model1.predict(X_test)
    RMES_models_svr_rbf.append(np.sqrt(mean_squared_error(Y_test, y_pred_model1)))
    models_rfr.append(model1)

    y_pred_model2 = model2.predict(X_test)
    RMES_models_svr_poly.append(np.sqrt(mean_squared_error(Y_test, y_pred_model2)))
    models_svr_poly.append(model2)

    y_pred_model3 = model3.predict(X_test)
    RMES_rfr.append(np.sqrt(mean_squared_error(Y_test, y_pred_model3)))
    models_rfr.append(model3)

    y_pred_model4 = model4.predict(X_test)
    RMES_gbr.append(np.sqrt(mean_squared_error(Y_test, y_pred_model4)))
    models_gbr.append(model4)




    #保存训练模型到文件
    joblib.dump(model1, f'models_svr_rbf/rbf_model_{i}.pkl') # 保存为 svr_model_0.pkl, svr_model_1.pkl, ...
    joblib.dump(model2, f'models_svr_poly/poly_model_{i}.pkl')
    joblib.dump(model3, f'models_rfr/rfr_model_{i}.pkl')
    joblib.dump(model4, f'models_gbr/gbr_model_{i}.pkl')

#根据RMSE选择最佳模型，并绘制折线图

# 生成 x 轴数据
x = np.linspace(0, 100, 100)


# 绘制曲线
plt.plot(x, RMES_models_svr_rbf, label='SVR (RBF)', color='blue')
plt.plot(x, RMES_models_svr_poly, label='SVR (Poly)', color='green')
plt.plot(x, RMES_rfr, label='Random Forest Regressor', color='red')
plt.plot(x, RMES_gbr, label='Gradient Boosting Regressor', color='orange')

# 添加图例
plt.legend()

# 添加标题
plt.title('Comparison of Regression Models')

# 添加坐标轴标签
plt.xlabel('X-axis Label')
plt.ylabel('Root Mean Squared Error (RMSE)')

# 为每条折线添加标签
plt.annotate('SVR (RBF)', xy=(x[50], RMES_models_svr_rbf[50]), xytext=(x[50] + 5, RMES_models_svr_rbf[50] + 0.1),
             arrowprops=dict(facecolor='blue', shrink=0.05))
plt.annotate('SVR (Poly)', xy=(x[50], RMES_models_svr_poly[50]), xytext=(x[50] + 5, RMES_models_svr_poly[50] + 0.1),
             arrowprops=dict(facecolor='green', shrink=0.05))
plt.annotate('Random Forest Regressor', xy=(x[50], RMES_rfr[50]), xytext=(x[50] + 5, RMES_rfr[50] + 0.1),
             arrowprops=dict(facecolor='red', shrink=0.05))
plt.annotate('Gradient Boosting Regressor', xy=(x[50], RMES_gbr[50]), xytext=(x[50] + 5, RMES_gbr[50] + 0.1),
             arrowprops=dict(facecolor='orange', shrink=0.05))

# 显示图形
plt.show()


#展示图片

plt.savefig('regression_comparison.png')

print(1)
