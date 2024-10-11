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

#载入116的json文件
with open('datasets_116.json', 'r') as f:
    loaded_data=json.load(f)

#将字典列表转化为dataframe列表
datasets_list_dataframe=[pd.DataFrame(dict) for dict in loaded_data]

#保存随机森林模型列表
models_gbr=[]

for i,dataset in enumerate(datasets_list_dataframe):
    #通过重新实例化，将每次训练的模型重置为初始状态
    gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    #svr_rbf = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    X = dataset.drop('strength', axis=1)
    Y = dataset['strength']
    # 将数据集划分为训练集和测试集

    model1 = gbr.fit(X, Y)

    models_gbr.append(model1)

    #保存训练模型到文件
    joblib.dump(model1, f'models_gbr_2/gbr_model2_{i}.pkl')