from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
import joblib
from sklearn.metrics import mean_squared_error

# 假设 datasets 是包含100个数据集的列表
datasets = [df1, df2, df3, ...]  # 你的 DataFrame 列表
models = []

# 设置 SVR 模型的超参数
svr_rbf = SVR(kernel='rbf', C=1.0, epsilon=0.1)

for i, dataset in enumerate(datasets):
    # 假设每个 dataset 都有特征 X 和标签 y
    X = dataset.drop('target', axis=1)  # 'target' 是标签列
    y = dataset['target']

    # 将数据集划分为训练集和测试集 (80%训练，20%测试)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 训练模型
    model = svr_rbf.fit(X_train, y_train)

    # 在测试集上评估模型
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Model {i} - Mean Squared Error: {mse}')

    # 将训练好的模型保存到列表中
    models.append(model)

    joblib.dump(model, f'models_svr/svr_model_{i}.pkl')  # 保存为 svr_model_0.pkl, svr_mod 保存模型到文件el_1.pkl, ...
