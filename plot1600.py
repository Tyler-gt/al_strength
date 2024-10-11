import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# 读取CSV文件
df = pd.read_csv('1600csv.csv')

# 特征列名称
feature1_column = 'Cu'
feature2_column = 'Mg'
y_column = 'Strength'

# 提取特征列
x1 = df[feature1_column].values
x2 = df[feature2_column].values
y = df[y_column].values

# 创建多项式特征 (这里使用三阶多项式)
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(np.column_stack((x1, x2)))

# 拟合多项式回归模型
model = LinearRegression()
model.fit(X_poly, y)

# 生成网格点以用于绘制曲面图
x1_grid, x2_grid = np.meshgrid(np.linspace(min(x1), max(x1), 1000),
                               np.linspace(min(x2), max(x2), 1000))

# 预测对应的 y 值
X_grid = poly.transform(np.column_stack((x1_grid.ravel(), x2_grid.ravel())))
y_grid_pred = model.predict(X_grid).reshape(x1_grid.shape)

# 创建图形
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制拟合的表面图
ax.plot_surface(x1_grid, x2_grid, y_grid_pred, cmap='viridis_r', alpha=0.7)

# (可选) 绘制原始数据点
# ax.scatter(x1, x2, y, color='r', label='Original Data')

# 设置标签
ax.set_xlabel(feature1_column)
ax.set_ylabel(feature2_column)
ax.set_zlabel(y_column)

# 设置视角
ax.view_init(elev=30, azim=240)

# 反转Z轴
ax.invert_xaxis()
#ax.invert_zaxis()

# 如果添加了原始数据点，可以取消注释并使用图例
# ax.legend()

# 添加标题
ax.set_title('test1600')

# 显示图形
plt.show()
