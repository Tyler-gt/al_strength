import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import numpy as np
#from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
# 读取CSV文件
df = pd.read_csv('new_header.csv')


feature1_column = 'Cu'
feature2_column = 'Mg'
y_column = 'Strength'

# 提取特征列
x1 = df[feature1_column].values
x2 = df[feature2_column].values
y = df[y_column].values

# 创建网格
xi, xj = np.meshgrid(np.linspace(min(x1), max(x1), 100),
                     np.linspace(min(x2), max(x2), 100))
zi = griddata((x1, x2), y, (xi, xj), method='linear')

# 创建图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制散点图
#ax.scatter(x1, x2, y, c='r', marker='o', label='Data points')

# 绘制表面图
ax.plot_surface(xi, xj, zi, cmap='viridis', alpha=0.6)

# 设置标签
ax.set_xlabel(feature1_column)
ax.set_ylabel(feature2_column)
ax.set_zlabel(y_column)

ax.set_zlim(290, 450)

plt.title('test60000')
plt.legend()
plt.show()