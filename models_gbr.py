import pandas as pd
import matplotlib.pyplot as plt

# 创建示例DataFrame
data = {'column1': [1, 2, 3, 4, 5],
        'column2': [10, 20, 30, 40, 50]}

df = pd.DataFrame(data)

# 绘制散点图
plt.scatter(df['column1'], df['column2'])

# 添加标签和标题
plt.xlabel('Column 1')
plt.ylabel('Column 2')
plt.title('Scatter Plot of Column 1 vs Column 2')

# 显示图像
plt.show()
