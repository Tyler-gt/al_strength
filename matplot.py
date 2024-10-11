import matplotlib
import pandas as pd

#打开csv文件
df = pd.read_csv('Cu_Mg_St3_1600.csv')


# 定义新的表头
new_header = ['Cu', 'Mg', 'Strength']  # 根据你的数据列数定义表头

# 添加表头到DataFrame
df.columns = new_header
print(df['Cu'])
# 将DataFrame保存到新的CSV文件
df.to_csv('1600csv.csv', index=False)