#做折线图
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
Cu=[2.5,2.7,2.9,3.1,3.3,3.5,3.7,3.9,4.1 ,4.3,4.5,5.1,5.2,5.3,5.5,6.0 ]
strength_cu=[373.44,379.16,384.64,389.89,394.97,399.79,404.36,408.31,410.83,415.2,418.67,415.63,415.14,413.61,412.57,410.97]
#已cu和strength做多项式拟合曲线


x=np.array(Cu)
y=np.array(strength_cu)
# 多项式拟合 (假设为二次曲线)
coefficients = np.polyfit(x, y, 3)  # 2表示二次多项式
polynomial = np.poly1d(coefficients)

# 画出拟合曲线
x_line = np.linspace(min(x), max(x), 100)
y_line = polynomial(x_line)
#为x,y轴命名
plt.xlabel('Cu')
plt.ylabel('Strength')

plt.scatter(x, y, label='Strength Points')
plt.plot(x_line, y_line, color='red', label='Fitted Curve')
plt.legend()
plt.show()
#保存图片
plt.savefig('strength_cu.png')


Mn=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1]
strength_mn=[408.0,413.64,414.94,415.2,412.38,409.57,406.81,404.85,402.8,400.52,398.03]
x=np.array(Mn)
y=np.array(strength_mn)
# 多项式拟合 (假设为二次曲线)
coefficients = np.polyfit(x, y, 3)  # 2表示二次多项式
polynomial = np.poly1d(coefficients)

# 画出拟合曲线
x_line = np.linspace(min(x), max(x), 100)
y_line = polynomial(x_line)
#为x,y轴命名
plt.xlabel('Mn')
plt.ylabel('Strength')

plt.scatter(x, y, label='Strength Points')
plt.plot(x_line, y_line, color='red', label='Fitted Curve')
plt.legend()
plt.show()
#保存图片
plt.savefig('strength_mn.png')




Mg=[0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.2]
strength_mg=[399.62,401.62,403.6,405.55,407.51,409.47,411.41,413.34,415.2,415.9,419,418.58,417.3,413.24]
x=np.array(Mg)
y=np.array(strength_mg)
# 多项式拟合 (假设为二次曲线)
coefficients = np.polyfit(x, y, 3)  # 2表示二次多项式
polynomial = np.poly1d(coefficients)

# 画出拟合曲线
x_line = np.linspace(min(x), max(x), 100)
y_line = polynomial(x_line)
#为x,y轴命名
plt.xlabel('Mg')
plt.ylabel('Strength')

plt.scatter(x, y, label='Strength Points')
plt.plot(x_line, y_line, color='red', label='Fitted Curve')
plt.legend()
plt.show()
#保存图片
plt.savefig('strength_mg.png')




Fe=[0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0,0.02]
strength_fe=[409.47,409.81,408.83,407.35,404.39,401.37,398.34,398.36,392.24,389.15,405.54,409.14]
x=np.array(Fe)
y=np.array(strength_fe)
# 多项式拟合 (假设为二次曲线)
coefficients = np.polyfit(x, y, 3)  # 2表示二次多项式
polynomial = np.poly1d(coefficients)

# 画出拟合曲线
x_line = np.linspace(min(x), max(x), 100)
y_line = polynomial(x_line)
#为x,y轴命名
plt.xlabel('Fe')
plt.ylabel('Strength_fe')

plt.scatter(x, y, label='Strength Points')
plt.plot(x_line, y_line, color='red', label='Fitted Curve')
plt.legend()
plt.show()
#保存图片
plt.savefig('strength_fe.png')


Zn=[0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4]
#整理strength_si
strength_zn=[409.45,409.75,409.95,410.2,410.46,410.66,410.93,411.12]
x=np.array(Zn)
y=np.array(strength_zn)
# 多项式拟合 (假设为二次曲线)
coefficients = np.polyfit(x, y, 3)  # 2表示二次多项式
polynomial = np.poly1d(coefficients)

# 画出拟合曲线
x_line = np.linspace(min(x), max(x), 100)
y_line = polynomial(x_line)
#为x,y轴命名
plt.xlabel('Zn')
plt.ylabel('Strength_zn')

plt.scatter(x, y, label='Strength Points')
plt.plot(x_line, y_line, color='red', label='Fitted Curve')
plt.legend()
plt.show()
#保存图片
plt.savefig('strength_zn.png')


Si=[0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
strength_si=[398.6,404.28,408,409.75,409.2,406.69,404.56,402.83,401.51,398.37]
x=np.array(Si)
y=np.array(strength_si)
# 多项式拟合 (假设为二次曲线)
coefficients = np.polyfit(x, y, 3)  # 2表示二次多项式
polynomial = np.poly1d(coefficients)

# 画出拟合曲线
x_line = np.linspace(min(x), max(x), 100)
y_line = polynomial(x_line)
#为x,y轴命名
plt.xlabel('Si')
plt.ylabel('Strength_si')

plt.scatter(x, y, label='Strength Points')
plt.plot(x_line, y_line, color='red', label='Fitted Curve')
plt.legend()
plt.show()
#保存图片
plt.savefig('strength_si.png')





