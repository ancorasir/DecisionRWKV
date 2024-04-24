import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# 加载数据
dt_time = np.load('dt_infTime.npy')
drwkv4_time = np.load('drwkv4_infTime.npy')
drwkv5_time = np.load('drwkv5_infTime.npy')
drwkv6_time = np.load('drwkv6_infTime.npy')

# 取数据
x = dt_time[0].tolist()
y_dt = dt_time[1].tolist()
y_drwkv4 = drwkv4_time[1].tolist()
y_drwkv5 = drwkv5_time[1].tolist()
y_drwkv6 = drwkv6_time[1].tolist()

# 定义拟合函数
def poly1(x, a, b):
    return a * x + b

def poly2(x, a, b, c):
    return a * x**2 + b * x + c

# 进行曲线拟合
params1, _ = curve_fit(poly2, x, y_dt)
params2, _ = curve_fit(poly1, x, y_drwkv4)
params3, _ = curve_fit(poly1, x, y_drwkv5)
params4, _ = curve_fit(poly1, x, y_drwkv6)

# 使用拟合参数生成拟合曲线上的点
x_fit = np.linspace(min(x), max(x), 100)
y1_fit = poly2(x_fit, *params1)
y2_fit = poly1(x_fit, *params2)
y3_fit = poly1(x_fit, *params3)
y4_fit = poly1(x_fit, *params4)

# 创建图表
fig, ax = plt.subplots()

# 绘制拟合曲线

ax.plot(x_fit, y1_fit, label='DT fit (quadratic)', color='darkred', linestyle='--', linewidth=3, alpha=0.7)
ax.plot(x_fit, y2_fit, label='DRWKV4 fit (linear)', color='darkblue', linestyle='--', linewidth=3, alpha=0.7)
ax.plot(x_fit, y3_fit, label='DRWKV5 fit (linear)', color='goldenrod', linestyle='--', linewidth=3, alpha=0.7)
ax.plot(x_fit, y4_fit, label='DRWKV6 fit (linear)', color='darkgreen', linestyle='--', linewidth=3, alpha=0.7)

# 设置y轴的标签
ax.set_ylabel('Time Cost (ms)', color='black')
ax.tick_params(axis='y', labelcolor='black')
ax.set_xlabel('Sequence Length')

# 添加网格
ax.grid(True)

# 添加图例
ax.legend(loc='upper left')

# 显示图表
plt.show()