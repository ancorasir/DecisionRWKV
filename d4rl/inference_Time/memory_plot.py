import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# 加载数据
dt_time = np.load('drwkv4_memory.npy')
dt_memory = np.load('dt_memory.npy')
drwkv4_memory = np.load('drwkv4_memory.npy')
drwkv5_memory = np.load('drwkv5_memory.npy')
drwkv6_memory = np.load('drwkv6_memory.npy')

# 取数据
x = dt_time[0].tolist()
y_dt_memory = dt_memory[1].tolist()

y_drwkv4_memory = drwkv4_memory[1].tolist()
y_drwkv5_memory = drwkv5_memory[1].tolist()
y_drwkv6_memory = drwkv6_memory[1].tolist()

# 定义拟合函数
def poly1(x, a, b):
    return a * x + b

def poly2(x, a, b, c):
    return a * x**2 + b * x + c

# 进行曲线拟合
params5, _ = curve_fit(poly2, x, y_dt_memory)
params6, _ = curve_fit(poly1, x, y_drwkv4_memory)
params7, _ = curve_fit(poly1, x, y_drwkv5_memory)
params8, _ = curve_fit(poly1, x, y_drwkv6_memory)

# 使用拟合参数生成拟合曲线上的点
x_fit = np.linspace(min(x), max(x), 100)
y5_fit = poly2(x_fit, *params5)
y6_fit = poly1(x_fit, *params6)
y7_fit = poly1(x_fit, *params7)
y8_fit = poly1(x_fit, *params8)

# 创建图表
fig, ax = plt.subplots()

# 绘制拟合曲线
ax.plot(x_fit, y5_fit, label='DT memory fit (quadratic)', color='orange', linestyle='--', linewidth=3, alpha=0.7)
ax.plot(x_fit, y6_fit, label='DRWKV4 memory fit (linear)', color='blue', linestyle='--', linewidth=3, alpha=0.7)
ax.plot(x_fit, y7_fit, label='DRWKV5 memory fit (linear)', color='green', linestyle='--', linewidth=3, alpha=0.7)
ax.plot(x_fit, y8_fit, label='DRWKV6 memory fit (linear)', color='red', linestyle='--', linewidth=3, alpha=0.7)

# 设置y轴的标签
ax.set_ylabel('Memory Usage (MB)', color='black')
ax.tick_params(axis='y', labelcolor='black')
ax.set_xlabel('Sequence Length')

# 添加网格
ax.grid(True)

# 添加图例
ax.legend(loc='upper left')

# 显示图表
plt.show()