# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.optimize import curve_fit
#
# # 加载数据
# dt_time = np.load('dt_infTime.npy')
# drwkv4_time = np.load('drwkv4_infTime.npy')
# drwkv5_time = np.load('drwkv5_infTime.npy')
# drwkv6_time = np.load('drwkv6_infTime.npy')
# dt_memory = np.load('dt_memory.npy')
# drwkv4_memory = np.load('drwkv4_memory.npy')
# drwkv5_memory = np.load('drwkv5_memory.npy')
# drwkv6_memory = np.load('drwkv6_memory.npy')
#
# # 提取数据
# x = dt_time[0].tolist()
# y_dt = dt_time[1].tolist()
# y_drwkv4 = drwkv4_time[1].tolist()
# y_drwkv5 = drwkv5_time[1].tolist()
# y_drwkv6 = drwkv6_time[1].tolist()
# y_dt_memory = dt_memory[1].tolist()
# y_drwkv4_memory = drwkv4_memory[1].tolist()
# y_drwkv5_memory = drwkv5_memory[1].tolist()
# y_drwkv6_memory = drwkv6_memory[1].tolist()
#
# # 定义拟合函数
# def poly2(x, a, b, c):
#     return a * x**2 + b * x + c
#
# def poly1(x, a, b):
#     return a * x + b
#
# # 进行曲线拟合
# params1, _ = curve_fit(poly2, x, y_dt)
# params2, _ = curve_fit(poly1, x, y_drwkv4)
# params3, _ = curve_fit(poly1, x, y_drwkv5)
# params4, _ = curve_fit(poly1, x, y_drwkv6)
#
# # 使用拟合参数生成拟合曲线上的点
# x_fit = np.linspace(min(x)+50, max(x), 100)
# y1_fit = poly2(x_fit, *params1)
# y2_fit = poly1(x_fit, *params2)
# y3_fit = poly1(x_fit, *params3)
# y4_fit = poly1(x_fit, *params4)
#
# # 创建图表和第一个y轴
# fig, ax1 = plt.subplots()
#
# # 绘制原始数据点和拟合曲线
# ax1.scatter(x, y_dt, color='red', label='DT inference time', s=5)
# ax1.scatter(x, y_drwkv4, color='blue', label='DRWKV4 inference time', s=5)
# ax1.scatter(x, y_drwkv5, color='yellow', label='DRWKV5 inference time', s=5)
# ax1.scatter(x, y_drwkv6, color='green', label='DRWKV6 inference time', s=5)
# ax1.plot(x_fit, y1_fit, label='DT fit (quadratic)', color='black', linestyle='--')
# ax1.plot(x_fit, y2_fit, label='DRWKV4 fit (linear)', color='grey', linestyle='--')
# ax1.plot(x_fit, y3_fit, label='DRWKV5 fit (linear)', color='grey', linestyle='--')
# ax1.plot(x_fit, y4_fit, label='DRWKV6 fit (linear)', color='grey', linestyle='--')
#
# # 设置第一个y轴的标签和标题
# ax1.set_xlabel('Sequence Length')
# ax1.set_ylabel('Average Time(ms)', color='black')
# ax1.set_title('Average CUDA Inference Time and Memory Usage')
# ax1.tick_params(axis='y', labelcolor='black')
#
# # 添加图例
# ax1.legend(loc='upper left')
#
# # 创建第二个y轴
# ax2 = ax1.twinx()
#
# # 绘制内存使用量的数据点
# ax2.scatter(x, y_dt_memory, color='red', label='DT memory usage', marker='x', s=5)
# ax2.scatter(x, y_drwkv4_memory, color='blue', label='DRWKV4 memory usage', marker='x', s=5)
# ax2.scatter(x, y_drwkv5_memory, color='yellow', label='DRWKV5 memory usage', marker='x', s=5)
# ax2.scatter(x, y_drwkv6_memory, color='green', label='DRWKV6 memory usage', marker='x', s=5)
#
# # 设置第二个y轴的标签
# ax2.set_ylabel('Memory Usage (MB)', color='black')
# ax2.tick_params(axis='y', labelcolor='black')
#
# # 添加图例
# ax2.legend(loc='upper right')
#
# # 显示图表
# plt.show()

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# 加载数据
dt_time = np.load('dt_infTime.npy')
drwkv4_time = np.load('drwkv4_infTime.npy')
drwkv5_time = np.load('drwkv5_infTime.npy')
drwkv6_time = np.load('drwkv6_infTime.npy')
dt_memory = np.load('dt_memory.npy')
drwkv4_memory = np.load('drwkv4_memory.npy')
drwkv5_memory = np.load('drwkv5_memory.npy')
drwkv6_memory = np.load('drwkv6_memory.npy')

# 提取数据
x = dt_time[0].tolist()
y_dt = dt_time[1].tolist()
y_drwkv4 = drwkv4_time[1].tolist()
y_drwkv5 = drwkv5_time[1].tolist()
y_drwkv6 = drwkv6_time[1].tolist()
y_dt_memory = dt_memory[1].tolist()
y_drwkv4_memory = drwkv4_memory[1].tolist()
y_drwkv5_memory = drwkv5_memory[1].tolist()
y_drwkv6_memory = drwkv6_memory[1].tolist()

# 定义拟合函数
def poly2(x, a, b, c):
    return a * x**2 + b * x + c

def poly1(x, a, b):
    return a * x + b

# 进行曲线拟合
params1, _ = curve_fit(poly2, x, y_dt)
params2, _ = curve_fit(poly1, x, y_drwkv4)
params3, _ = curve_fit(poly1, x, y_drwkv5)
params4, _ = curve_fit(poly1, x, y_drwkv6)
params5, _ = curve_fit(poly1, x, y_dt_memory)
params6, _ = curve_fit(poly1, x, y_drwkv4_memory)
params7, _ = curve_fit(poly1, x, y_drwkv5_memory)
params8, _ = curve_fit(poly1, x, y_drwkv6_memory)

# 使用拟合参数生成拟合曲线上的点
x_fit = np.linspace(min(x), max(x), 100)
y1_fit = poly2(x_fit, *params1)
y2_fit = poly1(x_fit, *params2)
y3_fit = poly1(x_fit, *params3)
y4_fit = poly1(x_fit, *params4)
y5_fit = poly1(x_fit, *params5)
y6_fit = poly1(x_fit, *params6)
y7_fit = poly1(x_fit, *params7)
y8_fit = poly1(x_fit, *params8)

# 创建图表和第一个y轴
fig, ax1 = plt.subplots()

# 绘制原始数据点和拟合曲线
ax1.scatter(x, y_dt, color='red', label='DT inference time', s=10)
ax1.scatter(x, y_drwkv4, color='blue', label='DRWKV4 inference time', s=10)
ax1.scatter(x, y_drwkv5, color='yellow', label='DRWKV5 inference time', s=10)
ax1.scatter(x, y_drwkv6, color='green', label='DRWKV6 inference time', s=10)
ax1.plot(x_fit, y1_fit, label='DT fit (quadratic)', color='darkred', linestyle='-', linewidth=2)
ax1.plot(x_fit, y2_fit, label='DRWKV4 fit (linear)', color='darkblue', linestyle='-', linewidth=2)
ax1.plot(x_fit, y3_fit, label='DRWKV5 fit (linear)', color='goldenrod', linestyle='-', linewidth=2)
ax1.plot(x_fit, y4_fit, label='DRWKV6 fit (linear)', color='darkgreen', linestyle='-', linewidth=2)

# 设置第一个y轴的标签和标题
ax1.set_xlabel('Sequence Length')
ax1.set_ylabel('Average Time(ms)', color='black')
ax1.set_title('Average CUDA Inference Time and Memory Usage')
ax1.tick_params(axis='y', labelcolor='black')

# 添加图例
ax1.legend(loc='upper left')

# 创建第二个y轴
ax2 = ax1.twinx()

# 绘制内存使用量的数据点和拟合曲线
ax2.scatter(x, y_dt_memory, color='red', label='DT memory usage', marker='x', s=10)
ax2.scatter(x, y_drwkv4_memory, color='blue', label='DRWKV4 memory usage', marker='x', s=10)
ax2.scatter(x, y_drwkv5_memory, color='yellow', label='DRWKV5 memory usage', marker='x', s=10)
ax2.scatter(x, y_drwkv6_memory, color='green', label='DRWKV6 memory usage', marker='x', s=10)
ax2.plot(x_fit, y5_fit, label='DT memory fit (linear)', color='darkred', linestyle=':', linewidth=2)
ax2.plot(x_fit, y6_fit, label='DRWKV4 memory fit (linear)', color='darkblue', linestyle=':', linewidth=2)
ax2.plot(x_fit, y7_fit, label='DRWKV5 memory fit (linear)', color='goldenrod', linestyle=':', linewidth=2)
ax2.plot(x_fit, y8_fit, label='DRWKV6 memory fit (linear)', color='darkgreen', linestyle=':', linewidth=2)

# 设置第二个y轴的标签
ax2.set_ylabel('Memory Usage (MB)', color='black')
ax2.tick_params(axis='y', labelcolor='black')

# 添加图例
ax2.legend(loc='upper right')

# 显示图表
plt.show()