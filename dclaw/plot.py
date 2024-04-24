import numpy as np
import matplotlib.pyplot as plt
# Load the data from the .npy files
plot_X = np.load('DRWKV4-DCLAW-128-12-0-plot_X.npy')
plot_Y = np.load('DRWKV4-DCLAW-128-12-0-plot_Y.npy')

X = []
Y = []
index = 0
eval = 1000
update = 3000
# Now you can use plot_X and plot_Y for further processing or analysis
for i in range(len(plot_X)):
    for j in range(int(update/eval)):
        X.append(eval*index)
        index = index + 1
X.append(eval*index)

for k in range(10):
    YY = []
    for i in range(len(plot_Y)):
        for j in range(int(update/eval)):
            YY.append(plot_Y[i][j][k])
            index = index + 1
    YY.append(plot_Y[len(plot_Y)-1][int(update/eval)][k])
    # print(YY)
    Y.append(YY)

Y = np.array(Y)
X = np.array(X)
plt.figure(figsize=(12, 8))

# 循环绘制每份Y轴数据
for i in range(10):
    # 计算每份Y数据的起始点
    start_index = i * 2
    # 生成对应的X轴数据
    x_data = X[start_index:]
    # 绘制数据
    if i ==0:
        plt.plot(x_data, Y[i, start_index:], label=f'Task {i + 1}', linewidth=5, alpha=1 )
    else:
        plt.plot(x_data, Y[i, start_index:], label=f'Task {i+1}', linewidth=2, alpha=0.5)

# 在X=200,400,600...1800处画一条竖直的虚线
for x in X[int(update/eval)::int(update/eval)]:  # 从第三个元素开始，每隔两个取一个
    plt.axvline(x=x, color='grey', linestyle='--', linewidth=1)

# 添加图例
plt.legend()

# 添加标题和轴标签
plt.title('Lifelong learning process')
plt.xlabel('Training step')
plt.ylabel('Task Reward')

# 显示图形
plt.show()