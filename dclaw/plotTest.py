import numpy as np
import matplotlib.pyplot as plt
import math

def calculate_mean_and_std(A, B, C):
    mean = (A + B + C) / 3
    variance = ((A - mean) ** 2 + (B - mean) ** 2 + (C - mean) ** 2) / 3
    std_dev = math.sqrt(variance)

    return mean, std_dev


methods = ["DT","DRWKV4","DRWKV5","DRWKV6"]
method = "DRWKV6"

seeds = [0,10,42]
scores = np.zeros([10,3])
# print(scores)
# seed = 0
# strX = method + "-DCLAW-128-12-" + str(seed) + "-plot_X.npy"
# plot_X = np.load(strX)
# lenth,_ = plot_X.shape
# print(lenth)
lenth = 10
#
for j in range(len(seeds)):
    # strX = methods + "-DCLAW-128-12-" + str(seed) + "-plot_X.npy"
    strY = method + "-DCLAW-128-12-" + str(seeds[j]) + "-plot_Y.npy"
    plot_Y = np.load(strY)
    # for i in range(10):

    for i in range(lenth):
        scores[i][j] = sum(plot_Y[i][1])/10
        # print(sum(plot_Y[i][1])/10)

for i in range(lenth):
    mean, std_dev = calculate_mean_and_std(scores[i][0],scores[i][1],scores[i][2])
    print(mean,std_dev)
# for seed in seeds:
#     strX = methods + "-DCLAW-128-12-" + str(seed) + "-plot_X.npy"
#     strY = methods + "-DCLAW-128-12-" + str(seed) + "-plot_Y.npy"
#     plot_X = np.load(strX)
#     plot_Y = np.load(strY)
#     for i in range(10):
#         print(sum(plot_Y[i][1])/10)