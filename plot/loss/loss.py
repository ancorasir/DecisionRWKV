# import csv
# import math
# x = []
# y1 = []
# y2 = []
# y3 = []
# average_plot = []
# std_deviation = []
# # Create an empty list to store the CSV data
# d4_0 = []
# d4_10 = []
# d4_42 = []
# # Open the CSV file
# with open('dRWKV4_train0.csv', 'r') as file:
#     reader = csv.reader(file)
#
#     # Iterate over each row in the CSV file and append it to the list
#     for row in reader:
#         d4_0.append(row)
# with open('dRWKV4_train10.csv', 'r') as file:
#     reader = csv.reader(file)
#
#     # Iterate over each row in the CSV file and append it to the list
#     for row in reader:
#         d4_10.append(row)
# with open('dRWKV4_train42.csv', 'r') as file:
#     reader = csv.reader(file)
#
#     # Iterate over each row in the CSV file and append it to the list
#     for row in reader:
#         d4_42.append(row)
#
# for i in range(1,1001):
#     x.append(float(d4_0[i][0]))
#     num1 = float(d4_0[i][1])
#     num2 = float(d4_10[i][1])
#     num3 = float(d4_42[i][1])
#     average = (num1 + num2 + num3) / 3
#     variance = ((num1 - average)**2 + (num2 - average)**2 + (num3 - average)**2) / 3
#     average_plot.append(average)
#     std_deviation.append(math.sqrt(variance))
# # print(y1[1])
#
# import matplotlib.pyplot as plt
# plt.plot(x, average_plot)
# # plt.xlabel('X轴')
# # plt.ylabel('Y轴')
# # plt.title('数据图像')
# plt.ylim(0, 0.8)
# plt.show()

import csv
import math
import matplotlib.pyplot as plt

x = []
average_plot_d4 = []
average_plot_d5 = []
average_plot_d6 = []
average_plot_dt = []
std_deviation_d4 = []
std_deviation_d5 = []
std_deviation_d6 = []
std_deviation_dt = []
# Create an empty list to store the CSV data
# d4_0 = []
# d4_10 = []
# d4_42 = []
# d5_0 = []
# d5_10 = []
# d5_42 = []
# d6_0 = []
# d6_10 = []
# d6_42 = []
# dt_0 = []
# dt_10 = []
# dt_42 = []
# Open the CSV files and read the data
def read_csv_data(filename):
    data = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(row)
    return data

d4_0 = read_csv_data('dRWKV4_train0.csv')
d4_10 = read_csv_data('dRWKV4_train10.csv')
d4_42 = read_csv_data('dRWKV4_train42.csv')
d5_0 = read_csv_data('d5_0.csv')
d5_10 = read_csv_data('d5_10.csv')
d5_42 = read_csv_data('d5_42.csv')
d6_0 = read_csv_data('d6_0.csv')
d6_10 = read_csv_data('d6_10.csv')
d6_42 = read_csv_data('d6_42.csv')
dt_0 = read_csv_data('dt_0.csv')
dt_10 = read_csv_data('dt_10.csv')
dt_42 = read_csv_data('dt_42.csv')
# Calculate average and standard deviation
x_number = 1001

for i in range(1, x_number):
    x.append(float(d4_0[i][0]))
    num1 = float(d4_0[i][1])
    num2 = float(d4_10[i][1])
    num3 = float(d4_42[i][1])
    average = (num1 + num2 + num3) / 3
    variance = ((num1 - average)**2 + (num2 - average)**2 + (num3 - average)**2) / 3
    average_plot_d4.append(average)
    std_deviation_d4.append(math.sqrt(variance))

for i in range(1, x_number):
    num1 = float(d5_0[i][1])
    num2 = float(d5_10[i][1])
    num3 = float(d5_42[i][1])
    average = (num1 + num2 + num3) / 3
    variance = ((num1 - average)**2 + (num2 - average)**2 + (num3 - average)**2) / 3
    average_plot_d5.append(average)
    std_deviation_d5.append(math.sqrt(variance))

for i in range(1, x_number):
    num1 = float(d6_0[i][1])
    num2 = float(d6_10[i][1])
    num3 = float(d6_42[i][1])
    average = (num1 + num2 + num3) / 3
    variance = ((num1 - average)**2 + (num2 - average)**2 + (num3 - average)**2) / 3
    average_plot_d6.append(average)
    std_deviation_d6.append(math.sqrt(variance))

for i in range(1, x_number):
    num1 = float(dt_0[i][1])
    num2 = float(dt_10[i][1])
    num3 = float(dt_42[i][1])
    average = (num1 + num2 + num3) / 3
    variance = ((num1 - average)**2 + (num2 - average)**2 + (num3 - average)**2) / 3
    average_plot_dt.append(average)
    std_deviation_dt.append(math.sqrt(variance))

# Plot the average
plt.plot(x, average_plot_dt, label='Average_dt')
plt.plot(x, average_plot_d4, label='Average_drwkv4')
plt.plot(x, average_plot_d5, label='Average_drwkv5')
plt.plot(x, average_plot_d6, label='Average_drwkv6')

# Plot the standard error regions (shaded)
plt.fill_between(x, [a - s for a, s in zip(average_plot_dt, std_deviation_dt)],
                 [a + s for a, s in zip(average_plot_dt, std_deviation_dt)], color='blue', alpha=0.2)

plt.fill_between(x, [a - s for a, s in zip(average_plot_d4, std_deviation_d4)],
                 [a + s for a, s in zip(average_plot_d4, std_deviation_d4)], color='yellow', alpha=0.2)

plt.fill_between(x, [a - s for a, s in zip(average_plot_d5, std_deviation_d5)],
                 [a + s for a, s in zip(average_plot_d5, std_deviation_d5)], color='green', alpha=0.2)

plt.fill_between(x, [a - s for a, s in zip(average_plot_d6, std_deviation_d6)],
                 [a + s for a, s in zip(average_plot_d6, std_deviation_d6)], color='red', alpha=0.2)



plt.ylim(0, 0.8)
plt.xlabel('updateSteps')  # Replace with your x-axis label
plt.ylabel('loss')  # Replace with your y-axis label
plt.title('Plot with Average and Standard Error Regions')
plt.legend()
plt.show()