import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file_path = 'D:\MCM\code\match1.csv'

play1_count = ['g_differ', 'p_differ','p1_ace','p1_fault','p1_dis', 'server']
data1 = pd.read_csv(file_path, usecols=play1_count)
play1 = data1.values
# 找到每列的最小值
min_values = np.min(play1, axis=0)
# 将第1列和第2列都减去最小值
play1[:, 0:2] -= min_values[0:2]
play1[:, 5][play1[:, 5] == 2] = 0
play1[:,0:4] += 1
play1[:,5] += 1


play2_count = ['g_differ', 'p_differ','p2_ace','p2_fault','p2_dis', 'server']
data2 = pd.read_csv(file_path, usecols=play2_count)
play2 = data2.values
play2[:, 0:2] *= -1
# 找到每列的最小值
min_values = np.min(play2, axis=0)
# 将第1列和第2列都减去最小值
play2[:, 0:2] -= min_values[0:2]
play2[:, 5][play2[:, 5] == 1] = 0
play2[:,0:4] += 1
play2[:,5] += 1

result_list = []
num_columns_play1 = len(play1)  # play1矩阵的列数

for i in range(num_columns_play1):
    matrix = np.vstack((play1[i,:], play2[i,:]))  # 垂直堆叠两个数组
    result_list.append(matrix)

print(result_list)

def TOPSIS(play):
    normalized_X = play/np.sqrt((play**2).sum(axis=0))
    entropy=-np.sum(normalized_X*np.log(normalized_X),axis=0)/np.log(play.shape[0])
    weight = (1-entropy)/np.sum(1-entropy)
    w_x=normalized_X*weight
    ideal_best=np.max(w_x,axis=0)
    ideal_worst=np.min(w_x,axis=0)
    d_best=np.sqrt(np.sum((w_x-ideal_best)**2,axis=1))
    d_worst=np.sqrt(np.sum((w_x-ideal_worst)**2,axis=1))

    performance=d_worst/(d_best+d_worst)

    return performance

list=[]
for item in result_list:
    t=TOPSIS(item)
    list.append(t)
# 创建曲线数据
print(list)
curve_data1 = [item[0] for item in list]  # 第一条曲线数据来源于list列表中每个矩阵的第一个元素的数值
curve_data2 = [item[1] for item in list]  # 第二条曲线数据来源于list列表中每个矩阵的第二个元素的数值
new_curve_data = [(x + y) / 2 for x, y in zip(curve_data1, curve_data2)]
# 创建横坐标
x_values = range(1, len(play1) + 1)

# 绘制曲线图
plt.plot(x_values, curve_data1, label='Curve 1')
plt.plot(x_values, curve_data2, label='Curve 2')
plt.plot(x_values, new_curve_data, label='New Curve', linestyle='--')

# 添加标题和图例
plt.title('Comparison of Curves')
plt.legend()

# 显示网格线
plt.grid(True)

# 显示图形
plt.show()

play1_out=TOPSIS(play1)
play2_out=TOPSIS(play2)
plt.figure()
plt.plot(x_values, play1_out, label='Curve 1')
plt.plot(x_values, play2_out, label='Curve 2')
plt.show()

# 将 play1_out 重塑为列向量
play1_out_column = play1_out.reshape(-1, 1)

# 将 play1 和 play1_out 水平堆叠
play1_with_out = np.hstack((play1, play1_out_column))

# 定义列名，包括原始列名和新增列名
column_names = play1_count + ['play1_out']

# 将数组转换为 DataFrame
play1_with_out_df = pd.DataFrame(play1_with_out, columns=column_names)

# 将 DataFrame 写入 CSV 文件
play1_with_out_df.to_csv('play1_with_out_data.csv', index=False)

###

# 将 play1_out 重塑为列向量
play2_out_column = play2_out.reshape(-1, 1)

# 将 play1 和 play1_out 水平堆叠
play2_with_out = np.hstack((play2, play2_out_column))

# 定义列名，包括原始列名和新增列名
column_names2 = play2_count + ['play1_out']

# 将数组转换为 DataFrame
play1_with_out_df2 = pd.DataFrame(play2_with_out, columns=column_names2)

# 将 DataFrame 写入 CSV 文件
play1_with_out_df2.to_csv('play2_with_out_data.csv', index=False)

print("CSV file with play1 and play1_out saved successfully.")
