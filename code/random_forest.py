import numpy as np
import pandas as pd
import os

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import seaborn as sns
from sklearn.model_selection import learning_curve

# 数据准备
def read_csv_folder(folder_path):
    # 初始化一个空列表，用于存储每个CSV文件的数据字典
    data_dicts = []

    # 获取文件夹中所有的CSV文件
    csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]

    # 循环遍历每个CSV文件
    for csv_file in csv_files:
        # 构造CSV文件的完整路径
        file_path = os.path.join(folder_path, csv_file)

        # 读取CSV文件
        data = pd.read_csv(file_path)

        # 从数据中提取X和Y
        X = data.iloc[:, :6].values
        Y = data.iloc[:, 6].values.reshape(-1, 1)

        # 构造数据字典
        data_dict = {'X': X, 'Y': Y}

        # 将数据字典添加到列表中
        data_dicts.append(data_dict)

    return data_dicts

# 将特征和目标变量分开
X=[]
Y=[]
t_data=read_csv_folder('train_data')

for data in t_data:
    for d in data['X']:
        X.append(d)
    for d_ in data['Y']:
        Y.append(d_)

# 将数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

y_train=np.array(y_train)
y_test=np.array(y_test)
# 初始化随机森林模型
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# 训练模型
rf_model.fit(X_train, np.ravel(y_train))

# 获取特征重要性
feature_importance = rf_model.feature_importances_
# 假设您有特征名称列表如下：
feature_names = ['g_differ', 'p_differ','p1_ace','p1_fault','p1_dis', 'server']
# 将特征重要性与特征名称一起组合
feature_importance_dict = dict(zip(feature_names, feature_importance))

# 打印每个特征对结果的影响
print("Feature Importance:")
for feature, importance in feature_importance_dict.items():
    print(f"{feature}: {importance}")

# 预测
y_pred_train=rf_model.predict(X_train)
y_pred_test = rf_model.predict(X_test)


# 设置全局绘图参数
plt.rcParams['figure.figsize'] = (25, 15)
plt.rcParams['axes.labelsize'] = 30
plt.rcParams['axes.titlesize'] = 30
plt.rcParams['font.size'] = 30
plt.rcParams['font.family'] = 'Times New Roman'

# 获取学习曲线数据
train_sizes, train_scores, test_scores = learning_curve(
    rf_model, X_train, np.ravel(y_train), cv=5, scoring='neg_mean_squared_error',
    train_sizes=np.linspace(0.1, 1.0, 10)
)

# 计算平均值和标准差
train_mean = -np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = -np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# 绘制学习曲线
plt.figure()
plt.plot(train_sizes, train_mean, color='blue', marker='o', linewidth=4, markersize=20, label='Training Error')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='blue')

plt.plot(train_sizes, test_mean, color='green', linestyle='--', linewidth=4, marker='s', markersize=20, label='Validation Error')
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.15, color='green')

plt.xlabel('Number of Trees')
plt.ylabel('Negative Mean Squared Error')
plt.title('Random Forest Learning Curve', pad=20)
plt.legend(loc='upper right')
plt.tight_layout()  # 优化布局
plt.show()

# 绘制特征重要性柱状图
# 设置图片大小
plt.figure()

# 绘制横向特征重要性柱状图
plt.barh(feature_names, feature_importance, edgecolor='black')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.title('Feature Importance', pad=20)

# 添加垂直网格线
plt.grid(axis='x', linestyle='--', alpha=0.6)

plt.tight_layout()  # 保证标签显示完整
plt.show()

# 训练集和测试集的预测准确率
train_accuracy = rf_model.score(X_train, y_train)
test_accuracy = rf_model.score(X_test, y_test)

# 绘制训练集折线图
plt.figure()
plt.plot(y_train, label='True', linestyle='-', linewidth=4, marker='o', markersize=20)
plt.plot(rf_model.predict(X_train), label='Pred', linestyle='--', linewidth=4, marker='s', markersize=20)
plt.xlabel('Samples')
plt.ylabel('Result')
plt.title(f'Result for train\nAccuracy: {train_accuracy*100:.2f}%', pad=20)
plt.legend()
plt.tight_layout()  # 优化布局
plt.grid(axis='both', linestyle='--', alpha=0.6)
plt.show()

# 绘制测试集折线图
plt.figure()
plt.plot(y_test, label='True', linestyle='-', linewidth=4, marker='o', markersize=20)
plt.plot(rf_model.predict(X_test), label='Pred', linestyle='--', linewidth=4, marker='s', markersize=20)
plt.xlabel('Samples')
plt.ylabel('Result')
plt.title(f'Result for test\nAccuracy: {test_accuracy*100:.2f}%', pad=20)
plt.legend()
plt.tight_layout()  # 优化布局
plt.grid(axis='both', linestyle='--', alpha=0.6)
plt.show()

# 训练集误差直方统计
train_errors = np.unique(y_train) - y_pred_train
plt.figure()
# 添加垂直网格线
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.hist(train_errors, bins=20, edgecolor='black')
plt.xlabel('Error')
plt.ylabel('Frequency')
plt.title('Training Set Error Histogram', pad=20)
plt.tight_layout()  # 优化布局
plt.show()

# 测试集误差直方统计
test_errors = np.unique(y_test) - y_pred_test
plt.figure()
# 添加垂直网格线
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.hist(test_errors, bins=20, edgecolor='black')
plt.xlabel('Error')
plt.ylabel('Frequency')
plt.title('Test Set Error Histogram', pad=20)
plt.tight_layout()  # 优化布局
plt.show()


# 设置为白色背景，带有网格线
sns.set(font="Times New Roman", style="whitegrid")
# 绘制训练集回归图
df = pd.DataFrame({'True Value': np.unique(y_train), 'Predicted Value': y_pred_train})
sns.lmplot(x='True Value', y='Predicted Value', data=df, height=15, aspect=1.666666667,
           scatter_kws={'s': 200},  # 设置散点的大小，这里设置为100
           line_kws={'linewidth': 4}  # 设置回归线的线宽，这里设置为4
           )
plt.title(f'Train Set Regression Coefficient: {r2_score(y_train, y_pred_train):.2f}', fontsize=30, pad=20)
plt.tick_params(axis='both', labelsize=30)
plt.xlabel('True Value', fontsize=30)
plt.ylabel('Predicted Value', fontsize=30)
plt.tight_layout()  # 优化布局
plt.show()

# 绘制测试集回归图
df = pd.DataFrame({'True Value': np.unique(y_test), 'Predicted Value': y_pred_test})
sns.lmplot(x='True Value', y='Predicted Value', data=df, height=15, aspect=1.666666667,
           scatter_kws={'s': 200},  # 设置散点的大小，这里设置为100
           line_kws={'linewidth': 4}  # 设置回归线的线宽，这里设置为4
           )
plt.title(f'Test Set Regression Coefficient: {r2_score(y_test, y_pred_test):.2f}', fontsize=30, pad=20)
plt.tick_params(axis='both', labelsize=30)
plt.xlabel('True Value', fontsize=30)
plt.ylabel('Predicted Value', fontsize=30)
plt.tight_layout()  # 优化布局
plt.show()

# 计算评估指标
sse = np.sum((y_test - y_pred_test) ** 2)
mae = mean_absolute_error(y_test, y_pred_test)
mse = mean_squared_error(y_test, y_pred_test)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100
r2 = r2_score(y_test, y_pred_test)

# 输出评估指标到控制台
print('SSE:', sse)
print('MAE:', mae)
print('MSE:', mse)
print('RMSE:', rmse)
print('MAPE:', mape)
print('R-squared:', r2)

# 计算均方误差
mse = mean_squared_error(y_test, y_pred_test)
print('Mean Squared Error:', mse)

