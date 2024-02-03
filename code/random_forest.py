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

# 初始化随机森林模型
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# 训练模型
rf_model.fit(X_train, y_train)

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

# 初始化决策树数目列表和MSE列表
n_estimators_list = [10, 20, 30, 40, 50]
mse_list = []

# 循环训练模型并记录MSE
for n_estimators in n_estimators_list:
    rf_model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_count = rf_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred_count)
    mse_list.append(mse)

# 绘制决策树数目和MSE曲线
plt.plot(n_estimators_list, mse_list, marker='o')
plt.xlabel('Number of Decision Trees')
plt.ylabel('Mean Squared Error')
plt.title('MSE vs. Number of Decision Trees')
plt.show()

# 获取学习曲线数据
train_sizes, train_scores, test_scores = learning_curve(
    rf_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error',
    train_sizes=np.linspace(0.1, 1.0, 10)
)

# 计算平均值和标准差
train_mean = -np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = -np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# 绘制学习曲线
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training Error')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='blue')

plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='Validation Error')
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.15, color='green')

plt.xlabel('Number of Trees')
plt.ylabel('Negative Mean Squared Error')
plt.title('Random Forest Learning Curve')
plt.legend(loc='upper right')
plt.show()

# 绘制特征重要性柱状图
plt.bar(feature_names, feature_importance)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Feature Importance')
plt.xticks(rotation=45)
plt.show()

# 训练集和测试集的预测准确率
train_accuracy = rf_model.score(X_train, y_train)
test_accuracy = rf_model.score(X_test, y_test)

# 绘制训练集折线图
plt.figure()
plt.plot(y_train, label='True')
plt.plot(rf_model.predict(X_train), label='Pred')
plt.xlabel('Samples')
plt.ylabel('Result')
plt.title(f'Result for train\nAccuracy: {train_accuracy*100:.2f}%')
plt.legend()
plt.show()

# 绘制测试集折线图
plt.figure()
plt.plot(y_test, label='True')
plt.plot(rf_model.predict(X_test), label='Pred')
plt.xlabel('Samples')
plt.ylabel('Result')
plt.title(f'Result for test\nAccuracy: {test_accuracy*100:.2f}%')
plt.legend()
plt.show()

sns.set(style="whitegrid",font_scale=1.2)
# 绘制训练集回归图
df = pd.DataFrame({'True Value': np.unique(y_train), 'Predicted Value': y_pred_train})
sns.lmplot(x='True Value', y='Predicted Value', data=df)
plt.title(f'Test Set Regression Coefficient: {r2_score(y_train, y_pred_train):.2f}')
plt.show()

# 绘制测试集回归图
df = pd.DataFrame({'True Value': np.unique(y_test), 'Predicted Value': y_pred_test})
sns.lmplot(x='True Value', y='Predicted Value', data=df)
plt.title(f'Test Set Regression Coefficient: {r2_score(y_test, y_pred_test):.2f}')
plt.show()

# 训练集误差直方统计
train_errors = np.unique(y_train) - y_pred_train
plt.figure()
plt.hist(train_errors, bins=20, edgecolor='black')
plt.xlabel('Error')
plt.ylabel('Frequency')
plt.title('Training Set Error Histogram')
plt.show()

# 测试集误差直方统计
test_errors = np.unique(y_test) - y_pred_test
plt.figure()
plt.hist(test_errors, bins=20, edgecolor='black')
plt.xlabel('Error')
plt.ylabel('Frequency')
plt.title('Test Set Error Histogram')
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
