import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


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
t_data=read_csv_folder('D:\\MCM\\code\\train_data')

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
y_pred = rf_model.predict(X_test)

# 计算均方误差
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)
