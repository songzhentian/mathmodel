import pandas as pd
import os

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
from pyswarm import pso

import numpy as np

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

# 定义PSO的目标函数，即交叉验证的负准确度（我们希望最小化负准确度）
def objective_function(params):
    n_estimators = int(params[0])
    max_depth = int(params[1])
    min_samples_split = int(params[2])
    min_samples_leaf = int(params[3])

    # 创建随机森林模型
    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )

    # 交叉验证
    scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='accuracy')

    # 返回负准确度（PSO最小化目标）
    return -np.mean(scores)


# 定义PSO搜索空间（参数的范围）
lb = [1, 1, 2, 1]  # 最小值
ub = [100, 10, 10, 10]  # 最大值

# 使用PSO进行优化
best_params, _ = pso(objective_function, lb, ub, swarmsize=10, maxiter=50)

# 输出最佳参数
print("Best Parameters:", best_params)

# 使用最佳参数训练最终模型
rf_model = RandomForestClassifier(
    n_estimators=int(best_params[0]),
    max_depth=int(best_params[1]),
    min_samples_split=int(best_params[2]),
    min_samples_leaf=int(best_params[3]),
    random_state=42
)
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

# 计算准确度
accuracy_train = accuracy_score(y_train, y_pred_train)
accuracy_test = accuracy_score(y_test, y_pred_test)

# 打印准确度
print(f"Train Accuracy: {accuracy_train:.2f}")
print(f"Test Accuracy: {accuracy_test:.2f}")

cla_name = ['0', '1']
# 输出精度报告
report_train = classification_report(y_train, y_pred_train, target_names=cla_name, digits=4)
report_test = classification_report(y_test, y_pred_test, target_names=cla_name, digits=4)
print("Train Classification Report:")
print(report_train)
print("\nTest Classification Report:")
print(report_test)

# 计算混淆矩阵
conf_matrix_train = confusion_matrix(y_train, y_pred_train)
conf_matrix_test = confusion_matrix(y_test, y_pred_test)

# 绘制训练集混淆矩阵热图
plt.figure()
sns.heatmap(conf_matrix_train, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Train Confusion Matrix', pad=20)
plt.show()

# 绘制测试集混淆矩阵热图
plt.figure()
sns.heatmap(conf_matrix_test, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Test Confusion Matrix', pad=20)
plt.show()

# 获取训练集 ROC 曲线数据
fpr_train, tpr_train, thresholds_train = roc_curve(y_train, rf_model.predict_proba(X_train)[:, 1])
roc_auc_train = auc(fpr_train, tpr_train)

# 获取测试集 ROC 曲线数据
fpr_test, tpr_test, thresholds_test = roc_curve(y_test, rf_model.predict_proba(X_test)[:, 1])
roc_auc_test = auc(fpr_test, tpr_test)

# 绘制训练集 ROC 曲线
plt.figure()
plt.plot(fpr_train, tpr_train, color='darkorange', lw=2, label=f'Train ROC curve (AUC = {roc_auc_train:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Train Receiver Operating Characteristic', pad=20)
plt.legend(loc="lower right")
plt.show()

# 绘制测试集 ROC 曲线
plt.figure()
plt.plot(fpr_test, tpr_test, color='darkorange', lw=2, label=f'Test ROC curve (AUC = {roc_auc_test:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Test Receiver Operating Characteristic', pad=20)
plt.legend(loc="lower right")
plt.show()

# 学习曲线
train_sizes, train_scores, test_scores = learning_curve(
    rf_model, X_train, y_train, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10)
)

# 计算平均值和标准差
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# 绘制学习曲线
plt.figure()
plt.plot(train_sizes, train_mean, color='blue', marker='o', linewidth=4, markersize=20, label='Training Accuracy')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='blue')

plt.plot(train_sizes, test_mean, color='green', linestyle='--', linewidth=4, marker='s', markersize=20, label='Validation Accuracy')
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.15, color='green')

plt.xlabel('Training Examples')
plt.ylabel('Accuracy')
plt.title('Random Forest Learning Curve', pad=20)
plt.legend(loc='lower right')
plt.tight_layout()
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