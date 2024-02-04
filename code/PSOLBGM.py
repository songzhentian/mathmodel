import pandas as pd
import os
from sklearn.utils import resample
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
from pyswarm import pso
import numpy as np
import chardet


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
        # 检测文件编码
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read())
        file_encoding = result['encoding']
        # 读取CSV文件
        data = pd.read_csv(file_path, encoding=file_encoding)

        # 从数据中提取X和Y
        X = data.iloc[:, :11].values
        Y = data.iloc[:, 11].values.reshape(-1, 1)

        # 构造数据字典
        data_dict = {'X': X, 'Y': Y}

        # 将数据字典添加到列表中
        data_dicts.append(data_dict)

    return data_dicts

# 将特征和目标变量分开
X=[]
Y=[]
t_data=read_csv_folder('match_point')

for data in t_data:
    for d in data['X']:
        X.append(d)
    for d_ in data['Y']:
        Y.append(d_)

# 将数据集划分为训练集和测试集
X_train, y_train = resample(X, Y, random_state=42)
y_train=np.array(y_train)

# 创建测试集（排除训练集中的样本）
X_train_set = set(tuple(x) for x in X_train)
X_test = [x for x in X if tuple(x) not in X_train_set]
y_test = [y for idx, y in enumerate(Y) if tuple(X[idx]) not in X_train_set]


# 将测试集转换为数组形式
X_test = np.array(X_test)
y_test = np.array(y_test)

# 定义PSO的目标函数，即交叉验证的负准确度（我们希望最小化负准确度）
def objective_function(params):
    n_estimators = int(params[0])
    max_depth = int(params[1])
    min_child_samples = int(params[2])
    min_child_weight = int(params[3])

    # 创建LGBM模型
    lgb_model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_child_samples=min_child_samples,
        min_child_weight=min_child_weight,
        random_state=42
    )

    # 交叉验证
    scores = cross_val_score(lgb_model, X_train, np.ravel(y_train), cv=5, scoring='accuracy')

    # 返回负准确度（PSO最小化目标）
    return -np.mean(scores)


# 定义PSO搜索空间（参数的范围）
lb = [10, 3, 2, 1]  # 最小值
ub = [100, 10, 7, 2]  # 最大值

# 使用PSO进行优化
best_params, _ = pso(objective_function, lb, ub, swarmsize=10, maxiter=50)

# 输出最佳参数
print("Best Parameters:", best_params)

# 使用最佳参数训练最终模型
lgb_model = lgb.LGBMClassifier(
    n_estimators=int(best_params[0]),
    max_depth=int(best_params[1]),
    min_child_samples=int(best_params[2]),
    min_child_weight=int(best_params[3]),
    random_state=42
)
# 训练模型
lgb_model.fit(X_train, np.ravel(y_train))

# 获取特征重要性
feature_importance = lgb_model.feature_importances_
# 假设您有特征名称列表如下：
feature_names = ['TPW_dif', 'COMPLETE_dif','rank_dif','ATPP_dif','AAG_dif', 'SERVEADV','res_dif','speed','serve_width','serve_depth','return_depth']
# 将特征重要性与特征名称一起组合
feature_importance_dict = dict(zip(feature_names, feature_importance))
sorted_feature_importance = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=False)

# 打印每个特征对结果的影响
print("Feature Importance:")
for feature, importance in sorted_feature_importance:
    print(f"{feature}: {importance}")

# 预测
y_pred_train=lgb_model.predict(X_train)
y_pred_test = lgb_model.predict(X_test)


# 设置全局绘图参数
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['axes.labelsize'] = 15
plt.rcParams['axes.titlesize'] = 15
plt.rcParams['font.size'] = 15
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
fpr_train, tpr_train, thresholds_train = roc_curve(y_train, lgb_model.predict_proba(X_train)[:, 1])
roc_auc_train = auc(fpr_train, tpr_train)

# 获取测试集 ROC 曲线数据
fpr_test, tpr_test, thresholds_test = roc_curve(y_test, lgb_model.predict_proba(X_test)[:, 1])
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
    lgb_model, X_train, np.ravel(y_train), cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10)
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
# 分离特征名称和重要性值
features, importances = zip(*sorted_feature_importance)
# 创建横向柱状图，并根据重要性渐变颜色
colors = plt.cm.viridis(np.linspace(1, 0, len(features)))  # 使用viridis色图，你可以根据需要更改色图

plt.barh(range(len(features)), importances, color=colors, edgecolor='black')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.title('Feature Importance (Sorted)',pad=20)
plt.yticks(range(len(features)), features)  # 使用特征名称设置 y 轴标签
plt.tight_layout()  # 优化布局
# 添加垂直网格线
plt.grid(axis='x', linestyle='--', alpha=0.6)

plt.tight_layout()  # 保证标签显示完整
plt.show()