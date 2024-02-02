import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve, auc


def plot_data(x_train, y_train, x_test, y_test):
    """
    绘制2D坐标
    Args:
        x_train: 训练数据的特征
        y_train: 训练数据的标签
        x_test: 测试数据的特征
        y_test: 测试数据的标签
    Returns:
        绘制好的图像
    """
    # 创建一个10x8的图像
    plt.figure(figsize=(10, 8))
    # 获取y为1和0的索引
    pos_train = np.where(y_train == 1)
    neg_train = np.where(y_train == 0)
    pos_test = np.where(y_test == 1)
    neg_test = np.where(y_test == 0)
    # 绘制标签为y=1的数据点
    plt.plot(np.ravel(x_train[pos_train, 0]), np.ravel(x_train[pos_train, 1]), 'ro', markersize=4, label="y_train=1")
    # 绘制标签为y=0的数据点
    plt.plot(np.ravel(x_train[neg_train, 0]), np.ravel(x_train[neg_train, 1]), 'g^', markersize=4, label="y_train=0")
    # 绘制标签为y=1的数据点
    plt.plot(np.ravel(x_test[pos_test, 0]), np.ravel(x_test[pos_test, 1]), 'co', markersize=4, label="y_test=1")
    # 绘制标签为y=0的数据点
    plt.plot(np.ravel(x_test[neg_test, 0]), np.ravel(x_test[neg_test, 1]), 'y^', markersize=4, label="y_test=0")
    # 设置x和y轴标签
    plt.xlabel("X1")
    plt.ylabel("X2")
    return plt


def plot_decision_boundary(x, y, train, test, model, ax=None):
    """
    决策边界可视化函数
    Args:
        x: 输入特征
        y: 标签
        train: 训练集
        test: 测试集
        model: 训练好的模型
        ax: 可选的轴对象

    Returns:
        plt: 绘制的图像对象
    """
    plt = plot_data(x[train], y[train], x[test], y[test])  # 绘制数据点
    w = model.coef_  # 决策函数中的特征系数
    b = model.intercept_  # 决策函数中的截距
    print('特征系数: ', w)
    print('截距: ', b)

    # 生成x轴坐标点
    xp = np.linspace(np.min(x[:, 0]), np.max(x[:, 0]), 100)
    yp = -(w[0, 0] * xp + b) / w[0, 1]  # 计算y轴坐标点

    # 绘制决策边界
    plt.plot(xp, yp, color='royalblue', linewidth=3.0, label="clf")
    plt.title(ax)
    plt.legend()
    plt.show(block=False)
    plt.savefig("./pics/%s_decision_boundary.png" % ax, dpi=600)
    return plt


def plot_decision_boundaries(x, y, train, test, models, id=None):
    """
    绘制多个决策边界在一张图内
    Args:
        x: 输入特征
        y: 标签
        train: 训练集
        test: 测试集
        models: 训练好的模型列表
    Returns:
        plt: 绘制的图像对象
    """
    plt = plot_data(x[train], y[train], x[test], y[test])  # 绘制数据点

    # 遍历每个模型
    for model in models:
        w = model.coef_  # 决策函数中的特征系数
        b = model.intercept_  # 决策函数中的截距
        print('特征系数: ', w)
        print('截距: ', b)
        # 生成x轴坐标点
        xp = np.linspace(np.min(x[:, 0]), np.max(x[:, 0]), 100)
        yp = -(w[0, 0] * xp + b) / w[0, 1]  # 计算y轴坐标点
        # 绘制决策边界
        plt.plot(xp, yp, linewidth=1.5, label=model.__class__.__name__)

    plt.show(block=False)
    plt.title("Decision Boundaries Fold %d" % id)
    plt.legend()
    plt.savefig("./pics/decision_boundaries_Fold-%d.png" % id, dpi=600)
    return plt


def plot_roc(fp, tp, a, name=None):
    """
    绘制ROC曲线
    Args:
        fp: 假正率
        tp: 真正率
        a: 曲线下面积
        name: 曲线名称

    Returns:
        绘制的ROC曲线
    """
    plt.figure(figsize=(10, 8))
    plt.plot(fp, tp, label='ROC curve (area = %0.2f)' % a)
    plt.plot([0, 1], [0, 1], 'k--')  # 绘制对角线
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic %s' % name)
    plt.legend(loc="lower right")
    plt.show(block=False)
    plt.savefig("./pics/%s_ROC.png" % name, dpi=600)
    return plt


def plot_pr_curve(y_true, y_scores, name=None):
    """
    绘制PR曲线
    Args:
        y_true: 真实标签
        y_scores: 预测得分
        name:  曲线名称
    Returns:
        plt: 绘制的图像对象
    """
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, label='PR curve (area = %0.2f)' % pr_auc)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve %s' % name)
    plt.legend(loc="lower right")
    plt.show(block=False)
    plt.savefig("./pics/%s_PR.png" % name, dpi=600)

    return plt
