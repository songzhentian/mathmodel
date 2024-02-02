from matplotlib import pyplot as plt  # 作图
from sklearn.datasets import make_classification  # 数据集生成
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # LDA
from sklearn.linear_model import LogisticRegression  # 逻辑回归
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split, KFold  # 分割数据z
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC  # 支持向量机
import draw

# 生成数据集
X, Y = make_classification(n_samples=300, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, n_classes=2,
                           n_clusters_per_class=1, random_state=4)

# 分割为训练集和测试集
cla_name = ['0', '1']
kf = KFold(n_splits=5)
i = 1
for train, test in kf.split(X, Y):
    # 朴素贝叶斯-高斯
    # 调用分类器
    nb_clf = GaussianNB()
    # 拟合模型
    nb_model = nb_clf.fit(X[train], Y[train])
    # 预测
    y_pre = nb_model.predict(X[test])
    # 计算预测概率
    y_scores = nb_model.predict_proba(X[test])[:, 1]
    # 计算ROC曲线
    fpr, tpr, thresholds = roc_curve(Y[test], y_scores)
    roc_auc = auc(fpr, tpr)
    print("Naive_Bayes-Gaussian\n")
    # 打印分类报告
    print(classification_report(Y[test], y_pre, target_names=cla_name))
    # 绘制ROC曲线
    draw.plot_roc(fpr, tpr, roc_auc, "Naive_Bayes-Gaussian_Fold-%d" % i)
    # 绘制PR曲线
    draw.plot_pr_curve(Y[test], y_scores, "Naive_Bayes-Gaussian_Fold-%d" % i)
    # 支持向量机
    svm = LinearSVC()
    svm.fit(X[train], Y[train])
    y_pre = svm.predict(X[test])
    y_scores = svm._predict_proba_lr(X[test])[:, 1]
    fpr, tpr, thresholds = roc_curve(Y[test], y_scores)
    roc_auc = auc(fpr, tpr)
    # 绘制ROC曲线
    draw.plot_roc(fpr, tpr, roc_auc, "LinerSVC_Fold-%d" % i)
    # 绘制决策边界
    draw.plot_decision_boundary(X, Y, train, test, svm, "LinerSVC_Fold-%d" % i)
    print("LinerSVC\n")
    # 打印分类报告
    print(classification_report(Y[test], y_pre, target_names=cla_name))
    # 绘制PR曲线
    draw.plot_pr_curve(Y[test], y_scores, "LinerSVC_Fold-%d" % i)

    # 逻辑分类器
    Logist = LogisticRegression()
    Logist.fit(X[train], Y[train])
    y_pre = svm.predict(X[test])
    y_scores = Logist.predict_proba(X[test])[:, 1]
    fpr, tpr, thresholds = roc_curve(Y[test], y_scores)
    roc_auc = auc(fpr, tpr)
    # 绘制ROC曲线
    draw.plot_roc(fpr, tpr, roc_auc, "Logist_Fold-%d" % i)
    # 绘制决策边界
    draw.plot_decision_boundary(X, Y, train, test, Logist, "Logist_Fold-%d" % i)
    print("Logist\n")
    # 打印分类报告
    print(classification_report(Y[test], y_pre, target_names=cla_name))
    # 绘制PR曲线
    draw.plot_pr_curve(Y[test], y_scores, "Logist_Fold-%d" % i)

    # 线性判别分析
    LDA = LinearDiscriminantAnalysis()
    LDA.fit(X[train], Y[train])
    y_pre = LDA.predict(X[test])
    y_scores = LDA.predict_proba(X[test])[:, 1]
    fpr, tpr, thresholds = roc_curve(Y[test], y_scores)
    roc_auc = auc(fpr, tpr)
    # 绘制ROC曲线
    draw.plot_roc(fpr, tpr, roc_auc, "LDA_Fold-%d" % i)
    # 绘制决策边界
    draw.plot_decision_boundary(X, Y, train, test, LDA, "LDA_Fold-%d" % i)
    print("LDA\n")
    # 打印分类报告
    print(classification_report(Y[test], y_pre, target_names=cla_name))
    # 绘制PR曲线
    draw.plot_pr_curve(Y[test], y_scores, "LDA_Fold-%d" % i)

    # 绘制所有决策边界
    models = [svm, Logist, LDA]
    draw.plot_decision_boundaries(X, Y, train, test, models, i)
    i += 1
plt.close("all")
print(1)
