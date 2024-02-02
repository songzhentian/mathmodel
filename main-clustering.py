import numpy as np
import sklearn.metrics
from matplotlib import pyplot as plt  # 作图
from mpl_toolkits.mplot3d import Axes3D  # 3D作图
from sklearn.datasets import make_blobs  # 聚类数据集生成
from sklearn.decomposition import PCA  # PCA降维
from sklearn.cluster import KMeans  # Kmeans模型
from sklearn.cluster import DBSCAN  # DBSCAN模型


def kmean_data(x_rd, y_true, k, f):
    k_mean_model = KMeans(n_clusters=k,  # 聚类数目
                          init='random',  # 随机初始向量
                          n_init=1).fit(x_rd)  # 运行1次取最好结果
    markers = ['o', '^', '+', 's', '>', '*', ]
    y_predict = k_mean_model.labels_  # 预测
    clu_centers = k_mean_model.cluster_centers_  # 聚类中心
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_axes(Axes3D(fig))
    for i in range(0, k):
        class_temp = np.where(y_predict == i)
        ax.scatter(np.ravel(x_rd[class_temp, 0]), np.ravel(x_rd[class_temp, 1]), np.ravel(x_rd[class_temp, 2]), s=25,
                   c=plt.cm.Set1(i), marker=markers[i], label="y=%d" % i)
    ax.scatter(clu_centers[:, 0], clu_centers[:, 1], clu_centers[:, 2], s=50, c='k', marker='x', label="Centers")
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_zlabel("X3")
    ax.legend()
    plt.show(block=False)
    plt.savefig("./pics/KMeans/k_%d.png" % k, dpi=600)
    f.write("K=%d\n" % k)
    print('FMI value : ', sklearn.metrics.fowlkes_mallows_score(y_true, y_predict))
    f.write('FMI value : %s\n' % sklearn.metrics.fowlkes_mallows_score(y_true, y_predict))
    print('PI value : ', sklearn.metrics.rand_score(y_true, y_predict))
    f.write('PI value : %s\n' % sklearn.metrics.rand_score(y_true, y_predict))
    print('DB value : ', sklearn.metrics.davies_bouldin_score(x_rd, y_predict))
    f.write('DB value : %s\n' % sklearn.metrics.davies_bouldin_score(x_rd, y_predict))


def draw_DBSCAN(x_rd, y_true, e, mini_sample, f):
    DBSCAN_model = DBSCAN(eps=e, min_samples=mini_sample, metric='euclidean').fit(x_rd)
    fig = plt.figure(figsize=(10, 8))
    markers = ['o', '^', '+', 's', '>', '*', ]
    colors = [plt.cm.Set1(0), plt.cm.Set1(1), plt.cm.Set1(2), plt.cm.Set1(3), plt.cm.Set1(4), plt.cm.Set1(5),
              plt.cm.Set1(6), plt.cm.Set1(7)]
    ax = fig.add_axes(Axes3D(fig))
    y_predict = DBSCAN_model.labels_
    y_s = np.unique(np.array(y_predict))
    y_s.sort()
    for i in y_s:
        class_temp = np.where(y_predict == i)
        ax.scatter(np.ravel(x_rd[class_temp, 0]), np.ravel(x_rd[class_temp, 1]), np.ravel(x_rd[class_temp, 2]), s=25,
                   c=colors[i], marker=markers[i], label="y=%d" % i)
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_zlabel("X3")
    ax.legend()
    plt.show(block=False)
    plt.savefig("./pics/DBSCAN/eps_%d_samples_%d.png" % (e, mini_sample), dpi=600)
    print('FMI value : ', sklearn.metrics.fowlkes_mallows_score(y_true, y_predict))
    print('PI value : ', sklearn.metrics.rand_score(y_true, y_predict))
    print('DB value : ', sklearn.metrics.davies_bouldin_score(x_rd, y_predict))
    f.write("eps=%d mini_samples=%d\n" % (e, mini_sample))
    f.write('FMI value : %s\n' % sklearn.metrics.fowlkes_mallows_score(y_true, y_predict))
    f.write('PI value : %s\n' % sklearn.metrics.rand_score(y_true, y_predict))
    f.write('DB value : %s\n' % sklearn.metrics.davies_bouldin_score(x_rd, y_predict))


X, y = make_blobs(n_samples=500,  # 样本个数
                  n_features=8,  # 特征个数
                  centers=3,  # 类别中心数目
                  cluster_std=4.0,  # 类别标准差
                  random_state=3  # 随机种子)
                  )

pca = PCA(n_components=3)
X_rd = pca.fit_transform(X)

fig1 = plt.figure(figsize=(10, 8))
ax1 = fig1.add_axes(Axes3D(fig1))
x1 = X_rd[:, 0]
x2 = X_rd[:, 1]
x3 = X_rd[:, 2]
ax1.scatter(x1, x2, x3, s=25, c='m', marker='o')
ax1.set_xlabel("X1")
ax1.set_ylabel("X2")
ax1.set_zlabel("X3")
plt.show(block=False)
plt.savefig("./pics/result.png", dpi=600)

file = open("results.txt", 'w')
for i in range(2, 6):
    kmean_data(X_rd, y, i, file)
plt.close("all")
for i in range(3, 8):
    for j in range(5, 16):
        draw_DBSCAN(X_rd, y, i, j, file)
    plt.close("all")
