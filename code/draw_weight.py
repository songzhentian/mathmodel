import tensorflow as tf
import numpy as np
import network

# 加载模型
model = network.neural_network(input_size=6, hidden_layer_sizes=[10, 20, 10], output_size=1)

# 加载最后一次训练得到的权重文件
checkpoint_dir = 'D:/MCM/code/training_checkpoints/'
latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
checkpoint = tf.train.Checkpoint(model=model)
checkpoint.restore(latest_checkpoint)

# 获取输入层和第一个隐藏层之间的权重
input_hidden_weights = model.layers[1].get_weights()[0]

# 对权重矩阵的每一行进行求和
input_hidden_weights_sum = np.sum(input_hidden_weights, axis=1, keepdims=True)

# 输出输入层和第一个隐藏层之间的权重的行求和结果
print("Input to Hidden Weights (Row Sums):")
print(input_hidden_weights_sum)
