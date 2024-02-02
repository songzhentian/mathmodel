import tensorflow as tf
import pandas as pd
import os
import matplotlib.pyplot as plt

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

# 定义神经网络模型
def neural_network(input_size, hidden_layer_sizes, output_size):
    inputs = tf.keras.Input(shape=(input_size,))
    x = inputs
    for size in hidden_layer_sizes:
        x = tf.keras.layers.Dense(size, activation='relu')(x)
    outputs = tf.keras.layers.Dense(output_size)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# 训练模型
def train(data_dicts):
    checkpoint_dir = 'D:/MCM\code/training_checkpoints/'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 定义神经网络的参数
    input_size = 6
    hidden_layer_sizes = [10, 20, 10]
    output_size = 1

    # 构建模型
    model = neural_network(input_size, hidden_layer_sizes, output_size)

    # 定义损失函数和优化器
    loss_fn = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    # 创建一个检查点对象
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

    # 创建一个检查点管理器（用于管理检查点）
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5)

    num_epochs = 10
    for epoch in range(num_epochs):
        for dataset in data_dicts:
            X, Y = dataset['X'], dataset['Y']
            with tf.GradientTape() as tape:
                predictions = model(X)
                loss = loss_fn(Y, predictions)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            print("Epoch:", epoch, "Loss:", loss.numpy())
        # 每个 epoch 结束后保存模型参数
        checkpoint_manager.save(checkpoint_number=epoch)


def predict(X_data, model_weight_path):
    # 定义模型结构
    model = neural_network(input_size=6, hidden_layer_sizes=[10, 20, 10], output_size=1)

    # 加载模型权重
    model.load_weights(model_weight_path)

    # 进行预测
    predictions = model.predict(X_data)

    return predictions

# 主函数
if __name__ == "__main__":
    train_path = 'D:\\MCM\\code\\train_data'
    train_Data = read_csv_folder(train_path)
    train(train_Data)
    #model_weight_path = 'D:/MCM/code/training_checkpoints/'
    #predictions = predict(X_test, model_weight_path)
    # 打印预测结果
    #print(predictions)
