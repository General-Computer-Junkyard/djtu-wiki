import tensorflow as tf
import numpy as np
import os

print("TensorFlow version:", tf.__version__)
print("NumPy version:", np.__version__)

# 加载 MNIST 数据集
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 数据预处理
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# 构建一个更简单的模型
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 编译模型
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 训练模型（减少训练轮数）
model.fit(
    x_train, y_train,
    epochs=3,
    batch_size=32,
    validation_data=(x_test, y_test)
)

# 创建保存目录
os.makedirs('mnist_model', exist_ok=True)

# 保存为 Keras H5 格式
model.save('mnist_model/model.h5')
print("Model saved successfully!")


