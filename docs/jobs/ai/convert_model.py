import tensorflowjs as tfjs
import tensorflow as tf
import os

print("TensorFlow.js version:", tfjs.__version__)

# 加载 H5 模型
model = tf.keras.models.load_model('mnist_model/model.h5')

# 确保输出目录存在
os.makedirs('public/model', exist_ok=True)

# 转换为 TensorFlow.js 格式
tfjs.converters.save_keras_model(model, 'public/model')
print("Model converted successfully!")
