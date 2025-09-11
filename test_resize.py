import tensorflow as tf
import numpy as np

print("TensorFlow version:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))

# 构造一个随机图像 [batch, height, width, channels]
x = np.random.rand(1, 512, 512, 3).astype(np.float32)

# 转换成 Tensor
x_tensor = tf.convert_to_tensor(x)

# 测试 GPU 上的 resize
with tf.device("/GPU:0"):  # 强制用 GPU 0
    try:
        y = tf.image.resize(x_tensor, [256, 256], method="bilinear")
        print("ResizeBilinear on GPU success, output shape:", y.shape)
    except Exception as e:
        print("ResizeBilinear on GPU FAILED:", e)

# 再测试 CPU 上的 resize
with tf.device("/CPU:0"):
    y_cpu = tf.image.resize(x_tensor, [256, 256], method="bilinear")
    print("ResizeBilinear on CPU success, output shape:", y_cpu.shape)
