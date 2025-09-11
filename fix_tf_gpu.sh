#!/bin/bash
# 修复 TensorFlow GPU 环境 (适配 RTX 4090 + Arch Linux + conda)

set -e

echo ">>> 激活 conda 环境: superpoint"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate superpoint

echo ">>> 卸载旧的 CUDA 包..."
conda remove --force cuda cuda-* -y || true
conda remove --force cudnn -y || true

echo ">>> 安装 TensorFlow 2.15 (支持 CUDA 12.x)..."
pip install --upgrade pip
pip install --upgrade tensorflow==2.15.*

echo ">>> 安装 CUDA 12.2 runtime + cuDNN (兼容 4090)"
conda install -c nvidia cuda=12.2 cudnn=8.9 -y

echo ">>> 配置环境变量"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

echo ">>> 测试 GPU 是否可见"
python - <<EOF
import tensorflow as tf
print("TensorFlow:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))
EOF

echo ">>> 完成！现在你可以用 GPU 加速 SuperPoint 了 🚀"
