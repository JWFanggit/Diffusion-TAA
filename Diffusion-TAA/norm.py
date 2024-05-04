import numpy as np
# 假设视频数据为x
# 计算每个通道的均值和标准差
mean = np.mean(x, axis=(1,2,3), keepdims=True)
std = np.std(x, axis=(1,2,3), keepdims=True)
# 对每个通道进行归一化
x_normalized = (x - mean) / std