import numpy as np
import cv2

# 根据实际情况生成一个示例的 hp_x 属性
hp_x = np.zeros((4, 4), dtype=np.float32)  # 假设的 hp_x 填充矩阵
P, Q = hp_x.shape  # 这里应该是 hp_x 的实际大小

# 打印 np.indices(hc_x.shape)[0] + np.indices(hc_x.shape)[1]
indices_sum = np.indices(hp_x.shape)[0] 
print("Computed Matrix (indices_sum):")
print(indices_sum)
