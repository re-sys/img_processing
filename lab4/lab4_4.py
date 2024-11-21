import numpy as np
import cv2
import matplotlib.pyplot as plt

def butterworth_notch_filter(image, D0, n, notch_centers):
    M, N = image.shape
    F = np.fft.fft2(image)
    F_shifted = np.fft.fftshift(F)

    u = np.linspace(0, N-1, N)
    v = np.linspace(0, M-1, M)
    U, V = np.meshgrid(u, v)

    H_total = np.ones((M, N))

    for center in notch_centers:
        D_k = np.sqrt((U - N / 2 - center[0])**2 + (V - M / 2 - center[1])**2) + 1e-10
        D_kk = np.sqrt((U - N / 2 + center[0])**2 + (V - M / 2 + center[1])**2) + 1e-10

        H = 1 / (1 + (D0 / D_k)**(2 * n)) * (1 / (1 + (D0 / D_kk)**(2 * n)))
        H[np.isnan(H)] = 0
        H_total *= H

    G = H_total * F_shifted

    gp = np.fft.ifftshift(G)
    filtered_image = np.fft.ifft2(gp)
    filtered_image = np.real(filtered_image)
    filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8)

    return filtered_image, H_total, F_shifted, G

# 读取输入图像
image = cv2.imread('Q5_3.tif', cv2.IMREAD_GRAYSCALE)
M, N = image.shape

D0 = 20
n = 2
notch_centers = [(29, 40), (-26.5, 40), (27, 78), (-27, 83)]

# 执行滤波
filtered_image, H_total, F_shifted, G = butterworth_notch_filter(image, D0, n, notch_centers)

# 删除中心周围的值
center_x, center_y = N // 2, M // 2
radius = 10
Y, X = np.ogrid[:M, :N]
mask = (X - center_x)**2 + (Y - center_y)**2 <= radius**2
F_shifted_masked = np.copy(F_shifted)
F_shifted_masked[mask] = 0

# 找到最大的10个值的坐标
top_indices = np.argsort(np.abs(F_shifted_masked).flatten())[-50:]  # 找到前50个值的索引
top_coordinates = np.unravel_index(top_indices, F_shifted_masked.shape)

# 使用集合来存储有效的最大值（忽略相近的点）
valid_points = []

# 设置最小距离
min_distance = 10  # 定义最小距离

for i in range(len(top_coordinates[0])):
    current_point = (top_coordinates[0][i], top_coordinates[1][i])
    if all(np.linalg.norm(np.array(current_point) - np.array(point)) >= min_distance for point in valid_points):
        valid_points.append(current_point)

# 创建可视化图像并归一化到0-255
F_shifted_vis = np.log1p(np.abs(F_shifted))
F_shifted_vis = (F_shifted_vis / F_shifted_vis.max() * 255).astype(np.uint8)

# 用 plt 画圈
plt.figure(figsize=(12, 12))

# 显示原图
plt.subplot(2, 2, 1)  
plt.imshow(image, cmap='gray')
plt.title('原始图像', fontsize=12)
plt.axis('off')

# 显示F_shifted并标记有效最大值
plt.subplot(2, 2, 2)
plt.imshow(F_shifted_vis, cmap='gray')
for max_y, max_x in valid_points:
    circle = plt.Circle((max_x, max_y), 5, color='red', fill=False, linewidth=2)
    plt.gca().add_patch(circle)
plt.title('F_shifted 及有效最大值', fontsize=12)
plt.axis('off')

# 显示过滤后的图像
plt.subplot(2, 2, 3)
plt.imshow(filtered_image, cmap='gray')
plt.title('过滤后的图像', fontsize=12)
plt.axis('off')

# 显示滤波后的频谱
plt.subplot(2, 2, 4)  # 添加过滤后的频谱的子图
G_vis = np.log1p(np.abs(G))  # 对滤波后的频谱取对数
G_vis = (G_vis / G_vis.max() * 255).astype(np.uint8)  # 归一化到0-255
plt.imshow(G_vis, cmap='gray')
plt.title('滤波后的频谱', fontsize=12)
plt.axis('off')

plt.tight_layout()
plt.show()
