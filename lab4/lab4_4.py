import numpy as np
import cv2
import matplotlib.pyplot as plt

def butterworth_notch_filter(image, D0, n, center):
    """
    实现巴特沃斯带阻滤波器
    :param image: 输入图像
    :param D0: 截止频率
    :param n: 滤波器阶数
    :param center: 中心坐标 (u0, v0)，确定带阻中心
    :return: 过滤后的图像和频域滤波器H
    """
    M, N = image.shape
    F = np.fft.fft2(image)
    F_shifted = np.fft.fftshift(F)

    # 创建频率坐标
    u = np.linspace(0, N-1, N)   # 从0到N-1
    v = np.linspace(0, M-1, M)   # 从0到M-1
    U, V = np.meshgrid(u, v)

    # 计算到中心的距离 D
    D_k = np.sqrt((U - N/2 - center[0])**2 + (V - M/2 - center[1])**2)
    D_kk = np.sqrt((U - N/2 + center[0])**2 + (V - M/2 + center[1])**2)

    # 巴特沃斯带阻过滤器的公式
    H = 1 / (1 + (D0 / D_k)**(2*n)) * (1 / (1 + (D0 / D_kk)**(2*n)))
    H[np.isnan(H)] = 0  # 处理 D 为 0 的情况

    # 应用滤波器
    G = H * F_shifted

    # 计算逆DFT并取实部
    gp = np.fft.ifftshift(G)
    filtered_image = np.fft.ifft2(gp)
    filtered_image = np.real(filtered_image)

    # 归一化处理以适应图像格式
    filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8)

    return filtered_image, H, F_shifted

# 读取输入图像
image = cv2.imread('Q5_3.tif', cv2.IMREAD_GRAYSCALE)
M, N = image.shape
# 设置参数
D0 = 30  # 截止频率
n = 2    # 滤波器阶数
notch_centers = [( 30, 0), (-30, 0)]  # 中心频率位置

# 创建图形
plt.figure(figsize=(18, 12))

# 对不同的带阻中心进行过滤
for i, center in enumerate(notch_centers):
    filtered_image, H, F_shifted = butterworth_notch_filter(image, D0, n, center)
    
    # 显示处理后的图像
    plt.subplot(3, len(notch_centers), i + 1)
    plt.imshow(filtered_image, cmap='gray')
    plt.title(f'处理后图像 (中心: {center})', fontsize=12)
    plt.axis('off')

    # 显示滤波器的频谱
    plt.subplot(3, len(notch_centers), i + len(notch_centers) + 1)
    plt.imshow(np.log1p(np.abs(H)), cmap='gray')  # 使用对数变换以增强可视化效果
    plt.title('滤波器 H 的频谱', fontsize=12)
    plt.axis('off')

    # 显示输入图像的频谱
    plt.subplot(3, len(notch_centers), i + 2 * len(notch_centers) + 1)
    plt.imshow(np.log1p(np.abs(F_shifted)), cmap='gray')  # 使用对数变换以增强可视化效果
    plt.title('原图的频谱', fontsize=12)
    plt.axis('off')

# 显示图形
plt.tight_layout()
plt.show()
