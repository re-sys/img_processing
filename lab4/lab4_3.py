import numpy as np
import cv2
import matplotlib.pyplot as plt

def gaussian_lowpass_filter(image, D0):
    """
    实现高斯低通滤波器
    :param image: 输入图像
    :param D0: 截止频率
    :return: 过滤后的图像
    """
    M, N = image.shape
    F = np.fft.fft2(image)
    F_shifted = np.fft.fftshift(F)

    # 创建高斯低通滤波器
    # 创建从零开始的频率坐标
    u = np.linspace(0, N-1, N)  # 从0到N-1, 然后移到中心
    v = np.linspace(0, M-1, M)   # 从0到M-1, 然后移到中心
    U, V = np.meshgrid(u, v)

    # 计算到中心的距离 D
    D = np.sqrt((U-N/2)**2 + (V-M/2)**2)

    H = np.exp(-(D**2) / (2 * (D0**2)))

    # 应用滤波器
    G = H * F_shifted

    # 计算逆DFT并取实部
    gp = np.fft.ifftshift(G)
    filtered_image = np.fft.ifft2(gp)
    filtered_image = np.real(filtered_image)

    # 归一化处理以适应图像格式
    filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8)

    return filtered_image

def gaussian_highpass_filter(image, D0):
    """
    实现高斯高通滤波器
    :param image: 输入图像
    :param D0: 截止频率
    :return: 过滤后的图像
    """
    M, N = image.shape
    F = np.fft.fft2(image)
    F_shifted = np.fft.fftshift(F)

    # 创建高斯高通滤波器
    # 创建从零开始的频率坐标
    u = np.linspace(0, N-1, N) - N/2  # 从0到N-1, 然后移到中心
    v = np.linspace(0, M-1, M) - M/2  # 从0到M-1, 然后移到中心
    U, V = np.meshgrid(u, v)

    # 计算到中心的距离 D
    D = np.sqrt((U)**2 + (V)**2)

    H = 1 - np.exp(-(D**2) / (2 * (D0**2)))  # 高通滤波器为1减去低通

    # 应用滤波器
    G = H * F_shifted

    # 计算逆DFT并取实部
    gp = np.fft.ifftshift(G)
    filtered_image = np.fft.ifft2(gp)
    filtered_image = np.real(filtered_image)

    # 归一化处理以适应图像格式
    filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8)

    return filtered_image

# 读取输入图像
image = cv2.imread('Q5_2.tif', cv2.IMREAD_GRAYSCALE)

# 设置截止频率
D0_values = [30, 60, 160]

# 创建图形
plt.figure(figsize=(18, 12))

# 对不同的截止频率进行高斯低通滤波
for i, D0 in enumerate(D0_values):
    filtered_image_lp = gaussian_lowpass_filter(image, D0)

    # 绘制结果
    plt.subplot(3, 2, 2 * i + 1)
    plt.imshow(filtered_image_lp, cmap='gray')
    plt.title(f'高斯低通滤波 (D0 = {D0})', fontsize=12)
    plt.axis('off')

# 对不同的截止频率进行高斯高通滤波
for i, D0 in enumerate(D0_values):
    filtered_image_hp = gaussian_highpass_filter(image, D0)

    # 绘制结果
    plt.subplot(3, 2, 2 * i + 2)
    plt.imshow(filtered_image_hp, cmap='gray')
    plt.title(f'高斯高通滤波 (D0 = {D0})', fontsize=12)
    plt.axis('off')

# 显示图形
plt.tight_layout()
plt.show()
