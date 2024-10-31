import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 指定字体路径
font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
font_prop = font_manager.FontProperties(fname=font_path)
def ideal_lowpass_filter(image, D0):
    """
    实现理想低通滤波器
    :param image: 输入图像
    :param D0: 截止频率
    :return: 过滤后的图像
    """
    # 获取图像的大小
    M, N = image.shape
    # 进行DFT并中心化
    F = np.fft.fft2(image)
    F_shifted = np.fft.fftshift(F)

    # 创建低通滤波器
    # 生成网格坐标
    x = np.linspace(-N/2, N/2-1, N)
    y = np.linspace(-M/2, M/2-1, M)
    X, Y = np.meshgrid(x, y)

    # 计算每个点到中心的距离
    D = np.sqrt(X**2 + Y**2)
    # 创建理想低通滤波器
    H = np.zeros_like(D)
    H[D <= D0] = 1  # 在截止频率内的部分为1，其余部分为0

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
D0_values = [10, 30, 60, 160, 460]

# 绘制结果
plt.figure(figsize=(18, 12))

# 对不同的截止频率进行过滤
for i, D0 in enumerate(D0_values):
    filtered_image = ideal_lowpass_filter(image, D0)
    
    # 绘制结果
    plt.subplot(2, 3, i + 1)
    plt.imshow(filtered_image, cmap='gray')
    plt.title(f'理想低通滤波器 D0 = {D0}', fontsize=12)
    plt.axis('off')

# 打开原始图像的子图
plt.subplot(2, 3, len(D0_values) + 1)
plt.imshow(image, cmap='gray')
plt.title('原始图像', fontsize=12)
plt.axis('off')

plt.tight_layout()
plt.show()
