import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# # 指定 Noto 字体路径
font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'  # 确保该路径是正确的
font_prop = fm.FontProperties(fname=font_path)


def is_nearly_purely_imaginary(H, threshold=1e-5):
    """
    判断H的每个元素是否是几乎纯虚数。
    
    参数:
    H: 待判断的复数数组
    threshold: 用于判断的阈值

    返回:
    nearly_purely_imaginary: 布尔数组，指示每个元素是否接近纯虚数
    """
    real_part = np.real(H)
    return np.abs(real_part) < threshold

# 使用示例

def get_Sobel_H(P, Q):
    """
    计算Sobel算子的传递函数 H(u,v).
    
    参数:
    P: 行数 (填充后的图像高度)
    Q: 列数 (填充后的图像宽度)
    
    返回:
    H_x: Sobel x方向的传递函数
    H_y: Sobel y方向的传递函数
    """
    # Sobel算子（水平和垂直部分）
   # 定义Sobel算子，方向为x
    # 注意：这里使用相反的符号是因为，在频域滤波中，Sobel算子的符号与空域中的定义需要一致。
    # 具体来说，负的x方向算子用于提取水平方向的边缘（高亮显示亮度下降的边缘），
    # 将其与空域滤波中的应用保持一致。
    sobel_x = -np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float32)

    # 定义Sobel算子，方向为y
    # 同样，这里也使用相反的符号，目的是确保在频域和空域中的操作效果一致。
    # 负的y方向算子用于提取垂直方向的边缘（高亮显示亮度上升的边缘）。
    sobel_y = -np.array([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]], dtype=np.float32)


    # 创建填充矩阵
    hp_x = np.zeros((P, Q), dtype=np.float32)
    hp_y = np.zeros((P, Q), dtype=np.float32)

    # 将Sobel算子放置到中心
    hp_x[P//2-1:P//2+2, Q//2-1:Q//2+2] = sobel_x
    hp_y[P//2-1:P//2+2, Q//2-1:Q//2+2] = sobel_y

    # 1. 进行频域中心化
    hp_x *= (-1) ** (np.indices(hp_x.shape)[0] + np.indices(hp_x.shape)[1])
    hp_y *= (-1) ** (np.indices(hp_y.shape)[0] + np.indices(hp_y.shape)[1])

    # 2. 计算DFT
    Hc_x = np.fft.fft2(hp_x)
    Hc_y = np.fft.fft2(hp_y)

    # 3. 解中心化
    Hc_x *= (-1) ** (np.indices(Hc_x.shape)[0] + np.indices(Hc_x.shape)[1])
    Hc_y *= (-1) ** (np.indices(Hc_y.shape)[0] + np.indices(Hc_y.shape)[1])

    # 4. 将实部置零
    H_x = np.copy(Hc_x)
    H_y = np.copy(Hc_y)
    # nearly_purely_imaginary_x = is_nearly_purely_imaginary(H_x)
    # nearly_purely_imaginary_y = is_nearly_purely_imaginary(H_y)

    # # 输出结果
    # print("H_x 接近纯虚数的元素：", nearly_purely_imaginary_x)
    # print("H_y 接近纯虚数的元素：", nearly_purely_imaginary_y)

    H_x.real = 0
    H_y.real = 0

    return H_x, H_y

def frequency_domain_filtering(H, image):
    M, N = image.shape

    # 2. 计算填充大小
    P = 2 * M
    Q = 2 * N

    # 3. 创建填充图像，并填充为0
    fp = np.zeros((P, Q), dtype=np.float32)
    fp[0:M, 0:N] = image

    # 4. 进行频域中心化
    fp *= (-1) ** (np.indices(fp.shape)[0] + np.indices(fp.shape)[1])

    # 5. 计算DFT
    F = np.fft.fft2(fp)

    # 7. 计算 G(u,v) = H(u,v) * F(u,v)
    G = H * F

    # 8. 计算逆DFT，得到过滤后的图像
    gp = np.fft.ifft2(G)

    # 9. 中心化逆变换结果
    gp *= (-1) ** (np.indices(gp.shape)[0] + np.indices(gp.shape)[1])

    # 10. 提取与原始图像相同大小的区域
    g = np.real(gp[0:M, 0:N])  # 取实部

    # 11. 归一化处理（0到255）
    g = np.clip(g, 0, 255).astype(np.uint8)

    # 12. 计算频谱的幅度
    F_magnitude = np.abs(F)
    F_magnitude_log = 20 * np.log(1 + F_magnitude)  # 对数变换以增强可视化效果

    # 计算滤波后的频谱幅度
    G_magnitude = np.abs(G)
    G_magnitude_log = 20 * np.log(1 + G_magnitude)  # 对数变换以增强可视化效果

    # 绘制结果
    # plt.figure(figsize=(24, 8))

    # 原始图像
    # plt.subplot(1, 4, 1)
    # plt.imshow(image, cmap='gray')
    # plt.title('原始图像', fontproperties=font_prop)
    # plt.axis('off')

    # # 频谱图像
    # plt.subplot(1, 4, 2)
    # plt.imshow(F_magnitude_log, cmap='gray')
    # plt.title('频谱', fontproperties=font_prop)
    # plt.axis('off')

    # # 滤波器 H
    # plt.subplot(1, 4, 3)
    # plt.imshow(np.abs(H), cmap='gray')
    # plt.title('滤波器 H', fontproperties=font_prop)
    # plt.axis('off')

    # # 频域滤波结果
    # plt.subplot(1, 4, 4)
    # plt.imshow(g, cmap='gray')
    # plt.title('频域滤波后的图像', fontproperties=font_prop)
    # plt.axis('off')

    # plt.show()
    return g

# 使用示例
image = cv2.imread('Q5_1.tif', cv2.IMREAD_GRAYSCALE)
# 计算填充大小
M, N = image.shape
P = 2 * M
Q = 2 * N
H_x, H_y = get_Sobel_H(P, Q)  # 在调用函数前，需要先计算 H_x 和 H_y
frequency_result_x = frequency_domain_filtering(H_x, image)
frequency_result_y = frequency_domain_filtering(H_y, image)
frequency_result = np.sqrt(frequency_result_x**2 + frequency_result_y**2)  # 合并两个方向的结果
# 使用Sobel算子进行卷积
# Sobel x方向的卷积核
sobel_x = np.array([[-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]], dtype=np.float32)

# Sobel y方向的卷积核
sobel_y = np.array([[1, 2, 1],
                     [0, 0, 0],
                     [-1, -2, -1]], dtype=np.float32)


filtered_x = cv2.filter2D(image, -1, sobel_x)
filtered_y = cv2.filter2D(image, -1, sobel_y)
sobel_result = np.sqrt(filtered_x**2 + filtered_y**2)

# 绘制结果对比
plt.figure(figsize=(18, 12))

# 原始图像
plt.subplot(3, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('原始图像', fontproperties=font_prop)
plt.axis('off')

# 频域滤波结果
plt.subplot(3, 3, 2)
plt.imshow(frequency_result_x, cmap='gray')
plt.title('频域滤波结果 (x方向)', fontproperties=font_prop)
plt.axis('off')

plt.subplot(3, 3, 3)
plt.imshow(frequency_result_y, cmap='gray')
plt.title('频域滤波结果 (y方向)', fontproperties=font_prop)
plt.axis('off')

plt.subplot(3, 3, 4)
plt.imshow(frequency_result, cmap='gray')
plt.title('频域滤波合并结果', fontproperties=font_prop)
plt.axis('off')

# 空域卷积结果
plt.subplot(3, 3, 5)
plt.imshow(filtered_x, cmap='gray')
plt.title('空域卷积结果 (x方向)', fontproperties=font_prop)
plt.axis('off')

plt.subplot(3, 3, 6)
plt.imshow(filtered_y, cmap='gray')
plt.title('空域卷积结果 (y方向)', fontproperties=font_prop)
plt.axis('off')

plt.subplot(3, 3, 7)
plt.imshow(sobel_result, cmap='gray')
plt.title('空域卷积合并结果', fontproperties=font_prop)
plt.axis('off')

plt.show()
