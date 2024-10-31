import cv2
import numpy as np
import matplotlib.pyplot as plt

def enhance_image(image, blurtype):
    # 根据 blurtype 选择平滑方法
    if blurtype == 'gaussian':
        smoothed_image = cv2.GaussianBlur(image, (5, 5), 1)
    elif blurtype == 'median':
        smoothed_image = cv2.medianBlur(image, 5)
    else:
        smoothed_image = image

    # 创建一个画布来显示所有图像
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    
    # 1. 显示平滑后的图像
    axes[0, 0].imshow(smoothed_image, cmap='gray')
    axes[0, 0].set_title('Smoothed Image')
    axes[0, 0].axis('off')

    # 2. 使用拉普拉斯算子细节增强

   # 定义3x3的拉普拉斯卷积核
    laplacian_kernel = np.array([[1,  1, 1],
                                [1, -8, 1],
                                [1,  1, 1]])

    # 使用卷积核进行卷积
    laplacian = cv2.filter2D(smoothed_image, -1, laplacian_kernel)
    laplacian = cv2.convertScaleAbs(laplacian)  # 转换为8位图像
    # if blurtype == 'median':
    #     laplacian = cv2.medianBlur(laplacian, 5)
    axes[0, 1].imshow(laplacian, cmap='gray')
    axes[0, 1].set_title('Laplacian Image')
    axes[0, 1].axis('off')


    # 3. 使用梯度增强显著边缘
    gradienttype = 'sobel'  # 选择使用的梯度算子

    if gradienttype == 'sobel':
        # Sobel算子
        # 定义Sobel算子的卷积核
        sobel_kernel_x = np.array([[1, 0, -1],
                                    [2, 0, -2],
                                    [1, 0, -1]])

        sobel_kernel_y = np.array([[1, 2, 1],
                                    [0, 0, 0],
                                    [-1, -2, -1]])

        # 使用卷积核进行卷积
        # 使用卷积核进行卷积，并指定边缘处理方式
        sobel_x = cv2.filter2D(smoothed_image, -1, sobel_kernel_x, borderType=cv2.BORDER_DEFAULT).astype(np.float32)
        sobel_y = cv2.filter2D(smoothed_image, -1, sobel_kernel_y, borderType=cv2.BORDER_DEFAULT).astype(np.float32)

        


        # 计算梯度幅值
        gradient = cv2.magnitude(sobel_x, sobel_y)

        title = 'Sobel Gradient Image'


    elif gradienttype == 'prewitt':
        # Prewitt算子
        prewitt_kernel_x = np.array([[1, 0, -1],
                                      [1, 0, -1],
                                      [1, 0, -1]])

        prewitt_kernel_y = np.array([[1, 1, 1],
                                      [0, 0, 0],
                                      [-1, -1, -1]])

        prewitt_x = cv2.filter2D(smoothed_image, -1, prewitt_kernel_x)
        prewitt_y = cv2.filter2D(smoothed_image, -1, prewitt_kernel_y)
        gradient = cv2.magnitude(prewitt_x, prewitt_y)
        title = 'Prewitt Gradient Image'

    elif gradienttype == 'roberts':
        # Roberts算子
        roberts_kernel_x = np.array([[1, 0],
                                      [0, -1]])

        roberts_kernel_y = np.array([[0, 1],
                                      [-1, 0]])

        roberts_x = cv2.filter2D(smoothed_image, -1, roberts_kernel_x)
        roberts_y = cv2.filter2D(smoothed_image, -1, roberts_kernel_y)
        gradient = cv2.magnitude(roberts_x, roberts_y)
        title = 'Roberts Gradient Image'
    else:
        raise ValueError("Invalid gradient type. Choose 'sobel', 'prewitt', or 'roberts'.")

    # 转换为8位图像并显示
    gradient = cv2.convertScaleAbs(gradient)
    axes[0, 2].imshow(gradient, cmap='gray')
    axes[0, 2].set_title(title)
    axes[0,2].axis('off')

    # 显示图像
    

# 调用函数并选择算子类型
# 假设smoothed_image已经定义好



    # 4. 将拉普拉斯和梯度组合
    enhanced_details = cv2.addWeighted(laplacian, 0.9, gradient, 0.1, 0)
    axes[1, 0].imshow(enhanced_details, cmap='gray')
    axes[1, 0].set_title('Enhanced Details')
    axes[1, 0].axis('off')

    # 5. 将平滑图像与增强细节图像进行融合，获得锐化图像
    sharpened_image = cv2.addWeighted(image, 1, enhanced_details, 0.5, 0)
    axes[1, 1].imshow(sharpened_image, cmap='gray')
    axes[1, 1].set_title('Sharpened Image')
    axes[1, 1].axis('off')

    # 6. 进行直方图均衡化增强对比度
    contrast_enhanced_image = cv2.equalizeHist(sharpened_image.astype(np.uint8))
    axes[1, 2].imshow(contrast_enhanced_image, cmap='gray')
    axes[1, 2].set_title('Contrast Enhanced Image (Histogram Equalization)')
    axes[1, 2].axis('off')

    # 显示原始图像
    axes[2, 0].imshow(image, cmap='gray')
    axes[2, 0].set_title('Original Image')
    axes[2, 0].axis('off')

    # 显示最终增强后的图像
    axes[2, 1].imshow(contrast_enhanced_image, cmap='gray')
    axes[2, 1].set_title('Final Enhanced Image')
    axes[2, 1].axis('off')

    plt.tight_layout()
    plt.show()

    return contrast_enhanced_image

# 读取图像
image1 = cv2.imread('Q4_1.tif', cv2.IMREAD_GRAYSCALE)  # 第一张图片
image2 = cv2.imread('Q4_2.tif', cv2.IMREAD_GRAYSCALE)  # 第二张图片

# 处理两个图像，指定不同类型的模糊
enhanced_image1 = enhance_image(image1, blurtype='no')
enhanced_image2 = enhance_image(image2, blurtype='median')

# 保存处理后的结果
cv2.imwrite('Enhanced_Q4_1.tif', enhanced_image1)
cv2.imwrite('Enhanced_Q4_2.tif', enhanced_image2)
