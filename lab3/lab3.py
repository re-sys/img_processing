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
        raise ValueError("Invalid blur type. Choose 'gaussian' or 'median'.")

    # 创建一个画布来显示所有图像
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    
    # 1. 显示平滑后的图像
    axes[0, 0].imshow(smoothed_image, cmap='gray')
    axes[0, 0].set_title('Smoothed Image')
    axes[0, 0].axis('off')

    # 2. 使用拉普拉斯算子细节增强
    laplacian = cv2.Laplacian(smoothed_image, cv2.CV_64F)
    laplacian = cv2.convertScaleAbs(laplacian)  # 转换为8位图像
    axes[0, 1].imshow(laplacian, cmap='gray')
    axes[0, 1].set_title('Laplacian Image')
    axes[0, 1].axis('off')

    # 3. 使用梯度增强显著边缘
    sobel_x = cv2.Sobel(smoothed_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(smoothed_image, cv2.CV_64F, 0, 1, ksize=3)
    gradient = cv2.magnitude(sobel_x, sobel_y)
    gradient = cv2.convertScaleAbs(gradient)  # 转换为8位图像
    axes[0, 2].imshow(gradient, cmap='gray')
    axes[0, 2].set_title('Gradient Image')
    axes[0, 2].axis('off')

    # 4. 将拉普拉斯和梯度组合
    enhanced_details = cv2.addWeighted(laplacian, 0.5, gradient, 0.5, 0)
    axes[1, 0].imshow(enhanced_details, cmap='gray')
    axes[1, 0].set_title('Enhanced Details')
    axes[1, 0].axis('off')

    # 5. 将平滑图像与增强细节图像进行融合，获得锐化图像
    sharpened_image = cv2.addWeighted(smoothed_image, 1, enhanced_details, 0.5, 0)
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
enhanced_image1 = enhance_image(image1, blurtype='gaussian')
enhanced_image2 = enhance_image(image2, blurtype='median')

# 保存处理后的结果
cv2.imwrite('Enhanced_Q4_1.tif', enhanced_image1)
cv2.imwrite('Enhanced_Q4_2.tif', enhanced_image2)
