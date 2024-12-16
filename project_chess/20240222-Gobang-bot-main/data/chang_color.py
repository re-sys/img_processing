from PIL import Image, ImageEnhance


def overlay_green(image_path, output_path, transparency=0.5):
    # 打开原始图像
    image = Image.open(image_path).convert("RGBA")

    # 创建一个绿色的覆盖层，RGBA模式下的绿色 (0, 255, 0) 和 透明度值
    green_layer = Image.new("RGBA", image.size, (255, 0, 0, int(255 * transparency)))

    # 将绿色覆盖层与原始图像合并
    combined = Image.alpha_composite(image, green_layer)

    # 保存或显示结果
    combined.save(output_path)
    combined.show()


# 调用函数处理图像
overlay_green('chess_black.png', "chess_red.png", transparency=0.5)
