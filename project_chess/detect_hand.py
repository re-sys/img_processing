import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 读取图像
image_path = 'hand.jpg'  # 替换为你的图像路径


def resize_image_with_padding(image, target_size):
    """
    Resize the image to the target size while maintaining the aspect ratio
    and padding it to fit the target dimensions.

    :param image: Input image (numpy array)
    :param target_size: Target size (int) for both width and height
    :return: Resized and padded image (numpy array)
    """
    # 获取原始图像的尺寸
    original_height, original_width = image.shape[:2]

    # 计算缩放比例
    scale = target_size / max(original_width, original_height)

    # 计算新尺寸
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    # 调整图像尺寸
    resized_img = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # 创建一个新的空白图像
    new_img = np.zeros((target_size, target_size, 3), dtype=np.uint8)

    # 计算放置位置
    x_offset = (target_size - new_width) // 2
    y_offset = (target_size - new_height) // 2

    # 将缩放后的图像放置到空白图像中
    new_img[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_img

    return new_img

def resize_and_pad_binary_image(binary_image, target_size):
    """
    Resize the binary image to the target size while maintaining the aspect ratio
    and padding it to fit the target dimensions.

    :param binary_image: Input binary image (numpy array)
    :param target_size: Target size (int) for both width and height
    :return: Resized and padded binary image (numpy array)
    """
    # 获取原始图像的尺寸
    original_height, original_width = binary_image.shape[:2]

    # 计算缩放比例
    scale = target_size / max(original_width, original_height)

    # 计算新尺寸
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    # 调整图像尺寸
    resized_img = cv2.resize(binary_image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

    # 创建一个新的空白图像（全黑），以目标大小填充
    new_img = np.zeros((target_size, target_size), dtype=np.uint8)

    # 计算放置位置
    x_offset = (target_size - new_width) // 2
    y_offset = (target_size - new_height) // 2

    # 将缩放后的二值图像放置到空白图像中
    new_img[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_img

    return new_img

def find_chessboard(image_path):
    img = cv2.imread(image_path)

    # 2. 图像预处理
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换为灰度图
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)  # 二值化
    # 膨胀
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=8)
    # plt.imshow(binary, cmap='gray')
    # plt.title('Binary Image')
    # plt.axis('off')
    # plt.show()

    # 3. 检测轮廓
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 检测轮廓

    # 初始化最大面积和相应的轮廓
    max_area = 0
    board_contour = None

    # 遍历所有轮廓，找到最大面积的轮廓
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:  # 如果当前区域大于最大区域
            max_area = area
            board_contour = contour  # 更新最大区域对应的轮廓

    # 如果找到了棋盘轮廓
    if board_contour is not None:
        # cv2.drawContours(img, [board_contour], -1, (0, 255, 0), 3)  # 绘制棋盘轮廓
        # 获取棋盘的边界框
        x, y, w, h = cv2.boundingRect(board_contour)
        chessboard = img[y:y+h, x:x+w]
    
    gray = cv2.cvtColor(chessboard, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # 使用高斯模糊
    canny = cv2.Canny(blurred, 50, 150)
    #先稀释一下然后膨胀
    kernel = np.ones((5, 5), np.uint8)
    # eroded = cv2.erode(canny, kernel, iterations=1)
    dilated = cv2.dilate(canny, kernel, iterations=2)
    
    gray_chessboard = resize_and_pad_binary_image(dilated, 640)
    return chessboard,gray_chessboard

def get_grid_info(gray_chessboard):
    contours, hierarchy = cv2.findContours(gray_chessboard,cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)
    point_array = []
    coordinates_dict = {}
    resize_chessboard = resize_image_with_padding(chessboard, 640)
    # 绘制小轮廓，大轮廓剔除
    for i, cnt in enumerate(contours):
        # 检查当前轮廓的层级，如果父轮廓的索引是 -1，说明它是第0层级
        level = 0
        parent_index = hierarchy[0][i][3]  # 父轮廓索引
        while parent_index != -1:
            level += 1
            parent_index = hierarchy[0][parent_index][3]  # 父轮廓索引
        if level == 1:  # 1rd dimension indicates parent index
            x, y, w, h = cv2.boundingRect(cnt)
            if w>200 or h>200 or w<25 or h<25:
                continue
            # 计算轮廓的边界框位置
            point_array.append((x,y))
            coordinates_dict[(x, y)] = (w, h)
            cv2.rectangle(resize_chessboard, (x, y), (x + w, y + h), (0, 255, 0), 10)
    return point_array,coordinates_dict,resize_chessboard

def arange_grid(point_array,threshold_y = 20,threshold_x = 20):
    # from scipy.spatial.distance import pdist, squareform
    points_with_sums = [(x, y, x + y) for (x, y) in point_array]
    min_point = min(points_with_sums, key=lambda p: p[2])  # p[2] 是 x + y 的值
    min_x, min_y = min_point[0], min_point[1]
    # chessboard_copy = resize_image_with_padding(chessboard, 640)
    points = point_array
    reference_y = min_y

    def is_floating_point1(points, reference_y, threshold):
        floating_points = []
        for point in points:
            if reference_y - threshold <= point[1] <= reference_y + threshold:
                floating_points.append(point)
            if floating_points:
                reference_y = floating_points[-1][1]
        return sorted(floating_points, key=lambda p: p[0])
    def is_floating_point2(points, reference_x, threshold):
        floating_points = []
        for point in points:
            if reference_x - threshold <= point[0] <= reference_x + threshold:
                floating_points.append(point)
            if floating_points:
                reference_x = floating_points[-1][0]
        return sorted(floating_points, key=lambda p: p[1])
        # 在图像中绘制这些点
    points = sorted(points, key=lambda p: p[0])
    floating_points_sorted = is_floating_point1(points, reference_y, threshold_y)
    lines = []
    points = sorted(points, key=lambda p: p[1])
    for floating_point in floating_points_sorted:
        lines.append(is_floating_point2(points, floating_point[0], threshold_x))

    return lines

def get_chessboard_states(lines,gray_chessboard,shrink_factor=7):
    h,w = len(lines[0]),len(lines)
    chessboard_states=np.zeros((h,w))
    white_ratio_board = np.zeros((h,w))
    # count = 0
    for i,line in enumerate(lines):
        for j,point in enumerate(line):
            x,y = point
            w,h = coordinates_dict[point]
            if w<42 or h<42:
                shrink_factor = 3
            else:
                shrink_factor = 7
            
            x_shrinked =  x + shrink_factor  # 边界不超出原图像
            y_shrinked = y + shrink_factor
            w_shrinked = max(0, w - 2 * shrink_factor)  # 收窄宽度
            h_shrinked = max(0, h - 2 * shrink_factor)  # 收窄高度

            # 确保收窄后的边界不会超出图像范围
            region = gray_chessboard[y_shrinked:y_shrinked + h_shrinked, x_shrinked:x_shrinked + w_shrinked]
            # 腐蚀
            
            total_pixels = region.size  # 总像素数
            white_pixels = np.sum(region == 255)  # 白色像素数

            # 计算白色像素占比
            white_ratio = white_pixels / total_pixels if total_pixels > 0 else 0
            white_ratio_board[j,i] = white_ratio
            if white_ratio > 0.1:
                #提取轮廓
                contours, hierarchy = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours) > 1:
                    logest_contour = max(contours, key=cv2.contourArea)
                    perimeter = cv2.arcLength(logest_contour, True)
                    area = cv2.contourArea(logest_contour)
                    if perimeter > 0 and area > 0 and perimeter / (2 * np.pi * np.sqrt(area / np.pi)) < 1.3:
                    
                        chessboard_states[j,i]=1
                    else:
                        chessboard_states[j,i]=2
                    
                    #用shrink_factor缩小轮廓
                    
                #判断是否为圆形
                if len(contours) == 1:
                    #计算轮廓的面积
                    area = cv2.contourArea(contours[0])
                    #计算周长
                    perimeter = cv2.arcLength(contours[0], True)
                
                    #判断是否为圆形
                    if perimeter > 0 and area > 0 and perimeter / (2 * np.pi * np.sqrt(area / np.pi)) < 1.5:
                        chessboard_states[j,i]=1
                    else:
                        chessboard_states[j,i]=2
    return chessboard_states,white_ratio_board

chessboard,gray_chessboard = find_chessboard(image_path)
plt.imshow(chessboard)
plt.show()
plt.imshow(gray_chessboard,cmap='gray')
plt.show()

point_array,coordinates_dict,resize_chessboard = get_grid_info(gray_chessboard)
contours, hierarchy = cv2.findContours(gray_chessboard,cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)
def show_hierarchy(contours,hierarchy):
    for i, cnt in enumerate(contours):
        # 绘制轮廓
        # 获取轮廓的边界框
        x, y, w, h = cv2.boundingRect(cnt)

        # 根据 hierachy 获取当前轮廓的层级
        level = 0
        parent_index = hierarchy[0][i][3]  # 父轮廓索引
        

        while parent_index != -1:  # -1 表示没有父轮廓
            level += 1
            parent_index = hierarchy[0][parent_index][3]  # 跳到父轮廓
        cv2.putText(resize_chessboard, str(level), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,), 1)
    # cv2.drawContours(resize_chessboard, contours, -1, (0, 255, 0), 3)  # 绘制棋盘轮廓
plt.imshow(resize_chessboard)
plt.show()
lines = arange_grid(point_array)
# print(lines)
chessboard_states,white_ratio_board = get_chessboard_states(lines,gray_chessboard)
print(chessboard_states)
# print(white_ratio_board)
# 3. 显示结果
chessboard_copy = resize_image_with_padding(chessboard, 640)
def show_lines(lines,chessboard_copy):
    for line in lines:
        for point in line:
            cv2.circle(chessboard_copy, point, 5, (255, 0, 0), -1)  # 绘制蓝色圆点
            x,y = point
            w,h = coordinates_dict[point]
            cv2.rectangle(chessboard_copy, (x, y), (x + w, y + h), (0, 255, 0), 10)  # 绘制矩形框
            plt.imshow(chessboard_copy)
            plt.show()


def show_region(lines,gray_chessboard,shrink_factor=7):
    for i,line in enumerate(lines):
        for j,point in enumerate(line):
            x,y = point
            w,h = coordinates_dict[point]
            # region = dilated[y:y+h,x:x+w]
            if w<42 or h<42:
                shrink_factor = 3
            else:
                shrink_factor = 7  # 定义收窄的像素值
            x_shrinked =  x + shrink_factor  # 边界不超出原图像
            y_shrinked = y + shrink_factor
            w_shrinked = max(0, w - 2 * shrink_factor)  # 收窄宽度
            h_shrinked = max(0, h - 2 * shrink_factor)  # 收窄高度

            # 确保收窄后的边界不会超出图像范围
            region = gray_chessboard[y_shrinked:y_shrinked + h_shrinked, x_shrinked:x_shrinked + w_shrinked]
            
            total_pixels = region.size  # 总像素数
            white_pixels = np.sum(region == 255)  # 白色像素数

            # 计算白色像素占比
            white_ratio = white_pixels / total_pixels if total_pixels > 0 else 0
            if white_ratio > 0.1:
                #提取轮廓
                contours, hierarchy = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if len(contours) > 1:
                    print(f"区域 ({j}, {i}) 包含多个轮廓")
                    logest_contour = max(contours, key=cv2.contourArea)
                    perimeter = cv2.arcLength(logest_contour, True)
                    area = cv2.contourArea(logest_contour)
                    print(perimeter / (2 * np.pi * np.sqrt(area / np.pi)))
                    #用膨胀
                    
                    plt.imshow(region,cmap='gray')
                    plt.show()
                    contour_image = np.zeros_like(region)
                    cv2.drawContours(contour_image, logest_contour, -1, (255,), 2)  # 填充为白色
                    plt.imshow(contour_image, cmap='gray')
                    plt.title('Contours')
                    plt.show()
                    
                    
                    print(f"缩小后区域 ({j}, {i}) 包含 {len(contours)} 个轮廓")
                #判断是否为圆形
                if len(contours) == 1:
                    #计算轮廓的面积
                    area = cv2.contourArea(contours[0])
                    #计算周长
                    perimeter = cv2.arcLength(contours[0], True)
                    #画出轮廓
                    contour_image = np.zeros_like(region)
                    # 画出轮廓
                    cv2.drawContours(contour_image, contours, -1, (255,), 2)  # 填充为白色
                    
                    print(perimeter / (2 * np.pi * np.sqrt(area / np.pi)))
                    
                    #判断是否为圆形
                    if perimeter > 0 and area > 0 and perimeter / (2 * np.pi * np.sqrt(area / np.pi)) < 1.3:
                        
                        chessboard_states[j,i]=1
                    else:
                        chessboard_states[j,i]=2
                    
            if j==3:
                print(f"区域 ({j}, {i}) 的白色占比: {white_ratio:.2f}")
                print(w,h)
                plt.imshow(region,cmap='gray')
                plt.show()

