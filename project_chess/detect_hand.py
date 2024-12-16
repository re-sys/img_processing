import cv2
import numpy as np
import matplotlib.pyplot as plt
class ChessboardDetector:
    def __init__(self, image_path, target_size=640, threshold_y=20, threshold_x=20):
        self.image_path = image_path
        self.target_size = target_size
        self.threshold_y = threshold_y
        self.threshold_x = threshold_x
        self.coordinates_dict = {}
        self.chessboard = None
        self.gray_chessboard = None
        self.resized_chessboard = None
        self.chessboard_states = None
        self.white_ratio_board = None
        self.lines = None
        self.process_image()
    def process_image(self):
        self.find_chessboard()
        point_array = self.get_grid_info()
        self.arange_grid(point_array)
        self.get_chessboard_states()
    def resize_image_with_padding(self, image):
        original_height, original_width = image.shape[:2]
        scale = self.target_size / max(original_width, original_height)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        resized_img = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        new_img = np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)
        x_offset = (self.target_size - new_width) // 2
        y_offset = (self.target_size - new_height) // 2
        new_img[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_img
        return new_img

    def resize_and_pad_binary_image(self, binary_image):
        original_height, original_width = binary_image.shape[:2]
        scale = self.target_size / max(original_width, original_height)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        resized_img = cv2.resize(binary_image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
        new_img = np.zeros((self.target_size, self.target_size), dtype=np.uint8)
        x_offset = (self.target_size - new_width) // 2
        y_offset = (self.target_size - new_height) // 2
        new_img[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_img
        return new_img

    def find_chessboard(self):
        img = cv2.imread(self.image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.dilate(binary, kernel, iterations=8)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # max_area = 0
        # board_contour = None
        if not contours:
            raise ValueError("No contours found in the image.")

        board_contour = max(contours, key=cv2.contourArea)
        # for contour in contours:
        #     area = cv2.contourArea(contour)
        #     if area > max_area:
        #         max_area = area
        #         board_contour = contour

        if board_contour is not None:
            x, y, w, h = cv2.boundingRect(board_contour)
            chessboard = img[y:y+h, x:x+w]
            gray_chessboard = cv2.cvtColor(chessboard, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray_chessboard, (5, 5), 0)
            canny = cv2.Canny(blurred, 50, 150)
            dilated = cv2.dilate(canny, kernel, iterations=2)
            self.gray_chessboard = self.resize_and_pad_binary_image(dilated)
            self.chessboard = chessboard
            self.resized_chessboard = self.resize_image_with_padding(chessboard)

    def get_grid_info(self):
        contours, hierarchy = cv2.findContours(self.gray_chessboard, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        point_array = []

        for i, cnt in enumerate(contours):
            level = 0
            parent_index = hierarchy[0][i][3]
            while parent_index != -1:
                level += 1
                parent_index = hierarchy[0][parent_index][3]
            if level == 1:
                x, y, w, h = cv2.boundingRect(cnt)
                if w > 200 or h > 200 or w < 25 or h < 25:
                    continue
                point_array.append((x, y))
                self.coordinates_dict[(x, y)] = (w, h)
                # cv2.rectangle(self.resized_chessboard, (x, y), (x + w, y + h), (0, 255, 0), 10)
        return point_array

    def arange_grid(self, point_array):
        points_with_sums = [(x, y, x + y) for (x, y) in point_array]
        min_point = min(points_with_sums, key=lambda p: p[2])
        min_x, min_y = min_point[0], min_point[1]
        points = sorted(point_array, key=lambda p: p[0])
        floating_points_sorted = self.is_floating_point1(points, min_y, self.threshold_y)
        lines = []
        points = sorted(points, key=lambda p: p[1])
        for floating_point in floating_points_sorted:
            lines.append(self.is_floating_point2(points, floating_point[0], self.threshold_x))
            if lines:
                row_length = len(lines[0])
                if not all(len(row) == row_length for row in lines):
                    raise ValueError("lines is not a square matrix, rows have different lengths.")
        self.lines = lines
        return lines

    def is_floating_point1(self, points, reference_y, threshold):
        floating_points = []
        for point in points:
            if reference_y - threshold <= point[1] <= reference_y + threshold:
                floating_points.append(point)
            # if floating_points:
                reference_y = floating_points[-1][1]
        return sorted(floating_points, key=lambda p: p[0])

    def is_floating_point2(self, points, reference_x, threshold):
        floating_points = []
        for point in points:
            if reference_x - threshold <= point[0] <= reference_x + threshold:
                floating_points.append(point)
            # if floating_points:
                reference_x = floating_points[-1][0]
        return sorted(floating_points, key=lambda p: p[1])

    def show_region(self,i,j,need_shrink_show=False,need_original=False):
        x, y = self.lines[i][j]
        w, h = self.coordinates_dict[(x, y)]
         
        if w < 42 or h < 42:
            shrink_factor = 3
        else:
            shrink_factor = 7
        x_shrinked = x + shrink_factor
        y_shrinked = y + shrink_factor
        w_shrinked = max(0, w - 2 * shrink_factor)
        h_shrinked = max(0, h - 2 * shrink_factor)

        region = self.gray_chessboard[y_shrinked:y_shrinked + h_shrinked, x_shrinked:x_shrinked + w_shrinked]
        
        if need_shrink_show:
            plt.imshow(region, cmap='gray')
            plt.show()
        if need_original:
            region_original = self.gray_chessboard[y:y+h, x:x+w]
            plt.imshow(region_original, cmap='gray')
            plt.show()
        return region

    def show_hierarchy(self):
        contours, hierarchy = cv2.findContours(self.gray_chessboard,cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)
        resize_chessboard = self.resized_chessboard.copy()
        for i, cnt in enumerate(contours):
            # 绘制轮廓
            # 获取轮廓的边界框
            x, y, w, h = cv2.boundingRect(cnt)

            # 根据 hierachy 获取当前轮廓的层级
            level = 0
            parent_index = hierarchy[0][i][3]  # 父轮廓索引
            # cv2.putText(resize_chessboard, str(parent_index), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,), 1)

            # if parent_index==2:
            #     cv2.drawContours(resize_chessboard, contours, i, (255,), 1)  # 绘制当前轮廓，白色

            while parent_index != -1:  # -1 表示没有父轮廓
                level += 1
                parent_index = hierarchy[0][parent_index][3]  # 跳到父轮廓
            cv2.putText(resize_chessboard, str(level), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,), 1)
        cv2.drawContours(resize_chessboard, contours, -1, (0, 255, 0), 3)  # 绘制棋盘轮廓
        plt.imshow(resize_chessboard)
        plt.show()
    
    def show_lines(self,show_block=False):
        chessboard_copy = self.resized_chessboard.copy()
        lines = self.lines
        for line in lines:
            for point in line:
                cv2.circle(chessboard_copy, point, 5, (255, 0, 0), -1)  # 绘制蓝色圆点
                x,y = point
                w,h = self.coordinates_dict[point]
                if show_block:
                    cv2.rectangle(chessboard_copy, (x, y), (x + w, y + h), (0, 255, 0), 10)  # 绘制矩形框
                # cv2.rectangle(chessboard_copy, (x, y), (x + w, y + h), (0, 255, 0), 10)  # 绘制矩形框
        plt.imshow(chessboard_copy)
        plt.show()

    def get_chessboard_states(self, shrink_factor=7):
        lines = self.lines
        h, w = len(lines[0]), len(lines)
        chessboard_states = np.zeros((h, w))
        white_ratio_board = np.zeros((h, w))

        for i, line in enumerate(lines):
            for j, point in enumerate(line):
                x, y = point
                w, h = self.coordinates_dict[point]
                if w < 42 or h < 42:
                    shrink_factor = 3
                else:
                    shrink_factor = 7
                x_shrinked = x + shrink_factor
                y_shrinked = y + shrink_factor
                w_shrinked = max(0, w - 2 * shrink_factor)
                h_shrinked = max(0, h - 2 * shrink_factor)

                region = self.gray_chessboard[y_shrinked:y_shrinked + h_shrinked, x_shrinked:x_shrinked + w_shrinked]
                total_pixels = region.size
                white_pixels = np.sum(region == 255)

                white_ratio = white_pixels / total_pixels if total_pixels > 0 else 0
                white_ratio_board[j, i] = white_ratio
                if white_ratio > 0.1:
                    contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if len(contours) > 1:
                        longest_contour = max(contours, key=cv2.contourArea)
                        perimeter = cv2.arcLength(longest_contour, True)
                        area = cv2.contourArea(longest_contour)
                        if perimeter > 0 and area > 0 and perimeter / (2 * np.pi * np.sqrt(area / np.pi)) < 1.3:
                            chessboard_states[j, i] = 1
                        else:
                            chessboard_states[j, i] = 2
                    elif len(contours) == 1:
                        area = cv2.contourArea(contours[0])
                        perimeter = cv2.arcLength(contours[0], True)
                        if perimeter > 0 and area > 0 and perimeter / (2 * np.pi * np.sqrt(area / np.pi)) < 1.5:
                            chessboard_states[j, i] = 1
                        else:
                            chessboard_states[j, i] = 2
        self.chessboard_states = chessboard_states
        self.white_ratio_board = white_ratio_board
        return chessboard_states, white_ratio_board

    def change_format(self,filename='filename.txt'):
        rows,cols = 15,15
        target_size = 15
        chessboard = []
        arr = [[0 for _ in range(target_size)] for _ in range(target_size)]
        # 填充数组，使其大小为15*15
        w,h = self.chessboard_states.shape
        padding_w = (15-w)//2
        padding_h = (15-h)//2
        for i in range(w):
            for j in range(h):
                if self.chessboard_states[i][j] == 1:
                    arr[i+padding_h][j+padding_w] = 1 
                elif self.chessboard_states[i][j] == 2:
                    arr[i+padding_h][j+padding_w] = 2
        # arr = np.pad(arr, ((padding_h,padding_h),(padding_w,padding_w)), 'constant', constant_values=0)
        
    # 遍历数组，将每个元素转换为OX格式
        for i in range(rows):
            row_str = []
            for j in range(cols):
                if arr[i][j] == 0:
                    row_str.append(' ')
                elif arr[i][j] == 1:
                    row_str.append('X')
                elif arr[i][j] == 2:
                    row_str.append('O')
            # 将每一行的字符串添加到棋盘列表中，使用点作为分隔符
            chessboard.append('.'.join(row_str))
        with open(filename, 'w') as file:
            for i, row in enumerate(chessboard):
                # 将每行转换为一个以 '.' 分隔的字符串
                # row_with_dots = '.'.join(row_str)  # 将每个格子用 . 连接
                if i == len(row_str) - 1:  # 如果是最后一行
                    file.write(row)  # 最后一行不加换行符
                else:
                    file.write(row + "\n")  # 每行之间加换行符
        for row in chessboard:
            print(row) 

# 使用示例
detector = ChessboardDetector('hand2.jpg')
# detector.find_chessboard()
# point_array, coordinates_dict, resized_chessboard = detector.get_grid_info()
# lines = detector.arange_grid(point_array)
# chessboard_states, white_ratio_board = detector.get_chessboard_states(lines)
print(detector.chessboard_states)
detector.show_lines(show_block=False)
detector.change_format()

# detector.show_region(0,1,need_shrink_show=True,need_original=True)
