import cv2
import numpy as np

def detect_chessboard(image_path, output_txt='board.txt'):


    """
    输入待识别图像路径, 棋盘txt文件待保存路径. 输出棋盘二维数组, 显示识别到的棋盘(可选)
    """

    # 读取图像
    img = cv2.imread(image_path)

    # 转换为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 高斯模糊
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 边缘检测
    edges = cv2.Canny(blur, 50, 150)

    # 轮廓提取
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 计算图像的宽度和高度
    height, width = gray.shape

    # 定义图像四个角的位置 (作为最小外接矩形)
    min_x, min_y = 0, 0
    max_x, max_y = width, height

    # 计算格子的大小
    grid_size_x = (max_x - min_x) // 15  # 每列格子的宽度
    grid_size_y = (max_y - min_y) // 15  # 每行格子的高度

    # 计算每个圆形所在的格子位置
    circles = cv2.HoughCircles(gray, method=cv2.HOUGH_GRADIENT, dp=1, minDist=25, param1=100, param2=19,
                               minRadius=10, maxRadius=18)

    if circles is not None:
        circles = np.uint16(np.around(circles))

        # 存储棋子的坐标和颜色信息
        stones = []
        for i in circles[0, :]:
            cx, cy, r = i[0], i[1], i[2]

            # 只处理位于图像四角的最小外接矩形内的棋子
            if min_x <= cx <= max_x and min_y <= cy <= max_y:
                # 将坐标映射到 14x14 网格
                row = int((cy - min_y) // grid_size_y)
                col = int((cx - min_x) // grid_size_x)

                # 确保坐标在合法范围内
                row = max(0, min(row, 14))
                col = max(0, min(col, 14))

                # 提取圆形区域的RGB值来判断颜色
                roi = img[cy - r:cy + r, cx - r:cx + r]

                # 转换ROI到HSV颜色空间
                hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

                # 计算ROI的平均HSV值
                avg_hsv = np.mean(hsv_roi, axis=(0, 1))  # 计算平均值

                hue = avg_hsv[0]
                value = avg_hsv[2]  # 获取亮度（Value）分量

                # 判断棋子颜色：结合亮度和色调判断
                if value > 120:  # 高亮度表示白棋
                    color = 'O'  # 白棋
                elif hue < 90:  # 较低的色调表示黑棋
                    color = 'X'  # 黑棋
                else:  # 如果亮度不高，但色调属于较高值的，可以判定为白棋
                    color = 'O'

                # 画圆并显示坐标
                cv2.circle(img, (cx, cy), r, (0, 255, 0), 2)
                cv2.circle(img, (cx, cy), 2, (0, 0, 255), 2)
                cv2.putText(img, f'({row}, {col})', (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

                stones.append((row, col, color))

        # 输出棋子位置
        board = np.full((15, 15), ' ', dtype='<U1')  # 初始化一个15x15的空棋盘，填充字符'.'
        for stone in stones:
            row, col, color = stone
            board[row, col] = color  # 在棋盘上设置棋子的颜色

        # 打印棋盘
        print(board)

        # 将棋盘保存到txt文件（用'.'分隔）
        with open(output_txt, 'w') as f:
            for row in board:
                f.write('.'.join(row) + '\n')

        # 显示图像
        cv2.imshow('Detected Chessboard', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 返回棋盘
        return board

    else:
        print("No circles found in the image.")
        return None

##   test
a = detect_chessboard('b_test/7.png')


# def check_win(board, x, y):
#     def check_dir(dx, dy):
#         cnt = 1
#         tx, ty = x + dx, y + dy
#         while tx >= 0 and tx <= 18 and ty >= 0 and ty <= 18 and board[tx][ty] == board[x][y]:
#             cnt += 1
#             tx += dx
#             ty += dy
#         return cnt
#
#     for dx, dy in [(0, 1), (1, 0), (1, 1), (1, -1)]:
#         if check_dir(dx, dy) + check_dir(-dx, -dy) - 1 >= 5:
#             return True
#     return False

# flag = False
# for stone in stones:
#     color, row, col = stone
#     if check_win(board, row, col):
#         print(f"{color} stone Win!!! palce({row+1}, {col+1})")
#         flag = True
#         break
#
# if not flag:
#     print("No Win")

