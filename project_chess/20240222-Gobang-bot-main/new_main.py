import pygame as pg
import os
from config_4 import *
import time
import datetime

class GameObject:
    # 具有棋子的图像、类别和坐标三个属性
    def __init__(self, image, color, pos):
        self.image = image
        self.color = color
        self.pos = image.get_rect(center=pos)


# 按钮类，生成了悔棋按钮和恢复按钮
class Button(object):
    # 具有图像surface，宽高和坐标属性
    def __init__(self, text, color, x=None, y=None):
        self.surface = font_big.render(text, True, color)
        self.WIDTH = self.surface.get_width()
        self.HEIGHT = self.surface.get_height()
        self.x = x
        self.y = y

    # 这个方法用于确定鼠标是否点击了对应的按钮
    def check_click(self, position):
        x_match = self.x < position[0] < self.x + self.WIDTH
        y_match = self.y < position[1] < self.y + self.HEIGHT
        if x_match and y_match:
            return True
        else:
            return False


def set_chess(board_inner, x, y, chr):
    if board_inner[x][y] != ' ':
        print('该位置已有棋子')
        print(x, y)
        return False
    else:
        board_inner[x][y] = chr
        print(x, y)
        # for _ in board_inner:
        #     print(_)
        # print()
        return True


def check_win(board_inner):
    for list_str in board_inner:
        if ''.join(list_str).find('O' * 5) != -1:
            print('白棋获胜')
            return 0
        elif ''.join(list_str).find('X' * 5) != -1:
            print('黑棋获胜')
            return 1
    else:
        return -1


def check_win_all(board_inner):
    board_c = [[] for _ in range(29)]
    for x in range(15):
        for y in range(15):
            board_c[x - y].append(board_inner[x][y])
    board_d = [[] for _ in range(29)]
    for x in range(15):
        for y in range(15):
            board_d[x + y].append(board_inner[x][y])
    return [check_win(board_inner), check_win([list(i) for i in zip(*board_inner)]), check_win(board_c),
            check_win(board_d)]


# 遍历棋盘获得每一行的分数
def value(board_inner, temp_list, value_model, chr):
    score = 0
    num = 0
    # 第一层循环，遍历棋盘，计算每一行的得分
    for list_str in board_inner:
        # 如果一行里指定棋子数量少于2个，则跳过这一行
        if ''.join(list_str).count(chr) < 2:
            continue
        a = 0  # a是一个跳过参数，识别到指定棋型后，需要跳过若干位
        # 第二层循环，双指针第一个指针。每一行需要逐位识别，因为最短的棋型是5位，因此range(11)就够了
        for i in range(11):
            if a == 0:  # 若a为0，则正常处理
                temp = []  # temp用于存储识别到的棋型
                # 第三层循环，双指针第二个指针。最短的棋型是5位，最长的棋型是11位；因此需要不断截取指定长度的切片与不同的棋型进行对比
                for j in range(5, 12):
                    # 如果超出本行长度，则跳出这层循环
                    if i + j > len(list_str):
                        break
                    num += 1  # num用于测试调试时计算循环次数
                    s = ''.join(list_str[i:i + j])  # s是本次用于对比的切片
                    s_num = min(s.count(chr), 5)  # s_num是s中本方棋子的数量
                    # 如果本方棋子少于2个，则跳过本次循环
                    if s_num < 2:
                        continue
                    else:
                        # 如果是本行的开头，则与开头棋型和通用棋型进行对比
                        if i == 0:
                            # 这个列表推导式从棋型文件中剔除了切片不可能匹配成功的棋型（比如切片只有3个X，则把包含4个X和5个X的棋型剔除）
                            for k in [t for _ in value_model[0].items() for t in _[1] if int(_[0]) <= s_num]:
                                # 如果切片与棋型完全相等，则把结果记录在temp中
                                if s == k[1][0]:
                                    temp.append((i, k))
                        else:
                            # 如果既不是开头也不是结尾，与通用棋型进行对比
                            if i + j < len(list_str):
                                for k in [t for _ in value_model[1].items() for t in _[1] if int(_[0]) <= s_num]:
                                    if s == k[1][0]:
                                        temp.append((i, k))
                            # 如果是结尾，与结尾棋型和通用棋型进行对比
                            elif i + j == len(list_str):
                                for k in [t for _ in value_model[2].items() for t in _[1] if int(_[0]) <= s_num]:
                                    if s == k[1][0]:
                                        temp.append((i, k))
            # 如果a不等于1，则相当于跳过本次比对，a-1，temp要记得重新赋值为[]
            else:
                a -= 1
                temp = []
            # 对temp进行判空操作，避免报错
            if temp:
                # 用列表推导式从temp中抽离出有效信息，获得切片匹配到的最高分
                max_value = max([i[1][1][1] for i in temp])
                # 基于最高分找到匹配到的棋型
                max_shape = [i for i in temp if i[1][1][1] == max_value][0]
                # 棋型特殊处理，若匹配到某些棋型，需要在匹配时跳过若干位
                if max_shape[1][0] in ['4_1_e', '4_1_1',
                                       '4_2_5', '4_2_6', '4_2_7', '4_2_8_e', '4_2_9',
                                       '4_3_4_s',
                                       '3p_0', '3p_0_1',
                                       '3p_1_3', '3_1_4_e', '3_1_5',
                                       '3_2_5_s',
                                       '3_3', '3_3_1', '3_3_2_e', '3_3_3',
                                       '2_0_5',
                                       '2_1',
                                       '2_2_1', '2_2_2_e', '2_2_3']:
                    a = 1
                elif max_shape[1][0] in ['4_2_1', '4_2_2', '4_2_3_e', '4_2_4',
                                         '4_3', '4_3_8', '4_3_9',
                                         '3p_1', '3_1_1_e', '3_1_2',
                                         '2_0',
                                         '2_2']:
                    a = 2
                elif max_shape[1][0] in ['3p_2']:
                    a = 3
                elif max_shape[1][0] in ['4_2']:
                    a = 5
                # 用temp_list保存每一行、每一位匹配到的棋型
                temp_list.append(max_shape)
                # 用score记录总分值
                score += max_value
    # print(temp_list)
    # print('value函数循环次数{}'.format(num))
    return score


# 计算某一方的附加分
def additional(te_list):
    score = 0
    # 对te_list做一些处理得到temp_list
    temp_list = [i[1][0][:2] for i in te_list]
    # 死四 + 活三 >= 2，则附加分加30分
    if sum([temp_list.count(i) for i in ['4_', '3p']]) >= 2:
        score += 30
    # 活三 + 死三 >= 2 且 活三 > 0，则附加分加15分
    elif sum([temp_list.count(i) for i in ['3p', '3_']]) >= 2 \
            and sum([temp_list.count(i) for i in ['3p']]) > 0:
        score += 15
    return score


# 计算棋盘横、竖、正斜、反斜四个方向分数，和附加分，加在一起成为最终总分
def value_all(board_inner, temp_list, value_model, chr):
    board_c = [[] for _ in range(29)]
    for x in range(15):
        for y in range(15):
            board_c[x + y].append(board_inner[x][y])
    board_d = [[] for _ in range(29)]
    for x in range(15):
        for y in range(15):
            board_d[x - y].append(board_inner[x][y])
    a = value(board_inner, temp_list, value_model, chr)
    b = value([list(i) for i in zip(*board_inner)], temp_list, value_model, chr)
    c = value(board_c, temp_list, value_model, chr)
    d = value(board_d, temp_list, value_model, chr)
    # 进行四个方向检测时，共用一个temp_list，因此附加分是考虑了全部四个方向
    add = additional(temp_list)
    # print(temp_list)
    # print('横{},竖{},正斜{},反斜{},附加{}'.format(a, b, c, d, add))
    return a + b + c + d + add


# 落子决策函数，通过遍历可落子区域，结合获胜、防守和进攻，从而输出最佳落子位置
def value_chess(board_inner):
    t1 = time.time()
    # 如果棋盘为空，则黑棋直接落在天元位置，分数为0
    if board_inner == [[' '] * 15 for _ in range(15)]:
        return 7, 7, 0
    # 一系列数据初始化
    temp_list_x = []
    temp_list_o = []
    tp_list_x_2 = []
    tp_list_o_2 = []
    tp_list_d = []
    score_x = value_all(board_inner, temp_list_x, value_model_X, 'X')  # 落子前，黑棋分数
    pos_x = (0, 0)
    score_o = value_all(board_inner, temp_list_o, value_model_O, 'O')  # 落子前，白棋分数
    pos_o = (0, 0)
    pos_d = (0, 0)
    score_x_2 = 0
    score_o_2 = 0
    score_diff = 0
    # 获得横竖两个方向，棋子落子范围；比如最左棋子在第5列，最右棋子在第9列；最上棋子在第5行，最下棋子在第9行；
    # 目的是为了缩小遍历范围；离当前棋子过远的区域，影响较小，就不再考虑遍历了；
    chess_range_x = [x for x in range(15) if ''.join(board_inner[x]).replace(' ', '') != '']
    chess_range_y = [y for y in range(15) if ''.join([list(i) for i in zip(*board_inner)][y]).replace(' ', '') != '']
    # 在棋子最大范围的基础上，做一些小小的拓展，得到落子检测区域；在上下左右四个方向各拓展2行/列；
    range_x = (max(0, min(chess_range_x) - 2), min(max(chess_range_x) + 2, 15))
    range_y = (max(0, min(chess_range_y) - 2), min(max(chess_range_y) + 2, 15))
    num = 0
    # 遍历落子检测区域所有的位置
    for x in range(*range_x):
        for y in range(*range_y):
            tp_list_x = []
            tp_list_o = []
            tp_list_c = []
            # 如果该位置已有棋子，则跳过
            if board_inner[x][y] != ' ':
                continue
            else:
                num += 1  # num用于循环次数计数，从而检测value_chess函数的性能
                # 假定在该位置落黑子
                board_inner[x][y] = 'X'
                score_a = value_all(board_inner, tp_list_x, value_model_X, 'X')  # 该位置落黑子，黑棋分数
                score_c = value_all(board_inner, tp_list_c, value_model_O, 'O')  # 该位置落黑子，白棋分数
                # score_x_2用于记录黑棋最高分对应的落子信息
                if score_a > score_x_2:
                    pos_x = x, y
                    tp_list_x_2 = tp_list_x
                    score_x_2 = score_a
                # 假定在该位置落白子
                board_inner[x][y] = 'O'
                score_b = value_all(board_inner, tp_list_o, value_model_O, 'O')  # 该位置落白子，白棋分数
                # score_x_2用于记录白棋最高分对应的落子信息
                if score_b > score_o_2:
                    pos_o = x, y
                    tp_list_o_2 = tp_list_o
                    score_o_2 = score_b
                # 将该位置棋子信息复原
                board_inner[x][y] = ' '
                # diff = = 1.1 * 黑棋分数增长 + （白棋原分数 - 落子后白棋分数） +  （白棋预期最高分 - 落子后白棋分数）
                # 之所以（白棋预期最高分 - 落子后白棋分数），是为了增加防守的逻辑；
                # 之所以设置1.1的系数，是为了鼓励进攻，毕竟只有进攻才能获得胜利
                diff = 1.1 * (score_a - score_x) + score_o - score_c + score_b - score_c
                # score_diff用于记录diff的最大值
                if diff > score_diff:
                    pos_d = x, y
                    tp_list_d = tp_list_x
                    score_diff = diff
    print('value_chess本次循环次数：{}'.format(num))
    print("value_chess循环遍历执行完毕，用时{}秒".format(round(time.time() - t1, 2)))
    # 三种不同的策略，打印出对应的信息
    if score_x_2 >= 1000:
        print('——' * 30)
        print('策略1棋面：')
        print('黑棋棋面：', temp_list_x)
        print('白棋棋面：', temp_list_o)
        score = score_x_2
        pos = pos_x
        x, y = pos
        board_inner[x][y] = 'X'
        # temp_list_x.clear()
        # score = value_all(board_inner, temp_list_x, value_model_X)
        score_o_e = value_all(board_inner, temp_list_o, value_model_O, 'O')
        board_inner[x][y] = ' '
        print('执行策略1、直接获胜')
        print('黑棋最佳落子：坐标{}，黑棋得分{}，白棋得分{}'.format(pos, score, score_o_e))
        # print('白棋最佳落子：坐标{}'.format(pos_o))
        print('白棋原分数{}，预期最高分数{}，分数差值{}'.format(score_o, score_o_2, score_o_2 - score_o))
        print('若白棋落子{}，白棋棋型{}'.format(pos_o, tp_list_o_2))
        print('黑棋原分数{}，预期最高分数{}，分数差值{}'.format(score_x, score_x_2, score_x_2 - score_x))
        print('若黑棋落子{}，黑棋棋型{}'.format(pos_x, tp_list_x_2))
        print('——' * 30)
        print("value_chess执行完毕，用时{}秒".format(round(time.time() - t1, 2)))
    elif score_o_2 >= 1000:
        print('——' * 30)
        print('策略2棋面：')
        print('黑棋棋面：', temp_list_x)
        print('白棋棋面：', temp_list_o)
        x, y = pos_o
        board_inner[x][y] = 'X'
        temp_list_x.clear()
        score = value_all(board_inner, temp_list_x, value_model_X, 'X')
        score_o_e = value_all(board_inner, temp_list_o, value_model_O, 'O')
        board_inner[x][y] = ' '
        pos = pos_o
        print('执行策略2、防守：防止对方获胜')
        print('黑棋最佳落子：坐标{}，黑棋得分{}，白棋得分{}'.format(pos, score, score_o_e))
        # print('白棋最佳落子：坐标{}'.format(pos_o))
        print('白棋原分数{}，预期最高分数{}，分数差值{}'.format(score_o, score_o_2, score_o_2 - score_o))
        print('若白棋落子{}，白棋棋型{}'.format(pos_o, tp_list_o_2))
        print('黑棋原分数{}，预期最高分数{}，分数差值{}'.format(score_x, score_x_2, score_x_2 - score_x))
        print('若黑棋落子{}，黑棋棋型{}'.format(pos_x, tp_list_x_2))
        print('——' * 30)
        print("value_chess执行完毕，用时{}秒".format(round(time.time() - t1, 2)))
    else:
        print('——' * 30)
        print('策略3棋面：')
        print('黑棋棋面：', temp_list_x)
        print('白棋棋面：', temp_list_o)
        x, y = pos_d
        board_inner[x][y] = 'X'
        temp_list_x.clear()
        temp_list_o.clear()
        score = value_all(board_inner, temp_list_x, value_model_X, 'X')
        score_o_e = value_all(board_inner, temp_list_o, value_model_O, 'O')
        board_inner[x][y] = 'O'
        score_test = value_all(board_inner, [], value_model_O, 'O')
        board_inner[x][y] = ' '
        pos = pos_d
        print('黑棋原得分', score_x)
        print('黑棋得分', score)
        print('白棋原得分', score_o)
        print('白棋得分', score_o_e)
        print('若该位置落白棋，白棋得分', score_test)
        print('落子后黑棋棋面', temp_list_x)
        print('执行策略3、防守：防守+进攻')
        print('我方增长得分+对方减少得分：{}'.format(score_diff))
        print('黑棋最佳落子：坐标{}，黑棋得分{}，白棋得分{}'.format(pos, score, score_o_e))
        # print('白棋最佳落子：坐标{}'.format(pos_o))
        print('白棋原分数{}，预期最高分数{}，分数差值{}'.format(score_o, score_o_2, score_o_2 - score_o))
        print('若白棋落子{}，白棋棋型{}'.format(pos_o, tp_list_o_2))
        print('黑棋原分数{}，预期最高分数{}，分数差值{}'.format(score_x, score_x_2, score_x_2 - score_x))
        print('若黑棋落子{}，黑棋棋型{}'.format(pos_x, tp_list_x_2))
        print('——' * 30)
        print("value_chess执行完毕，用时{}秒".format(round(time.time() - t1, 2)))
    # 返回最终策略对应的黑棋落子坐标和黑棋得分
    return *pos, score



def save_game(board_inner, objects, screen):
    # 创建 save 文件夹（如果它不存在的话）
    save_folder = "save"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # 获取当前时间，用于文件名
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = os.path.join(save_folder, f"game_{current_time}.txt")
    image_filename = os.path.join(save_folder, f"game_{current_time}.png")

    with open(filename, 'w') as file:
        for i, row in enumerate(board_inner):
            # 将每行转换为一个以 '.' 分隔的字符串
            row_with_dots = '.'.join(row)  # 将每个格子用 . 连接
            if i == len(board_inner) - 1:  # 如果是最后一行
                file.write(row_with_dots)  # 最后一行不加换行符
            else:
                file.write(row_with_dots + "\n")  # 每行之间加换行符

        # 保存棋子对象信息
        # file.write("objects:\n")
        # for obj in objects:
        #     # 保存每个棋子的位置和颜色
        #     file.write(f"{obj.color} {obj.pos.x} {obj.pos.y}\n")

    # 绘制棋子
    for obj in objects:
        screen.blit(obj.image, obj.pos)

    # 更新屏幕，确保渲染所有元素
    pg.display.update()

    # 保存当前游戏界面为图片
    pg.image.save(screen, image_filename)

    print(f"棋局数据已保存到 {filename}")
    print(f"当前游戏界面已保存为 {image_filename}")


def expand_board(board_inner, target_size=15):
    # 获取原棋盘的行数和列数
    original_rows = len(board_inner)
    original_cols = len(board_inner[0]) if original_rows > 0 else 0

    # 打印原棋盘信息（用于调试）
    print(f"原棋盘大小: {original_rows}x{original_cols}, 目标棋盘大小: {target_size}x{target_size}")

    # 创建一个 target_size x target_size 的空白棋盘
    expanded_board = [[' ' for _ in range(target_size)] for _ in range(target_size)]

    # 计算行和列的偏移量，使得原棋盘的中心与目标棋盘的中心对齐
    offset_row = (target_size - original_rows) // 2
    offset_col = (target_size - original_cols) // 2

    # 打印偏移量（用于调试）
    print(f"行偏移量: {offset_row}, 列偏移量: {offset_col}")

    # 将原棋盘复制到扩展后的棋盘中间
    for i in range(original_rows):
        for j in range(original_cols):
            target_i = offset_row + i
            target_j = offset_col + j

            # 将原棋盘的元素填充到扩展棋盘
            expanded_board[target_i][target_j] = board_inner[i][j]

    # 打印扩展后的棋盘（用于调试）
    print("扩展后的棋盘：")
    for row in expanded_board:
        print(row)

    return expanded_board


def load_board(filename, objects, chess_list, letter_list):
    board_inner = []  # 用于存储棋盘数据
    objects.clear()  # 清空之前的棋子对象
    black_count = 0  # 统计黑棋数量
    white_count = 0  # 统计白棋数量

    with open(filename, 'r') as file:
        for row_idx, line in enumerate(file):
            # 去掉每行的首尾空格，然后按 `.` 分割
            row = line.strip().split('.')  # 按 `.` 分隔每个格子

            # 处理空格格子，确保空的格子被填充为空格
            row_with_filled_spaces = [cell if cell != '' else ' ' for cell in row]

            board_inner.append(row_with_filled_spaces)  # 将每行数据添加到棋盘列表

    # 扩展棋盘到 15x15
    expanded_board = expand_board(board_inner, target_size=15)

    # 恢复棋子对象
    for row_idx, row in enumerate(expanded_board):
        for col_idx, cell in enumerate(row):
            if cell != ' ' and cell != '':  # 如果当前格子不为空（跳过空格和空字符串）
                letter = cell  # 获取棋子的类型
                if letter in letter_list:  # 确保棋子类型在列表中
                    chess_image = chess_list[letter_list.index(letter)]  # 获取对应的棋子图像
                    pos = (27 + col_idx * 40, 27 + row_idx * 40)  # 计算棋子在窗口中的显示位置
                    # 创建并添加棋子对象到列表
                    objects.append(GameObject(chess_image, letter, pos))
                    # 统计黑棋和白棋的数量
                    if letter == 'X':
                        black_count += 1
                    elif letter == 'O':
                        white_count += 1

    if black_count <= white_count:
        flag = 0  # 黑棋先下
    else:
        flag = 1  # 白棋先下

    if black_count != white_count and abs(black_count - white_count) != 1:
        print(f"棋盘不符合规则，黑棋：{black_count}，白棋：{white_count}，清空棋盘！")
        objects.clear()  # 清空棋子对象
        expanded_board = [[' ' for _ in range(15)] for _ in range(15)]  # 清空棋盘
        flag = 0

    # 判断哪个玩家该下棋


    return expanded_board, objects, flag


def main(board_inner):
    pg.init()
    # 一系列数据初始化
    clock = pg.time.Clock()  # pygame时钟
    objects = []  # 下棋记录列表
    recover_objects = []  # 恢复棋子时用到的列表，即悔棋记录列表
    ob_list = [objects, recover_objects]  # 将以上两个列表放到一个列表中，主要是增强抽象度，简少了代码行数
    screen = pg.display.set_mode((WIDTH, HEIGHT))  # 游戏窗口
    black = pg.image.load("data/chess_black.png").convert_alpha()  # 黑棋棋子图像
    white = pg.image.load("data/chess_white.png").convert_alpha()  # 白棋棋子图像
    green = pg.image.load("data/chess_green.png").convert_alpha()
    red   = pg.image.load("data/chess_red.png").convert_alpha()
    background = pg.image.load("data/bg_old.png").convert_alpha()  # 棋盘背景图像
    forecast_button = Button('预测', RED, 665, 200)  # 创建悔棋按钮
    save_button = Button('保存', BLUE, 665, 300)  # 创建恢复按钮
    load_button = Button('读取',YELLOW, 665, 400 )
    restart_button = Button('重新开始', GREEN, 625, 500)  # 创建重新开始按钮
    screen.blit(forecast_button.surface, (forecast_button.x, forecast_button.y))  # 把悔棋按钮打印游戏窗口
    screen.blit(save_button.surface, (save_button.x, save_button.y))  # 把恢复按钮打印游戏窗口
    screen.blit(restart_button.surface, (restart_button.x, restart_button.y))
    screen.blit(load_button.surface, (load_button.x, load_button.y))# 把重新开始按钮打印游戏窗口
    pg.display.set_caption("五子棋")  # 窗体的标题
    flag = 0  # 回合变量，用于识别当前是哪一方回合
    going = True  # 主循环变量，用于控制主循环继续或者结束
    chess_list = [black, white,red,green]  # 棋子图像列表，主要是增强抽象度，简少了代码行数
    letter_list = ['X', 'O']  # 棋子类型列表，主要是增强抽象度，简少了代码行数
    word_list = ['黑棋', '白棋']  # 棋子文字名称列表，主要是增强抽象度，简少了代码行数
    word_color = [(0, 0, 0), (255, 255, 255)]  # 棋子文字颜色列表，主要是增强抽象度，简少了代码行数
    while going:
        screen.blit(background, (0, 0))  # 将棋盘背景打印到游戏窗口
        # text = font.render("{}回合".format(word_list[flag]), True, word_color[flag])  # 创建一个文本对象，显示当前是哪方的回合
        # text_pos = text.get_rect(centerx=background.get_width() / 2, y=2)  # 确定文本对象的显示位置
        # screen.blit(text, text_pos)  # 将文本对象打印到游戏窗口
        # 通过循环不断识别玩家操作
        for event in pg.event.get():
            # 如果关闭窗口，主循环结束
            if event.type == pg.QUIT:
                going = False
            # 如果点击键盘ESC键，主循环结束
            elif event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
                going = False
            # 如果玩家进行了鼠标点击操作
            elif event.type == pg.MOUSEBUTTONDOWN:
                pos = pg.mouse.get_pos()  # 获取鼠标点击坐标
                # 如果点击了悔棋按钮或者恢复按钮
                if forecast_button.check_click(pos):
                    y, x = value_chess(board_inner)[:2]
                    if set_chess(board_inner, y, x, letter_list[flag]):
                        objects.append(GameObject(chess_list[flag+2], letter_list[flag], (27 + x * 40, 27 + y * 40)))
                    if 0 in check_win_all(board_inner) or 1 in check_win_all(board_inner):
                        # 将下棋记录的棋子打印到游戏窗口
                        for o in objects:
                            screen.blit(o.image, o.pos)
                        # 根据flag获取到当前获胜方，生成获胜文案
                        win_text = font.render("{}获胜，游戏5秒后重新开始".format(word_list[flag]), True,
                                               word_color[flag])
                        # 设定获胜文案的位置
                        win_text_pos = win_text.get_rect(centerx=background.get_width() / 2, y=200)
                        screen.blit(win_text, win_text_pos)  # 把获胜文案打印到游戏窗口
                        pg.display.update()  # 对游戏窗口进行刷新
                        pg.time.delay(5000)  # 暂停5秒，保证文案能够清晰展示
                        board_inner = [[' '] * 15 for _ in range(15)]  # 对board进行初始化
                        objects.clear()  # 下棋记录列表初始化
                        recover_objects.clear()  # 悔棋记录列表初始化
                        flag = 0  # flag初始化
                        continue
                    flag = [1, 0][flag]
                elif save_button.check_click(pos):
                    save_game(board_inner, objects, screen)
                elif load_button.check_click(pos):
                    # board_inner, objects, flag = load_board('save/wrong.txt', objects, chess_list, letter_list)
                     board_inner, objects, flag = load_board('save/test2.txt', objects, chess_list,letter_list)
                # board_inner,objects,flag = load_board('save/game_2024-12-12_16-49-36.txt', objects, chess_list, letter_list)
                elif restart_button.check_click(pos):
                    hint_text = font.render("游戏重新开始", True, word_color[flag])  # 提示文案
                    hint_text_pos = hint_text.get_rect(centerx=background.get_width() / 2, y=200)  # 提示文案位置
                    screen.blit(hint_text, hint_text_pos)  # 把提示文案打印到游戏窗口
                    pg.display.update()  # 对游戏窗口进行刷新
                    pg.time.delay(1000)  # 暂停1秒，保证文案能够清晰展示
                    board_inner = [[' '] * 15 for _ in range(15)]  # 对board进行初始化
                    objects.clear()  # 下棋记录列表初始化
                    recover_objects.clear()  # 悔棋记录列表初始化
                    flag = 0  # flag初始化
                    continue  # 通过continue跳过下一行代码，从而保证flag赋值不会异常
                else:
                    # 若用户点击的不是悔棋、恢复按钮，则进行落子操作
                    a, b = round((pos[0] - 27) / 40), round((pos[1] - 27) / 40)  # 将用户鼠标点击位置的坐标，换算为board坐标
                    # 若坐标非法（即点击到了黑色区域），则不做处理
                    if a >= 15 or b >= 15:
                        continue
                    else:
                        x, y = max(0, a) if a < 0 else min(a, 14), max(0, b) if b < 0 else min(b, 14)  # 将a、b进行处理得到x和y
                        # 若落子操作合法，则进行落子
                        if set_chess(board_inner, y, x, letter_list[flag]):
                            # 下棋记录列表添加指定棋子
                            objects.append(GameObject(chess_list[flag], letter_list[flag], (27 + x * 40, 27 + y * 40)))
                            # 一旦成功落子，则将悔棋记录列表清空；不这么做，一旦在悔棋和恢复中间掺杂落子操作，就会有问题
                            recover_objects.clear()
                            for obj in objects:
                                if obj.color == 'X':
                                    # 选择黑色棋子图像
                                    obj.image = black
                                    obj.color = 'black'
                                elif obj.color == 'O':
                                    # 选择白色棋子图像
                                    obj.image = white
                                    obj.color = 'white'

                            # 输出修改后的 objects 列表
                            # for obj in objects:
                            #     print(f"棋子 {obj.color} 对应的图像是 {obj.image}, 位置是 {obj.pos}")
                            print(board_inner)
                            # recover_objects = []


                            # 判断是否出现获胜方
                            if 0 in check_win_all(board_inner) or 1 in check_win_all(board_inner):
                                # 将下棋记录的棋子打印到游戏窗口
                                for o in objects:
                                    screen.blit(o.image, o.pos)
                                # 根据flag获取到当前获胜方，生成获胜文案
                                win_text = font.render("{}获胜，游戏5秒后重新开始".format(word_list[flag]), True,
                                                       word_color[flag])
                                # 设定获胜文案的位置
                                win_text_pos = win_text.get_rect(centerx=background.get_width() / 2, y=200)
                                screen.blit(win_text, win_text_pos)  # 把获胜文案打印到游戏窗口
                                pg.display.update()  # 对游戏窗口进行刷新
                                pg.time.delay(5000)  # 暂停5秒，保证文案能够清晰展示
                                board_inner = [[' '] * 15 for _ in range(15)]  # 对board进行初始化
                                objects.clear()  # 下棋记录列表初始化
                                recover_objects.clear()  # 悔棋记录列表初始化
                                flag = 0  # flag初始化
                                continue  # 通过continue跳过下一行代码，从而保证flag赋值不会异常
                            flag = [1, 0][flag]

                        # 若落子位置已经有棋子，则进行提示
                        else:
                            hint_text = font.render("该位置已有棋子", True, word_color[flag])  # 提示文案
                            hint_text_pos = hint_text.get_rect(centerx=background.get_width() / 2, y=200)  # 提示文案位置
                            # 将下棋记录的棋子打印到游戏窗口
                            for o in objects:
                                screen.blit(o.image, o.pos)
                            screen.blit(hint_text, hint_text_pos)  # 把提示文案打印到游戏窗口
                            pg.display.update()  # 对游戏窗口进行刷新
                            pg.time.delay(300)  # 暂停0.3秒，保证文案能够清晰展示

        for o in objects:
            screen.blit(o.image, o.pos)
        clock.tick(60)  # 游戏帧率每秒60帧
        pg.display.update()  # 对游戏窗口进行刷新


if __name__ == '__main__':
    pg.init()
    main_dir = os.path.split(os.path.abspath(__file__))[0]
    font = pg.font.Font('font/12345.TTF', 20)
    font_big = pg.font.Font('font/12345.TTF', 40)
    main(board)
    pg.quit()
