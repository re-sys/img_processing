import cv2
import numpy as np
import math

# --- 参数配置 ---
# 1. 橙色的 HSV 范围 (这些值需要根据实际环境光照和橙色物体的具体颜色进行调整)
# 你可以使用一些工具（如 OpenCV 的颜色选择器脚本）来帮助确定这些值
# H: 色调 (0-179 在 OpenCV中), S: 饱和度 (0-255), V: 明度 (0-255)
LOWER_ORANGE = np.array([0, 50, 70])  # 橙色的 HSV 阈值下限
UPPER_ORANGE = np.array([12, 270, 200]) # 橙色的 HSV 阈值上限
# 注意: 如果橙色跨越 H=0/180 的边界 (例如红色), 可能需要定义两个范围并合并掩码

# 2. 椭圆拟合的最小轮廓点数
MIN_CONTOUR_POINTS_FOR_ELLIPSE = 50

# 3. 轮廓面积过滤阈值 (避免检测到过小的噪点)
MIN_CONTOUR_AREA = 500 # 单位：像素平方

# 4. 三维坐标估算参数 (重要：这些参数需要通过相机标定获得！)
# 如果没有标定，以下为示例值，估算结果可能不准确
CAMERA_FOCAL_LENGTH_X = 800  # fx: x轴方向的焦距 (像素单位)
CAMERA_FOCAL_LENGTH_Y = 800  # fy: y轴方向的焦距 (像素单位)
CAMERA_PRINCIPAL_POINT_X = 320 # cx: 主点x坐标 (图像中心x)
CAMERA_PRINCIPAL_POINT_Y = 240 # cy: 主点y坐标 (图像中心y)

# 圆环的真实半径 (单位：米)。例如，标准篮球框内径约为 0.45 米，半径为 0.225 米
REAL_HOOP_RADIUS_METERS = 0.225

def estimate_3d_coordinates(ellipse_center_px, ellipse_axes_px, real_radius_m, fx, fy, cx, cy):
    """
    根据椭圆参数和相机内参估算圆环中心的三维坐标。
    这是一个简化的估算方法，基于一些假设。

    参数:
    ellipse_center_px (tuple): 椭圆中心在图像上的 (x, y) 坐标 (像素)。
    ellipse_axes_px (tuple): 椭圆的 (短轴长度, 长轴长度) (像素)。注意是轴长，不是半轴。
    real_radius_m (float): 圆环的真实半径 (米)。
    fx, fy (float): 相机焦距 (像素)。
    cx, cy (float): 相机主点坐标 (像素)。

    返回:
    tuple: (X, Y, Z) 估算的相机坐标系下的三维坐标 (米)，如果无法估算则返回 None。
    """
    u, v = ellipse_center_px
    minor_axis_len, major_axis_len = ellipse_axes_px

    # 使用长轴作为圆直径在图像上的投影长度的近似
    # a_px 是长半轴的像素长度
    a_px = major_axis_len / 2.0
    # b_px 是短半轴的像素长度
    b_px = minor_axis_len / 2.0

    if a_px <= 0:
        return None

    # 估算深度 Z (距离相机中心的距离)
    # 简化假设1: 椭圆的长轴近似对应了圆的直径在图像平面上的投影
    # 这种情况下，Z ≈ (f * R_real) / a_px
    # 为了更稳健，可以使用平均焦距
    f_avg = (fx + fy) / 2.0
    Z = (f_avg * real_radius_m) / a_px

    # 另一种估算Z的方法，考虑了圆平面与图像平面的夹角 (透视缩短效应)
    # cos_theta = b_px / a_px (theta 是圆平面法线与光轴的夹角)
    # 如果圆平面几乎平行于图像平面，b_px ≈ a_px, cos_theta ≈ 1
    # 如果圆平面倾斜，b_px < a_px
    # Z_alternative = (f_avg * real_radius_m * cos_theta) / b_px if b_px > 0 else Z
    # Z = Z_alternative # 可以尝试这种方法，但 a_px 通常更稳定些

    # 根据针孔相机模型反推 X 和 Y
    X = ((u - cx) * Z) / fx
    Y = ((v - cy) * Z) / fy

    return (X, Y, Z)

def main():
    # 打开默认摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("错误：无法打开摄像头。")
        return

    # 获取摄像头的帧宽度和高度，用于设置主点 (如果未手动指定)
    # frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # global CAMERA_PRINCIPAL_POINT_X, CAMERA_PRINCIPAL_POINT_Y
    # if CAMERA_PRINCIPAL_POINT_X is None: CAMERA_PRINCIPAL_POINT_X = frame_width / 2
    # if CAMERA_PRINCIPAL_POINT_Y is None: CAMERA_PRINCIPAL_POINT_Y = frame_height / 2


    while True:
        ret, frame = cap.read()
        if not ret:
            print("错误：无法读取视频帧。")
            break

        # 1. 预处理：高斯模糊减少噪声
        blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)

        # 2. 颜色分割：转换到 HSV 颜色空间
        hsv_frame = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)

        # 根据定义的橙色范围创建掩码
        orange_mask = cv2.inRange(hsv_frame, LOWER_ORANGE, UPPER_ORANGE)

        # 可选：对掩码进行形态学操作，去除小的噪声点，连接断开的区域
        kernel = np.ones((5, 5), np.uint8)
        orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_OPEN, kernel)
        orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_CLOSE, kernel)

        # === 新增：结合canny.py的边缘检测和形态学操作来改善轮廓质量 ===
        # 对orange_mask进行Canny边缘检测
        canny_edges = cv2.Canny(orange_mask, 100, 150)
        
        # 定义形态学操作的核
        morph_kernel = np.ones((3, 3), np.uint8)
        # 按照canny.py中的步骤进行形态学操作
        canny_morph = canny_edges.copy()
        
        # 膨胀3次
        for _ in range(5):
            canny_morph = cv2.dilate(canny_morph, morph_kernel)
        
        # 腐蚀6次
        for _ in range(9):
            canny_morph = cv2.erode(canny_morph, morph_kernel)
        
        kernel = np.ones((6, 6), np.uint8)
        orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_OPEN, kernel)
        orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_OPEN, kernel)
        # 3. 轮廓检测 - 使用经过canny+形态学处理的图像
        contours, _ = cv2.findContours(canny_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # === 新增：收集所有有效轮廓点，进行整体椭圆拟合 ===
        valid_contours = []
        total_contour_area = 0
        
        # 过滤和收集有效轮廓
        for contour in contours:
            contour_area = cv2.contourArea(contour)
            # 过滤掉面积过小的轮廓
            valid_contours.append(contour)
            # if contour_area >= MIN_CONTOUR_AREA:
            #     total_contour_area += contour_area
        
        detected_ellipses_info = []
        
        # Corner case处理：确保有足够的有效轮廓
        if len(valid_contours) > 0:
            try:
                # 将所有有效轮廓的点合并为一个大的点集
                all_points = np.vstack(valid_contours)
                
                # Corner case处理：确保有足够的点进行椭圆拟合
                if len(all_points) >= MIN_CONTOUR_POINTS_FOR_ELLIPSE:
                    # 对合并后的所有点进行椭圆拟合
                    ellipse = cv2.fitEllipse(all_points)
                    # ellipse: ((center_x, center_y), (minor_axis_len, major_axis_len), angle)

                    center_px = (int(ellipse[0][0]), int(ellipse[0][1]))
                    axes_len_px = (ellipse[1][0], ellipse[1][1]) # (短轴, 长轴)
                    angle_deg = ellipse[2]

                    # a 为长半轴, b 为短半轴
                    a_semi_axis_px = axes_len_px[1] / 2.0
                    b_semi_axis_px = axes_len_px[0] / 2.0
                    
                    # 额外的去噪验证：检查椭圆的合理性
                    # 避免过于扁平或过于小的椭圆
                    aspect_ratio = a_semi_axis_px / b_semi_axis_px if b_semi_axis_px > 0 else float('inf')
                    if aspect_ratio <= 5.0 and a_semi_axis_px >= 10:  # 合理的纵横比和最小尺寸
                        # 绘制椭圆和中心点
                        cv2.ellipse(frame, ellipse, (0, 255, 0), 2) # 绿色椭圆
                        cv2.circle(frame, center_px, 5, (0, 0, 255), -1) # 红色中心点

                        # 存储检测到的椭圆信息
                        info_text_ellipse = f"Combined Ellipse - Center: ({center_px[0]},{center_px[1]}), a: {a_semi_axis_px:.1f}px, b: {b_semi_axis_px:.1f}px"
                        detected_ellipses_info.append({
                            "text_display_y_offset": 0,
                            "info_text_ellipse": info_text_ellipse,
                            "center_px": center_px,
                            "axes_len_px": axes_len_px
                        })
                        
                        # 显示轮廓统计信息
                        stats_text = f"Valid contours: {len(valid_contours)}, Total area: {total_contour_area:.0f}px²"
                        cv2.putText(frame, stats_text, (10, frame.shape[0] - 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            except cv2.error as e:
                # fitEllipse 可能会因为轮廓点共线等原因失败
                # print(f"椭圆拟合失败: {e}")
                pass # 忽略拟合失败的情况

        # 显示检测到的椭圆信息和估算的3D坐标
        y_offset_start = 30
        for i, info in enumerate(detected_ellipses_info):
            cv2.putText(frame, info["info_text_ellipse"], (10, y_offset_start + i * 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # 4. 估算三维坐标
            coords_3d = estimate_3d_coordinates(
                info["center_px"],
                info["axes_len_px"],
                REAL_HOOP_RADIUS_METERS,
                CAMERA_FOCAL_LENGTH_X, CAMERA_FOCAL_LENGTH_Y,
                CAMERA_PRINCIPAL_POINT_X, CAMERA_PRINCIPAL_POINT_Y
            )

            if coords_3d:
                X, Y, Z = coords_3d
                info_text_3d = f"3D Est: X={X:.2f}m, Y={Y:.2f}m, Z={Z:.2f}m"
                cv2.putText(frame, info_text_3d, (10, y_offset_start + 20 + i * 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                cv2.putText(frame, "3D Est: N/A", (10, y_offset_start + 20 + i * 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


        # 显示结果帧
        cv2.imshow("Original Frame", frame)
        cv2.imshow("Orange Mask", orange_mask) # 用于调试颜色分割
        cv2.imshow("Canny + Morphology", canny_morph) # 新增：显示处理后的边缘图像

        # 按 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 清理
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()