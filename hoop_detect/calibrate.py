import numpy as np
import cv2
import glob
import os

# 设置棋盘格参数
BOARD_WIDTH = 8    # 棋盘格横向内角点数
BOARD_HEIGHT = 6   # 棋盘格纵向内角点数
SQUARE_SIZE = 2.5  # 棋盘格方格实际尺寸（单位：厘米）

# 准备世界坐标系中的角点坐标 (0,0,0), (1,0,0), ..., (7,5,0)
objp = np.zeros((BOARD_HEIGHT * BOARD_WIDTH, 3), np.float32)
objp[:, :2] = np.mgrid[0:BOARD_WIDTH, 0:BOARD_HEIGHT].T.reshape(-1, 2) * SQUARE_SIZE

# 存储3D点和2D点
obj_points = []  # 世界坐标系中的点
img_points = []  # 图像坐标系中的点

# 读取标定图像
images = glob.glob('calibration_images/*.jpg')  # 替换为你的图像路径

for i, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 查找棋盘格角点
    ret, corners = cv2.findChessboardCorners(
        gray, 
        (BOARD_WIDTH, BOARD_HEIGHT), 
        None
    )
    
    if ret:
        obj_points.append(objp)
        
        # 亚像素级角点精确化
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_refined = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria
        )
        img_points.append(corners_refined)
        
        # 可视化角点（可选）
        cv2.drawChessboardCorners(img, (BOARD_WIDTH, BOARD_HEIGHT), corners_refined, ret)
        cv2.imshow(f'Corners Detected {i+1}', img)
        cv2.waitKey(500)
    else:
        print(f"警告：未能在图像 {os.path.basename(fname)} 中找到棋盘格角点")

cv2.destroyAllWindows()

# 执行相机标定
if len(obj_points) > 5:  # 至少需要6张有效图像
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, 
        img_points, 
        gray.shape[::-1],  # 图像尺寸 (width, height)
        None, 
        None
    )
    
    # 输出标定结果
    print("\n=== 相机标定结果 ===")
    print(f"重投影误差: {ret:.4f} 像素 (值越小越好)")
    print("\n内参矩阵 (K):")
    print(mtx)
    print("\n畸变系数 (k1, k2, p1, p2, k3):")
    print(dist)
    
    # 评估标定质量
    mean_error = 0
    for i in range(len(obj_points)):
        imgpoints_projected, _ = cv2.projectPoints(
            obj_points[i], rvecs[i], tvecs[i], mtx, dist
        )
        error = cv2.norm(img_points[i], imgpoints_projected, cv2.NORM_L2) / len(imgpoints_projected)
        mean_error += error
    
    print(f"\n平均重投影误差: {mean_error/len(obj_points):.5f} 像素")
    
    # 保存标定结果
    np.savez('camera_params.npz', mtx=mtx, dist=dist)
    print("\n标定参数已保存到 camera_params.npz")
    
    # 可视化畸变校正效果
    test_img = cv2.imread(images[0])
    h, w = test_img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    dst = cv2.undistort(test_img, mtx, dist, None, newcameramtx)
    
    # 裁剪并显示结果
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv2.imshow('Original vs Undistorted', np.hstack((test_img, dst)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("错误：有效图像数量不足，至少需要6张包含完整棋盘格的图像")