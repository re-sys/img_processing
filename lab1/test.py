import cv2
import numpy as np
src_image = cv2.imread('rice.tif', cv2.IMREAD_UNCHANGED)
def bilinear_wufengyang(src_image, dst_dim):
    dst_image = np.zeros(dst_dim, dtype=src_image.dtype)
    src_width, src_height = src_image.shape[1], src_image.shape[0]
    dst_width, dst_height = dst_image.shape[1], dst_image.shape[0]
    src_y = np.linspace(0, src_width-1, dst_width)
    src_x = np.linspace(0, src_height-1, dst_height)
    src_x1 = np.floor(src_x).astype(np.int32)
    src_x2 = np.ceil(src_x).astype(np.int32)
    src_y1 = np.floor(src_y).astype(np.int32)
    src_y2 = np.ceil(src_y).astype(np.int32)
    for i in range(dst_height):
        Q11 = src_image[src_x1[i], src_y1]
        Q21 = src_image[src_x2[i], src_y1]
        Q12 = src_image[src_x1[i], src_y2]
        Q22 = src_image[src_x2[i], src_y2]
        if src_x1[i] != src_x2[i]:
            w11 = (src_x2[i] - src_x[i]) * (src_y2 - src_y)
            w21 = (src_x[i] - src_x1[i]) * (src_y2 - src_y)
            w12 = (src_x2[i] - src_x[i]) * (src_y - src_y1)
            w22 = (src_x[i] - src_x1[i]) * (src_y - src_y1)
            dst_image[i,:] = w11 * Q11 + w21 * Q21 + w12 * Q12 + w22 * Q22+(src_y1==src_y2)*((src_x2[i] - src_x[i])*Q11+(src_x[i] - src_x1[i])*Q21)
            # dst_image[i,src_y1==src_y2] = Q11[src_y1==src_y2]
        else:
            w11 = (src_y2 - src_y)
            w21 = (src_y - src_y1)
            dst_image[i,:] = w11 * Q11 + w21 * Q12 +(src_y1==src_y2)*Q11
            
    return dst_image
dst_dim = (round(src_image.shape[0]*1.4), round(src_image.shape[1]*1.4))
dst2 = bilinear_wufengyang(src_image, dst_dim)
dst3 = cv2.resize(src_image, dst_dim, interpolation=cv2.INTER_LINEAR)