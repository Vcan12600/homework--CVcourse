import cv2
import numpy as np
image_path = r"../left2.jpg"
image_path1 = r"right2.jpg"

# 读入图片
left = cv2.imread(image_path)
right = cv2.imread(image_path1)
left = cv2.resize(left, fx=0.4, fy=0.4, dsize=None)
right = cv2.resize(right, fx=0.4, fy=0.4, dsize=None)

# SIFT算子获得关键点和描述子
sift = cv2.SIFT_create()
KeyPoints_L, descriptors_L = sift.detectAndCompute(left, None)
KeyPoints_R, descriptors_R = sift.detectAndCompute(right, None)
np_KeyPoints_L = np.float32([kp.pt for kp in KeyPoints_L])
np_KeyPoints_R = np.float32([kp.pt for kp in KeyPoints_R])

# 匹配描述子，并找到描述子对应的关键点
bf = cv2.BFMatcher(crossCheck=True)
matches = bf.match(descriptors_L, descriptors_R)
matches = sorted(matches, key=lambda x: x.distance)
matches = matches[: 40]
PointIdx = []
for m in matches:
    PointIdx.append((m.trainIdx, m.queryIdx))
tuple_list_kpl = []
tuple_list_kpr = []

for (i, j) in PointIdx:
    tuple_list_kpl.append(KeyPoints_L[j])
    tuple_list_kpr.append(KeyPoints_R[i])


kpl_sort = tuple(tuple_list_kpl)
kpr_sort = tuple(tuple_list_kpr)

img_with_keypointsL = cv2.drawKeypoints(left, kpl_sort, None)
img_with_keypointsR = cv2.drawKeypoints(right, kpr_sort, None)
cv2.imshow('left', img_with_keypointsL)
cv2.imshow('right', img_with_keypointsR)
cv2.waitKey(0)
cv2.destroyAllWindows()

ptsL = np.float32([np_KeyPoints_L[i] for (_, i) in PointIdx])
ptsR = np.float32([np_KeyPoints_R[i] for (i, _) in PointIdx])


# 计算单应矩阵，将右边图片投影到左边图片的坐标系下
H, status = cv2.findHomography(ptsR, ptsL, cv2.RANSAC)
result = cv2.warpPerspective(right, H, (right.shape[1] + left.shape[1], right.shape[0]))
result_clip = result[0:left.shape[0], 0:left.shape[1]]
left_fuse = result_clip


# # 做像素融合
(cols, rows) = (left.shape[0], left.shape[1])
for row in range(0, rows-1):
    for col in range(0, cols-1):
        if result_clip[col][row].any() != 0:
            left_fuse[col][row] = 0.1*result_clip[col][row] + 0.9*left[col][row]
        else:
            left_fuse[col][row] = left[col][row]

result[0:left.shape[0], 0:left.shape[1]] = left_fuse
cv2.imshow('image', result)
cv2.waitKey(0)
cv2.destroyAllWindows()


