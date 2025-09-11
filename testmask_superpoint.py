import cv2
import numpy as np

# 假设我们有一张原图和掩膜
img_file = "/data1/home/limingzhe/abigzhe_dinopro/navi_v1.0/schleich_hereford_bull/multiview-02-pixel_5/images/002.jpg"
mask_file = "/data1/home/limingzhe/abigzhe_dinopro/navi_v1.0/schleich_hereford_bull/multiview-02-pixel_5/masks/002.png"
img = cv2.imread(img_file)
mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)

h, w = mask.shape[:2]

# 生成 1000 个随机 KeyPoint
num_points = 1000
xs = np.random.randint(0, w, size=num_points)
ys = np.random.randint(0, h, size=num_points)
kps = [cv2.KeyPoint(float(x), float(y), 1) for x, y in zip(xs, ys)]
# 随机生成描述子 (1000 x 256)
desc = np.random.rand(len(kps), 256).astype(np.float32)

# 过滤在掩膜内的点
filtered_kps = []
filtered_desc = []
for kp, d in zip(kps, desc):
    x, y = int(round(kp.pt[0])), int(round(kp.pt[1]))
    if mask[y, x] > 0:  # 掩膜值大于 0
        filtered_kps.append(kp)
        filtered_desc.append(d)

filtered_desc = np.array(filtered_desc, dtype=np.float32)

print(f"原始点数: {len(kps)}, 掩膜内点数: {len(filtered_kps)}")

# 可视化结果
img_vis = img.copy()
for kp in filtered_kps:
    x, y = int(round(kp.pt[0])), int(round(kp.pt[1]))
    cv2.circle(img_vis, (x, y), 3, (0, 255, 0), -1)  # 绿色实心点

cv2.imwrite("test_filtered_points.jpg", img_vis)
print("过滤后的点已画在 test_filtered_points.jpg 上")
