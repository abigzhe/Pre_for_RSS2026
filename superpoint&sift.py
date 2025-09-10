import cv2
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from pathlib import Path
import random


# 固定参数
# 模型权重目录（绝对路径）
weights_dir = Path("/data1/home/limingzhe/abigzhe_dinopro/SuperPoint/pretrained_models/sp_v6")

# 输入 / 输出路径
input_dir = Path("/data1/home/limingzhe/abigzhe_dinopro/navi_v1.0")
output_dir = Path("/data1/home/limingzhe/abigzhe_dinopro/result/all/superpoint/point")
vis_dir = output_dir / "vis"
output_dir.mkdir(parents=True, exist_ok=True)
vis_dir.mkdir(parents=True, exist_ok=True)



def extract_superpoint_keypoints_and_descriptors(keypoint_map, descriptor_map,
                                                 keep_k_points=1000):
    def select_k_best(points, k):
        idx = points[:, 2].argsort()
        pts_sorted = points[idx, :2]
        start = min(k, points.shape[0])
        return pts_sorted[-start:, :]

    ys, xs = np.where(keypoint_map > 0)
    if len(xs) == 0:
        return [], np.empty((0, descriptor_map.shape[-1]), dtype=np.float32)

    prob = keypoint_map[ys, xs]
    pts = np.stack([ys, xs, prob], axis=-1)

    pts = select_k_best(pts, keep_k_points).astype(int)
    desc = descriptor_map[pts[:, 0], pts[:, 1]]

    kps = [cv2.KeyPoint(int(x), int(y), 1) for y, x in pts]
    return kps, desc.astype(np.float32)


def preprocess_image(img_file):
    img = cv2.imread(img_file, cv2.IMREAD_COLOR)
    assert img is not None, f"无法读取图像: {img_file}"
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float32) / 255.0
    gray = np.expand_dims(gray, axis=-1)  # [H,W,1]
    return gray, img


def save_features(out_file, keypoints, descriptors):
    out_file.parent.mkdir(parents=True, exist_ok=True)  # 确保目录存在
    kps = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints], dtype=np.float32)
    np.savez(out_file, keypoints=kps, descriptors=descriptors)
    print(f"✅ 保存特征到 {out_file}")


def extract_SIFT_keypoints_and_descriptors(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if hasattr(cv2, "SIFT_create"):
        sift = cv2.SIFT_create()
    else:
        sift = cv2.xfeatures2d.SIFT_create()
    kp, desc = sift.detectAndCompute(gray, None)
    return kp, desc


def visualize_side_by_side(img, sp_kps, sift_kps):
    """生成左右拼接对比图"""
    # SuperPoint
    sp_vis = cv2.drawKeypoints(img, sp_kps, None, color=(0, 255, 0))
    cv2.putText(sp_vis, "SuperPoint", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    # SIFT
    sift_vis = cv2.drawKeypoints(img, sift_kps, None, color=(0, 0, 255))
    cv2.putText(sift_vis, "SIFT", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

    # 拼接
    vis = np.hstack((sp_vis, sift_vis))
    return vis
def visualize_and_save(img_bgr, sp_kps, sift_kps, out_path):
    # 拷贝两份
    img_sp = img_bgr.copy()
    img_sift = img_bgr.copy()

    # --- SuperPoint点 ---
    for kp in sp_kps:
        x, y = map(int, kp.pt)
        cv2.circle(img_sp, (x, y), radius=6, color=(0, 0, 255), thickness=-1)  # 绿色实心点

    # --- SIFT点 ---
    for kp in sift_kps:
        x, y = map(int, kp.pt)
        cv2.circle(img_sift, (x, y), radius=6, color=(255, 0, 0), thickness=-1)  # 蓝色实心点

    # 拼接图像
    h = max(img_sp.shape[0], img_sift.shape[0])
    w = img_sp.shape[1] + img_sift.shape[1]
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    vis[:img_sp.shape[0], :img_sp.shape[1]] = img_sp
    vis[:img_sift.shape[0], img_sp.shape[1]:] = img_sift

    # 左上角标签
    cv2.putText(vis, "SuperPoint", (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 4.0, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(vis, "SIFT", (img_sp.shape[1] + 30, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 4.0, (255, 255, 255), 3, cv2.LINE_AA)

    # 保存
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)
    print(f"💾 可视化保存到 {out_path}")


from tqdm import tqdm   # ✅ 新增，用来显示进度条

def main():
    img_paths = sorted(input_dir.rglob("*.jpg"))
    print(f"找到 {len(img_paths)} 张图片")

    # ✅ 随机选 5 张做可视化
    vis_samples = random.sample(img_paths, min(5, len(img_paths)))

    graph = tf.Graph()
    with tf.Session(graph=graph) as sess:
        tf.saved_model.loader.load(
            sess,
            [tf.saved_model.tag_constants.SERVING],
            str(weights_dir)
        )

        input_img_tensor = graph.get_tensor_by_name('superpoint/image:0')
        output_prob_nms_tensor = graph.get_tensor_by_name('superpoint/prob_nms:0')
        output_desc_tensors = graph.get_tensor_by_name('superpoint/descriptors:0')

        # ✅ tqdm 进度条
        for idx, img_file in enumerate(tqdm(img_paths, desc="Processing images")):
            gray, img_orig = preprocess_image(str(img_file))

            # ---- SuperPoint ----
            prob, descmap = sess.run(
                [output_prob_nms_tensor, output_desc_tensors],
                feed_dict={input_img_tensor: np.expand_dims(gray, 0)}
            )
            keypoint_map = np.squeeze(prob)
            descriptor_map = np.squeeze(descmap)

            sp_kps, sp_desc = extract_superpoint_keypoints_and_descriptors(
                keypoint_map, descriptor_map, keep_k_points=1000
            )

            # ✅ 去掉 "images" 文件夹
            rel_path = img_file.relative_to(input_dir)
            if "images" in rel_path.parts:
                parts = list(rel_path.parts)
                parts.remove("images")
                rel_path = Path(*parts).with_suffix("")
            else:
                rel_path = rel_path.with_suffix("")

            out_file = output_dir / rel_path

            # ---- 保存 npz ----
            save_features(out_file.with_suffix(".superpoint.npz"), sp_kps, sp_desc)

            # ---- SIFT ----
            sift_kps, sift_desc = extract_SIFT_keypoints_and_descriptors(img_orig)
            if sift_desc is None:
                sift_desc = np.empty((0, 128), dtype=np.float32)
                sift_kps = []
            save_features(out_file.with_suffix(".sift.npz"), sift_kps, sift_desc)

            # ---- 只可视化随机5张 ----
            if img_file in vis_samples:
                vis_img_path = vis_dir / (rel_path.name + "_compare.jpg")
                visualize_and_save(img_orig, sp_kps, sift_kps, vis_img_path)

        print("✅ 全部处理完成！")




if __name__ == "__main__":
    main()