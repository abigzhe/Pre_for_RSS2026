#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
from pathlib import Path

# ========== 配置 ==========
root_dir = Path("/data1/home/limingzhe/abigzhe_dinopro/navi_v1.0")
output_file = root_dir / "pairs.txt"
num_pairs_per_dir = 10  # 每个 images 目录抽取的配对数

def main():
    random.seed(42)  # 固定随机种子

    # 找到所有 .../images 目录（两级子目录下 images 文件夹）
    image_dirs = list(root_dir.rglob("images"))
    print(f"找到 {len(image_dirs)} 个 images 目录")

    all_pairs = []

    for img_dir in image_dirs:
        images = sorted([p for p in img_dir.iterdir()
                         if p.is_file() and p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
        if len(images) < 2:
            print(f"⚠️ 目录 {img_dir} 图片不足 2 张，跳过")
            continue

        # 所有可能的不重复组合
        all_combinations = [(images[i], images[j])
                            for i in range(len(images))
                            for j in range(i + 1, len(images))]

        # 随机抽取
        sample_num = min(num_pairs_per_dir, len(all_combinations))
        sampled_pairs = random.sample(all_combinations, sample_num)

        # 添加到总列表
        for img1, img2 in sampled_pairs:
            all_pairs.append((img1, img2))

    # 写入 pairs.txt
    with open(output_file, "w") as f:
        for img1, img2 in all_pairs:
            f.write(f"{img1} {img2}\n")

    print(f"总共写入 {len(all_pairs)} 对图片到 {output_file}")

if __name__ == "__main__":
    main()
