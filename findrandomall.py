# import os
# from pathlib import Path
# import shutil
# import pandas as pd

# # 输入文件与路径
# pairs_file = Path("result/randomall/pairs_eval.txt")
# navi_root = Path("navi_v1.0")
# output_root = Path("result/randomall/images")
# output_root.mkdir(parents=True, exist_ok=True)

# # 读取pairs_eval.txt
# pairs = pd.read_csv(pairs_file)

# # 遍历每一对
# for _, row in pairs.iterrows():
#     scene_id = row["scene_id"]
#     img0 = row["img0"]
#     img1 = row["img1"]

#     # 构造源路径
#     img0_path = navi_root / scene_id / "images" / img0
#     img1_path = navi_root / scene_id / "images" / img1

#     # 构造目标路径（保持原目录结构）
#     img0_out = output_root / scene_id / "images" / img0
#     img1_out = output_root / scene_id / "images" / img1

#     # 确保目录存在
#     img0_out.parent.mkdir(parents=True, exist_ok=True)
#     img1_out.parent.mkdir(parents=True, exist_ok=True)

#     # 复制
#     if img0_path.exists():
#         shutil.copy(img0_path, img0_out)
#     else:
#         print(f"缺失文件: {img0_path}")

#     if img1_path.exists():
#         shutil.copy(img1_path, img1_out)
#     else:
#         print(f"缺失文件: {img1_path}")

# print("✅ 所有配对图片已复制到 result/randomall/（保持原始目录结构）")


# import shutil
# from pathlib import Path
# import pandas as pd

# # 输入文件与路径
# # 输入文件与路径
# pairs_file = Path("result/randomall/pairs_eval.txt")
# navi_root = Path("navi_v1.0")
# output_root = Path("result/randomall/images")
# output_root.mkdir(parents=True, exist_ok=True)

# # 读取pairs_eval.txt
# pairs = pd.read_csv(pairs_file)

# # 用集合去重，避免重复拷贝
# all_images = set()

# for _, row in pairs.iterrows():
#     scene_id = row["scene_id"]
#     img0 = row["img0"]
#     img1 = row["img1"]

#     all_images.add((scene_id, img0))
#     all_images.add((scene_id, img1))

# # 遍历所有需要的图片
# for scene_id, img in all_images:
#     src = navi_root / scene_id / "images" / img
#     dst = output_root / scene_id / "images" / img

#     if src.exists():
#         dst.parent.mkdir(parents=True, exist_ok=True)
#         shutil.copy(src, dst)
#     else:
#         print(f"⚠️ 缺失文件: {src}")

# print("✅ 已拷贝 pairs_eval.txt 中所有图片到 result/randomall，保持原始目录层级")
#!/usr/bin/env python3
# import shutil
# from pathlib import Path
# from csv import DictReader

# # 输入文件与路径
# pairs_file = Path("result/randomall/pairs_eval.txt")
# navi_root = Path("navi_v1.0")
# output_root = Path("result/randomall/images")
# output_root.mkdir(parents=True, exist_ok=True)

# def normalize_scene(scene_id: str) -> str:
#     """把 scene_id 中的 -multiview- 规范化为 /multiview-（只替换第一次出现）"""
#     s = scene_id.strip().strip("/").replace("\\", "/")
#     if "-multiview-" in s and "/multiview-" not in s:
#         s = s.replace("-multiview-", "/multiview-", 1)
#     return s

# # 读 pairs，并去重
# need = set()
# with pairs_file.open("r", newline="", encoding="utf-8") as f:
#     reader = DictReader(f)
#     for row in reader:
#         scene_raw = row["scene_id"].strip()
#         img0 = row["img0"].strip()
#         img1 = row["img1"].strip()
#         scene_norm = normalize_scene(scene_raw)
#         need.add((scene_norm, img0))
#         need.add((scene_norm, img1))

# missing = []

# for scene_norm, img in sorted(need):
#     # 主要候选：规范化后的路径
#     src1 = navi_root / scene_norm / "images" / img
#     # 备用候选：原始 scene_id（万一本来就带 /）
#     src2 = navi_root / scene_norm.replace("/multiview-", "-multiview-") / "images" / img

#     src = src1 if src1.exists() else (src2 if src2.exists() else None)

#     if src is None:
#         missing.append(str(src1))
#         print(f"⚠️ 缺失文件: {src1}")
#         continue

#     dst = output_root / scene_norm / "images" / img
#     dst.parent.mkdir(parents=True, exist_ok=True)
#     # 已存在就不重复复制
#     if not dst.exists():
#         shutil.copy(src, dst)

# # 写一个缺失清单，方便排查
# if missing:
#     miss_file = output_root / "missing_paths.txt"
#     miss_file.write_text("\n".join(missing), encoding="utf-8")
#     print(f"完成，但有 {len(missing)} 个缺失，清单见: {miss_file}")
# else:
#     print("✅ 全部拷贝完成，且无缺失。目标目录：", output_root)



#!/usr/bin/env python3
import shutil
from pathlib import Path
from csv import DictReader

# 输入文件与路径
pairs_file = Path("result/randomall/pairs_eval.txt")
navi_root = Path("navi_v1.0")
output_root = Path("result/randomall/images")
output_root.mkdir(parents=True, exist_ok=True)

def normalize_scene(scene_id: str) -> str:
    """
    规范化 scene_id：
    1) 把第一次出现的 '-multiview-' 改成 '/multiview-'
    2) 把 '-wild_set' 改成 '/wild_set'
    仅在目标片段尚未是目录形式时才替换。
    """
    s = scene_id.strip().strip("/").replace("\\", "/")
    if "-multiview-" in s and "/multiview-" not in s:
        s = s.replace("-multiview-", "/multiview-", 1)
    if "-wild_set" in s and "/wild_set" not in s:
        s = s.replace("-wild_set", "/wild_set")
    return s

# 收集需要的所有图片（去重）
need = set()
with pairs_file.open("r", newline="", encoding="utf-8") as f:
    reader = DictReader(f)
    for row in reader:
        scene_raw = row["scene_id"].strip()
        img0 = row["img0"].strip()
        img1 = row["img1"].strip()
        scene_norm = normalize_scene(scene_raw)
        need.add((scene_norm, img0))
        need.add((scene_norm, img1))

missing = []

for scene_norm, img in sorted(need):
    # 优先用规范化后的路径
    candidates = [
        navi_root / scene_norm / "images" / img,
    ]
    # 回退到原始 scene_id（如果原始是不同写法，避免漏拷）
    # 反向把 /multiview- 改回 -multiview-，/wild_set 改回 -wild_set
    scene_back = scene_norm.replace("/multiview-", "-multiview-").replace("/wild_set", "-wild_set")
    if scene_back != scene_norm:
        candidates.append(navi_root / scene_back / "images" / img)

    src = next((p for p in candidates if p.exists()), None)

    if src is None:
        # 记录缺失（按期望的规范化路径）
        expected = navi_root / scene_norm / "images" / img
        print(f"⚠️ 缺失文件: {expected}")
        missing.append(str(expected))
        continue

    dst = output_root / scene_norm / "images" / img
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy(src, dst)

# 输出缺失清单，方便排查
if missing:
    miss_file = output_root / "missing_paths.txt"
    miss_file.write_text("\n".join(missing), encoding="utf-8")
    print(f"完成，但有 {len(missing)} 个缺失，清单见: {miss_file}")
else:
    print(f"✅ 全部拷贝完成，无缺失。目标目录：{output_root}")

