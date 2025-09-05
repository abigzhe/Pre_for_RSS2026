# Imports (Check requirements.txt)
%load_ext autoreload
%autoreload 2

import glob
import json
import os
import random
from typing import Tuple, Dict

from IPython.display import display
import ipywidgets as widgets
from PIL import Image
import cv2
import colour
from matplotlib import pyplot as plt
import mediapy as media
import numpy as np
import torch as t
import trimesh

# NAVI imports.
import data_util
import mesh_util
import transformations
import visualization
from gl import scene_renderer
from gl import camera_util

disp = visualization.display_images
#resize_by：匿名函数，用来把图像缩小 y 倍。
resize_by = lambda x, y: x.resize((x.size[0]//y, x.size[1]//y))

navi_release_root = '../navi_v1.0/'
# Load the mesh and the images.

# Multi-view scene examples.
# query = 'duck_bath_yellow_s-multiview-00-pixel_4a'
query = 'duck_bath_yellow_s-multiview-01-pixel_4a'

# Video scene example (added at navi_v1.5).
#query = 'schleich_lion_action_figure-video-05-pixel_7-PXL_20230728_004915170.TS'

# Load the multi-view scene data.
#加载场景数据：
# annotations → 每张图像的标注（相机内参、外参）。
# mesh → 对应物体的高精度 3D 扫描网格。
# images → 图像集合。
# video → 如果是视频场景会返回视频帧；这里是 multiview，所以 video=None。
annotations, mesh, images, video = data_util.load_scene_data(
  query, navi_release_root, load_video=False)

# 从 mesh 中提取出渲染所需信息：
# triangles → 三角面片坐标。
# triangle_colors → 三角形颜色。
# material_ids → 材质信息。
triangles, triangle_colors, material_ids = (
    visualization.prepare_mesh_rendering_info(mesh))
  
# Create a sphere (for the camera centers).
'''创建一个半径为 5. 的小球。
把小球转成三角形表示，再转成 tensor。
后面会把小球放到 相机位置，直观显示每个相机的采集点。
'''
sphere = trimesh.primitives.Sphere(radius=5.)
sphere_vert = data_util.convert_to_triangles(
    sphere.vertices, sphere.faces)
sphere_vert = t.tensor(sphere_vert)

# Get the camera center spheres.
'''遍历每个相机标注：
camera_matrices_from_annotation(anno) → 得到 object_to_world 矩阵（物体坐标到世界坐标）。
取逆 → 得到 camera_to_world（相机在世界中的位置）。
把小球 sphere_vert 变换到 camera_to_world 位置。
camera_spheres 最终存储了所有相机位置的小球。'''
camera_spheres = []
for i_anno, anno in enumerate(annotations):
  #object_to_world这个矩阵本质上是**把“世界/物体坐标”变换到“相机坐标”**的外参
  object_to_world, _ = data_util.camera_matrices_from_annotation(anno)
  #这样 camera_to_world 的平移部分就是相机中心在世界系中的位置，旋转部分就是相机坐标轴在世界系中的朝向。
  camera_to_world = t.linalg.inv(object_to_world)
  camera_spheres.append(
      transformations.transform_mesh(sphere_vert, camera_to_world))
# For Depth visualization.
min_depth = 100  # in mm.
max_depth = 400  # in mm.
#disp_width、resize_factor 只是显示用的，不影响真实渲染/计算。
disp_width = 200  # 左侧小图在 notebook 中的显示宽度（像素）
resize_factor = 8 # 用于把原图缩小 1/8 进行预览

if video is not None:
  media.show_video(video, width=disp_width, title='Video')

# @widgets.interact 会生成一个滑动条（frame_index），范围从 0 到 len(annotations)-1，步长 1。
#当你拖动滑条或输入索引，函数会被重新调用并更新可视化：这是 Notebook 里常用的交互模式。
@widgets.interact
def display_multiview(frame_index=(0, len(annotations)-1, 1)):
  '''
  image：PIL Image 或类似对象（RGB 图）。
  anno：该帧的 annotation（字典），包含相机参数、图像尺寸等。
  camera_matrices_from_annotation(anno) 返回两个矩阵：
  **第一个（这里命名为 object_to_world）是把 mesh 从“物体/世界 坐标系”变换到“相机坐标系”**的 4×4 齐次矩阵（即 world→camera）；
  第二个 intrinsics 在这里被当作 view_projection_matrix（用于投影渲染）。
  h, w = anno['image_size']：取图像尺寸。
  '''
  image = images[frame_index]
  anno = annotations[frame_index]
  object_to_world, intrinsics = data_util.camera_matrices_from_annotation(anno)
  h, w = anno['image_size']
  
  # Render the 3D model alignment.
  #把 mesh 对齐到相机并渲染（得到对齐图+掩码+深度）
  triangles_aligned = transformations.transform_mesh(
      triangles, object_to_world)
  rend = scene_renderer.render_scene(
      triangles_aligned, view_projection_matrix=intrinsics,
      material_ids=material_ids, diffuse_coefficients=triangle_colors,
      output_type=t.float32, clear_color=(0,0,0,0),
      image_size=(h, w), cull_back_facing=False, return_rgb=False)
  rendering = rend[:,:,:3].numpy()
  rendering = Image.fromarray((255*rendering/rendering.max()).astype(np.uint8))
  mask = rend.numpy().mean(axis=2) > 0
  mask = Image.fromarray((mask * 255).astype(np.uint8))
  depth = visualization.apply_colors_to_depth_map(rend[:,:,3].numpy(), minn=100, maxx=max_depth)
  depth = Image.fromarray(depth)
  

  # Resize and display the images.
  image_resize = resize_by(image, resize_factor)
  rendering_resize = resize_by(rendering, resize_factor)
  mask_resize = resize_by(mask, resize_factor)
  depth_resize = resize_by(depth, resize_factor)
  disp(image_resize, disp_width, 'RGB Image', rendering_resize, disp_width, 'Alignment',
        mask_resize, disp_width, 'Binary Mask', depth, disp_width, "Metric Depth")


  # Render the 3D object with the camera poses.
  camera_triangles = t.concat(camera_spheres, axis=0)
  camera_colors = t.tensor([[0.5, 0., 0.], [0., 1., 0]])
  camera_material_ids = t.ones(camera_triangles.shape[0], dtype=t.int32)
  vert_index_start = frame_index * len(sphere_vert)
  vert_index_end = (frame_index + 1) * len(sphere_vert)
  camera_material_ids[vert_index_start:vert_index_end] += 1
  triangles_all = t.concat((triangles, camera_triangles), axis=0)
  material_ids_all = t.concat(
      (material_ids, material_ids.numpy().max() + camera_material_ids), axis=0)
  colors_all = t.concat((triangle_colors, camera_colors), axis=0)

  view_projection_matrix = camera_util.get_default_camera_for_mesh(
    triangles_all, move_away_mul=1.3, camera_index=3)
  rend_with_cameras = scene_renderer.render_scene(
      triangles_all, diffuse_coefficients=colors_all,
      material_ids=material_ids_all,
      view_projection_matrix=view_projection_matrix,
      image_size=(1024, 1024),
      cull_back_facing=False, clear_color=(1, 1, 1))
  disp(rend_with_cameras, 512, 'Multiview scene with camera positions.')
# Load all 'wild_set' annotations.
annotation_paths_wild = glob.glob(os.path.join(
    navi_release_root, '*', 'wild_set', 'annotations.json'))
annotations_wild = []
for annotation_path in annotation_paths_wild:
  with open(annotation_path, 'r') as f:
    annotations_wild.append(json.load(f))
random.shuffle(annotations_wild)
max_num_objects = 4                # 最多显示多少个物体
max_num_images_per_object = 5      # 每个物体最多显示多少张 in-the-wild 照片
disp_size = 180                    # 显示图片时的宽度（像素）
resize_factor = 16                 # 缩小倍数，加快显示（分辨率降低）


for i_object, anno_object in enumerate(annotations_wild):
  if i_object >= max_num_objects:
    break

  # Load the scene data.
  object_id = anno_object[0]['object_id']
  query = f'{object_id}-wild_set'
  annotations, mesh, images_wild_set, _ = data_util.load_scene_data(
    query, navi_release_root, max_num_images=max_num_images_per_object)

  # Load all images.
  #把所有原图缩小 1/16，加快 Notebook 展示。
  images_resize = [resize_by(image, resize_factor) for image in images_wild_set]

  overlays = []
  for i_anno in range(len(images_wild_set)):
    #anno_object[i_anno] → 当前图片的相机参数。
    #w, h → 当前缩小后图片的宽高（注意顺序是 (width, height)）。
    anno = anno_object[i_anno]
    w, h = images_resize[i_anno].size
  
    # Convert camera poses and intrinsics into matrices.
    object_to_world, intrinsics = data_util.camera_matrices_from_annotation(anno)
  
    # Render the mesh.
    rend = visualization.render_navi_scan(
        mesh, extrinsics=object_to_world, intrinsics=intrinsics,
        image_size=(h, w), with_texture=True)
    overlay = visualization.overlay_images(
        np.array(images_resize[i_anno]), rend)
    overlays.append(overlay)
  visualization.display_multiple_images(images_resize, disp_size, 'RGB', 'width')
  visualization.display_multiple_images(overlays, disp_size, 'Alignment', 'width')
  print('\n')
  #把 3D 点投影到图像上，并检查哪些点在当前相机视角下可见。
def project_and_filter_sample_coordinates(
    mesh_triangles: t.tensor, annotation, sampled_points: t.tensor,
    image: Image.Image) -> Dict[int, Tuple[int, int]]:
  """Returns the sampled points, projected on the image, that are visible from the current view."""
  """返回那些投影到当前图像上、且在该视角下可见的采样点坐标"""
  object_to_world, intrinsics = data_util.camera_matrices_from_annotation(annotation)

  # Render the 3D model alignment.
  #渲染深度图
  '''把 mesh 变换到相机坐标系，调用 scene_renderer 渲染。
  输出 rend 通常是 (H, W, 4)，前三通道 RGB，第四通道是深度。
  depth = rend[:,:,3] 提取每个像素的深度值（OpenGL 深度缓冲）。'''
  mesh_triangles_aligned = transformations.transform_mesh(
      mesh_triangles, object_to_world)
  rend = scene_renderer.render_scene(
      mesh_triangles_aligned, view_projection_matrix=intrinsics,
      output_type=t.float32, clear_color=(0,0,0,0),
      image_size=image.size[::-1], cull_back_facing=False, return_rgb=False)
  depth = rend[:, :, 3].numpy()

  # Align the sampled points.
  #把采样点投影到图像平面
  '''sampled_points：3D 空间中的采样点（在物体坐标系里）。
  第一步 transform_points(..., object_to_world) → 把它们变换到相机坐标系。
  第二步 transform_points(..., intrinsics) → 应用相机投影，把点投到标准化设备坐标 (NDC)。'''
  sampled_points_world = transformations.transform_points(
      sampled_points, object_to_world)
  sampled_points_screen = transformations.transform_points(
      sampled_points_world, intrinsics)

  # Convert from OpenGL space to image space.
  #从 NDC 转换到图像像素坐标
  sampled_points_screen += t.tensor([1., 1., 0])
  sampled_points_screen *= t.tensor([image.size[0]/2, image.size[1]/2, 1])
  #拼接坐标和深度
  samples = t.concat(
      (sampled_points_screen[:, :2], sampled_points_world[:, 2:3]),
      dim=1).numpy()

  # Discard points where the depth doesn't match the OpenGL depth buffer.
  #可见性过滤
  coords = {}
  for i_sample, sample in enumerate(samples):
    y = round(sample[1])
    x = round(sample[0])
    z = sample[2]
    if abs(depth[y, x] - z) < 1:
      coords[i_sample] = (y, x)
  return coords

#把两张图的对应点画出来，方便可视化。
def show_correspondences(image_1: Image.Image, image_2: Image.Image,
                         corresp_dict_1: Dict[int, Tuple[int, int]],
                         corresp_dict_2: Dict[int, Tuple[int, int]], resize_factor=1) -> None:
  """Display the intersection of valid correspondences between two images."""
  '''输入：两张图和它们各自的点坐标字典。字典格式是：点编号 → (y, x)，即像素位置。'''

  '''转成 numpy 数组。如果两张图高度不同，用 pad 填充到相同高度。拼接在一起（左边是图1，右边是图2）。再整体缩小一倍（加快显示）。'''
  image_1 = np.array(image_1)
  image_2 = np.array(image_2)
  h1, w1 = image_1.shape[:2]
  h2, w2 = image_2.shape[:2]

  # Handle images of different shapes (in the wild_set images).
  if h1 != h2:
    h_max = max(h1, h2)
    image_1 = np.pad(image_1, [[0, h_max-h1], [0, 0], [0, 0]])
    image_2 = np.pad(image_2, [[0, h_max-h2], [0, 0], [0, 0]])

  # Concatenate the two images to display the correspondences.
  img_corresp = np.concatenate((image_1, image_2), axis=1)
  img_corresp = cv2.resize(
      img_corresp,
      (img_corresp.shape[1] // resize_factor, img_corresp.shape[0] // resize_factor))



  '''把图1的点做成列表 (点编号, y, x)。按 y 排序（这样画出来颜色渐变更美观）。生成从红色到蓝色的渐变色，数量等于点的数量。'''
  # Sort the correspondences of the left images by Y-coordinate
  corresp_1_as_list = [(k, *v) for k, v in corresp_dict_1.items()]
  corresp_1_as_list = sorted(corresp_1_as_list, key=lambda x: x[1])

  # Create the color gradient.
  red = colour.Color("red")
  colors = list(red.range_to(colour.Color("blue"), len(corresp_1_as_list)))
  #绘制对应线
  plt.figure(figsize=(12, 17))
  plt.axis('off')
  plt.imshow(img_corresp)
  for color_idx, (corresp_idx, y1, x1) in enumerate(corresp_1_as_list):
    if corresp_idx in corresp_dict_2:
      y2, x2 = corresp_dict_2[corresp_idx]
      x = [x1 / resize_factor, (x2 + w1) / resize_factor]
      y = [y1 / resize_factor, y2 / resize_factor]
      plt.plot(x, y, color=colors[color_idx].rgb, marker='o')