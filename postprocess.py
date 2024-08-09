import bpy
import numpy as np
import os
import shutil
import albumentations as A
import cv2
from functools import partial
def postprocess(path, name, output_path = "./results/animal_pod/", monochrome = False, overexpose = False, blur_strength = 3):

    bpy.ops.wm.open_mainfile(filepath="/home/justin/repos/animal-pointodyssey/data/blender_assets/postprocess.blend")
    vid_path = os.path.join(output_path, name)
    image_path = os.path.join(path, "frames")
    images = os.listdir(image_path)
    images.sort()
    im = bpy.data.images.load(os.path.join(image_path, images[0]))
    im.source = "SEQUENCE"
    bpy.context.scene.node_tree.nodes["Image"].image = im
    bpy.data.scenes["Scene"].node_tree.nodes["Image"].frame_start = 0
    bpy.data.scenes["Scene"].node_tree.nodes["Image"].frame_offset = -1
    bpy.data.scenes["Scene"].render.resolution_x = im.size[0]
    bpy.data.scenes["Scene"].render.resolution_y = im.size[1]
    bpy.data.scenes["Scene"].frame_start = 0
    bpy.data.scenes["Scene"].frame_end = len(images) - 1
    bpy.context.scene.render.filepath = os.path.join(vid_path, "frames/")
    bw_node = bpy.data.scenes['Scene'].node_tree.nodes["RGB to BW"]
    image_node = bpy.data.scenes['Scene'].node_tree.nodes["Image"]
    gamma_node = bpy.data.scenes['Scene'].node_tree.nodes["Gamma"]
    if monochrome:
        bpy.context.scene.node_tree.links.new(bw_node.outputs[0], gamma_node.inputs[0])
    else:
        bpy.context.scene.node_tree.links.new(image_node.outputs[0], gamma_node.inputs[0])

    for i in range(0, bpy.context.scene.frame_end, 5):
        if overexpose:
            bpy.data.scenes["Scene"].node_tree.nodes["Exposure"].inputs[1].default_value = np.random.normal(5, 0.1)
            bpy.data.scenes["Scene"].node_tree.nodes["Exposure"].inputs[1].keyframe_insert("default_value", frame = i)
            bpy.data.scenes["Scene"].node_tree.nodes["Gamma"].inputs[1].default_value = np.random.normal(2.5, 0.1)
            bpy.data.scenes["Scene"].node_tree.nodes["Gamma"].inputs[1].keyframe_insert("default_value", frame = i)
        bpy.data.scenes["Scene"].node_tree.nodes["Blur"].size_x = int(np.round(np.random.uniform(0, blur_strength)))
        bpy.data.scenes["Scene"].node_tree.nodes["Blur"].size_y = int(np.round(np.random.uniform(0, blur_strength)))
        bpy.data.scenes["Scene"].node_tree.nodes["Blur"].keyframe_insert("size_x", frame = i)
        bpy.data.scenes["Scene"].node_tree.nodes["Blur"].keyframe_insert("size_y", frame=i)
    bpy.ops.wm.save_as_mainfile(filepath="/home/justin/repos/animal-pointodyssey/p.blend")
    bpy.ops.render.render(animation=True)
    traj_file = [x for x in os.listdir(path) if ".npy" in x][0]
    shutil.copy(os.path.join(path, traj_file), os.path.join(output_path, name + "/" + name + ".npy"))
    video_script = f"ffmpeg -f image2 -r 12 -pattern_type glob -i '{vid_path}/frames/*.png' -vcodec libx264 -crf 22 '{output_path}/{name}.mp4'"
    os.system(video_script)




if __name__ == "__main__":
    pre_path = "./results/dataset"
    output_path = "./results/animal_pod/"
    os.makedirs(output_path, exist_ok=True)
    videos = os.listdir(pre_path)
    videos.sort()
    existing = [x for x in os.listdir(output_path) if os.path.isdir(os.path.join(output_path, x))]
    existing.sort()

    n = 0
    if len(existing) != 0:
        n = int(existing[-1]) + 1
    # Declare an augmentation pipeline
    # bw_transform = A.ToGray(always_apply=True)
    # exposure_transform = A.Compose([])
    # for video in videos:
    #     cur_path = os.path.join(pre_path, video)
    #     frames = sorted(os.listdir(cur_path))
    #     for f in frames
    for video in videos:
        cur_path = os.path.join(pre_path, video)
        print(cur_path)
        postprocess(path = cur_path, name = str(n).zfill(4), monochrome = False, overexpose = False)
        n += 1
        postprocess(path=cur_path, name=str(n).zfill(4), monochrome=True, overexpose=False)
        n += 1
        postprocess(path=cur_path, name=str(n).zfill(4), monochrome=False, overexpose=True)
        n += 1
        postprocess(path=cur_path, name=str(n).zfill(4), monochrome=True, overexpose=True)
        n += 1