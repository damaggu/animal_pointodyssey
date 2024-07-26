import bpy
import numpy as np

bpy.ops.wm.open_mainfile(filepath="/home/justin/repos/animal-pointodyssey/data/blender_assets/postprocess.blend")


bpy.context.scene.node_tree.nodes["Movie Clip"].clip = bpy.data.movieclips.load("/home/justin/repos/animal-pointodyssey/results/gen2/mouse_6_3c_07-19T03:52:18/video.mp4")


bpy.context.scene.render.filepath = "bw_output"
for i in range(0, bpy.context.scene.frame_end, 20):
    bpy.data.scenes["Scene"].node_tree.nodes["Exposure"].inputs[1].default_value = np.random.normal(5, 0.1)
    bpy.data.scenes["Scene"].node_tree.nodes["Exposure"].inputs[1].keyframe_insert("default_value", frame = i)
    bpy.data.scenes["Scene"].node_tree.nodes["Gamma"].inputs[1].default_value = np.random.normal(2.5, 0.1)
    bpy.data.scenes["Scene"].node_tree.nodes["Gamma"].inputs[1].keyframe_insert("default_value", frame = i)
    bpy.data.scenes["Scene"].node_tree.nodes["Blur"].size_x = int(np.round(np.random.uniform(2, 3)))
    bpy.data.scenes["Scene"].node_tree.nodes["Blur"].size_y = int(np.round(np.random.uniform(2, 3)))
    bpy.data.scenes["Scene"].node_tree.nodes["Blur"].keyframe_insert("size_x", frame = i)
    bpy.data.scenes["Scene"].node_tree.nodes["Blur"].keyframe_insert("size_y", frame=i)
bpy.ops.render.render(animation=True)