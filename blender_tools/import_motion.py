import bpy
from bpy.props import StringProperty, BoolProperty
from bpy.types import Operator
from bpy_extras.io_utils import ExportHelper

import os
import numpy as np
import mathutils
from mathutils import Matrix, Vector, Quaternion


posebones = bpy.context.object.pose.bones
bones = bpy.context.object.data.bones

quats = np.load('/home/justin/repos/mujoco-python-viewer/quat_data.npz')
pos = np.load('/home/justin/repos/mujoco-python-viewer/pos_data.npz')

def set_quat(pb, t):
    q = Quaternion(quats[pb.name][t])
    cur_orient = bones[pb.name].matrix_local.to_quaternion()
    q.rotate(cur_orient)
    pb.matrix = q.to_matrix().to_4x4()    
    bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
    for child in pb.children:
        set_quat(child, t)
        
def set_pos(pb, t):
    translate = Matrix.Translation(pos[pb.name][t])
    pb.matrix = translate @ pb.matrix

for i in range(0, len(quats["Root"]), 10):
    set_quat(posebones["Root"], i)
    set_pos(posebones["Root"], i)
    frame_n = i/2
    for pb in posebones:
        pb.keyframe_insert("rotation_quaternion", frame= frame_n)
        pb.keyframe_insert("location", frame = frame_n)

