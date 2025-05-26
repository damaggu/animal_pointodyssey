import bpy
from bpy.props import StringProperty, BoolProperty
from bpy.types import Operator
from bpy_extras.io_utils import ExportHelper

import os
import numpy as np
import mathutils
from mathutils import Matrix, Vector, Quaternion
import itertools



def find_root_bone(armature):
    bones = armature.data.bones
    root_bone = None
    for b in bones:
        if b.parent == None:
            root_bone = b.name
    return root_bone
def reset_quat(pb):
    q = Quaternion((1, 0, 0, 0))
    pb.rotation_quaternion = q
    for child in pb.children:
        reset_quat(child)


def set_quat(pb, traj, t):
    if pb.name not in traj.keys():
        pb.bone.use_inherit_rotation = True
        pb.rotation_quaternion = Quaternion()
    else:
        pb.bone.use_inherit_rotation = False
        q = pb.bone.matrix_local.to_quaternion()
        q.rotate(Quaternion((traj[pb.name][t][3:])))
        q.rotate(pb.bone.matrix_local.to_quaternion().inverted())
        # q = Quaternion((1, 0, 0, 0))
        #    if cur_orient is not None:
        #        q = cur_orient.rotation_difference(q)
        #        #q = q.rotate(cur_orient.inverted())
        pb.rotation_quaternion = q
    for child in pb.children:
        set_quat(child, traj, t)


def old_set_quat(pb, traj, t):
    if pb.name not in traj.keys():
        return

    q = Quaternion((traj[pb.name][t][3:]))
    cur_orient = pb.bone.matrix_local.to_quaternion()
    cur_orient.rotate(q)
    pb.matrix = cur_orient.to_matrix().to_4x4()
    bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=0)
    for child in pb.children:
        old_set_quat(child, traj, t)


def set_pos(armature, pb, traj, t):
    armature.location = Vector(traj[pb.name][t][:3]) - pb.bone.head


def old_set_pos(pb, traj, t):
    translate = Matrix.Translation(traj[pb.name][t][:3])
    pb.matrix = translate @ pb.matrix


def set_pose(armature, traj, t):
    pb = armature.pose.bones[find_root_bone(armature)]
    set_pos(armature, pb, traj, t)
    set_quat(pb, traj, t)


def old_set_pose(pb, traj, t):
    old_set_quat(pb, traj, t)
    old_set_pos(pb, traj, t)



def map_traj(armature, traj, frames_per_key = 10, speedup = 1):

    posebones = armature.pose.bones
    root_bone = find_root_bone(armature)
    for ob in bpy.context.selected_objects:
        ob.animation_data_clear()

    for i in range(0, len(traj[root_bone]), frames_per_key):
        set_pose(armature, traj, i)
        frame_n = i // speedup
        for pb in posebones:
            pb.keyframe_insert("rotation_quaternion", frame=frame_n)
            armature.keyframe_insert("location", frame=frame_n)
    bpy.context.scene.frame_end = len(traj[root_bone]) // speedup

if __name__ == "__main__":
    armature = bpy.context.object
    trajectory = np.load('/home/justin/repos/animal-pointodyssey/trajectories.npz')
    map_traj(armature, trajectory)

