import bpy
from bpy.props import StringProperty, BoolProperty
from bpy.types import Operator
from bpy_extras.io_utils import ExportHelper

import os
import numpy as np
import mathutils
from mathutils import Matrix, Vector, Quaternion
import itertools

armature = bpy.context.object
posebones = armature.pose.bones

bones = armature.data.bones

traj = np.load('/home/justin/repos/animal-pointodyssey/trajectories.npz')


def reset_quat(pb):
    q = Quaternion((1, 0, 0, 0))
    pb.rotation_quaternion = q
    for child in pb.children:
        reset_quat(child)


def set_quat(pb, t, cur_orient=None):
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
        set_quat(child, t, pb.rotation_quaternion)


def old_set_quat(pb, t):
    if pb.name not in traj.keys():
        return

    q = Quaternion((traj[pb.name][t][3:]))
    cur_orient = bones[pb.name].matrix_local.to_quaternion()
    cur_orient.rotate(q)
    pb.matrix = cur_orient.to_matrix().to_4x4()
    bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=0)
    for child in pb.children:
        old_set_quat(child, t)


def set_pos(pb, t):
    armature.location = Vector(traj[pb.name][t][:3]) - pb.bone.head


def old_set_pos(pb, t):
    translate = Matrix.Translation(traj[pb.name][t][:3])
    pb.matrix = translate @ pb.matrix


def set_pose(pb, t):
    set_pos(pb, t)
    set_quat(pb, t)


def old_set_pose(pb, t):
    old_set_quat(pb, t)
    old_set_pos(pb, t)


frames_per_key = 16
speedup = 8

bpy.ops.anim.keyframe_clear_v3d()

root_bone = None
for b in bones:

    if b.parent == None:
        root_bone = b.name

# set_pose(posebones[root_bone], 200)

for i in range(0, len(traj[root_bone]), frames_per_key):
    set_pose(posebones[root_bone], i)
    frame_n = i // speedup
    for pb in posebones:
        pb.keyframe_insert("rotation_quaternion", frame=frame_n)
        armature.keyframe_insert("location", frame=frame_n)
bpy.context.scene.frame_end = len(traj[root_bone]) // speedup

