import bpy
from bpy.props import StringProperty, BoolProperty
from bpy.types import Operator
from bpy_extras.io_utils import ExportHelper

import os
import math

import mathutils
from mathutils import Matrix, Vector, Quaternion

armature = bpy.context.object
posebones = armature.pose.bones

bones = armature.data.bones
root_bone = None
for b in bones:
    if b.parent == None:
        root_bone = b.name


def write_mjcf(dir_path, model_file_name, selected_objects, export=False):
    # Open the file for writing
    with open(os.path.join(dir_path, model_file_name), "w") as file:

        # Write the MJCF header
        file.write('<mujoco>\n')
        file.write('  <compiler meshdir="./mesh"/>\n')
        file.write('  <worldbody>\n')
        file.write('      <geom name="floor" size="0 0 .05" type="plane" material="grid" condim="3"/>\n')

        # Write the bodies recursively
        for obj in selected_objects:

            if obj.parent is None:
                print(obj)
                mesh_file_names = write_body(obj, bones[root_bone].head, file, 2, dir_path, export)

        # Write the closing tags for the XML
        file.write('  </worldbody>\n')

        mesh_elements = [f'\n    <mesh name="{os.path.splitext(filename)[0]}" file="{filename}"/>' for filename in
                         mesh_file_names]
        mesh_string = "".join(mesh_elements)
        asset_string = """<asset>
<texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1=".1 .2 .3" rgb2=".2 .3 .4"/>
<material name="grid" texture="grid" texrepeat="1 1" texuniform="true" reflectance=".2"/>
  </asset>"""
        file.write(asset_string)
        file.write('</mujoco>\n')


def write_body(obj, last_pos, file, level, dir_path, export=False):
    # Set the indentation for this level
    indent = "  " * level
    filepaths = []

    pos = obj.tail - obj.head
    rot = obj.rotation_quaternion

    # Write the body element with the local position and rotation
    file.write(
        f'{indent}<body name="{obj.name}" pos="{last_pos[0]} {last_pos[1]} {last_pos[2]}">\n')
    if level == 2:
        file.write(f'{indent}<joint type="free"/>\n')

    if obj.is_in_ik_chain:
        is_locked = [obj.lock_ik_x, obj.lock_ik_y, obj.lock_ik_z]
        use_limits = [obj.use_ik_limit_x, obj.use_ik_limit_y, obj.use_ik_limit_z]
        limits = [
            (math.degrees(obj.ik_min_x), math.degrees(obj.ik_max_x)),
            (math.degrees(obj.ik_min_y), math.degrees(obj.ik_max_y)),
            (math.degrees(obj.ik_min_z), math.degrees(obj.ik_max_z)),
        ]
        names = [
            'rx_{}'.format(obj.name),
            'ry_{}'.format(obj.name),
            'rz_{}'.format(obj.name),
        ]
        axes = ["1 0 0", "0 1 0", "0 0 1"]
        for i in range(3):
            if not is_locked[i]:
                file.write(
                    f'{indent}<joint type="hinge" name="{names[i]}" axis="{axes[i]}" range="{limits[i][0]} {limits[i][1]}"/>\n')
    # Write the geom element for the object
    file.write(
        f'{indent}<geom type="capsule" name="{obj.name}_geom" size="0.05" fromto="0 0 0 {pos[0]} {pos[1]} {pos[2]}"/>\n')

    child_file_paths = []
    # Recursively write the children
    for child in obj.children:
        child_file_paths.extend(write_body(child, pos, file, level + 1, dir_path, export))

    file.write(f'{indent}</body>\n')

    return [*filepaths, *child_file_paths]


def reset_pose(pb):
    if pb.name == root_bone:
        armature.location = Vector((0, 0, 0))
    q = Quaternion((1, 0, 0, 0))
    pb.rotation_quaternion = q
    for child in pb.children:
        reset_pose(child)


bpy.ops.anim.keyframe_clear_v3d()
reset_pose(posebones[root_bone])
write_mjcf("/home/justin/repos/animal-pointodyssey/export", "mouse_export.xml", posebones, export=False)