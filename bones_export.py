import bpy
from bpy.props import StringProperty, BoolProperty
from bpy.types import Operator
from bpy_extras.io_utils import ExportHelper

import os

import mathutils
from mathutils import Matrix, Vector




def write_mjcf(dir_path, model_file_name, selected_objects,  export=False):
    # Open the file for writing
    with open(os.path.join(dir_path, model_file_name), "w") as file:

        # Write the MJCF header
        file.write('<mujoco>\n')
        file.write('  <compiler meshdir="./mesh"/>\n')
        file.write('  <worldbody>\n')

        # Write the bodies recursively
        for obj in selected_objects:

            if obj.parent is None:
                print(obj)
                mesh_file_names = write_body(obj, [0, 0, 0], file, 2, dir_path, export)

        # Write the closing tags for the XML
        file.write('  </worldbody>\n')
        
        mesh_elements = [f'\n    <mesh name="{os.path.splitext(filename)[0]}" file="{filename}"/>' for filename in mesh_file_names]
        mesh_string = "".join(mesh_elements)
        asset_string = f"<asset>{mesh_string}\n  </asset>\n"
        file.write(asset_string)
        file.write('</mujoco>\n')



def write_body(obj, last_pos, file, level, dir_path, export=False):
    # Set the indentation for this level
    indent = "  " * level
    filepaths = []
    
    pos = obj.tail_local - obj.head_local

    # Write the body element with the local position and rotation
    file.write(
        f'{indent}<body name="{obj.name}" pos="{last_pos[0]} {last_pos[1]} {last_pos[2]}">\n')

    # Write the geom element for the object
    file.write(f'{indent}  <geom type="capsule" name="{obj.name} geom" size="0.05" fromto="0 0 0 {pos[0]} {pos[1]} {pos[2]}"/>\n')

    child_file_paths = []
    # Recursively write the children
    for child in obj.children:
        child_file_paths.extend(write_body(child, pos, file, level+1, dir_path, export))

    file.write(f'{indent}</body>\n')
    
    return [*filepaths, *child_file_paths]

selected_objects = bpy.context.object.data.bones

write_mjcf("", "asdf.xml", selected_objects, export = False)