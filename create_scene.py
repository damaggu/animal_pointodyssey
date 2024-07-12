import numpy as np
import json
import os
import math
import argparse

from typing import Any, Dict, Optional, Sequence, Union
import random

import bpy
import mathutils
from import_motion import *
from import_mapped import *
class BlenderScene():
    def __init__(self,
                 scratch_dir=None,
                 base=None):
        self.scratch_dir = scratch_dir
        self.blender_scene = bpy.context.scene
        if base is None:
            bpy.context.scene.world = bpy.data.worlds.new("World")
        else:
            print("Loading scene from '%s'" % base)
            bpy.ops.wm.open_mainfile(filepath=base)

    def save_scene(self, filename = 'scene.blend'):
        absolute_path = os.path.abspath(self.scratch_dir)
        try:
            bpy.ops.wm.save_as_mainfile(filepath=os.path.join(absolute_path, filename))
        except:
            print('error saving blend file, skipping')

    def add_character(self, character_path, traj_path = None, data_file = None, t0 = 0, tf = 1000, frames_per_key = 10, speedup = 1):
        if traj_path is not None and data_file is not None:
            return
        names = []
        with bpy.data.libraries.load(character_path) as (data_from, data_to):
            names = [name for name in data_from.collections]
        bpy.ops.wm.append(
            filepath=os.path.join(character_path, 'Collection', names[0]),
            directory=os.path.join(character_path, 'Collection'),
            filename=names[0]
        )
        for obj in bpy.context.selected_objects:
            obj.select_set(obj.type == "ARMATURE")
        armature = bpy.context.selected_objects[0]
        bpy.context.view_layer.objects.active = armature
        if traj_path is not None:
            trajectory = np.load(traj_path)
            map_traj(armature, trajectory, frames_per_key, speedup)
        if data_file is not None:
            map_file_onto_armature(armature, data_file, t0, tf, speedup = speedup, frames_per_key = frames_per_key)






if __name__ == "__main__":
    import sys

    argv = sys.argv

    if "--" not in argv:
        argv = []
    else:
        argv = argv[argv.index("--") + 1:]

    print("argsv:{0}".format(argv))
    parser = argparse.ArgumentParser(description='Render Motion in 3D Environment for HuMoR Generation.')
    parser.add_argument('--base', type=str, default='data/blender_assets/hdri.blend')
    parser.add_argument('--output-dir', type=str, default='results/animal')
    parser.add_argument('--traj-path', type=str, default=None)
    parser.add_argument('--data-path', type=str, default=None)
    parser.add_argument('--character-path', type=str, default=None)
    parser.add_argument('--speedup', type=int, default=1)
    parser.add_argument('--frames-per-key', type=int, default=10)
    parser.add_argument('--t0', type=int, default=0)
    parser.add_argument('--tf', type=int, default=1000)
    args = parser.parse_args(argv)
    print("args:{0}".format(args))

    ## Load the world
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    scene = BlenderScene(scratch_dir= args.output_dir, base = args.base)
    if args.character_path is not None:
        scene.add_character(args.character_path, args.traj_path, data_file = args.data_path, t0 = args.t0, tf = args.tf, speedup = args.speedup, frames_per_key= args.frames_per_key)
    scene.save_scene()

