import bpy
import mathutils
import pip

try:
    import h5py
except ModuleNotFoundError:
    pip.main(['install', 'h5py', '--user'])
    import h5py

def find_root_bone(armature):
    bones = armature.data.bones
    root_bone = None
    for b in bones:
        if b.parent == None:
            root_bone = b.name
    return root_bone




def map_file_onto_armature(armature, file, origin = (0, 0, 0), t0 = 0, tf = 1000, speedup = 2, frames_per_key = 6):
    f = h5py.File(file, 'r')
    pose = f["pose"]
    ephys = f["ephys"]
    behavior = f["behavior"]
    keypoints = pose["keypoints"]
    qpos = pose["qpos"]
    ground_height = 0.0
    root_bone = armature.pose.bones[find_root_bone(armature)]
    armature.animation_data_clear()
    ob = bpy.data.objects.new( "empty", None )
    armature.users_collection[0].objects.link(ob)
    points = []

    for i in range(23):
        point = ob.copy()
        point.name = f"Empty_{i}"
        point.scale = (0.1, 0.1, 0.1)
        armature.users_collection[0].objects.link(point)
        #point.parent = armature
        points.append(point)
        motion = keypoints[t0:tf, :, i]
        for t, p in enumerate(motion[::frames_per_key]):
            cur_frame = t*frames_per_key//speedup
            point.location = p / 100 + origin
            point.location.z -= ground_height
            point.keyframe_insert("location", frame=cur_frame)
            if i == 5:
                armature.location = (point.location - root_bone.head*armature.scale)
                
                armature.keyframe_insert("location", frame=cur_frame)
    bpy.context.scene.frame_end = (tf - t0)// speedup
    

    bones = armature.pose.bones
    tail = bones["Tail_12"]
    for b in bones:
        if b == tail:
            b.constraints[0].target = None
            continue
        if "target_id" in b.keys():
            b.constraints[0].target = points[b["target_id"]]
        if "pole_target_id" in b.keys():
            b.constraints[0].pole_target = points[b["pole_target_id"]]
            b.constraints[0].pole_angle = b["pole_angle"]
            
    
    tailpoint = ob.copy()
    tailpoint.name = "Empty_100"
    armature.users_collection[0].objects.link(tailpoint)
    for t, p in enumerate(motion[::frames_per_key]):
        cur_frame = t*frames_per_key//speedup
        bpy.context.scene.frame_set(cur_frame)
        tailpoint.location = tail.tail * armature.scale + armature.location
        tailpoint.location.z = 0
        tailpoint.keyframe_insert("location", frame=cur_frame)
    tail.constraints[0].target = tailpoint
    bpy.data.objects.remove(ob, do_unlink = True)


if __name__ == "__main__":
    objects = bpy.data.objects

    for obj in objects:
        if obj.type == "EMPTY":
            objects.remove(obj, do_unlink=True)


    armature = bpy.context.selected_objects[0]
    start = 182836
    length = 1000
    map_file_onto_armature(armature,'/home/justin/Downloads/2020_12_24_2.h5', t0 = start, tf = start + length, origin = (-2, 0, 0))