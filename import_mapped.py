import bpy
import mathutils
import pip
pip.main(['install', 'h5py', '--user'])
import h5py

def find_root_bone(armature):
    bones = armature.data.bones
    root_bone = None
    for b in bones:
        if b.parent == None:
            root_bone = b.name
    return root_bone




def map_file_onto_armature(armature, file, t0, tf, speedup = 1.5, frames_per_key = 6):
    f = h5py.File(file, 'r')
    pose = f["pose"]
    ephys = f["ephys"]
    behavior = f["behavior"]
    keypoints = pose["keypoints"]
    qpos = pose["qpos"]
    
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
            cur_frame = t*frames_per_key/speedup
            point.location = p / 100
            if i == 6:
                point.location.z += 0.2
            point.keyframe_insert("location", frame=cur_frame)
            if i == 5:
                armature.location = (point.location - root_bone.head*armature.scale)
                
                armature.keyframe_insert("location", frame=cur_frame)
    bpy.context.scene.frame_end = (tf - t0)// speedup
    bpy.data.objects.remove(ob, do_unlink = True)

    bones = armature.pose.bones

    for b in bones:
        if "target_id" in b.keys():
            b.constraints[0].target = points[b["target_id"]]
        if "pole_target_id" in b.keys():
            b.constraints[0].pole_target = points[b["pole_target_id"]]
            b.constraints[0].pole_angle = b["pole_angle"]

if __name__ == "__main__":
    armature = bpy.context.selected_objects[0]

    map_file_onto_armature(armature,'/home/justin/Downloads/2020_12_22_1.h5', 0, 5000)