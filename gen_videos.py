import datetime
import os
import h5py
import numpy as np
import random
from create_scene import BlenderScene
import matplotlib.pyplot as plt
import json
import argparse
def get_active_start(keypoints, timesteps, threshold = 5):
    for i in range(1000):
        start = np.random.randint(0, len(keypoints) - timesteps)
        data = keypoints[start:start+timesteps, :2, :]
        diffs = data.max(0) - data.min(0)
        s = np.sum(diffs)
        if s/timesteps > threshold:
            return start
    return None

def convert_to_kubric(file_path):
    annotations = np.load(os.path.join(file_path, "annotations.npz"))

    kubric = {}
    n_points = annotations["trajs_2d"].shape[1]
    idx = np.random.choice(n_points, 2048, replace = False)
    kubric["coords"] = annotations["trajs_2d"][:, idx, :].astype(np.float32)
    kubric["coords"] = np.swapaxes(kubric["coords"], 0, 1)
    kubric["visibility"] = annotations["visibilities"][:, idx] == 1.
    kubric["visibility"] = np.swapaxes(kubric["visibility"], 0, 1)
    print(kubric["visibility"])
    with open(os.path.join(file_path, "kubric.npy"), "wb") as f:
        np.save(f, kubric)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--render_only',  default=False, action='store_true')
    args = parser.parse_args()
    run_info = json.load(open('default_args.json', 'r'))
    timesteps = run_info["vid_length"] * run_info["speedup"]
    f = h5py.File(run_info["file"], 'r')
    pose = f["pose"]
    keypoints = pose["keypoints"]
    behavior = f["behavior"]
    run_info["start"] = datetime.datetime.now().strftime("%m-%dT%H:%M:%S")

    for i in range(run_info["vid_num"]):
        run_info["seed"] = np.random.randint(0, 1000000)
        np.random.seed(run_info["seed"])
        scene = BlenderScene(scratch_dir = run_info["scratch_dir"], base = run_info["base"], render_args=run_info["render_args"])
        character = None
        starts = []
        run_info["track"] = np.random.rand() <run_info["track_prop"]
        run_info["fog"] = np.random.rand() < run_info["fog_prop"]
        run_info["shake"] = np.random.rand() < run_info["shake_prop"]
        run_info["render_args"]["use_motion_blur"] = np.random.rand() < run_info["blur_prop"]
        run_info["brightness"] = np.random.uniform(0.3, 1.1)
        run_info["background"] = np.random.choice(os.listdir(run_info["background_folder"]))
        run_info["num_characters"] = np.random.randint(run_info["min_characters"], run_info["max_characters"]+1)
        run_info["character_samples"] = run_info["samples_per_character"] * run_info["num_characters"]
        print(run_info["background"])
        save_dir = f'./results/mouse_{run_info["start"]}/{str(i).zfill(4)}/'
        center_origin = np.random.uniform(-run_info["origin_range"], run_info["origin_range"], [2])
        character = None
        for j in range(run_info["num_characters"]):
            start = get_active_start(keypoints, timesteps, threshold=8)
            origin = np.random.uniform(-1, 1, [3]) * 2
            origin[:2] += center_origin
            origin[2] = 0
            character = scene.add_character(run_info["character"], data_file=run_info["file"],
                            origin = origin,
                            t0=start,
                            tf=start + timesteps,
                            speedup=run_info["speedup"], frames_per_key=6)
            scene.randomize_materials(character, run_info["material_path"])
            starts.append(start)
        run_info["starts"] = starts

        cam_pos = np.zeros(3)
        for s in starts:
            data = keypoints[start:start + timesteps, :2, :]
            avg = np.average(data, axis=(0, 2))/100
            cam_pos[:2] += avg + center_origin
        cam_pos = cam_pos/len(starts)
        cam_pos[2] = 5 + np.random.uniform(-2, 2)

        if run_info["track"]:
            scene.target_cam(cam_pos, character, track = True)
        else:
            scene.set_cam(pos=cam_pos, dir=np.random.uniform(-1, 1, [3]))

        if run_info["shake"]:
            scene.shake_cam(intensity = 1, min_height=run_info["min_cam_height"])

        scene.set_brightness(run_info["brightness"])
        scene.set_background(os.path.join(run_info["background_folder"], run_info["background"]))
        scene.set_render_args(run_info["render_args"])
        print(os.path.join(run_info["background_folder"], run_info["background"]))
        scene.save_scene()
        os.makedirs(save_dir, exist_ok=True)
        json.dump(run_info, open(os.path.join(save_dir, 'run_info.json'), 'w'), indent=4)
        if args.render_only:
            render_script = f"python export_annotation.py --scene_dir results/animal/scene.blend --save_dir {save_dir} --rendering --samples_per_pixel 48 --use_gpu {'--add_fog' if run_info['fog'] else ''}"
            os.system(render_script)
        else:
            render_script = f"python export_annotation.py --scene_dir results/animal/scene.blend --save_dir {save_dir} --rendering --samples_per_pixel 48  \
                --exr --export_obj \
                --use_gpu --export_tracking --sampling_character_num {run_info['character_samples']} --sampling_scene_num {run_info['scene_samples']} {'--add_fog' if run_info['fog'] else ''}"
            os.system(render_script)
            convert_to_kubric(save_dir)
        video_script = f"ffmpeg -f image2 -r {run_info['render_args']['fps']} -pattern_type glob -i '{save_dir}/images/*.png' -vcodec libx264 -crf 22 '{save_dir}/video.mp4'"
        os.system(video_script)

