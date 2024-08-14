# import OS module
import os
import numpy as np
import shutil
import json

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

def fix_annotations(file_path, num_characters = 3):
    fix_depth = f"python -m po_utils.openexr_utils --data_dir {file_path} --output_dir {os.path.join(file_path, 'exr_img')} --batch_size 64 --frame_idx 0"
    os.system(fix_depth)
    fix_script = f"python -m po_utils.gen_tracking_indoor --data_root {file_path}  --cp_root {file_path} --sampling_scene_num 100000 --sampling_character_num {num_characters * 1000}"
    os.system(fix_script)
    convert_to_kubric(file_path)

# Get the list of all files and directories
def add_videos(path, fix = True):
    output_folder = "./results/datasets/aug_11"
    os.makedirs(output_folder, exist_ok=True)
    dir_list = [x for x in os.listdir(path) if os.path.isdir(os.path.join(path, x))]
    dir_list.sort()
    print(dir_list)
    existing = os.listdir(output_folder)
    existing.sort()
    cur = 0
    if len(existing) != 0:
        cur = int(existing[-1]) + 1
    print(cur)

    for d in dir_list:
        full_path = os.path.join(path, d)
        run_info = json.load(open(os.path.join(full_path, "run_info.json"), 'r'))
        if fix:
            try:
                if "num_characters" in run_info.keys():
                    fix_annotations(full_path, run_info["num_characters"])
                else:
                    fix_annotations(full_path, 3)
            except:
                print(d, "failed")
                continue
        if "kubric.npy" in os.listdir(full_path):
            output_path = os.path.join(output_folder, str(cur).zfill(4))
            images = os.path.join(full_path, "images")
            frame_dir = os.path.join(output_path, "frames")
            os.makedirs(frame_dir, exist_ok=True)
            for im in os.listdir(images):
                frame_id = im[6:10] + ".png"
                shutil.copy(os.path.join(images, im), os.path.join(output_path, "frames/" + frame_id))
            shutil.copy(os.path.join(full_path, "kubric.npy"), os.path.join(output_path, str(cur).zfill(4) + ".npy"))
        cur += 1


paths = [
    "/home/justin/repos/animal-pointodyssey/results/mouse_08-10T00:38:15",
]
for p in paths:
    add_videos(p, fix = False)