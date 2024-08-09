import os
import shutil



path = "./results/dataset_processed"
videos = os.listdir(path)
videos.sort()
for video in videos:
    vid_path = os.path.join(path, video)
    # video_script = f"ffmpeg -f image2 -r 12 -pattern_type glob -i '{vid_path}/frames/*.png' -vcodec libx264 -crf 22 '{path}/{video}.mp4'"
    # print(video_script)
    # os.system(video_script)
    if os.path.isdir(vid_path):
        shutil.move(os.path.join(vid_path, "kubric.npy"), os.path.join(vid_path, video + ".npy"))