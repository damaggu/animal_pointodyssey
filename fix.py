# import OS module
import os
# Get the list of all files and directories
path = "./results/gen2"
dir_list = os.listdir(path)
print(dir_list)
for n, d in enumerate(dir_list):
    try:
        video_script = f"mv '{path+'/'+d}/video.mp4' ./results/videos/vid_{n}.mp4"
        os.system(video_script)
    except:
        pass
