
import os
if __name__ == '__main__':
    vid_length = 1000
    for i in range(100):
        create_script = f"python create_scene.py -- --character-path data/harvard_motion/mapped_mouse.blend --data-path /home/justin/Downloads/2020_12_22_1.h5 --speedup 2 --t0 {i*vid_length} --tf {(i+1)*vid_length}"
        os.system(create_script)
        render_script = "python export_annotation.py --scene_dir results/animal/scene.blend --save_dir ./results/mouse --rendering --samples_per_pixel 64  \
	--exr --export_obj \
	--use_gpu --export_tracking --sampling_character_num 5000 --sampling_scene_num 2000 --timestamp"
        os.system(render_script)
