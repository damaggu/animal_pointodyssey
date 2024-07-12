#!/bin/sh

python export_annotation.py --scene_dir results/animal/scene.blend --save_dir ./results/mouse_vertical --rendering --samples_per_pixel 64  \
	--exr --export_obj \
	--use_gpu --export_tracking --sampling_character_num 5000 --sampling_scene_num 2000
