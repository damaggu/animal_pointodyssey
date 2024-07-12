#!/bin/sh

python point_odyssey/export_annotation_animal.py --output_dir ./results/animal --rendering \
	--exr --export_obj \
	--export_tracking --sampling_points 5000 --sampling_scene_points 2000