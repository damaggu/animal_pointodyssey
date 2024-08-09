import numpy as np
import cv2
import time
from colour import Color
import os
def main(image_path, annotation_path, style = "dot"):
    traj_length = 5
    init_color = Color("yellow")
    end_color = Color("blue")
    color_range = list(init_color.range_to(end_color, traj_length))
    frame_n = 0
    height = 540
    width = 960
    display_num = 150
    annotations = np.load(annotation_path, allow_pickle= True).item()


    trajs = annotations["coords"]
    vis = annotations["visibility"]
    print(trajs.shape)
    output = cv2.VideoWriter(
        "output.mp4", cv2.VideoWriter_fourcc(*'mp4v'),
        12, (width, height))
    i = 0
    n_points = trajs.shape[0]
    idxs = np.random.choice(n_points, display_num, replace=False)
    idxs = np.arange(display_num)
    images = sorted(os.listdir(image_path))
    while i < trajs.shape[1]:
        frame = cv2.imread(os.path.join(image_path, images[i]))

        for idx in idxs:
            if style == "dot":
                if vis[idx, i]:
                    cv2.circle(frame, trajs[idx, i, :].astype(int), 0, [c*256 for c in init_color.rgb][::-1], 3)
                else:
                    cv2.circle(frame, trajs[idx, i, :].astype(int), 0, [c * 256 for c in end_color.rgb][::-1], 3)
            elif style == "line":
                for j in range(max(0, i - traj_length), i):

                    cv2.line(frame, trajs[idx, j, :].astype(int), trajs[idx, j+1, :].astype(int), [c*256 for c in color_range[i-j-1].rgb][::-1], 2)

        # writing the new frame in output
        output.write(frame)
        cv2.imshow("output", frame)
        if cv2.waitKey(0) & 0xFF == ord('s'):
            break
        i += 1
        print(i)

    cv2.destroyAllWindows()
    output.release()


if __name__ == "__main__":
    image_path = "/home/justin/repos/animal-pointodyssey/results/animal_pod/0003/frames"
    annotation_path = "/home/justin/repos/animal-pointodyssey/results/animal_pod/0003/0003.npy"
    main(image_path, annotation_path, style = "dot")
