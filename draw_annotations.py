import numpy as np
import cv2
import time
from colour import Color
import os
def main(image_path, annotation_path, style = "dot"):
    traj_length = 6
    init_color = Color("blue")
    end_color = Color("yellow")
    color_range = list(init_color.range_to(end_color, traj_length))
    frame_n = 0
    height = 800
    width = 1000
    display_num = 100
    annotations = np.load(annotation_path, allow_pickle= True).item()


    trajs = annotations["coords"]
    mask = np.array([np.inf not in np.abs(t) for t in trajs])
    trajs = trajs[mask, ...]
    print(len(trajs))
    #trajs = np.clip(trajs, -100000, 100000)
    vis = annotations["visibility"]
    print(trajs.shape)
    output = cv2.VideoWriter(
        "output.mp4", cv2.VideoWriter_fourcc(*'mp4v'),
        24, (width, height))
    i = 0
    n_points = trajs.shape[0]
    idxs = np.random.choice(n_points, display_num, replace=False)
    idxs = np.arange(display_num)
    images = sorted(os.listdir(image_path))
    while i < trajs.shape[1]:
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        image = cv2.imread(os.path.join(image_path, images[i]))
        offset = (np.array(frame.shape[:2]) - np.array(image.shape[:2])) //2
        frame[offset[0]:offset[0] + image.shape[0],offset[1]:offset[1] + image.shape[1], :] = image

        for idx in idxs:
            if np.inf in trajs[idx, i, :] or -np.inf in trajs[idx, i, :]:
                continue

            if style == "dot":

                if vis[idx, i]:
                    cv2.circle(frame, trajs[idx, i, :].astype(int) + offset[::-1], 0, [c*256 for c in init_color.rgb][::-1], 3)
                else:
                    cv2.circle(frame, trajs[idx, i, :].astype(int)+ offset[::-1], 0, [c * 256 for c in end_color.rgb][::-1], 2)
            elif style == "line":

                for j in range(max(0, i - traj_length), i):
                    if np.inf in trajs[idx, j, :] or -np.inf in trajs[idx, j, :]:
                        continue
                    cv2.line(frame, trajs[idx, j, :].astype(int) + offset[::-1], trajs[idx, j+1, :].astype(int)+ offset[::-1], [c*256 for c in color_range[i-j-1].rgb][::-1], 2)

        # writing the new frame in output

        output.write(frame)
        cv2.imshow("output", frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break
        i += 1
        print(i)

    cv2.destroyAllWindows()
    output.release()


if __name__ == "__main__":
    image_path = "/home/justin/repos/animal-pointodyssey/results/datasets/animal_pod_aug19/1524/frames"
    annotation_path = "/home/justin/repos/animal-pointodyssey/results/datasets/animal_pod_aug19/1524/1524.npy"
    main(image_path, annotation_path, style = "line")
