import numpy as np
import cv2
import time
from colour import Color
def main(file_path):
    traj_length = 5
    init_color = Color("yellow")
    end_color = Color("blue")
    color_range = list(init_color.range_to(end_color, traj_length))
    frame_n = 0
    height = 540
    width = 960
    display_num = 3000
    annotations = np.load(file_path + "annotations.npz")

    trajs = annotations["trajs_2d"]
    output = cv2.VideoWriter(
        "output.mp4", cv2.VideoWriter_fourcc(*'mp4v'),
        12, (width, height))
    i = 0
    n_points = annotations["trajs_2d"].shape[1]
    idxs = np.random.choice(n_points, display_num, replace=False)
    while i < len(trajs):

        frame = cv2.imread(file_path + "images/frame_{:04}.png".format(i))


        for idx in idxs:
            for j in range(max(0, i - traj_length), i):

                cv2.line(frame, trajs[j, idx, :].astype(int), trajs[j+1, idx, :].astype(int), [c*256 for c in color_range[i-j-1].rgb][::-1], 2)

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
    main("/home/justin/repos/animal-pointodyssey/results/mouse_07-29T19:03:44/0026/")
