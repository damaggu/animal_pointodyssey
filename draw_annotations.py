import numpy as np
import cv2
import time

def main(file_path):
    traj_length = 30
    init_color = (240, 100, 50)
    col_diff = np.uint8(np.array((-8, -3, 5)))
    final_color = col_diff * traj_length + init_color
    final_color = np.uint8(np.where(final_color<=0, 0, final_color))
    frame_n = 0
    height = 540
    width = 960
    display_num = 0.02
    annotations = np.load(file_path + "annotations.npz")

    trajs = annotations["trajs_2d"]
    frame = np.zeros((height,width,3), np.uint8)
    output = cv2.VideoWriter(
        "output.mp4", cv2.VideoWriter_fourcc(*'mp4v'),
        50, (width, height))
    i = 0
    print(final_color)
    while i < len(trajs):

        frame = frame
        frame = np.where(frame<=0, 0, frame + col_diff)
        frame = np.where(np.repeat((np.sum(frame, axis = 2) == sum(final_color))[:, :, np.newaxis], 3, axis=2), 0, frame )
        cur_pos = trajs[i]
        for pos in cur_pos[::int(1/display_num)]:
            cv2.circle(frame, pos.astype(int), 0, init_color, 3)

        # writing the new frame in output
        image = cv2.imread(file_path + "images/frame_{:04}.png".format(i))
        final_frame = np.where(np.repeat((np.sum(frame, axis = 2) == 0)[:, :, np.newaxis], 3, axis=2), image, frame )
        output.write(final_frame)
        cv2.imshow("output", final_frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break
        i += 1
        print(i)

    cv2.destroyAllWindows()
    output.release()


if __name__ == "__main__":
    main("./results/mouse_vertical/")
