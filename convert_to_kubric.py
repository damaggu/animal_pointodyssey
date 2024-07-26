import numpy as np

file_path = "/home/justin/repos/animal-pointodyssey/results/mouse_07-25T15:29:25/0000/"

annotations = np.load(file_path + "annotations.npz")

kubric = {}
n_points = annotations["trajs_2d"].shape[1]
idx = np.random.choice(n_points, 2048, replace=False)
print(idx)
kubric["coords"] = annotations["trajs_2d"][:, idx, :].astype(np.float32)
kubric["coords"] = np.swapaxes(kubric["coords"], 0, 1)
kubric["visibility"] = annotations["visibilities"][:, idx] == 1.
kubric["visibility"] = np.swapaxes(kubric["visibility"], 0, 1)
print(kubric["visibility"].shape)
with open(file_path + "kubric.npy", "wb") as f:
    np.save(f, kubric)

