import mujoco
import mujoco_viewer
import numpy as np
import os
model = mujoco.MjModel.from_xml_path('./data/mujoco/mouse.xml')
body_names = [model.body(i).name for i in range(model.nbody)]
print(body_names)
data = mujoco.MjData(model)

# create the viewer object
viewer = mujoco_viewer.MujocoViewer(model, data)

# simulate and render
quats = []
posl = []

for _ in range(10000):

    if viewer.is_alive:

        mujoco.mj_step(model, data)
        try:
            quat, pos = viewer.render()
        except:
            pass
        if quat is not None:

            quats.append(quat.copy())
            posl.append(pos.copy())
    else:
        break
quats = quats[:-1]
posl = posl[:-1]
quats = np.stack(quats)
posl = np.stack(posl)
print(posl.shape)
print(quats.shape)
print(quats[:, 0].shape)
assert len(body_names) == quats.shape[1]
final_traj = {name: np.concatenate((posl[:, i], quats[:, i]), axis = 1)for i, name in enumerate(body_names)}
np.savez_compressed("export/trajectories.npz", **final_traj)
# close
viewer.close()