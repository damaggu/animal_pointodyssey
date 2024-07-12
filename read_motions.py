import h5py
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')
f = h5py.File('/home/justin/Downloads/2020_12_22_1.h5', 'r')
pose = f["pose"]
ephys = f["ephys"]
behavior = f["behavior"]
keypoints = pose["keypoints"]
qpos = pose["qpos"]
ax = plt.figure().add_subplot(projection='3d')
for i in range(2):
    motion = keypoints[:10000, :, i]
    ax.plot(*motion.T)
plt.show()