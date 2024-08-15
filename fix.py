import os
import shutil



path = "./ttemplogs"
runs = os.listdir(path)
runs.sort()
for r in runs:
    x = os.listdir(os.path.join(path, r))
    if "evaluations.npz" in x:
        print(r)
    else:
        delete_script = f"rm -rf {path}/{r}"
        os.system(delete_script)