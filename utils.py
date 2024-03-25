from typing import Union
from pathlib import Path
import os
import json


def log_to_dir(base_dir: str, file_name_to_data: dict[str, Union[str, dict]]):
    if base_dir is None:
        return
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    for file_name, data in file_name_to_data.items():
        with open(os.path.join(base_dir, file_name), "w") as f:
            if file_name.endswith(".json"):
                json.dump(data, f, indent=2)
            else:
                f.write(data)

