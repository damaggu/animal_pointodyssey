import itertools
import os
from typing import Sequence

from absl import app

from dm_control import mjcf
from dm_control.mujoco.wrapper.mjbindings import mjlib
from lxml import etree
import numpy as np


ASSET_RELPATH = 'assets/'
ASSET_DIR = os.path.dirname(__file__) + '/' + ASSET_RELPATH
BASE_MODEL = 'dog_test.xml'
DEFAULT_MODEL = 'dog_defaults.xml'
OUT_MODEL = "dog.xml"
DEFAULT_GEAR = 100
DEFAULT_GAINPRM = 5


with open(os.path.join(ASSET_DIR, BASE_MODEL), 'r') as f:
    basetree = etree.XML(f.read(), etree.XMLParser(remove_blank_text=True))

model = mjcf.from_xml_string(etree.tostring(basetree, pretty_print=True),
                                 model_dir=ASSET_DIR)

model.option.timestep = "0.001"
model.option.integrator = "implicitfast"
model.compiler.angle = "radian"

joints = model.find_all('joint')
model.tendon.add("fixed", name = "back_x", limited = "true", range = "-1 0")
model.tendon.add("fixed", name = "back_z")
for j in joints:
    if j.name == None:
        continue
    j.stiffness = "0.01"
    j.damping = "0.01"
    if "rx_Back" in j.name:
        model.tendon.fixed["back_x"].add("joint", joint=j.name, coef="1")
    elif "rz_Back" in j.name:
        model.tendon.fixed["back_z"].add("joint", joint=j.name, coef="1")
    else:
        model.actuator.add("motor", name = j.name + "_motor", joint = j.name, gear = str(DEFAULT_GEAR))

model.actuator.add("general", name = "back_x_motor", tendon = "back_x", gainprm = str(DEFAULT_GAINPRM))
model.actuator.add("general", name = "back_z_motor", tendon = "back_z", gainprm = str(DEFAULT_GAINPRM))


mjcf.export_with_assets(model, "assets/mujoco/dog_model", OUT_MODEL)