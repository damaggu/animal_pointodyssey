import itertools
import os
from typing import Sequence

from absl import app

from dm_control import mjcf
from dm_control.mujoco.wrapper.mjbindings import mjlib
from lxml import etree
import numpy as np


ASSET_RELPATH = 'export/'
ASSET_DIR = os.path.dirname(__file__) + '/' + ASSET_RELPATH
BASE_MODEL = 'mouse_export.xml'
DEFAULT_MODEL = 'mouse_defaults.xml'
OUT_DIR = "data/mujoco/"
OUT_MODEL = "mouse.xml"
DEFAULT_GEAR = 120
DEFAULT_GAINPRM = 5
DEFAULT_COEF = 0.01

with open(os.path.join(ASSET_DIR, BASE_MODEL), 'r') as f:
    basetree = etree.XML(f.read(), etree.XMLParser(remove_blank_text=True))

model = mjcf.from_xml_string(etree.tostring(basetree, pretty_print=True),
                                 model_dir=ASSET_DIR)

model.option.timestep = "0.001"
model.option.integrator = "RK4"
model.compiler.angle = "degree"

model.default.general.ctrllimited="true"
model.default.general.ctrlrange="-1 1"
model.default.general.forcelimited="false"
model.default.general.gainprm = str(DEFAULT_GAINPRM)
model.default.tendon.range = "-1 1"
model.default.motor.gear = str(DEFAULT_GEAR)

model.worldbody.add("camera", name="cam0", pos="5 -17 7", xyaxes="1 0.25 0 0 0.3 1", mode="trackcom")



model.tendon.add("fixed", name = "back_x")
model.tendon.add("fixed", name = "back_y")
model.tendon.add("fixed", name = "back_z")
# model.tendon.add("fixed", name = "tail_x")
# model.tendon.add("fixed", name = "tail_z")

joints = model.find_all('joint')

for j in joints:
    if j.name == None:
        continue
    if "Tail" in j.name:
        j.remove()
        continue
    j.stiffness = "5"
    j.damping = "0.4"
    #r = j.range/2
    #j.range = str(r[0]) + " " + str(r[1])
    if "Back" in j.name:
        j.stiffness = "20"
        j.damping = "0.5"
    if "rx_Back" in j.name:
        model.tendon.fixed["back_x"].add("joint", joint=j.name, coef=str(DEFAULT_COEF))
    elif "ry_Back" in j.name:
        model.tendon.fixed["back_y"].add("joint", joint=j.name, coef=str(DEFAULT_COEF))
    elif "rz_Back" in j.name:
        model.tendon.fixed["back_z"].add("joint", joint=j.name, coef=str(DEFAULT_COEF))
    elif "rx_Tail" in j.name:
        model.tendon.fixed["tail_x"].add("joint", joint=j.name, coef=str(DEFAULT_COEF))
    elif "rz_Tail" in j.name:
        model.tendon.fixed["tail_z"].add("joint", joint=j.name, coef=str(DEFAULT_COEF))
    else:
        model.actuator.add("motor", name = j.name + "_motor", joint = j.name, gear = str(DEFAULT_GEAR))

model.actuator.add("general", name = "back_x_motor", tendon = "back_x", gainprm = str(DEFAULT_GAINPRM))
model.actuator.add("general", name = "back_y_motor", tendon = "back_y", gainprm = str(DEFAULT_GAINPRM))
model.actuator.add("general", name = "back_z_motor", tendon = "back_z", gainprm = str(DEFAULT_GAINPRM))
# model.actuator.add("general", name = "tail_x_motor", tendon = "tail_x", gainprm = str(DEFAULT_GAINPRM))
# model.actuator.add("general", name = "tail_z_motor", tendon = "tail_z", gainprm = str(DEFAULT_GAINPRM))

touch_sites = ["L_B_Finger_3_3", "R_B_Finger_3_3", "L_F_Finger_3_3", "R_F_Finger_3_3"]

bodies = model.find_all('body')
parts_to_remove = ["Finger", "Tail", "Rib"]
to_remove = []
for b in bodies:
    for p in parts_to_remove:
        if p in b.name:
            to_remove.append(b.name)
for bn in to_remove:
    b = model.find("body", bn)
    if b is not None:
        b.remove()

bodies = model.find_all('body')
for b in bodies:
    if "Head" in b.name:
        b.add("site", name = "Head", type = "box", size = "0.002 0.002 0.002")
    elif b.name in touch_sites:
        b.add("site", name=b.name, type="box", size="0.004 0.004 0.002")
        model.sensor.add("touch", name=b.name, site=b.name)


head_sensors = ["accelerometer", "velocimeter", "gyro"]
for s in head_sensors:
    model.sensor.add(s, name=s, site="Head")

geoms = model.find_all("geom")

for g in geoms:

    if "Finger" in g.name or "Rib" in g.name:
        g.size= "0.02"
    elif "floor" not in g.name:
        g.size = "0.06"
    g.density = "1000"


mjcf.export_with_assets(model, OUT_DIR, OUT_MODEL)