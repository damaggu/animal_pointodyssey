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
DEFAULT_GEAR = 1000


with open(os.path.join(ASSET_DIR, BASE_MODEL), 'r') as f:
    basetree = etree.XML(f.read(), etree.XMLParser(remove_blank_text=True))

model = mjcf.from_xml_string(etree.tostring(basetree, pretty_print=True),
                                 model_dir=ASSET_DIR)

joints = model.find_all('joint')

print(joints)

for j in joints:
    if j.name == None:
        continue
    j.stiffness = "0.01"
    j.damping = "0.01"
    m = model.actuator.add("motor", name = j.name + "_motor", joint = j.name, gear = str(DEFAULT_GEAR))

mjcf.export_with_assets(model, "final_model", OUT_MODEL)