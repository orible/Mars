"""Install (or update) and enable the Mars body add-on in Blender's user preferences.

    blender -b -P tools/blender/install_addon.py

The setup scripts run this. Run it again after changing mars_body.py: Blender
keeps its own copy of the add-on.
"""

import os

import addon_utils
import bpy

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mars_body.py")
bpy.ops.preferences.addon_install(filepath=path, overwrite=True)
bpy.ops.preferences.addon_enable(module="mars_body")
bpy.ops.wm.save_userpref()
loaded, enabled = addon_utils.check("mars_body")
print("MARS_ADDON %s" % ("enabled" if loaded and enabled else "FAILED"))
