"""Round-trip test for mars_body.py.

In Blender (the full test: import, export, edit, export):

    blender -b --factory-startup -P tools/blender/test_roundtrip.py -- sim/bodies/biped.json

With plain Python (the JSON writer and coordinate maths only):

    python tools/blender/test_roundtrip.py sim/bodies/biped.json

Exits non-zero on failure.
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mars_body as mb  # noqa: E402

args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
spec = os.path.abspath(args[0] if args else os.path.join(os.path.dirname(__file__), "..", "..", "sim", "bodies", "biped.json"))
failures = []


def check(ok, what):
    print(("ok   " if ok else "FAIL ") + what)
    if not ok:
        failures.append(what)


def diff(a, b, path=""):
    """Paths at which two parsed specs differ."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            out += diff(a.get(k), b.get(k), "%s/%s" % (path, k))
        return out
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += diff(x, y, "%s/%d" % (path, i))
        return out
    return [] if a == b else [path]


with open(spec, encoding="utf-8") as f:
    raw = f.read()
doc = json.loads(raw)

# Core: the writer reproduces Go's output, and the mapping is a rotation.
check(mb.dump_spec(doc) == raw, "Go-style JSON writer reproduces %s byte for byte" % os.path.basename(spec))
for v in [(1, 0, 0), (0, 1, 0), (0, 0, 1), (0.3, -0.2, 0.7)]:
    check(mb.from_blender(mb.to_blender(v)) == tuple(float(x) for x in v), "from_blender(to_blender(%s))" % (v,))
check(mb.to_blender((1, 0, 0)) == (0, -1, 0) and mb.to_blender((0, 1, 0)) == (0, 0, 1) and mb.to_blender((0, 0, 1)) == (-1, 0, 0),
      "sim forward/up/right are Blender -Y/Z/-X")
for x, want in [(0.00001, "0.00001"), (3e-7, "3e-7"), (2e8, "200000000"), (1e21, "1e+21"), (-0.5, "-0.5"), (0.1 + 0.2, "0.30000000000000004")]:
    check(mb._go_float(x) == want, "go float %r -> %s" % (x, mb._go_float(x)))

if mb.bpy is not None:
    import bpy
    from mathutils import Vector

    bpy.ops.wm.read_factory_settings(use_empty=True)
    mb.register()
    tmp = tempfile.mkdtemp()
    arm = mb.import_body(bpy.context, spec)
    names = [o.name for o in arm.children_recursive]
    check(len(arm.data.bones) == len(doc["links"]), "one bone per link (%d)" % len(arm.data.bones))
    n_cap = sum(len(l.get("colliders", [])) for l in doc["links"])
    check(sum(1 for n in names if n.startswith("collider.")) == n_cap, "one capsule per collider (%d)" % n_cap)

    out = os.path.join(tmp, "same.json")
    warnings = mb.export_body(arm, out)
    check(not warnings, "no export warnings %s" % warnings)
    with open(out, encoding="utf-8") as f:
        same = f.read()
    check(same == raw, "import then export reproduces the spec byte for byte")
    if same != raw:
        print("  differences:", diff(doc, json.loads(same))[:20])

    # Posing must not change the export (it reads the rest pose).
    pb = arm.pose.bones["thigh_l"]
    pb.rotation_mode = "XYZ"
    pb.rotation_euler = (0.7, 0.2, -0.3)
    bpy.context.view_layer.update()
    mb.export_body(arm, out)
    with open(out, encoding="utf-8") as f:
        check(f.read() == raw, "a posed armature exports the same spec")
    pb.rotation_euler = (0, 0, 0)
    bpy.context.view_layer.update()

    # Move the left knee cylinder's rod pin 1 cm to Blender +X (sim -z).
    rod = bpy.data.objects["cyl.knee_l.rod"]
    w = rod.matrix_world.copy()
    w.translation += Vector((0.01, 0, 0))
    rod.matrix_world = w
    bpy.context.view_layer.update()
    mb.export_body(arm, out)
    with open(out, encoding="utf-8") as f:
        moved = json.loads(f.read())
    k = next(i for i, c in enumerate(doc["cylinders"]) if c["name"] == "knee_l")
    d = diff(doc, moved)
    check(d == ["/cylinders/%d/rod/point/2" % k], "moving one pin changes only that number: %s" % d)
    dz = moved["cylinders"][k]["rod"]["point"][2] - doc["cylinders"][k]["rod"]["point"][2]
    check(abs(dz + 0.01) < 2e-6, "the pin moved 1 cm to the robot's left (sim z %+.6f)" % dz)

    # Grow the right thigh's capsule radius by 20% via its X/Y scale.
    cap = bpy.data.objects["collider.thigh_r.0"]
    cap.scale = (1.2, 1.2, 1.0)
    bpy.context.view_layer.update()
    mb.export_body(arm, out)
    with open(out, encoding="utf-8") as f:
        grown = json.loads(f.read())
    li = next(i for i, l in enumerate(doc["links"]) if l["name"] == "thigh_r")
    d = [p for p in diff(doc, grown) if not p.startswith("/cylinders/%d/" % k)]
    check(d == ["/links/%d/colliders/0/radius" % li], "scaling a capsule changes only its radius: %s" % d)
    r0, r1 = doc["links"][li]["colliders"][0]["radius"], grown["links"][li]["colliders"][0]["radius"]
    check(abs(r1 - 1.2 * r0) < 2e-6, "radius %.6f -> %.6f" % (r0, r1))

print("%d failure(s)" % len(failures))
sys.exit(1 if failures else 0)
