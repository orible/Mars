"""Mars body spec round trip for Blender 4.x.

Import a Mars body spec (sim/bodies/biped.json, format in sim/body/body.go)
as an armature you can edit, and export it back to the same JSON.

What you get, in the neutral pose (every joint at 0):

* an armature with one bone per link: the head is the joint centre, the
  tail points at the child joint or along the centre of mass (a short bone
  along the joint axis for light brackets). Joint axis, limits, damping,
  stiffness, mass, CoM and inertia are custom properties on each bone
  (``mars_*``, in the simulator's frame and units).
* an empty for every cylinder pin (``cyl.<name>.base`` / ``.rod``), cable
  point (``cyl.<name>.cable<j>.<k>``), tendon point (``tendon.<name>.<k>``)
  and contact point (``contact.<link>.<i>``, ``mars_sensor`` marks a pad),
  each parented to its link's bone.
* a capsule mesh for every collider (``collider.<link>.<i>``), parented to
  its bone: move, rotate and scale the object (Z scale stretches it, X/Y
  scale changes the radius; *Mars > Apply capsule scale* rebuilds the mesh).
* a line per cylinder hooked to its pins, so posing the armature shows the
  cylinders follow (display only).

Export reads it all back in the rest pose (posing does not change the
export) and writes every point in its link's own frame. Anything the add-on
does not edit (hydraulics, tendon stiffness, supply, ...) is copied from the
spec stored with the import, and a value that moved by less than 0.01 mm
keeps its original number exactly, so exporting without edits reproduces
the file byte for byte and an edit changes only what you touched. Which
link a point is on is the bone it is parented to: re-parent (in the rest
pose) to move it to another link. Duplicate a contact or collider to add
one; delete it to remove it.

Coordinates: the simulator has X forward, Y up and Z to the robot's right.
A Blender character faces -Y with Z up and its right side at -X, so a sim
point (x, y, z) is Blender (-z, -x, y), and Blender (X, Y, Z) is sim
(-Y, Z, -X). Both are right-handed, so this is a rotation.

Install: Edit > Preferences > Add-ons > Install... and pick this file (or
copy it into your scripts/addons folder). File > Import / Export >
"Mars body spec (.json)", and the Mars tab in the 3D view's sidebar.
"""

bl_info = {
    "name": "Mars body spec",
    "author": "Mars",
    "version": (1, 0, 0),
    "blender": (4, 0, 0),
    "location": "File > Import/Export > Mars body spec (.json); 3D View > Sidebar > Mars",
    "description": "Edit the Mars robot body spec (joints, cylinder anchors, contacts, collision capsules) and write it back",
    "category": "Import-Export",
}

import json
import math
import os
from decimal import Decimal

# ---------------------------------------------------------------------------
# Pure maths and JSON (no bpy), shared with the tests.

KEEP_TOL = 1e-5  # m (and rad, kg, ...): smaller changes keep the original number


def to_blender(v):
    """Sim (x forward, y up, z right) to Blender (-z, -x, y)."""
    return (-v[2], -v[0], v[1])


def from_blender(v):
    """Blender (X, Y, Z) to sim (-Y, Z, -X)."""
    return (-v[1], v[2], -v[0])


def _round6(x):
    v = round(float(x), 6)
    return 0.0 if v == 0 else v


def keep_vec(orig, new):
    """orig unchanged if new is within KEEP_TOL of it everywhere, else new rounded to the micrometre."""
    if isinstance(orig, list) and len(orig) == len(new) and all(abs(o - n) <= KEEP_TOL for o, n in zip(orig, new)):
        return orig
    return [_round6(x) for x in new]


def keep_num(orig, new):
    if isinstance(orig, (int, float)) and not isinstance(orig, bool) and abs(orig - new) <= KEEP_TOL:
        return orig
    return _round6(new)


def _go_float(x):
    """Format a float the way Go's encoding/json does."""
    if x != x or x in (float("inf"), float("-inf")):
        raise ValueError("JSON cannot hold %r" % x)
    r = repr(float(x))
    a = abs(x)
    if a != 0 and (a < 1e-6 or a >= 1e21):
        mant, exp = r.split("e")
        sign, digits = exp[0], exp[1:].lstrip("0") or "0"
        return "%se%s%s" % (mant, sign if sign == "-" else "+", digits)
    s = format(Decimal(r), "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def _go_string(s):
    out = json.dumps(s, ensure_ascii=False)
    for c, e in (("<", "\\u003c"), (">", "\\u003e"), ("&", "\\u0026"), (" ", "\\u2028"), (" ", "\\u2029")):
        out = out.replace(c, e)
    return out


def go_json(v, indent=""):
    """json.MarshalIndent(v, "", "  ") as Go writes it (without the final newline)."""
    if isinstance(v, dict):
        if not v:
            return "{}"
        inner = indent + "  "
        return "{\n" + ",\n".join(inner + _go_string(k) + ": " + go_json(x, inner) for k, x in v.items()) + "\n" + indent + "}"
    if isinstance(v, list):
        if not v:
            return "[]"
        inner = indent + "  "
        return "[\n" + ",\n".join(inner + go_json(x, inner) for x in v) + "\n" + indent + "]"
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return _go_float(v)
    if isinstance(v, str):
        return _go_string(v)
    raise TypeError("cannot write %r" % (v,))


def dump_spec(doc):
    return go_json(doc) + "\n"


def link_origins(doc):
    """Each link's joint centre in the pelvis frame in the neutral pose."""
    out = {}
    for l in doc["links"]:
        if l.get("parent"):
            o, p = l["joint"]["origin"], out[l["parent"]]
            out[l["name"]] = (p[0] + o[0], p[1] + o[1], p[2] + o[2])
        else:
            out[l["name"]] = (0.0, 0.0, 0.0)
    return out


def add3(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def sub3(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def stand_height(doc):
    """Pelvis height that puts the lowest contact or capsule on the floor."""
    origin = link_origins(doc)
    lo = None
    for l in doc["links"]:
        o = origin[l["name"]]
        ys = [o[1] + p[1] for p in l.get("contacts", [])]
        ys += [o[1] + min(c["a"][1], c["b"][1]) - c["radius"] for c in l.get("colliders", [])]
        for y in ys:
            lo = y if lo is None else min(lo, y)
    return 1.0 if lo is None else -lo


# Field order of body.Link in Go, so new keys land where bodygen puts them.
LINK_KEYS = ["name", "parent", "joint", "mass", "com", "inertia", "contacts", "sensors", "colliders"]


def order_link(l):
    out = {k: l[k] for k in LINK_KEYS if k in l}
    out.update({k: v for k, v in l.items() if k not in out})
    return out


def anchor_refs(doc):
    """Every anchor in the spec: (ref path, display name, anchor dict)."""
    out = []
    for k, c in enumerate(doc.get("cylinders", [])):
        out.append((["cylinders", k, "base"], "cyl.%s.base" % c["name"], c["base"]))
        out.append((["cylinders", k, "rod"], "cyl.%s.rod" % c["name"], c["rod"]))
        for j, cab in enumerate(c.get("cables", []) or []):
            for p, a in enumerate(cab["path"]):
                out.append((["cylinders", k, "cables", j, "path", p], "cyl.%s.cable%d.%d" % (c["name"], j, p), a))
    for k, t in enumerate(doc.get("tendons", []) or []):
        for p, a in enumerate(t["path"]):
            out.append((["tendons", k, "path", p], "tendon.%s.%d" % (t["name"], p), a))
    return out


def get_ref(doc, ref):
    for part in ref:
        doc = doc[part]
    return doc


def set_ref(doc, ref, value):
    for part in ref[:-1]:
        doc = doc[part]
    doc[ref[-1]] = value


def capsule_geometry(r, h, seg=16, rings=4):
    """Vertices and faces of a capsule along Z: radius r, axis from -h to +h."""
    verts = [(0.0, 0.0, h + r)]
    rows = []
    for i in range(1, rings + 1):  # top hemisphere down to the equator
        t = (math.pi / 2) * i / rings
        rows.append((h + r * math.cos(t), r * math.sin(t)))
    for i in range(0 if h > 1e-9 else 1, rings):  # bottom hemisphere from the equator
        t = (math.pi / 2) * i / rings
        rows.append((-h - r * math.sin(t), r * math.cos(t)))
    for z, rr in rows:
        for s in range(seg):
            a = 2 * math.pi * s / seg
            verts.append((rr * math.cos(a), rr * math.sin(a), z))
    verts.append((0.0, 0.0, -h - r))
    faces = []
    row = lambda i, s: 1 + i * seg + (s % seg)
    for s in range(seg):
        faces.append((0, row(0, s), row(0, s + 1)))
    for i in range(len(rows) - 1):
        for s in range(seg):
            faces.append((row(i, s), row(i + 1, s), row(i + 1, s + 1), row(i, s + 1)))
    last = len(verts) - 1
    for s in range(seg):
        faces.append((row(len(rows) - 1, s), last, row(len(rows) - 1, s + 1)))
    return verts, faces


# ---------------------------------------------------------------------------
# Blender.

try:
    import bpy
    from bpy.props import StringProperty
    from bpy_extras.io_utils import ExportHelper, ImportHelper
    from mathutils import Matrix, Vector
except ImportError:  # imported outside Blender (tests of the maths above)
    bpy = None


if bpy is not None:

    def _collection(parent, name):
        c = bpy.data.collections.new(name)
        parent.children.link(c)
        return c

    def _bone_parent_matrix(arm, bone_name):
        """World matrix an object parented to this bone sits in, in the rest pose (the bone's tail)."""
        b = arm.data.bones[bone_name]
        return arm.matrix_world @ b.matrix_local @ Matrix.Translation((0.0, b.length, 0.0))

    def _parent_to_bone(obj, arm, bone_name, world):
        obj.parent = arm
        obj.parent_type = "BONE"
        obj.parent_bone = bone_name
        obj.matrix_parent_inverse = Matrix.Identity(4)
        obj.matrix_basis = _bone_parent_matrix(arm, bone_name).inverted() @ world

    def _rest_world(obj, arm):
        """obj's world matrix with the armature in its rest pose, or None if it is not on a bone of arm."""
        if obj.parent != arm or obj.parent_type != "BONE" or obj.parent_bone not in arm.data.bones:
            return None
        return _bone_parent_matrix(arm, obj.parent_bone) @ obj.matrix_parent_inverse @ obj.matrix_basis

    def _capsule_mesh(name, r, h):
        verts, faces = capsule_geometry(r, h)
        me = bpy.data.meshes.new(name)
        me.from_pydata(verts, [], faces)
        me.update()
        return me

    def _capsule_world(arm, a, b):
        """World matrix of a capsule mesh (axis along local Z) between Blender armature-space points a and b."""
        a, b = Vector(a), Vector(b)
        d = b - a
        rot = d.to_track_quat("Z", "Y").to_matrix().to_4x4() if d.length > 1e-9 else Matrix.Identity(4)
        return arm.matrix_world @ Matrix.Translation((a + b) / 2) @ rot

    def _tail(link, children, origin, axis):
        head = Vector(to_blender(origin[link["name"]]))
        far = [c for c in children.get(link["name"], []) if (Vector(to_blender(origin[c])) - head).length > 0.01]
        if len(far) == 1:
            return Vector(to_blender(origin[far[0]]))
        com = Vector(to_blender(link.get("com", [0, 0, 0])))
        if com.length > 0.01:
            return head + 2 * com
        if axis is not None and Vector(to_blender(axis)).length > 0:
            return head + 0.04 * Vector(to_blender(axis)).normalized()
        return head + Vector((0.0, 0.0, 0.1))

    def import_body(context, path):
        """Build the body in path as an armature with empties and capsules; returns the armature object."""
        with open(path, encoding="utf-8") as f:
            raw = f.read()
        doc = json.loads(raw)
        name = doc.get("name", os.path.splitext(os.path.basename(path))[0])
        text = bpy.data.texts.new("mars_%s.json" % name)
        text.write(raw)

        top = _collection(context.scene.collection, "Mars %s" % name)
        cols = {k: _collection(top, "%s %s" % (name, k)) for k in ("anchors", "contacts", "colliders", "cylinder lines")}

        origin = link_origins(doc)
        children = {}
        for l in doc["links"]:
            if l.get("parent"):
                children.setdefault(l["parent"], []).append(l["name"])

        arm_data = bpy.data.armatures.new(name)
        arm_data.display_type = "STICK"
        arm = bpy.data.objects.new(name, arm_data)
        arm.location = (0.0, 0.0, stand_height(doc))
        arm.show_in_front = True
        top.objects.link(arm)
        arm["mars_text"] = text.name
        arm["mars_path"] = os.path.abspath(path)

        if context.object and context.object.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        for o in context.view_layer.objects:
            o.select_set(False)
        context.view_layer.objects.active = arm
        arm.select_set(True)
        bpy.ops.object.mode_set(mode="EDIT")
        ebs = {}
        for l in doc["links"]:
            eb = arm_data.edit_bones.new(l["name"])
            eb.head = Vector(to_blender(origin[l["name"]]))
            tail = _tail(l, children, origin, (l.get("joint") or {}).get("axis"))
            if (tail - eb.head).length < 0.02:
                tail = eb.head + Vector((0.0, 0.0, 0.02))
            eb.tail = tail
            eb.use_connect = False
            if l.get("parent"):
                eb.parent = ebs[l["parent"]]
            ebs[l["name"]] = eb
        bpy.ops.object.mode_set(mode="OBJECT")
        context.view_layer.update()

        for l in doc["links"]:
            b = arm_data.bones[l["name"]]
            j = l.get("joint")
            if j:
                b["mars_axis"] = [float(x) for x in j["axis"]]
                b["mars_limits"] = [float(x) for x in j["limits"]]
                b["mars_damping"] = float(j["damping"])
                b["mars_stiffness"] = float(j.get("stiffness", 0))
            b["mars_mass"] = float(l["mass"])
            b["mars_com"] = [float(x) for x in l["com"]]
            b["mars_inertia"] = [float(x) for x in l["inertia"]]

        def at(link, p):  # world matrix of a link-frame point
            q = to_blender(add3(origin[link], p))
            return arm.matrix_world @ Matrix.Translation(q)

        def empty(col, name, link, p, kind, shape, size):
            o = bpy.data.objects.new(name, None)
            o.empty_display_type = shape
            o.empty_display_size = size
            col.objects.link(o)
            o["mars_kind"] = kind
            _parent_to_bone(o, arm, link, at(link, p))
            return o

        for l in doc["links"]:
            sensors = set(l.get("sensors", []) or [])
            for i, p in enumerate(l.get("contacts", []) or []):
                o = empty(cols["contacts"], "contact.%s.%d" % (l["name"], i), l["name"], p, "contact", "PLAIN_AXES", 0.012)
                o["mars_index"] = i
                o["mars_sensor"] = i in sensors
            for i, c in enumerate(l.get("colliders", []) or []):
                a, b = to_blender(add3(origin[l["name"]], c["a"])), to_blender(add3(origin[l["name"]], c["b"]))
                h = (Vector(b) - Vector(a)).length / 2
                nm = "collider.%s.%d" % (l["name"], i)
                o = bpy.data.objects.new(nm, _capsule_mesh(nm, c["radius"], h))
                o.display_type = "WIRE"
                o.show_in_front = True
                cols["colliders"].objects.link(o)
                o["mars_kind"] = "collider"
                o["mars_index"] = i
                o["mars_radius"] = float(c["radius"])
                o["mars_half"] = h
                _parent_to_bone(o, arm, l["name"], _capsule_world(arm, a, b))

        anchors = {}
        for ref, nm, a in anchor_refs(doc):
            o = empty(cols["anchors"], nm, a["link"], a["point"], "anchor", "SPHERE", 0.006)
            o["mars_ref"] = "/".join(str(x) for x in ref)
            anchors[o["mars_ref"]] = o

        # Display lines through each cylinder's pins (or cable runs), hooked
        # to the anchor empties so they follow a posed armature.
        def line(nm, objs):
            pts = [o.matrix_world.translation.copy() for o in objs]
            me = bpy.data.meshes.new(nm)
            me.from_pydata([tuple(p) for p in pts], [(i, i + 1) for i in range(len(pts) - 1)], [])
            lo = bpy.data.objects.new(nm, me)
            lo["mars_kind"] = "view"
            cols["cylinder lines"].objects.link(lo)
            for i, o in enumerate(objs):
                m = lo.modifiers.new("hook%d" % i, "HOOK")
                m.object = o
                m.vertex_indices_set([i])
                m.matrix_inverse = o.matrix_world.inverted()

        context.view_layer.update()
        for k, c in enumerate(doc.get("cylinders", [])):
            try:
                runs = [[anchors["cylinders/%d/base" % k], anchors["cylinders/%d/rod" % k]]]
                runs += [[anchors["cylinders/%d/cables/%d/path/%d" % (k, j, p)] for p in range(len(cab["path"]))]
                         for j, cab in enumerate(c.get("cables", []) or [])]
                for j, run in enumerate(runs):
                    line("line.%s%s" % (c["name"], "" if j == 0 else ".cable%d" % (j - 1)), run)
            except Exception as e:  # display only: never fail the import over it
                print("Mars: no line for cylinder %s: %s" % (c["name"], e))
        return arm

    def find_armature(context):
        o = context.object
        if o is not None and o.type == "ARMATURE" and "mars_text" in o:
            return o
        for o in context.scene.objects:
            if o.type == "ARMATURE" and "mars_text" in o:
                return o
        return None

    def export_body(arm, path):
        """Write the spec of armature arm to path. Returns a list of warnings."""
        warnings = []
        doc = json.loads(bpy.data.texts[arm["mars_text"]].as_string())
        bones = arm.data.bones
        inv = arm.matrix_world.inverted()
        pelvis = doc["links"][0]["name"]
        head0 = bones[pelvis].head_local

        # Link origins from the bone heads (armature space, rest pose).
        origin = {}
        for l in doc["links"]:
            if l["name"] in bones:
                origin[l["name"]] = from_blender(bones[l["name"]].head_local - head0)
            else:
                warnings.append("no bone for link %s: kept as it was" % l["name"])
        old_origin = link_origins(doc)
        for n in old_origin:
            origin.setdefault(n, old_origin[n])

        def local(world, link):  # world point to link frame
            p = from_blender(inv @ world.translation - head0) if hasattr(world, "translation") else from_blender(inv @ world - head0)
            return sub3(p, origin[link])

        # Gather the objects by what they are.
        contacts, colliders, anchors = {}, {}, {}
        for o in arm.children_recursive:
            kind = o.get("mars_kind")
            if kind in (None, "view"):
                continue
            w = _rest_world(o, arm)
            if w is None:
                warnings.append("%s is not parented to a bone of %s: ignored" % (o.name, arm.name))
                continue
            if kind == "contact":
                contacts.setdefault(o.parent_bone, []).append((o.get("mars_index", 1 << 30), o.name, o, w))
            elif kind == "collider":
                colliders.setdefault(o.parent_bone, []).append((o.get("mars_index", 1 << 30), o.name, o, w))
            elif kind == "anchor":
                anchors[o["mars_ref"]] = (o, w)

        for n in bones.keys():
            if n not in old_origin:
                warnings.append("bone %s is not a link of the spec: ignored (add links with bodygen or in the JSON)" % n)

        links = []
        for l in doc["links"]:
            l = dict(l)
            n = l["name"]
            if n in bones:
                b = bones[n]
                if l.get("joint"):
                    j = dict(l["joint"])
                    j["origin"] = keep_vec(j["origin"], sub3(origin[n], origin[l["parent"]]))
                    for key, prop in (("axis", "mars_axis"), ("limits", "mars_limits")):
                        if prop in b:
                            j[key] = keep_vec(j[key], list(b[prop]))
                    if "mars_damping" in b:
                        j["damping"] = keep_num(j["damping"], b["mars_damping"])
                    if "mars_stiffness" in b:
                        s = keep_num(j.get("stiffness", 0), b["mars_stiffness"])
                        if s:
                            j["stiffness"] = s
                        else:
                            j.pop("stiffness", None)
                    l["joint"] = j
                if "mars_mass" in b:
                    l["mass"] = keep_num(l["mass"], b["mars_mass"])
                for key, prop in (("com", "mars_com"), ("inertia", "mars_inertia")):
                    if prop in b:
                        l[key] = keep_vec(l[key], list(b[prop]))

            old = l.get("contacts", []) or []
            pts, sensors = [], []
            for i, (_, _, o, w) in enumerate(sorted(contacts.get(n, []), key=lambda t: t[:2])):
                p = local(w, n)
                idx = o.get("mars_index")
                pts.append(keep_vec(old[idx], p) if isinstance(idx, int) and idx < len(old) else [_round6(x) for x in p])
                if o.get("mars_sensor"):
                    sensors.append(i)
            if pts:
                l["contacts"] = pts
            else:
                l.pop("contacts", None)
            if sensors:
                l["sensors"] = l["sensors"] if l.get("sensors") == sensors else sensors
            else:
                l.pop("sensors", None)

            old = l.get("colliders", []) or []
            caps = []
            for _, _, o, w in sorted(colliders.get(n, []), key=lambda t: t[:2]):
                m = inv @ w
                h = float(o.get("mars_half", 0.0))
                a = sub3(from_blender(m @ Vector((0.0, 0.0, -h)) - head0), origin[n])
                bb = sub3(from_blender(m @ Vector((0.0, 0.0, h)) - head0), origin[n])
                sc = m.to_scale()
                r = float(o.get("mars_radius", 0.05)) * (sc[0] + sc[1]) / 2
                idx = o.get("mars_index")
                if isinstance(idx, int) and idx < len(old):
                    oc = old[idx]
                    caps.append({"a": keep_vec(oc["a"], a), "b": keep_vec(oc["b"], bb), "radius": keep_num(oc["radius"], r)})
                else:
                    caps.append({"a": [_round6(x) for x in a], "b": [_round6(x) for x in bb], "radius": _round6(r)})
            if caps:
                l["colliders"] = caps
            else:
                l.pop("colliders", None)
            links.append(order_link(l))

        out = dict(doc)
        out["links"] = links
        out = json.loads(json.dumps(out))  # deep copy before editing anchors in place
        for ref, nm, a in anchor_refs(out):
            key = "/".join(str(x) for x in ref)
            if key not in anchors:
                warnings.append("anchor %s has no empty: kept as it was" % nm)
                continue
            o, w = anchors[key]
            link = o.parent_bone
            p = local(w, link)
            point = keep_vec(a["point"], p) if link == a["link"] else [_round6(x) for x in p]
            set_ref(out, ref, {"link": link, "point": point})

        data = dump_spec(out)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(data)
        text = bpy.data.texts[arm["mars_text"]]
        text.clear()
        text.write(data)
        arm["mars_path"] = os.path.abspath(path)
        return warnings

    class MARS_OT_import(bpy.types.Operator, ImportHelper):
        """Import a Mars body spec as an armature with anchors, contacts and collision capsules"""

        bl_idname = "mars.import_body"
        bl_label = "Import Mars body spec"
        bl_options = {"REGISTER", "UNDO"}
        filename_ext = ".json"
        filter_glob: StringProperty(default="*.json", options={"HIDDEN"})

        def execute(self, context):
            arm = import_body(context, self.filepath)
            self.report({"INFO"}, "Imported %s (%d links)" % (arm.name, len(arm.data.bones)))
            return {"FINISHED"}

    class MARS_OT_export(bpy.types.Operator, ExportHelper):
        """Write the Mars body spec back to JSON (every point in its link's frame)"""

        bl_idname = "mars.export_body"
        bl_label = "Export Mars body spec"
        filename_ext = ".json"
        filter_glob: StringProperty(default="*.json", options={"HIDDEN"})

        @classmethod
        def poll(cls, context):
            return find_armature(context) is not None

        def invoke(self, context, event):
            arm = find_armature(context)
            if arm is not None and arm.get("mars_path"):
                self.filepath = arm["mars_path"]
            context.window_manager.fileselect_add(self)
            return {"RUNNING_MODAL"}

        def execute(self, context):
            arm = find_armature(context)
            if context.object and context.object.mode != "OBJECT":
                bpy.ops.object.mode_set(mode="OBJECT")
            warnings = export_body(arm, self.filepath)
            for w in warnings:
                self.report({"WARNING"}, w)
            self.report({"INFO"}, "Wrote %s" % self.filepath)
            return {"FINISHED"}

    class MARS_OT_add_collider(bpy.types.Operator):
        """Add a collision capsule along the active bone"""

        bl_idname = "mars.add_collider"
        bl_label = "Add capsule to bone"
        bl_options = {"REGISTER", "UNDO"}

        @classmethod
        def poll(cls, context):
            arm = find_armature(context)
            return arm is not None and arm.data.bones.active is not None

        def execute(self, context):
            arm = find_armature(context)
            if context.object and context.object.mode != "OBJECT":
                bpy.ops.object.mode_set(mode="OBJECT")
            b = arm.data.bones.active
            r = 0.04
            h = max(0.0, b.length / 2 - r)
            n = sum(1 for o in arm.children_recursive if o.get("mars_kind") == "collider" and o.parent_bone == b.name)
            nm = "collider.%s.%d" % (b.name, n)
            o = bpy.data.objects.new(nm, _capsule_mesh(nm, r, h))
            o.display_type = "WIRE"
            o.show_in_front = True
            col = next((c for c in arm.users_collection), context.scene.collection)
            col.objects.link(o)
            o["mars_kind"] = "collider"
            o["mars_index"] = 1 << 20  # after the existing ones
            o["mars_radius"] = r
            o["mars_half"] = h
            _parent_to_bone(o, arm, b.name, _capsule_world(arm, b.head_local, b.tail_local))
            return {"FINISHED"}

    class MARS_OT_apply_capsule_scale(bpy.types.Operator):
        """Rebuild the selected capsules' meshes at their current size, with unit scale"""

        bl_idname = "mars.apply_capsule_scale"
        bl_label = "Apply capsule scale"
        bl_options = {"REGISTER", "UNDO"}

        def execute(self, context):
            for o in context.selected_objects:
                if o.get("mars_kind") != "collider":
                    continue
                loc, rot, sc = o.matrix_basis.decompose()
                r = float(o["mars_radius"]) * (sc[0] + sc[1]) / 2
                h = float(o["mars_half"]) * sc[2]
                old = o.data
                o.data = _capsule_mesh(o.name, r, h)
                if old.users == 0:
                    bpy.data.meshes.remove(old)
                o["mars_radius"], o["mars_half"] = r, h
                o.matrix_basis = Matrix.LocRotScale(loc, rot, None)
            return {"FINISHED"}

    class MARS_PT_panel(bpy.types.Panel):
        bl_label = "Mars body"
        bl_space_type = "VIEW_3D"
        bl_region_type = "UI"
        bl_category = "Mars"

        def draw(self, context):
            col = self.layout.column(align=True)
            col.operator(MARS_OT_import.bl_idname, icon="IMPORT")
            col.operator(MARS_OT_export.bl_idname, icon="EXPORT")
            col.separator()
            col.operator(MARS_OT_add_collider.bl_idname, icon="MESH_CAPSULE")
            col.operator(MARS_OT_apply_capsule_scale.bl_idname, icon="FULLSCREEN_ENTER")
            arm = find_armature(context)
            if arm is not None:
                col.separator()
                col.prop(arm.data, "pose_position", expand=True)
                b = arm.data.bones.active
                if b is not None:
                    box = self.layout.box()
                    box.label(text=b.name)
                    for k in ("mars_axis", "mars_limits", "mars_damping", "mars_stiffness", "mars_mass", "mars_com", "mars_inertia"):
                        if k in b:
                            box.prop(b, '["%s"]' % k, text=k[5:])

    def _menu_import(self, context):
        self.layout.operator(MARS_OT_import.bl_idname, text="Mars body spec (.json)")

    def _menu_export(self, context):
        self.layout.operator(MARS_OT_export.bl_idname, text="Mars body spec (.json)")

    _classes = (MARS_OT_import, MARS_OT_export, MARS_OT_add_collider, MARS_OT_apply_capsule_scale, MARS_PT_panel)

    def register():
        for c in _classes:
            bpy.utils.register_class(c)
        bpy.types.TOPBAR_MT_file_import.append(_menu_import)
        bpy.types.TOPBAR_MT_file_export.append(_menu_export)

    def unregister():
        bpy.types.TOPBAR_MT_file_export.remove(_menu_export)
        bpy.types.TOPBAR_MT_file_import.remove(_menu_import)
        for c in reversed(_classes):
            bpy.utils.unregister_class(c)

else:

    def register():
        raise RuntimeError("mars_body: register() needs Blender")

    def unregister():
        pass
