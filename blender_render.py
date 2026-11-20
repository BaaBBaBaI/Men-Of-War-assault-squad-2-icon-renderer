"""
Blender 3.5 headless renderer for Men of War: Assault Squad 2 vehicle icons.
Usage:
  blender -b --python blender_render.py -- --vehicle <dir> --mod <mod_dir> --out <out_dir> [options]
"""

import os
import sys
import math
import argparse
import tempfile
import numpy as np
import bpy
from mathutils import Vector, Matrix

# Ensure local modules are importable inside Blender's Python
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from gem2_loader import load_vehicle, TextureResolver, Gem2Material
from icon_style import compose_icon_set, save_tga_rgba, ICON_W, ICON_H


# ---------------------------------------------------------------------------
# Scene setup helpers
# ---------------------------------------------------------------------------

def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def setup_cycles(samples: int = 32, width: int = ICON_W * 4, height: int = ICON_H * 4):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = samples
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.02
    scene.cycles.use_denoising = False  # Faster and preserves crisp micro-detail at 104x52
    scene.cycles.max_bounces = 4
    scene.cycles.diffuse_bounces = 2
    scene.cycles.glossy_bounces = 2
    scene.cycles.transmission_bounces = 2
    scene.cycles.transparent_max_bounces = 6

    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"

    # Standard sRGB color management (matches MoW:AS2 direct-lit look better than Filmic)
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0


def setup_lighting(
    sun_energy: float = 5.8,
    ambient_strength: float = 0.22,
    sun_yaw_deg: float = 53.0,
    sun_pitch_deg: float = 67.0,
):
    scene = bpy.context.scene

    # Low studio world ambient so dark undersides/tracks stay punchy like MoW:AS2
    world = bpy.data.worlds.new("StudioWorld")
    scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Color"].default_value = (0.95, 0.93, 0.87, 1.0)
    bg.inputs["Strength"].default_value = ambient_strength
    out = nt.nodes.new("ShaderNodeOutputWorld")
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])

    # Key Sun light positioned by (sun_yaw_deg, sun_pitch_deg):
    # yaw=0 is front (+X), yaw=90 is left (+Y), pitch=90 is straight overhead (+Z)
    syaw = math.radians(sun_yaw_deg)
    spitch = math.radians(sun_pitch_deg)
    sun_from = Vector((
        math.cos(spitch) * math.cos(syaw),
        math.cos(spitch) * math.sin(syaw),
        math.sin(spitch),
    ))
    key_data = bpy.data.lights.new("KeySun", type="SUN")
    key_data.energy = sun_energy
    key_data.color = (1.0, 0.96, 0.86)
    key_data.angle = math.radians(7.0)
    key_obj = bpy.data.objects.new("KeySun", key_data)
    scene.collection.objects.link(key_obj)
    key_dir = (-sun_from).normalized()
    key_obj.rotation_euler = key_dir.to_track_quat("-Z", "Y").to_euler()

    # Specular sheen Area light follows sun azimuth so highlights rotate with the sun
    spec_data = bpy.data.lights.new("SpecArea", type="AREA")
    spec_data.energy = 110.0
    spec_data.color = (1.0, 0.98, 0.92)
    spec_data.size = 3.2
    spec_obj = bpy.data.objects.new("SpecArea", spec_data)
    scene.collection.objects.link(spec_obj)
    spec_r = 5.0
    spec_obj.location = Vector((
        spec_r * math.cos(syaw),
        spec_r * math.sin(syaw),
        4.6,
    ))
    spec_obj.rotation_euler = (Vector((0, 0, 0.8)) - spec_obj.location).to_track_quat("-Z", "Y").to_euler()

    # Subtle fill light from front-left so hull front stays readable
    fill_data = bpy.data.lights.new("FillArea", type="AREA")
    fill_data.energy = 24.0
    fill_data.color = (0.90, 0.93, 1.0)
    fill_data.size = 6.0
    fill_obj = bpy.data.objects.new("FillArea", fill_data)
    scene.collection.objects.link(fill_obj)
    fill_obj.location = Vector((4.8, 2.8, 2.0))
    fill_obj.rotation_euler = (Vector((0, 0, 0.6)) - fill_obj.location).to_track_quat("-Z", "Y").to_euler()

    # Rim light from upper-rear-left to pop upper armor edges & turret top
    rim_data = bpy.data.lights.new("RimArea", type="AREA")
    rim_data.energy = 55.0
    rim_data.color = (1.0, 0.97, 0.88)
    rim_data.size = 3.5
    rim_obj = bpy.data.objects.new("RimArea", rim_data)
    scene.collection.objects.link(rim_obj)
    rim_obj.location = Vector((-4.2, 2.2, 4.8))
    rim_obj.rotation_euler = (Vector((0, 0, 0.9)) - rim_obj.location).to_track_quat("-Z", "Y").to_euler()


# ---------------------------------------------------------------------------
# Material builder
# ---------------------------------------------------------------------------

_IMAGE_CACHE = {}


def _load_image(path: str, non_color: bool = False):
    key = (os.path.abspath(path), non_color)
    if key in _IMAGE_CACHE:
        return _IMAGE_CACHE[key]
    try:
        img = bpy.data.images.load(path, check_existing=False)
        img.colorspace_settings.name = "Non-Color" if non_color else "sRGB"
        _IMAGE_CACHE[key] = img
        return img
    except Exception as e:
        print(f"[warn] Could not load image {path}: {e}")
        _IMAGE_CACHE[key] = None
        return None


def build_blender_material(
    mtl_name: str,
    gmat: Gem2Material,
    vehicle_dir: str,
    resolver: TextureResolver,
) -> bpy.types.Material:
    mat = bpy.data.materials.new(name=mtl_name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()

    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Metallic"].default_value = 0.18
    bsdf.inputs["Roughness"].default_value = 0.30
    bsdf.inputs["Specular"].default_value = 0.95
    bsdf.inputs["Clearcoat"].default_value = 0.55
    bsdf.inputs["Clearcoat Roughness"].default_value = 0.18
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    diff_path = resolver.resolve(gmat.diffuse, vehicle_dir)
    bump_path = resolver.resolve(gmat.bump, vehicle_dir)
    spec_path = resolver.resolve(gmat.specular, vehicle_dir)

    is_glass = ("glass" in mtl_name.lower()) or (gmat.diffuse and "glass" in gmat.diffuse.lower())
    is_enumerator = (gmat.diffuse and "enumerator" in gmat.diffuse.lower())

    if diff_path:
        diff_img = _load_image(diff_path, non_color=False)
        if diff_img is not None:
            tex = nt.nodes.new("ShaderNodeTexImage")
            tex.image = diff_img
            nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])

            if is_glass:
                bsdf.inputs["Roughness"].default_value = 0.18
                bsdf.inputs["Specular"].default_value = 0.75
                bsdf.inputs["Alpha"].default_value = 0.35
            elif gmat.blend == "test":
                gt = nt.nodes.new("ShaderNodeMath")
                gt.operation = "GREATER_THAN"
                gt.inputs[1].default_value = 0.45
                nt.links.new(tex.outputs["Alpha"], gt.inputs[0])
                nt.links.new(gt.outputs["Value"], bsdf.inputs["Alpha"])
            elif gmat.blend == "blend":
                nt.links.new(tex.outputs["Alpha"], bsdf.inputs["Alpha"])
                if is_enumerator:
                    bsdf.inputs["Roughness"].default_value = 0.85
            elif gmat.blend == "add":
                bsdf.inputs["Alpha"].default_value = 0.0
    else:
        bsdf.inputs["Base Color"].default_value = (0.35, 0.37, 0.30, 1.0)
        if is_glass:
            bsdf.inputs["Alpha"].default_value = 0.35

    # Specular / Roughness modulation
    if spec_path and not is_glass:
        spec_img = _load_image(spec_path, non_color=True)
        if spec_img is not None:
            stex = nt.nodes.new("ShaderNodeTexImage")
            stex.image = spec_img
            # Map specular brightness [0..1] -> roughness [0.42..0.12] for punchy military sheen
            mr = nt.nodes.new("ShaderNodeMapRange")
            mr.inputs["From Min"].default_value = 0.0
            mr.inputs["From Max"].default_value = 1.0
            mr.inputs["To Min"].default_value = 0.42
            mr.inputs["To Max"].default_value = 0.12
            nt.links.new(stex.outputs["Color"], mr.inputs["Value"])
            nt.links.new(mr.outputs["Result"], bsdf.inputs["Roughness"])

    # Normal / Bump map (subtle strength for crisp military armor detail)
    if bump_path and not is_glass:
        bump_img = _load_image(bump_path, non_color=True)
        if bump_img is not None:
            btex = nt.nodes.new("ShaderNodeTexImage")
            btex.image = bump_img
            nmap = nt.nodes.new("ShaderNodeNormalMap")
            nmap.inputs["Strength"].default_value = 0.40
            nt.links.new(btex.outputs["Color"], nmap.inputs["Color"])
            nt.links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])

    return mat


# ---------------------------------------------------------------------------
# Vehicle importer & auto-normalizer
# ---------------------------------------------------------------------------

def import_vehicle_to_blender(
    vehicle_dir: str,
    mod_dir: str = None,
    game_dir: str = None,
    mdl_path: str = None,
    common_tex_dir: str = None,
):
    instances, materials = load_vehicle(vehicle_dir, mdl_path=mdl_path)
    if not instances:
        raise RuntimeError(f"No meshes loaded from {vehicle_dir}")

    resolver = TextureResolver(mod_dir=mod_dir, game_dir=game_dir, common_tex_dir=common_tex_dir)
    _IMAGE_CACHE.clear()

    # Pre-transform all vertices to compute global bounding box for normalization
    transformed = []
    all_pts = []
    core_pts = []  # Non-gun-barrel points for tight MoW:AS2 framing

    for inst in instances:
        ply = inst.ply
        if len(ply.vertices) == 0 or len(ply.indices) == 0:
            continue
        # Sanitize corrupted/uninitialized vertices (e.g. 1e+27 from some mod exporters)
        v_raw = ply.vertices.astype(np.float64)
        valid_v = np.all(np.isfinite(v_raw), axis=1) & (np.max(np.abs(v_raw), axis=1) < 1e5)
        tris = ply.indices
        in_range = np.all(tris < len(v_raw), axis=1)
        valid_tri = in_range.copy()
        if np.any(in_range):
            valid_tri[in_range] = np.all(valid_v[tris[in_range]], axis=1)
        tris = tris[valid_tri]
        if len(tris) == 0:
            continue
        # Clamp invalid unreferenced vertices to 0 so Blender doesn't see 1e+27
        if not np.all(valid_v):
            v_raw = np.where(valid_v[:, None], v_raw, 0.0)

        ones = np.ones((len(v_raw), 1), dtype=np.float64)
        v_hom = np.concatenate([v_raw, ones], axis=1)
        v_world = (v_hom @ inst.world_matrix)[:, :3]

        det = float(np.linalg.det(inst.world_matrix[:3, :3]))
        flip = (det < 0.0) ^ ply.mirrored
        tris = tris[:, [0, 2, 1]] if flip else tris

        ref_idx = np.unique(tris)
        ref_pts = v_world[ref_idx]

        transformed.append((inst, v_world, tris, ref_idx))
        all_pts.append(ref_pts)

        bname = inst.bone_name.lower()
        pname = os.path.basename(inst.ply_path).lower()
        # Exclude long gun barrels / antennas from core framing box so hull fills frame tightly
        if not any(k in bname or k in pname for k in ("gun", "barrel", "antenna", "mgun")):
            core_pts.append(ref_pts)

    all_pts_arr = np.concatenate(all_pts, axis=0)
    core_pts_arr = np.concatenate(core_pts, axis=0) if core_pts else all_pts_arr

    # Center X/Y on core hull, rest lowest point on Z=0, scale so core max extent = 4.0 units
    c_min = core_pts_arr.min(axis=0)
    c_max = core_pts_arr.max(axis=0)
    g_min = all_pts_arr.min(axis=0)

    center_x = 0.5 * (c_min[0] + c_max[0])
    center_y = 0.5 * (c_min[1] + c_max[1])
    base_z = g_min[2]

    core_size = c_max - c_min
    scale = 4.0 / max(float(core_size.max()), 1e-4)

    # Build Blender materials
    bl_mats = {}
    for mname, gmat in materials.items():
        bl_mats[mname] = build_blender_material(mname, gmat, vehicle_dir, resolver)

    # Create Blender mesh objects
    vehicle_objs = []
    norm_all_pts = []
    norm_core_pts = []

    for idx, (inst, v_world, tris, ref_idx) in enumerate(transformed):
        ply = inst.ply
        pts = np.empty_like(v_world)
        pts[:, 0] = (v_world[:, 0] - center_x) * scale
        pts[:, 1] = (v_world[:, 1] - center_y) * scale
        pts[:, 2] = (v_world[:, 2] - base_z) * scale

        norm_all_pts.append(pts[ref_idx])
        bname = inst.bone_name.lower()
        pname = os.path.basename(inst.ply_path).lower()
        if not any(k in bname or k in pname for k in ("gun", "barrel", "antenna", "mgun")):
            norm_core_pts.append(pts[ref_idx])

        mesh = bpy.data.meshes.new(f"{inst.bone_name}_{idx}")
        mesh.from_pydata(pts.tolist(), [], tris.tolist())

        # Assign materials per submesh
        mat_slot_map = {}
        for sm in ply.submeshes:
            if sm.mtl_name not in mat_slot_map:
                mat_slot_map[sm.mtl_name] = len(mesh.materials)
                mesh.materials.append(bl_mats.get(sm.mtl_name))
            slot = mat_slot_map[sm.mtl_name]
            start = sm.first_face
            end = min(start + sm.num_faces, len(mesh.polygons))
            for fi in range(start, end):
                mesh.polygons[fi].material_index = slot

        # Smooth shading with auto-smooth for crisp armor bevels
        mesh.polygons.foreach_set("use_smooth", [True] * len(mesh.polygons))
        mesh.use_auto_smooth = True
        mesh.auto_smooth_angle = math.radians(42.0)

        # Assign UVs
        uv_layer = mesh.uv_layers.new(name="UVMap")
        uvs = ply.uvs
        loops = mesh.loops
        uv_data = np.empty((len(loops), 2), dtype=np.float32)
        loop_vidx = np.empty(len(loops), dtype=np.int32)
        loops.foreach_get("vertex_index", loop_vidx)
        uv_data = uvs[loop_vidx]
        uv_layer.data.foreach_set("uv", uv_data.ravel())

        mesh.update()

        obj = bpy.data.objects.new(f"{inst.bone_name}_{idx}", mesh)
        bpy.context.scene.collection.objects.link(obj)
        vehicle_objs.append(obj)

    norm_all = np.concatenate(norm_all_pts, axis=0)
    norm_core = np.concatenate(norm_core_pts, axis=0) if norm_core_pts else norm_all
    return vehicle_objs, norm_all, norm_core


# ---------------------------------------------------------------------------
# Auto-framing camera (matches MoW:AS2 t34_3_00.tga front-left 3/4 view)
# ---------------------------------------------------------------------------

def setup_camera_auto_framed(
    all_pts: np.ndarray,
    core_pts: np.ndarray,
    yaw_deg: float = 48.0,
    pitch_deg: float = 20.0,
    zoom: float = 1.0,
    shift_x: float = 0.04,
    shift_y: float = 0.0,
    focal_mm: float = 95.0,
):
    """
    Place a telephoto camera at front-left-above (+X, +Y, +Z) looking at the vehicle
    and automatically solve camera distance & target offset so the vehicle fills
    the 104x52 MoW:AS2 icon interior tightly.
    shift_x > 0 shifts the vehicle right; shift_y > 0 shifts the vehicle up.
    """
    scene = bpy.context.scene
    cam_data = bpy.data.cameras.new("IconCam")
    cam_data.type = "PERSP"
    cam_data.lens = focal_mm
    cam_data.sensor_width = 36.0
    cam_data.shift_x = -shift_x
    cam_data.shift_y = -shift_y
    cam_data.clip_start = 0.05
    cam_data.clip_end = 500.0

    cam_obj = bpy.data.objects.new("IconCam", cam_data)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj

    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)

    # Unit vector from target toward camera (front-left-above: +X, +Y, +Z)
    cam_dir = np.array([
        math.cos(pitch) * math.cos(yaw),
        math.cos(pitch) * math.sin(yaw),
        math.sin(pitch),
    ], dtype=np.float64)

    # Camera basis vectors in world space
    fwd = -cam_dir  # looking from camera toward target
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(fwd, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    up /= np.linalg.norm(up)

    # Project core points onto camera right/up/depth to center and fit tightly
    # Also include 25% weight of full bounds (e.g. gun barrel) horizontally so it frames naturally
    c_r = core_pts @ right
    c_u = core_pts @ up
    c_d = core_pts @ cam_dir  # positive toward camera

    # Slight horizontal shift right if barrel sticks far left, matching t34_3_00.tga
    r_min, r_max = float(c_r.min()), float(c_r.max())
    u_min, u_max = float(c_u.min()), float(c_u.max())

    # Target point in world space
    r_mid = 0.5 * (r_min + r_max)
    u_mid = 0.5 * (u_min + u_max)
    d_mid = 0.5 * (float(c_d.min()) + float(c_d.max()))
    target = r_mid * right + u_mid * up + d_mid * cam_dir

    # Compute exact perspective distance needed to fit core_pts into safe interior:
    # Interior of 104x52 is ~96x44 -> ~84% width, ~84% height
    aspect = float(ICON_W) / float(ICON_H)  # 2.0
    fov_x_half = math.atan(0.5 * cam_data.sensor_width / cam_data.lens)
    tan_x = math.tan(fov_x_half)
    tan_y = tan_x / aspect

    fill_x = 0.97 * zoom
    fill_y = 1.01 * zoom

    rel = core_pts - target[None, :]
    pr = np.abs(rel @ right)
    pu = np.abs(rel @ up)
    pd = rel @ cam_dir  # positive = closer to camera

    # For each point, required distance D satisfies: pr / (D - pd) <= fill_x * tan_x
    req_d_x = float(np.max(pd + pr / (fill_x * tan_x)))
    req_d_y = float(np.max(pd + pu / (fill_y * tan_y)))
    dist = max(req_d_x, req_d_y, 2.0)

    cam_pos = target + cam_dir * dist
    cam_obj.location = Vector(cam_pos.tolist())
    look_vec = Vector((target - cam_pos).tolist())
    cam_obj.rotation_euler = look_vec.to_track_quat("-Z", "Y").to_euler()
    return cam_obj


# ---------------------------------------------------------------------------
# Ground shadow catcher & rendering
# ---------------------------------------------------------------------------

def create_shadow_catcher():
    bpy.ops.mesh.primitive_plane_add(size=40.0, location=(0.0, 0.0, -0.002))
    plane = bpy.context.active_object
    plane.name = "ShadowCatcher"
    plane.is_shadow_catcher = True

    mat = bpy.data.materials.new("ShadowMat")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.7, 0.7, 0.7, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.9
    plane.data.materials.append(mat)
    return plane


def render_to_rgba_array(tmp_png: str) -> np.ndarray:
    bpy.context.scene.render.filepath = tmp_png
    bpy.ops.render.render(write_still=True)
    img = bpy.data.images.load(tmp_png, check_existing=False)
    W, H = img.size
    px = np.array(img.pixels[:], dtype=np.float32).reshape(H, W, 4)
    bpy.data.images.remove(img)
    # Flip vertically (Blender images are bottom-up) and convert to [0..255]
    px = px[::-1, :, :] * 255.0
    return np.clip(px, 0.0, 255.0)


def render_vehicle_icons(
    vehicle_dir: str,
    mod_dir: str = None,
    out_dir: str = ".",
    name: str = None,
    game_dir: str = None,
    mdl_path: str = None,
    common_tex_dir: str = None,
    yaw_deg: float = 48.0,
    pitch_deg: float = 20.0,
    zoom: float = 1.0,
    shift_x: float = 0.04,
    shift_y: float = 0.0,
    samples: int = 32,
    sun_energy: float = 5.8,
    sun_yaw_deg: float = 53.0,
    sun_pitch_deg: float = 67.0,
    ambient: float = 0.22,
    sharpen: float = 0.45,
    save_preview: bool = False,
):
    vehicle_dir = os.path.abspath(vehicle_dir)
    if not name:
        name = os.path.basename(os.path.normpath(vehicle_dir))

    clear_scene()
    setup_cycles(samples=samples, width=ICON_W * 4, height=ICON_H * 4)
    setup_lighting(
        sun_energy=sun_energy,
        ambient_strength=ambient,
        sun_yaw_deg=sun_yaw_deg,
        sun_pitch_deg=sun_pitch_deg,
    )

    vehicle_objs, all_pts, core_pts = import_vehicle_to_blender(
        vehicle_dir=vehicle_dir,
        mod_dir=mod_dir,
        game_dir=game_dir,
        mdl_path=mdl_path,
        common_tex_dir=common_tex_dir,
    )
    setup_camera_auto_framed(
        all_pts=all_pts,
        core_pts=core_pts,
        yaw_deg=yaw_deg,
        pitch_deg=pitch_deg,
        zoom=zoom,
        shift_x=shift_x,
        shift_y=shift_y,
    )
    shadow_plane = create_shadow_catcher()

    with tempfile.TemporaryDirectory(prefix="mowas2_render_") as td:
        veh_png = os.path.join(td, "veh.png")
        sh_png = os.path.join(td, "shadow.png")

        # Pass 1: Vehicle RGBA only (hide shadow catcher)
        shadow_plane.hide_render = True
        veh_rgba = render_to_rgba_array(veh_png)

        # Pass 2: Ground contact/cast shadow only (hide vehicle from camera rays, keep shadow rays)
        shadow_plane.hide_render = False
        for obj in vehicle_objs:
            obj.visible_camera = False
        bpy.context.scene.cycles.samples = max(16, samples // 2)
        sh_rgba = render_to_rgba_array(sh_png)
        shadow_alpha = np.clip(sh_rgba[:, :, 3] / 255.0, 0.0, 1.0)

    icons = compose_icon_set(veh_rgba, shadow_alpha_hi=shadow_alpha, sharpen=sharpen)

    os.makedirs(out_dir, exist_ok=True)
    saved = []
    for suffix in ("00", "01", "02", "03"):
        out_path = os.path.join(out_dir, f"{name}_{suffix}.tga")
        save_tga_rgba(out_path, icons[suffix])
        saved.append(out_path)
        print(f"[ok] Saved {out_path}")

    if save_preview:
        # Save a 4x upscaled PNG preview sheet using Blender's image saver
        scale = 4
        pw, ph = ICON_W * scale, ICON_H * scale
        pad = 8
        sheet_w = pw + pad * 2
        sheet_h = (ph + pad) * 4 + pad
        sheet = np.zeros((sheet_h, sheet_w, 4), dtype=np.float32)
        sheet[:, :, :3] = 40.0 / 255.0
        sheet[:, :, 3] = 1.0
        for i, suffix in enumerate(("00", "01", "02", "03")):
            im = icons[suffix].astype(np.float32) / 255.0
            up = np.repeat(np.repeat(im, scale, axis=0), scale, axis=1)
            y0 = pad + i * (ph + pad)
            # Alpha blend onto dark gray preview sheet
            a = up[:, :, 3:4]
            sheet[y0:y0 + ph, pad:pad + pw, :3] = up[:, :, :3] * a + sheet[y0:y0 + ph, pad:pad + pw, :3] * (1.0 - a)
        preview_path = os.path.join(out_dir, f"{name}_preview.png")
        bimg = bpy.data.images.new("PreviewSheet", width=sheet_w, height=sheet_h, alpha=True)
        bimg.pixels.foreach_set(sheet[::-1, :, :].ravel())
        bimg.filepath_raw = preview_path
        bimg.file_format = "PNG"
        bimg.save()
        print(f"[ok] Saved preview {preview_path}")

    return saved


def main():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    p = argparse.ArgumentParser(description="Render MoW:AS2 vehicle TGA icons via Blender.")
    p.add_argument("--vehicle", required=True, help="Path to vehicle folder (containing .def/.mdl/.ply)")
    p.add_argument("--mod", default=None, help="Optional path to mod root folder")
    p.add_argument("--common-tex", default=None, help="Optional path to resource/texture/common folder")
    p.add_argument("--game", default=None, help="Optional path to MoW:AS2 base game folder")
    p.add_argument("--mdl", default=None, help="Optional specific .mdl file inside vehicle folder")
    p.add_argument("--out", required=True, help="Output folder for _00.._03.tga files")
    p.add_argument("--name", default=None, help="Base icon filename (defaults to vehicle folder name)")
    p.add_argument("--yaw", type=float, default=48.0, help="Camera yaw angle in degrees (default: 48)")
    p.add_argument("--pitch", type=float, default=20.0, help="Camera pitch angle in degrees (default: 20)")
    p.add_argument("--zoom", type=float, default=1.0, help="Framing zoom multiplier (default: 1.0)")
    p.add_argument("--shift-x", type=float, default=0.04, help="Horizontal shift right as fraction of width (default: 0.04)")
    p.add_argument("--shift-y", type=float, default=0.0, help="Vertical shift up as fraction of width (default: 0.0)")
    p.add_argument("--samples", type=int, default=32, help="Cycles samples (default: 32)")
    p.add_argument("--sun", type=float, default=5.8, help="Sun light energy (default: 5.8)")
    p.add_argument("--sun-yaw", type=float, default=53.0, help="Sun azimuth angle in degrees (default: 53)")
    p.add_argument("--sun-pitch", type=float, default=67.0, help="Sun elevation angle in degrees (default: 67)")
    p.add_argument("--ambient", type=float, default=0.22, help="World ambient strength (default: 0.22)")
    p.add_argument("--sharpen", type=float, default=0.45, help="Unsharp mask amount (default: 0.45)")
    p.add_argument("--preview", action="store_true", help="Also save a 4x PNG preview sheet")

    args = p.parse_args(argv)
    render_vehicle_icons(
        vehicle_dir=args.vehicle,
        mod_dir=args.mod,
        out_dir=args.out,
        name=args.name,
        game_dir=args.game,
        mdl_path=args.mdl,
        common_tex_dir=args.common_tex,
        yaw_deg=args.yaw,
        pitch_deg=args.pitch,
        zoom=args.zoom,
        shift_x=args.shift_x,
        shift_y=args.shift_y,
        samples=args.samples,
        sun_energy=args.sun,
        sun_yaw_deg=args.sun_yaw,
        sun_pitch_deg=args.sun_pitch,
        ambient=args.ambient,
        sharpen=args.sharpen,
        save_preview=args.preview,
    )


if __name__ == "__main__":
    main()
