"""
GEM2 (Men of War: Assault Squad 2) vehicle loader.
Parses .def, .mdl, .ply (EPLY), .mtl, and resolves textures from mod folders and base game .pak (ZIP) archives.
"""

import os
import re
import struct
import zipfile
import tempfile
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


# ---------------------------------------------------------------------------
# S-expression tokenizer & parser for .def / .mdl / .mtl
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r';[^\r\n]*|"([^"]*)"|(\{)|(\})|([^\s\{\}";]+)')


def parse_gem2_text(text: str):
    """Parse GEM2 curly-brace format into nested lists of strings/lists."""
    root = []
    stack = [root]
    for m in _TOKEN_RE.finditer(text):
        quoted, lbrace, rbrace, bare = m.groups()
        if lbrace is not None:
            node = []
            stack[-1].append(node)
            stack.append(node)
        elif rbrace is not None:
            if len(stack) > 1:
                stack.pop()
        elif quoted is not None:
            stack[-1].append(quoted)
        elif bare is not None:
            stack[-1].append(bare)
    return root


def _read_text(path: str) -> str:
    with open(path, "rb") as f:
        raw = f.read()
    for enc in ("utf-8-sig", "cp1251", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Gem2Material:
    name: str
    shader: str = "simple"
    diffuse: Optional[str] = None
    bump: Optional[str] = None
    specular: Optional[str] = None
    color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    blend: str = "none"  # none, test, blend, add
    emissive: bool = False


@dataclass
class Gem2SubMesh:
    first_face: int
    num_faces: int
    mtl_name: str
    flags: int
    color: Optional[Tuple[int, int, int, int]] = None


@dataclass
class Gem2Ply:
    path: str
    bounds: Tuple[float, float, float, float, float, float]
    vertices: np.ndarray        # (N, 3) float32
    normals: np.ndarray         # (N, 3) float32
    uvs: np.ndarray             # (N, 2) float32
    indices: np.ndarray         # (M, 3) uint32
    submeshes: List[Gem2SubMesh]
    skin_bones: List[str] = field(default_factory=list)
    mirrored: bool = False


@dataclass
class Gem2MeshInstance:
    bone_name: str
    ply_path: str
    world_matrix: np.ndarray    # (4, 4) float64, row-vector convention: v_world = v_local @ M
    ply: Gem2Ply


# ---------------------------------------------------------------------------
# Texture & asset resolver (supports mod dir + base game .pak archives)
# ---------------------------------------------------------------------------

TEXTURE_EXTS = (".dds", ".tga", ".bmp", ".png", ".jpg")


class TextureResolver:
    def __init__(
        self,
        mod_dir: Optional[str] = None,
        game_dir: Optional[str] = None,
        cache_dir: Optional[str] = None,
        common_tex_dir: Optional[str] = None,
    ):
        self.mod_dir = os.path.abspath(mod_dir) if mod_dir and os.path.isdir(mod_dir) else None
        self.common_tex_dir = (
            os.path.abspath(common_tex_dir) if common_tex_dir and os.path.isdir(common_tex_dir) else None
        )
        if game_dir is None and self.mod_dir:
            # Auto-detect game_dir from ...\mods\<mod_name>
            parent = os.path.dirname(self.mod_dir)
            if os.path.basename(parent).lower() == "mods":
                game_dir = os.path.dirname(parent)
        self.game_dir = os.path.abspath(game_dir) if game_dir and os.path.isdir(game_dir) else None
        self.cache_dir = cache_dir or os.path.join(tempfile.gettempdir(), "mowas2_icon_tex_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        self._pak_index: Optional[Dict[str, Tuple[str, str]]] = None  # norm_lower_path -> (pak_path, member_name)
        self._resolved_cache: Dict[Tuple[str, str], Optional[str]] = {}

    def _build_pak_index(self):
        if self._pak_index is not None:
            return
        self._pak_index = {}
        if not self.game_dir:
            return
        res_dir = os.path.join(self.game_dir, "resource")
        if not os.path.isdir(res_dir):
            return
        # Prioritize texture*.pak, then others
        paks = []
        for fn in os.listdir(res_dir):
            if fn.lower().endswith(".pak"):
                paks.append(os.path.join(res_dir, fn))
        paks.sort(key=lambda p: (0 if "texture" in os.path.basename(p).lower() else 1, os.path.basename(p).lower()))

        for pak_path in paks:
            try:
                with zipfile.ZipFile(pak_path, "r") as zf:
                    for name in zf.namelist():
                        low = name.replace("\\", "/").lower()
                        if low not in self._pak_index:
                            self._pak_index[low] = (pak_path, name)
            except Exception:
                continue

    def _try_file(self, base_path: str) -> Optional[str]:
        if os.path.isfile(base_path):
            return base_path
        root, ext = os.path.splitext(base_path)
        if ext.lower() in TEXTURE_EXTS:
            return None
        for e in TEXTURE_EXTS:
            cand = base_path + e
            if os.path.isfile(cand):
                return cand
        return None

    def _try_pak(self, rel_path: str) -> Optional[str]:
        self._build_pak_index()
        assert self._pak_index is not None
        norm = rel_path.replace("\\", "/").lstrip("/").lower()
        candidates = [norm]
        root, ext = os.path.splitext(norm)
        if ext not in TEXTURE_EXTS:
            for e in TEXTURE_EXTS:
                candidates.append(norm + e)
        for cand in candidates:
            hit = self._pak_index.get(cand)
            if hit:
                pak_path, member = hit
                h = hashlib.md5(f"{pak_path}:{member}".encode("utf-8")).hexdigest()[:12]
                out_name = f"{h}_{os.path.basename(member)}"
                out_path = os.path.join(self.cache_dir, out_name)
                if not os.path.isfile(out_path):
                    with zipfile.ZipFile(pak_path, "r") as zf:
                        with zf.open(member) as src, open(out_path, "wb") as dst:
                            dst.write(src.read())
                return out_path
        return None

    def resolve(self, tex_ref: Optional[str], vehicle_dir: str) -> Optional[str]:
        if not tex_ref:
            return None
        key = (tex_ref.lower(), vehicle_dir.lower())
        if key in self._resolved_cache:
            return self._resolved_cache[key]

        ref = tex_ref.replace("\\", "/").strip()
        result = None

        if ref.startswith("$/"):
            rel = ref[2:]
            # 1. Explicit common_tex_dir override
            if self.common_tex_dir:
                result = self._try_file(os.path.join(self.common_tex_dir, rel))
            # 2. Mod resource/texture/common/<rel>
            if not result and self.mod_dir:
                result = self._try_file(os.path.join(self.mod_dir, "resource", "texture", "common", rel))
            # 3. Base game loose resource/texture/common/<rel>
            if not result and self.game_dir:
                result = self._try_file(os.path.join(self.game_dir, "resource", "texture", "common", rel))
            # 4. Base game pak: common/<rel> or texture/common/<rel>
            if not result:
                result = self._try_pak("common/" + rel) or self._try_pak("texture/common/" + rel)
        elif ref.startswith("/"):
            rel = ref.lstrip("/")
            if self.mod_dir:
                result = self._try_file(os.path.join(self.mod_dir, "resource", rel))
            if not result and self.game_dir:
                result = self._try_file(os.path.join(self.game_dir, "resource", rel))
            if not result:
                result = self._try_pak(rel)
        else:
            # 1. Relative to vehicle_dir
            result = self._try_file(os.path.join(vehicle_dir, ref))
            # 2. Explicit common_tex_dir override
            if not result and self.common_tex_dir:
                result = self._try_file(os.path.join(self.common_tex_dir, ref))
            # 3. Mod resource/texture/common/<ref>
            if not result and self.mod_dir:
                result = self._try_file(os.path.join(self.mod_dir, "resource", "texture", "common", ref))
            # 4. Base game pak relative to entity path or common
            if not result and self.mod_dir:
                # Compute vehicle_dir relative to resource/entity/
                try:
                    rel_veh = os.path.relpath(vehicle_dir, os.path.join(self.mod_dir, "resource", "entity"))
                    if not rel_veh.startswith(".."):
                        result = self._try_pak(os.path.join(rel_veh, ref))
                except ValueError:
                    pass
            if not result:
                result = self._try_pak("common/" + ref)

        self._resolved_cache[key] = result
        return result


# ---------------------------------------------------------------------------
# .mtl parser
# ---------------------------------------------------------------------------

def parse_mtl(mtl_path: str) -> Gem2Material:
    name = os.path.basename(mtl_path)
    mat = Gem2Material(name=name)
    if not os.path.isfile(mtl_path):
        return mat
    nodes = parse_gem2_text(_read_text(mtl_path))
    for node in nodes:
        if isinstance(node, list) and node and str(node[0]).lower() == "material":
            if len(node) > 1 and isinstance(node[1], str):
                mat.shader = node[1].lower()
            for item in node[1:]:
                if not isinstance(item, list) or not item:
                    continue
                key = str(item[0]).lower()
                if key == "diffuse" and len(item) > 1:
                    mat.diffuse = str(item[1])
                elif key == "bump" and len(item) > 1:
                    mat.bump = str(item[1])
                elif key == "specular" and len(item) > 1:
                    mat.specular = str(item[1])
                elif key == "blend" and len(item) > 1:
                    mat.blend = str(item[1]).lower()
                elif key == "emissive":
                    mat.emissive = True
                elif key == "color" and len(item) > 1:
                    parts = str(item[1]).split()
                    if len(parts) >= 3:
                        try:
                            r = float(parts[0]) / 255.0
                            g = float(parts[1]) / 255.0
                            b = float(parts[2]) / 255.0
                            a = float(parts[3]) / 255.0 if len(parts) >= 4 else 1.0
                            mat.color = (r, g, b, a)
                        except ValueError:
                            pass
    return mat


# ---------------------------------------------------------------------------
# .ply (EPLY) binary parser
# ---------------------------------------------------------------------------

def parse_ply(ply_path: str) -> Gem2Ply:
    with open(ply_path, "rb") as f:
        data = f.read()

    if data[:4] != b"EPLY":
        raise ValueError(f"Not a GEM2 EPLY file: {ply_path}")

    pos = 4
    bounds = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    skin_bones: List[str] = []
    submeshes: List[Gem2SubMesh] = []
    vertices = np.zeros((0, 3), dtype=np.float32)
    normals = np.zeros((0, 3), dtype=np.float32)
    uvs = np.zeros((0, 2), dtype=np.float32)
    indices = np.zeros((0, 3), dtype=np.uint32)
    mirrored = False

    n = len(data)
    while pos + 4 <= n:
        tag = data[pos:pos + 4]
        pos += 4

        if tag == b"BNDS":
            bounds = struct.unpack_from("<6f", data, pos)
            pos += 24

        elif tag == b"SKIN":
            nbones, = struct.unpack_from("<I", data, pos)
            pos += 4
            for _ in range(nbones):
                slen = data[pos]
                pos += 1
                skin_bones.append(data[pos:pos + slen].decode("ascii", errors="replace"))
                pos += slen

        elif tag == b"MESH":
            u0, first_face, num_faces, flags = struct.unpack_from("<IIII", data, pos)
            pos += 16
            color = None
            if flags & 0x0200:
                color = struct.unpack_from("<BBBB", data, pos)
                pos += 4
            slen = data[pos]
            pos += 1
            mtl_name = data[pos:pos + slen].decode("ascii", errors="replace")
            pos += slen
            # Skinned submesh bone index table
            if (u0 & 0x1000) or skin_bones:
                nbmap = data[pos]
                pos += 1 + nbmap
            submeshes.append(Gem2SubMesh(
                first_face=first_face,
                num_faces=num_faces,
                mtl_name=mtl_name,
                flags=flags,
                color=color,
            ))

        elif tag == b"VERT":
            nverts, stride, vflags = struct.unpack_from("<IHH", data, pos)
            pos += 8
            vbytes = data[pos:pos + nverts * stride]
            pos += nverts * stride

            if nverts > 0:
                raw = np.frombuffer(vbytes, dtype=np.uint8).reshape(nverts, stride)
                # Position is always first 12 bytes (3 float32)
                vertices = raw[:, 0:12].view(np.float32).reshape(nverts, 3).copy()

                # Determine normal + uv offsets based on stride & skinning
                if skin_bones:
                    # Standard skinned strides:
                    # 40 = pos(12) + w(4) + idx(4) + norm(12) + uv(8)
                    # 56 = pos(12) + w(4) + idx(4) + norm(12) + uv(8) + tan(16)
                    # Extra weights (rare): +4 bytes per extra weight
                    if stride in (40, 56):
                        n_off = 20
                    elif stride in (44, 60):
                        n_off = 24
                    elif stride in (48, 64):
                        n_off = 28
                    else:
                        n_off = 20
                else:
                    # Standard rigid strides:
                    # 32 = pos(12) + norm(12) + uv(8)
                    # 48 = pos(12) + norm(12) + uv(8) + tan(16)
                    n_off = 12

                if n_off + 12 <= stride:
                    normals = raw[:, n_off:n_off + 12].view(np.float32).reshape(nverts, 3).copy()
                else:
                    normals = np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (nverts, 1))

                uv_off = n_off + 12
                if uv_off + 8 <= stride:
                    uvs = raw[:, uv_off:uv_off + 8].view(np.float32).reshape(nverts, 2).copy()
                    # Flip V for OpenGL/Blender convention (DirectX V=0 top -> Blender V=0 bottom)
                    uvs[:, 1] = 1.0 - uvs[:, 1]
                else:
                    uvs = np.zeros((nverts, 2), dtype=np.float32)

        elif tag == b"INDX":
            nidx, = struct.unpack_from("<I", data, pos)
            pos += 4
            idx_arr = np.frombuffer(data, dtype=np.uint16, count=nidx, offset=pos).astype(np.uint32)
            pos += nidx * 2
            n_tris = nidx // 3
            indices = idx_arr[:n_tris * 3].reshape(n_tris, 3).copy()

        elif tag == b"MROR":
            mirrored = True

        else:
            # Unknown trailing chunk; stop safely
            break

    return Gem2Ply(
        path=ply_path,
        bounds=bounds,
        vertices=vertices,
        normals=normals,
        uvs=uvs,
        indices=indices,
        submeshes=submeshes,
        skin_bones=skin_bones,
        mirrored=mirrored,
    )


# ---------------------------------------------------------------------------
# .def and .mdl parser
# ---------------------------------------------------------------------------

def find_mdl_for_vehicle(vehicle_dir: str) -> str:
    """Find the primary .mdl file in a vehicle directory using its .def or folder name."""
    folder_name = os.path.basename(os.path.normpath(vehicle_dir)).lower()
    defs = [f for f in os.listdir(vehicle_dir) if f.lower().endswith(".def")]
    # Prefer <folder_name>.def
    defs.sort(key=lambda f: (0 if os.path.splitext(f)[0].lower() == folder_name else 1, len(f), f.lower()))

    for df in defs:
        text = _read_text(os.path.join(vehicle_dir, df))
        nodes = parse_gem2_text(text)
        for node in nodes:
            if isinstance(node, list):
                mdl = _find_extension_mdl(node)
                if mdl:
                    cand = os.path.join(vehicle_dir, mdl)
                    if os.path.isfile(cand):
                        return cand

    mdls = [f for f in os.listdir(vehicle_dir) if f.lower().endswith(".mdl")]
    if not mdls:
        raise FileNotFoundError(f"No .mdl file found in {vehicle_dir}")
    mdls.sort(key=lambda f: (0 if os.path.splitext(f)[0].lower() == folder_name else 1, len(f), f.lower()))
    return os.path.join(vehicle_dir, mdls[0])


def _find_extension_mdl(node) -> Optional[str]:
    if not isinstance(node, list) or not node:
        return None
    if str(node[0]).lower() == "extension" and len(node) > 1 and isinstance(node[1], str):
        if node[1].lower().endswith(".mdl"):
            return node[1]
    for child in node[1:]:
        if isinstance(child, list):
            hit = _find_extension_mdl(child)
            if hit:
                return hit
    return None


def _parse_local_matrix(bone_node: list, is_basis: bool = False) -> np.ndarray:
    """Build 4x4 row-vector transform matrix from {Position}, {Orientation}, or {Matrix34}."""
    m = np.eye(4, dtype=np.float64)
    for item in bone_node:
        if not isinstance(item, list) or not item:
            continue
        key = str(item[0]).lower()
        if key == "position" and len(item) >= 4:
            m[3, 0] = float(item[1])
            m[3, 1] = float(item[2])
            m[3, 2] = float(item[3])
        elif key == "orientation" and len(item) >= 10:
            if is_basis:
                # GEM2 basis has Orientation (1,0,0 / 0,-1,0 / 0,0,1) to convert right-handed 3dsMax
                # to left-handed GEM2. Keeping identity preserves right-handed Blender coords (+X fwd, +Y left, +Z up).
                continue
            vals = [float(x) for x in item[1:10]]
            m[0, :3] = vals[0:3]
            m[1, :3] = vals[3:6]
            m[2, :3] = vals[6:9]
        elif key == "matrix34" and len(item) >= 13:
            vals = [float(x) for x in item[1:13]]
            m[0, :3] = vals[0:3]
            m[1, :3] = vals[3:6]
            m[2, :3] = vals[6:9]
            m[3, :3] = vals[9:12]
    return m


# Bone/mesh names that are interior/hidden/damage/collision proxies we should skip if needed
_SKIP_BONE_PREFIXES = ("fx", "emit", "foresight", "seat", "visor", "crew", "inventory", "ram", "body1", "body2")


def load_vehicle(vehicle_dir: str, mdl_path: Optional[str] = None) -> Tuple[List[Gem2MeshInstance], Dict[str, Gem2Material]]:
    """
    Load all visible VolumeView meshes and their world transforms from a vehicle's .mdl file.
    Returns (instances, materials_dict).
    """
    vehicle_dir = os.path.abspath(vehicle_dir)
    if mdl_path is None:
        mdl_path = find_mdl_for_vehicle(vehicle_dir)

    nodes = parse_gem2_text(_read_text(mdl_path))
    instances: List[Gem2MeshInstance] = []
    ply_cache: Dict[str, Gem2Ply] = {}
    materials: Dict[str, Gem2Material] = {}

    def walk_bone(bone_node: list, parent_world: np.ndarray, depth: int = 0):
        # Bone header can be {bone "name" ...} or {bone revolute "name" ...}
        bone_name = ""
        for tok in bone_node[1:4]:
            if isinstance(tok, str) and tok.lower() not in ("revolute", "prismatic", "fixed"):
                bone_name = tok
                break

        is_basis = (depth == 0 and bone_name.lower() == "basis")
        local_m = _parse_local_matrix(bone_node, is_basis=is_basis)
        world_m = local_m @ parent_world

        # Check visibility / VolumeView
        for item in bone_node:
            if not isinstance(item, list) or not item:
                continue
            key = str(item[0]).lower()
            if key == "volumeview" and len(item) > 1 and isinstance(item[1], str):
                # Check if {Visible 0} or {Ground} or {NoCastShadow}
                hidden = False
                for sub in item[2:]:
                    if isinstance(sub, list) and sub and str(sub[0]).lower() == "visible":
                        if len(sub) > 1 and str(sub[1]) == "0":
                            hidden = True
                if hidden:
                    continue

                ply_name = item[1]
                # Skip enumerator number plates if they use missing/placeholder textures?
                # No — we'll load them and handle missing/alpha textures cleanly in Blender!
                ply_full = os.path.join(vehicle_dir, ply_name)
                if not os.path.isfile(ply_full):
                    # Try case-insensitive match
                    for fn in os.listdir(vehicle_dir):
                        if fn.lower() == ply_name.lower():
                            ply_full = os.path.join(vehicle_dir, fn)
                            break
                if os.path.isfile(ply_full):
                    if ply_full not in ply_cache:
                        try:
                            ply_cache[ply_full] = parse_ply(ply_full)
                        except Exception as e:
                            print(f"[warn] Failed to parse {ply_full}: {e}")
                            continue
                    ply = ply_cache[ply_full]
                    instances.append(Gem2MeshInstance(
                        bone_name=bone_name,
                        ply_path=ply_full,
                        world_matrix=world_m,
                        ply=ply,
                    ))
                    for sm in ply.submeshes:
                        if sm.mtl_name and sm.mtl_name not in materials:
                            mtl_full = os.path.join(vehicle_dir, sm.mtl_name)
                            if not os.path.isfile(mtl_full):
                                for fn in os.listdir(vehicle_dir):
                                    if fn.lower() == sm.mtl_name.lower():
                                        mtl_full = os.path.join(vehicle_dir, fn)
                                        break
                            materials[sm.mtl_name] = parse_mtl(mtl_full)
            elif key == "bone":
                walk_bone(item, world_m, depth + 1)

    for node in nodes:
        if isinstance(node, list) and node and str(node[0]).lower() == "skeleton":
            for child in node[1:]:
                if isinstance(child, list) and child and str(child[0]).lower() == "bone":
                    walk_bone(child, np.eye(4, dtype=np.float64), 0)

    return instances, materials
