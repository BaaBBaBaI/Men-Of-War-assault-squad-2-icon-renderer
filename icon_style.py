"""
MoW:AS2 104x52 4-state unit icon compositor (_00, _01, _02, _03) and 32-bit RGBA TGA writer.
Reproduces the exact studio backdrop, horizon line, inner bevel, rounded metallic frame,
and state transforms of Men of War: Assault Squad 2 vehicle icons.
"""

import os
import struct
from typing import Dict, Optional, Tuple
import numpy as np


ICON_W = 104
ICON_H = 52

_HERE = os.path.dirname(os.path.abspath(__file__))
_FRAME_NPZ = os.path.join(_HERE, "frame_overlay.npz")


def _load_frame_overlay() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if os.path.isfile(_FRAME_NPZ):
        d = np.load(_FRAME_NPZ)
        return d["frame_rgba"].astype(np.uint8), d["frame_mask"].astype(bool), d["alpha_mask"].astype(np.uint8)
    # Fallback parametric rounded frame if npz is missing
    rgba = np.zeros((ICON_H, ICON_W, 4), dtype=np.uint8)
    mask = np.ones((ICON_H, ICON_W), dtype=bool)
    mask[3:ICON_H - 3, 3:ICON_W - 3] = False
    rgba[:, :, :3] = 205
    rgba[:, :, 3] = 255
    for y in (0, 1, ICON_H - 2, ICON_H - 1):
        for x in (0, 1, ICON_W - 2, ICON_W - 1):
            rgba[y, x, 3] = 0
    return rgba, mask, rgba[:, :, 3].copy()


def build_studio_backgrounds() -> Tuple[np.ndarray, np.ndarray]:
    """
    Build clean 104x52 MoW:AS2 studio backdrops:
      - bg_00: warm parchment/beige studio wall + horizon seam + floor gradient
      - bg_02: terracotta/red selected studio wall + horizon seam + floor gradient
    Returns float32 arrays in [0..255] of shape (52, 104, 3).
    """
    H, W = ICON_H, ICON_W
    bg00 = np.zeros((H, W, 3), dtype=np.float32)

    # Subtle left-to-right lighting gradient (~0.90 at left x=3 to 1.00 at right x=100)
    xs = np.linspace(0.905, 1.0, W, dtype=np.float32)[None, :, None]

    # Wall: y = 0..13
    wall_top = np.array([181.0, 174.0, 150.0], dtype=np.float32)
    wall_bot = np.array([184.0, 177.0, 152.0], dtype=np.float32)
    for y in range(14):
        t = y / 13.0
        bg00[y, :, :] = wall_top * (1.0 - t) + wall_bot * t

    # Horizon seam: y = 14 (dark crease), y = 15 (subtle highlight)
    bg00[14, :, :] = np.array([165.0, 158.0, 137.0], dtype=np.float32)
    bg00[15, :, :] = np.array([187.0, 180.0, 160.0], dtype=np.float32)

    # Floor: y = 16..51 (gentle gradient brightening toward foreground)
    floor_top = np.array([179.0, 171.0, 148.0], dtype=np.float32)
    floor_bot = np.array([209.0, 201.0, 173.0], dtype=np.float32)
    for y in range(16, H):
        t = min(1.0, (y - 16) / 31.0)
        bg00[y, :, :] = floor_top * (1.0 - t) + floor_bot * t

    bg00 *= xs

    # Selected (_02) red/terracotta studio backdrop
    bg02 = bg00.copy()
    bg02[:14, :, :] *= np.array([0.998, 0.785, 0.781], dtype=np.float32)
    bg02[14, :, :] *= np.array([0.994, 0.468, 0.453], dtype=np.float32)
    bg02[15, :, :] *= np.array([0.995, 0.556, 0.556], dtype=np.float32)
    bg02[16:, :, :] *= np.array([0.991, 0.467, 0.452], dtype=np.float32)

    return bg00, bg02


def build_inner_bevel() -> np.ndarray:
    """Build inner frame drop-shadow multiplier of shape (52, 104, 1)."""
    H, W = ICON_H, ICON_W
    m = np.ones((H, W), dtype=np.float32)

    # Top inner shadow (stronger, light comes from top-left)
    m[3, :] *= 0.54
    m[4, :] *= 0.80
    m[5, :] *= 0.94

    # Bottom inner shadow
    m[H - 4, :] *= 0.80
    m[H - 5, :] *= 0.95

    # Left inner shadow
    m[:, 3] *= 0.56
    m[:, 4] *= 0.82
    m[:, 5] *= 0.94

    # Right inner shadow
    m[:, W - 4] *= 0.80
    m[:, W - 5] *= 0.95

    return m[:, :, None]


def downsample_box(img: np.ndarray, target_h: int = ICON_H, target_w: int = ICON_W) -> np.ndarray:
    """High-quality area downsampling for integer multiples (e.g., 4x supersampled render)."""
    H, W = img.shape[:2]
    if H == target_h and W == target_w:
        return img.astype(np.float32)
    sy = H // target_h
    sx = W // target_w
    if sy * target_h == H and sx * target_w == W:
        if img.ndim == 3:
            return img.reshape(target_h, sy, target_w, sx, img.shape[2]).mean(axis=(1, 3)).astype(np.float32)
        return img.reshape(target_h, sy, target_w, sx).mean(axis=(1, 3)).astype(np.float32)
    # Bilinear fallback
    ys = np.linspace(0, H - 1, target_h)
    xs = np.linspace(0, W - 1, target_w)
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    return img[np.ix_(y0, x0)].astype(np.float32)


def _unsharp_mask_rgb(rgb: np.ndarray, amount: float = 0.45) -> np.ndarray:
    """Subtle 3x3 unsharp mask to preserve crisp MoW:AS2 pixel detail at 104x52."""
    padded = np.pad(rgb, ((1, 1), (1, 1), (0, 0)), mode="edge")
    blur = (
        padded[:-2, :-2] + 2.0 * padded[:-2, 1:-1] + padded[:-2, 2:]
        + 2.0 * padded[1:-1, :-2] + 4.0 * padded[1:-1, 1:-1] + 2.0 * padded[1:-1, 2:]
        + padded[2:, :-2] + 2.0 * padded[2:, 1:-1] + padded[2:, 2:]
    ) / 16.0
    return np.clip(rgb + amount * (rgb - blur), 0.0, 255.0)


def _make_hover_state(rgb00: np.ndarray) -> np.ndarray:
    """Transform _00 RGB [0..255] into _01 (hover/highlighted) matching original game icons."""
    norm = np.clip(rgb00 / 255.0, 0.0, 1.0)
    # Bright lift matching MoW:AS2 hover state (bright cream backdrop + lifted vehicle highlights)
    lifted = np.power(norm, 0.64) * 255.0
    lifted = lifted * 0.98 + 14.0
    return np.clip(lifted, 0.0, 255.0)


def _make_disabled_state(rgb00: np.ndarray) -> np.ndarray:
    """Transform _00 RGB [0..255] into _03 (disabled grayscale) matching original game icons."""
    luma = 0.299 * rgb00[:, :, 0] + 0.587 * rgb00[:, :, 1] + 0.114 * rgb00[:, :, 2]
    gray = np.clip(luma * 0.78 + 23.0, 0.0, 255.0)
    return np.stack([gray, gray, gray], axis=-1)


def compose_icon_set(
    veh_rgba_hi: np.ndarray,
    shadow_alpha_hi: Optional[np.ndarray] = None,
    sharpen: float = 0.42,
) -> Dict[str, np.ndarray]:
    """
    Composite a supersampled vehicle render (straight or premultiplied RGBA in [0..255] float/uint8)
    and optional shadow-catcher alpha [0..1] onto the 4 MoW:AS2 icon states.

    Note: veh_rgba_hi is expected to be straight-alpha RGBA in [0..255].
    Returns dict {"00": rgba, "01": rgba, "02": rgba, "03": rgba} of (52, 104, 4) uint8.
    """
    veh_hi = veh_rgba_hi.astype(np.float32)
    # Downsample in premultiplied space to avoid dark halo fringes
    a_hi = np.clip(veh_hi[:, :, 3:4] / 255.0, 0.0, 1.0)
    premul_hi = np.concatenate([veh_hi[:, :, :3] * a_hi, a_hi], axis=-1)
    premul_lo = downsample_box(premul_hi, ICON_H, ICON_W)

    a_lo = np.clip(premul_lo[:, :, 3:4], 0.0, 1.0)
    rgb_lo = np.where(a_lo > 1e-4, premul_lo[:, :, :3] / np.maximum(a_lo, 1e-4), 0.0)

    # Subtle MoW:AS2 icon contrast punch (deepens shadows, pops specular highlights)
    rgb_lo = np.clip((rgb_lo - 78.0) * 1.28 + 72.0, 0.0, 255.0)

    if sharpen > 0.0:
        rgb_sharp = _unsharp_mask_rgb(rgb_lo, amount=sharpen)
        # Apply sharpening only on opaque/semi-opaque vehicle pixels
        rgb_lo = rgb_lo * (1.0 - a_lo) + rgb_sharp * a_lo

    if shadow_alpha_hi is not None:
        sh_lo = downsample_box(shadow_alpha_hi.astype(np.float32), ICON_H, ICON_W)
        if sh_lo.ndim == 2:
            sh_lo = sh_lo[:, :, None]
        sh_lo = np.clip(sh_lo, 0.0, 0.85)
    else:
        sh_lo = np.zeros((ICON_H, ICON_W, 1), dtype=np.float32)

    bg00, bg02 = build_studio_backgrounds()
    bevel = build_inner_bevel()
    frame_rgba, frame_mask, alpha_mask = _load_frame_overlay()

    # Ground shadow darkens the backdrop before compositing the vehicle
    bg00_sh = bg00 * (1.0 - sh_lo * 0.72)
    bg02_sh = bg02 * (1.0 - sh_lo * 0.72)

    # Composite vehicle over backdrops
    comp00 = (rgb_lo * a_lo + bg00_sh * (1.0 - a_lo)) * bevel
    comp02 = (rgb_lo * a_lo + bg02_sh * (1.0 - a_lo)) * bevel

    # Derive _01 (hover) and _03 (disabled) from _00
    comp01 = _make_hover_state(comp00)
    comp03 = _make_disabled_state(comp00)

    out: Dict[str, np.ndarray] = {}
    for key, comp in (("00", comp00), ("01", comp01), ("02", comp02), ("03", comp03)):
        rgba = np.zeros((ICON_H, ICON_W, 4), dtype=np.uint8)
        rgba[:, :, :3] = np.clip(np.round(comp), 0, 255).astype(np.uint8)
        rgba[:, :, 3] = alpha_mask
        # Apply exact rounded metallic frame on top
        rgba[frame_mask] = frame_rgba[frame_mask]
        out[key] = rgba

    return out


def save_tga_rgba(path: str, rgba: np.ndarray):
    """
    Save an (H, W, 4) uint8 RGBA image as uncompressed 32-bit BGRA TGA
    with bottom-left origin (desc=0x08), matching MoW:AS2 original .tga files.
    """
    H, W, C = rgba.shape
    assert C == 4
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    header = struct.pack(
        "<BBBHHBHHHHBB",
        0,      # id length
        0,      # colormap type
        2,      # image type: uncompressed true-color
        0, 0, 0,# colormap spec
        0, 0,   # x_origin, y_origin
        W, H,   # width, height
        32,     # bits per pixel
        0x08,   # descriptor: 8-bit alpha, bottom-left origin
    )
    # Flip vertically (bottom-left origin) and swap R<->B for BGRA
    flipped = rgba[::-1, :, :]
    bgra = flipped[:, :, [2, 1, 0, 3]].tobytes()
    with open(path, "wb") as f:
        f.write(header)
        f.write(bgra)
