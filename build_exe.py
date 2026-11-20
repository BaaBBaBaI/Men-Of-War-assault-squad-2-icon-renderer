"""
Build a standalone Windows executable (MoWAS2_Icon_Renderer.exe) using PyInstaller.
Note: Blender 3.5+ must still be installed on the target machine for headless Cycles rendering.
"""

import os
import sys
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    os.chdir(HERE)
    sep = ";" if os.name == "nt" else ":"
    datas = [
        ("blender_render.py", "."),
        ("gem2_loader.py", "."),
        ("icon_style.py", "."),
        ("frame_overlay.npz", "."),
    ]
    cmd = [
        sys.executable,
        "-m", "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onefile",
        "--windowed",
        "--name", "MoWAS2_Icon_Renderer",
    ]
    for src, dst in datas:
        cmd.extend(["--add-data", f"{src}{sep}{dst}"])
    cmd.append("app.py")

    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)
    exe_path = os.path.join(HERE, "dist", "MoWAS2_Icon_Renderer.exe")
    print(f"[ok] Built: {exe_path}")


if __name__ == "__main__":
    main()
