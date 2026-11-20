# MoW:AS2 Vehicle Icon Renderer

Desktop GUI and CLI tool for rendering `104x52` 32-bit RGBA `.tga` unit icons (`_00` normal, `_01` hover, `_02` selected, `_03` disabled) from Men of War: Assault Squad 2 GEM2 vehicle models (`.def`, `.mdl`, `.ply`, `.mtl`) in the original game style using headless Blender Cycles.

## Requirements

- Windows
- Blender 3.5+ (`blender.exe`)
- Python 3.9+ with `numpy` (only needed when running from source)

## How to Run

### Using the EXE

1. Download or build `MoWAS2_Icon_Renderer.exe`.
2. Double-click `MoWAS2_Icon_Renderer.exe`.
3. Set **Mod Folder** (or individually set **Vehicles Folder** and **Textures Folder**), verify **Blender.exe**, select one or more vehicles, and click **Render Selected**.

### From Source

GUI:
```bat
run_app.bat
```
or:
```powershell
python app.py
```

CLI:
```powershell
python app.py --vehicle "<path_to_vehicle_folder>" --mod "<path_to_mod>" --out "<output_dir>"
```
Or without a full mod folder:
```powershell
python app.py --vehicle "<path_to_vehicle_folder>" --common-tex "<path_to_texture_common>" --out "<output_dir>"
```

## Building the EXE

```powershell
pip install pyinstaller numpy
python build_exe.py
```

The output executable is written to `dist/MoWAS2_Icon_Renderer.exe`.
