"""
MoW:AS2 Vehicle Icon Renderer — Desktop Microapp & CLI.
Renders 104x52 4-state (_00, _01, _02, _03) TGA unit icons directly from
Men of War: Assault Squad 2 mod vehicles using Blender 3.5 in headless mode.
"""

import os
import sys
import glob
import shutil
import subprocess
import threading
import argparse
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

_FROZEN = getattr(sys, "frozen", False)
_BUNDLE_DIR = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
_APP_DIR = os.path.dirname(os.path.abspath(sys.executable)) if _FROZEN else os.path.dirname(os.path.abspath(__file__))
_BLENDER_SCRIPT = os.path.join(_BUNDLE_DIR, "blender_render.py")

DEFAULT_MOD_DIR = r"D:\SteamLibrary\steamapps\common\Men of War Assault Squad 2\mods\A world At War upgraded dev"
DEFAULT_OUT_DIR = os.path.join(_APP_DIR, "renders")


def find_blender_exe() -> str:
    candidates = [
        shutil.which("blender"),
        r"C:\Program Files\Blender Foundation\Blender 3.5\blender.exe",
    ]
    # Also scan Blender Foundation folder for any installed version
    bf_dir = r"C:\Program Files\Blender Foundation"
    if os.path.isdir(bf_dir):
        for entry in sorted(os.listdir(bf_dir), reverse=True):
            cand = os.path.join(bf_dir, entry, "blender.exe")
            if os.path.isfile(cand):
                candidates.append(cand)
    for c in candidates:
        if c and os.path.isfile(c):
            return c
    return ""


def scan_vehicles(mod_dir: str = "", vehicles_dir: str = ""):
    """Return list of (display_label, vehicle_dir) from vehicles_dir or <mod_dir>/resource/entity/-vehicle."""
    veh_root = ""
    if vehicles_dir and os.path.isdir(vehicles_dir):
        veh_root = vehicles_dir
    elif mod_dir and os.path.isdir(mod_dir):
        veh_root = os.path.join(mod_dir, "resource", "entity", "-vehicle")

    results = []
    if not veh_root or not os.path.isdir(veh_root):
        return results
    for root, dirs, files in os.walk(veh_root):
        # Skip destroyed wreck subfolders 'x'
        dirs[:] = [d for d in sorted(dirs) if d.lower() != "x"]
        has_mdl = any(f.lower().endswith(".mdl") for f in files)
        if has_mdl:
            if os.path.abspath(root) == os.path.abspath(veh_root):
                rel = os.path.basename(os.path.normpath(root))
            else:
                rel = os.path.relpath(root, veh_root).replace("\\", "/")
            results.append((rel, root))
    results.sort(key=lambda x: x[0].lower())
    return results


def run_blender_render(
    blender_exe: str,
    vehicle_dir: str,
    mod_dir: str = "",
    common_tex_dir: str = "",
    out_dir: str = ".",
    name: str = "icon",
    yaw: float = 48.0,
    pitch: float = 20.0,
    zoom: float = 1.0,
    shift_x: float = 0.04,
    shift_y: float = 0.0,
    samples: int = 32,
    sun: float = 5.8,
    sun_yaw: float = 53.0,
    sun_pitch: float = 67.0,
    ambient: float = 0.22,
    preview: bool = True,
):
    cmd = [
        blender_exe,
        "-b",
        "--python", _BLENDER_SCRIPT,
        "--",
        "--vehicle", vehicle_dir,
        "--out", out_dir,
        "--name", name,
        "--yaw", str(yaw),
        "--pitch", str(pitch),
        "--zoom", str(zoom),
        "--shift-x", str(shift_x),
        "--shift-y", str(shift_y),
        "--samples", str(samples),
        "--sun", str(sun),
        "--sun-yaw", str(sun_yaw),
        "--sun-pitch", str(sun_pitch),
        "--ambient", str(ambient),
    ]
    if mod_dir and os.path.isdir(mod_dir):
        cmd.extend(["--mod", mod_dir])
    if common_tex_dir and os.path.isdir(common_tex_dir):
        cmd.extend(["--common-tex", common_tex_dir])
    if preview:
        cmd.append("--preview")

    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        creationflags=creationflags,
    )
    # Check whether the 4 TGAs were produced (Blender on Windows may exit non-zero on TBBmalloc warning)
    tgas = [os.path.join(out_dir, f"{name}_{s}.tga") for s in ("00", "01", "02", "03")]
    ok = all(os.path.isfile(p) for p in tgas)
    return ok, proc.stdout, tgas


class IconRendererApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MoW:AS2 Vehicle Icon Renderer")
        self.geometry("980x640")
        self.minsize(860, 560)

        self.blender_exe = tk.StringVar(value=find_blender_exe())
        self.mod_dir = tk.StringVar(value=DEFAULT_MOD_DIR if os.path.isdir(DEFAULT_MOD_DIR) else "")
        self.vehicles_dir = tk.StringVar(value="")
        self.common_tex_dir = tk.StringVar(value="")
        self.out_dir = tk.StringVar(value=DEFAULT_OUT_DIR)
        self.search_var = tk.StringVar()
        self.icon_name = tk.StringVar()

        self.yaw_var = tk.DoubleVar(value=48.0)
        self.pitch_var = tk.DoubleVar(value=20.0)
        self.zoom_var = tk.DoubleVar(value=1.0)
        self.shift_x_var = tk.DoubleVar(value=0.04)
        self.shift_y_var = tk.DoubleVar(value=0.0)
        self.sun_var = tk.DoubleVar(value=5.8)
        self.sun_yaw_var = tk.DoubleVar(value=53.0)
        self.sun_pitch_var = tk.DoubleVar(value=67.0)
        self.ambient_var = tk.DoubleVar(value=0.22)
        self.samples_var = tk.IntVar(value=32)

        self.all_vehicles = []
        self.filtered_vehicles = []
        self._preview_img = None
        self._rendering = False

        self._build_ui()
        self._refresh_vehicles()

    def _build_ui(self):
        # Top paths frame
        top = ttk.LabelFrame(self, text="Paths", padding=8)
        top.pack(fill=tk.X, padx=10, pady=(10, 5))

        for row, (label, var, is_file, refresh, clearable) in enumerate((
            ("Mod Folder:", self.mod_dir, False, True, True),
            ("Vehicles Folder:", self.vehicles_dir, False, True, True),
            ("Textures Folder:", self.common_tex_dir, False, False, True),
            ("Output Folder:", self.out_dir, False, False, False),
            ("Blender.exe:", self.blender_exe, True, False, False),
        )):
            ttk.Label(top, text=label, width=15).grid(row=row, column=0, sticky=tk.W, pady=2)
            ttk.Entry(top, textvariable=var).grid(row=row, column=1, sticky=tk.EW, padx=5, pady=2)
            bframe = ttk.Frame(top)
            bframe.grid(row=row, column=2, pady=2)
            ttk.Button(
                bframe,
                text="Browse...",
                command=lambda v=var, f=is_file, r=refresh: self._browse(v, f, refresh=r),
            ).pack(side=tk.LEFT)
            if clearable:
                ttk.Button(
                    bframe,
                    text="Clear",
                    width=6,
                    command=lambda v=var, r=refresh: self._clear_path(v, refresh=r),
                ).pack(side=tk.LEFT, padx=(4, 0))
        top.columnconfigure(1, weight=1)

        # Main split pane
        mid = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        mid.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Left: Vehicle browser
        left = ttk.LabelFrame(mid, text="Mod Vehicles (resource/entity/-vehicle)", padding=8)
        mid.add(left, weight=2)

        srow = ttk.Frame(left)
        srow.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(srow, text="Filter:").pack(side=tk.LEFT)
        sentry = ttk.Entry(srow, textvariable=self.search_var)
        sentry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
        self.search_var.trace_add("write", lambda *_: self._apply_filter())
        ttk.Button(srow, text="Reload", command=self._refresh_vehicles).pack(side=tk.RIGHT)

        lframe = ttk.Frame(left)
        lframe.pack(fill=tk.BOTH, expand=True)
        self.listbox = tk.Listbox(lframe, selectmode=tk.EXTENDED, activestyle="none")
        scroll = ttk.Scrollbar(lframe, orient=tk.VERTICAL, command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=scroll.set)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox.bind("<<ListboxSelect>>", self._on_select_vehicle)

        self.count_label = ttk.Label(left, text="0 vehicles")
        self.count_label.pack(anchor=tk.W, pady=(4, 0))

        # Right: Controls & Preview
        right = ttk.Frame(mid)
        mid.add(right, weight=3)

        ctrl = ttk.LabelFrame(right, text="Render Settings", padding=8)
        ctrl.pack(fill=tk.X, pady=(0, 6))

        nrow = ttk.Frame(ctrl)
        nrow.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(nrow, text="Icon Base Name:", width=15).pack(side=tk.LEFT)
        ttk.Entry(nrow, textvariable=self.icon_name).pack(side=tk.LEFT, fill=tk.X, expand=True)

        sliders = [
            ("Camera Yaw (°):", self.yaw_var, -180, 180, 1.0),
            ("Camera Pitch (°):", self.pitch_var, 5, 65, 1.0),
            ("Zoom:", self.zoom_var, 0.5, 1.6, 0.02),
            ("Cycles Samples:", self.samples_var, 8, 96, 4),
            ("Shift Right:", self.shift_x_var, -0.25, 0.25, 0.01),
            ("Shift Up:", self.shift_y_var, -0.25, 0.25, 0.01),
            ("Sun Yaw (°):", self.sun_yaw_var, -180, 180, 5.0),
            ("Sun Pitch (°):", self.sun_pitch_var, 10, 90, 2.0),
            ("Sun Energy:", self.sun_var, 1.0, 12.0, 0.2),
            ("Ambient:", self.ambient_var, 0.05, 1.5, 0.05),
        ]
        sgrid = ttk.Frame(ctrl)
        sgrid.pack(fill=tk.X)
        for i, (lbl, var, mn, mx, step) in enumerate(sliders):
            r = i // 2
            c = (i % 2) * 3
            ttk.Label(sgrid, text=lbl, width=15).grid(row=r, column=c, sticky=tk.W, pady=2)
            sb = ttk.Spinbox(sgrid, from_=mn, to=mx, increment=step, textvariable=var, width=8)
            sb.grid(row=r, column=c + 1, sticky=tk.W, padx=(2, 12), pady=2)

        brow = ttk.Frame(ctrl)
        brow.pack(fill=tk.X, pady=(8, 0))
        self.render_btn = ttk.Button(brow, text="Render Selected", command=self._start_render)
        self.render_btn.pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(brow, text="Reset Defaults", command=self._reset_defaults).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(brow, text="Open Output Folder", command=self._open_out_dir).pack(side=tk.RIGHT)

        # Preview panel
        prev = ttk.LabelFrame(right, text="4-State Icon Preview (_00 / _01 / _02 / _03)", padding=8)
        prev.pack(fill=tk.BOTH, expand=True)

        self.preview_canvas = tk.Canvas(prev, bg="#282828", highlightthickness=0)
        self.preview_canvas.pack(fill=tk.BOTH, expand=True)

        # Bottom status bar
        self.status_var = tk.StringVar(value="Ready. Select one or more vehicles and click Render Selected.")
        ttk.Label(self, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, padding=(6, 3)).pack(
            fill=tk.X, side=tk.BOTTOM
        )

    def _browse(self, var, is_file=False, refresh=False):
        if is_file:
            p = filedialog.askopenfilename(filetypes=[("Blender", "blender.exe"), ("All files", "*.*")])
        else:
            p = filedialog.askdirectory(initialdir=var.get() or None)
        if p:
            var.set(os.path.normpath(p))
            if refresh:
                self._refresh_vehicles()

    def _clear_path(self, var, refresh=False):
        var.set("")
        if refresh:
            self._refresh_vehicles()

    def _refresh_vehicles(self):
        self.all_vehicles = scan_vehicles(
            mod_dir=self.mod_dir.get().strip(),
            vehicles_dir=self.vehicles_dir.get().strip(),
        )
        self._apply_filter()

    def _apply_filter(self):
        q = self.search_var.get().strip().lower()
        self.filtered_vehicles = [v for v in self.all_vehicles if not q or q in v[0].lower()]
        self.listbox.delete(0, tk.END)
        for label, _ in self.filtered_vehicles:
            self.listbox.insert(tk.END, label)
        self.count_label.config(text=f"{len(self.filtered_vehicles)} / {len(self.all_vehicles)} vehicles")

    def _on_select_vehicle(self, _event=None):
        sel = self.listbox.curselection()
        if len(sel) == 1:
            _, vdir = self.filtered_vehicles[sel[0]]
            self.icon_name.set(os.path.basename(vdir))
        elif len(sel) > 1:
            self.icon_name.set("(auto: folder name)")

    def _reset_defaults(self):
        self.yaw_var.set(48.0)
        self.pitch_var.set(20.0)
        self.zoom_var.set(1.0)
        self.shift_x_var.set(0.04)
        self.shift_y_var.set(0.0)
        self.sun_var.set(5.8)
        self.sun_yaw_var.set(53.0)
        self.sun_pitch_var.set(67.0)
        self.ambient_var.set(0.22)
        self.samples_var.set(32)

    def _open_out_dir(self):
        d = self.out_dir.get()
        os.makedirs(d, exist_ok=True)
        if os.name == "nt":
            os.startfile(d)

    def _show_preview_png(self, png_path: str):
        if not os.path.isfile(png_path):
            return
        try:
            # Use tk.PhotoImage (supports PNG natively in Tk 8.6+) and subsample 2x (416 -> 208) so all 4 fit nicely
            raw = tk.PhotoImage(file=png_path)
            img = raw.subsample(2, 2)
            self._preview_img = img
            self.preview_canvas.delete("all")
            self.preview_canvas.update_idletasks()
            cw = max(self.preview_canvas.winfo_width(), 240)
            ch = max(self.preview_canvas.winfo_height(), 240)
            self.preview_canvas.create_image(cw // 2, ch // 2, image=img, anchor=tk.CENTER)
        except Exception as e:
            self.status_var.set(f"Preview load warning: {e}")

    def _start_render(self):
        if self._rendering:
            return
        blender = self.blender_exe.get().strip()
        if not blender or not os.path.isfile(blender):
            messagebox.showerror("Blender Not Found", "Please select a valid blender.exe path.")
            return
        sel = self.listbox.curselection()
        if not sel:
            messagebox.showinfo("No Selection", "Select one or more vehicles from the list first.")
            return

        targets = [self.filtered_vehicles[i] for i in sel]
        self._rendering = True
        self.render_btn.config(state=tk.DISABLED)

        def worker():
            last_preview = None
            ok_count = 0
            for idx, (label, vdir) in enumerate(targets, 1):
                base_name = (
                    self.icon_name.get().strip()
                    if len(targets) == 1 and self.icon_name.get().strip() and not self.icon_name.get().startswith("(")
                    else os.path.basename(vdir)
                )
                sub_out = os.path.join(self.out_dir.get(), base_name)
                self.after(0, lambda m=f"Rendering [{idx}/{len(targets)}] {label}...": self.status_var.set(m))
                ok, logs, _ = run_blender_render(
                    blender_exe=blender,
                    vehicle_dir=vdir,
                    mod_dir=self.mod_dir.get().strip(),
                    common_tex_dir=self.common_tex_dir.get().strip(),
                    out_dir=sub_out,
                    name=base_name,
                    yaw=self.yaw_var.get(),
                    pitch=self.pitch_var.get(),
                    zoom=self.zoom_var.get(),
                    shift_x=self.shift_x_var.get(),
                    shift_y=self.shift_y_var.get(),
                    samples=self.samples_var.get(),
                    sun=self.sun_var.get(),
                    sun_yaw=self.sun_yaw_var.get(),
                    sun_pitch=self.sun_pitch_var.get(),
                    ambient=self.ambient_var.get(),
                    preview=True,
                )
                if ok:
                    ok_count += 1
                    p = os.path.join(sub_out, f"{base_name}_preview.png")
                    if os.path.isfile(p):
                        last_preview = p
                        self.after(0, lambda pp=p: self._show_preview_png(pp))
                else:
                    print(f"[ERROR] Failed rendering {label}:\n{logs[-1500:]}")

            def done():
                self._rendering = False
                self.render_btn.config(state=tk.NORMAL)
                self.status_var.set(f"Done! Rendered {ok_count}/{len(targets)} vehicle(s) to {self.out_dir.get()}")

            self.after(0, done)

        threading.Thread(target=worker, daemon=True).start()


def main():
    p = argparse.ArgumentParser(description="MoW:AS2 Vehicle Icon Renderer (GUI or CLI)")
    p.add_argument("--vehicle", help="CLI mode: path to vehicle folder")
    p.add_argument("--mod", default="", help="Optional path to mod folder")
    p.add_argument("--common-tex", default="", help="Optional path to resource/texture/common folder")
    p.add_argument("--out", default=DEFAULT_OUT_DIR, help="Output folder")
    p.add_argument("--name", default=None, help="Output icon base name")
    p.add_argument("--yaw", type=float, default=48.0)
    p.add_argument("--pitch", type=float, default=20.0)
    p.add_argument("--zoom", type=float, default=1.0)
    p.add_argument("--shift-x", type=float, default=0.04)
    p.add_argument("--shift-y", type=float, default=0.0)
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--sun", type=float, default=5.8)
    p.add_argument("--sun-yaw", type=float, default=53.0)
    p.add_argument("--sun-pitch", type=float, default=67.0)
    p.add_argument("--ambient", type=float, default=0.22)
    p.add_argument("--blender", default=None, help="Path to blender.exe")
    args = p.parse_args()

    if args.vehicle:
        blender = args.blender or find_blender_exe()
        if not blender:
            print("ERROR: Could not find blender.exe. Pass --blender <path>.")
            sys.exit(1)
        name = args.name or os.path.basename(os.path.normpath(args.vehicle))
        ok, logs, tgas = run_blender_render(
            blender_exe=blender,
            vehicle_dir=args.vehicle,
            mod_dir=args.mod,
            common_tex_dir=args.common_tex,
            out_dir=args.out,
            name=name,
            yaw=args.yaw,
            pitch=args.pitch,
            zoom=args.zoom,
            shift_x=args.shift_x,
            shift_y=args.shift_y,
            samples=args.samples,
            sun=args.sun,
            sun_yaw=args.sun_yaw,
            sun_pitch=args.sun_pitch,
            ambient=args.ambient,
            preview=True,
        )
        if not ok:
            print(logs)
            sys.exit(1)
        for t in tgas:
            print("Saved:", t)
    else:
        app = IconRendererApp()
        app.mainloop()


if __name__ == "__main__":
    main()
