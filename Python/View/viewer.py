import tkinter as tk
from tkinter import filedialog
from pathlib import Path
import pyvista as pv
from typing import Optional
import numpy as np
import mmap
import json
import xml.etree.ElementTree as ET

def load_file():
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Select a VTK/VTI file",
        filetypes=[("VTK Files", "*.vti *.vtp"), ("All files", "*.*")]
    )
    return Path(path) if path else None

def get_gprmax_metadata(file_path):
    materials, srcs, rxs, pmls = {},{},{},{}

    with open(file_path, 'rb') as f:
        mm = mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ)
        try:
            xml_pos = mm.find(b'<gprMax')
            mm.seek(xml_pos)
            root = ET.fromstring(mm.read(mm.size()-xml_pos))
            for e in root.findall('Material'): materials[e.get('name')] = int(e.text)
            for e in root.findall('Sources'): srcs[e.get('name')] = int(e.text)
            for e in root.findall('Receivers'): rxs[e.get('name')] = int(e.text)
            for e in root.findall('PML'): pmls[e.get('name')] = int(e.text)
        except Exception as e:
            print(f"Note: Detailed metadata couldn't be extracted ({e})")
    return materials, srcs, rxs, pmls


def info(dataset):
    print("\n=== Dataset Information ===")

    dims = dataset.dimensions
    bounds = dataset.bounds
    spacing = getattr(dataset, 'spacing', None)

    print(f"- Dimensions (nx, ny, nz): {dims}")
    print(f"- Spatial limits (xmin, xmax, ymin, ymax, zmin, zmax): {bounds}")
    if spacing is not None:
        print(f"- Spacing (dx, dy, dz): {spacing}")
    else:
        print("- Spacing: Not available")

    print("\n=== Available fields (Cell Data) ===")
    for name in dataset.cell_data.keys():
        arr = dataset.cell_data[name]
        kind = "Vector" if arr.ndim > 1 else "Scalar"
        print(f"  - {name}: {kind}, shape={arr.shape}, type VTK={arr.GetDataTypeAsString()}")

def plot_gprmax(dataset, materials, srcs: Optional[dict] = None, rxs : Optional[dict]=None, pmls: Optional[dict]=None):
    
    def add_component(name, idx, array_name, color: Optional[str] = None, cmap: Optional[str] = None, opacity = 1.0, is_mat = False):
        if array_name not in dataset.cell_data: return
        thresh = dataset.threshold([idx, idx], scalars=array_name)
        if thresh.n_cells > 0:
            if is_mat:
                actors[name] = plotter.add_mesh(
                    thresh, 
                    name=name,
                    scalars=array_name, 
                    cmap='coolwarm',
                    clim=[0, max_id+1], 
                    opacity=opacity, 
                    show_scalar_bar=False
                )
            else:
                actors[name] = plotter.add_mesh(
                    thresh,
                    name=name, 
                    color=color, 
                    cmap=cmap, 
                    opacity=opacity, 
                    show_scalar_bar=False,
                    render_points_as_spheres=True, 
                    point_size=12
                )
    
    def toggle_vis(flag, name):
        if name in actors: actors[name].visibility = flag

    def apply_slice(normal, origin):
        slc = dataset.slice(normal=normal, origin=origin)
        plotter.add_mesh(slc, name="InternalSlice", cmap='coolwarm',
                         clim=[0, max_id+1], opacity=1.0, show_scalar_bar=False)
        
    win_w, win_h = 1200, 800
    plotter = pv.Plotter(notebook=False, window_size=[win_w, win_h])
    plotter.set_background("black")

    srcs = srcs or {}
    pmls = pmls or {}
    rxs = rxs or {}

    srcs_pmls = dict(srcs)
    srcs_pmls.update(pmls)

    actors = {}
    max_id = int(max(materials.values()))
    material_range = range(0, max_id+1)


    for name, idx in sorted(materials.items(), key=lambda x:x[1]):
        if idx in material_range:
            # if idx == 1: continue
            add_component(f"{name}", idx, 'Material', is_mat=True)
    if srcs_pmls:
        for name, idx in srcs_pmls.items():
            opacity = 1.0
            if idx == 1 : opacity = 0.5
            add_component(f"{name}", idx, 'Sources_PML', color="blue", opacity=opacity)
    if rxs:
        for name, idx in rxs.items():
            add_component(f"{name}", idx, 'Receivers', color="lime")
    
    #plotter.add_plane_widget(callback=apply_slice, normal='z', color="cyan")
    
    start_x = 20
    start_y = win_h - 30 
    btn_size = 18
    y_spacing = 28       
    col_width = 160
    rows_per_col = (win_h - 60) // y_spacing

    dropped = []
    for i, name in enumerate(actors.keys()):
        col = i // rows_per_col
        row = i % rows_per_col

        current_x = start_x + (col * col_width)
        current_y = start_y - (row * y_spacing)

        if current_x + col_width < win_w:
            plotter.add_checkbox_button_widget(
            callback=lambda f, n=name: toggle_vis(f, n),
            value=True,
            position=(current_x, current_y),
            size=btn_size,
            color_on="#4CAF50",
            color_off="#F44336"
            )
            plotter.add_text(name, position=(current_x + 25, current_y),
                             font_size=8, color="white")
        else:
            dropped.append(name)

    if dropped:
        print(f"Warning: {len(dropped)} component(s) have no toggle (window too small): "
              f"{', '.join(dropped)}")
            
    plotter.add_axes(color="white")
    plotter.show_bounds(
        grid='back',           
        location='outer',      
        all_edges=True,        
        xtitle='X [m]', 
        ytitle='Y [m]', 
        ztitle='Z [m]',
        color='white',
        font_size=10,
        fmt="%.2f",
        padding=0.0,          
        use_3d_text=False      
    )
    plotter.reset_camera()
    plotter.show()


if __name__ == "__main__":
    file_path = load_file()
    if file_path:
        mats, srcs, rxs, pmls = get_gprmax_metadata(file_path)
        data = pv.read(str(file_path))
        plot_gprmax(data, mats, srcs, rxs, pmls)