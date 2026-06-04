import numpy as np
import pytcad.Quantities as qs
import time
import plotly.graph_objects as go
import pyvista as pv
import matplotlib.pyplot as plt


def display_quantity_pyvista(sim, quantity='potential', show_edges=True, show_nodes=True):
    """
    Visualize 2D/3D mesh with color mapped to a node-based physical quantity using PyVista.
    Args:
        sim: Simulation object with mesh and quantities
        quantity: str, name of the quantity to display (e.g., 'potential', 'electron_density', etc.)
        show_edges: bool, whether to show mesh edges
        show_nodes: bool, whether to show mesh nodes
    """
    # Get node coordinates
    x, y, z = sim.mesh.get_coordinate()
    points = np.column_stack([x, y, z])

    # Always plot all regions as separate actors and always use 'rainbow' colormap
    plotter = pv.Plotter()
    # plotter = BackgroundPlotter()
    x, y, z = sim.mesh.get_coordinate()
    points = np.column_stack([x, y, z])
    # Get node-based quantity values
    if quantity == 'potential':
        value = qs.get_potential(sim)
        colorbar_title = 'Potential [V]'
    elif quantity == 'electron_density':
        value = qs.get_electron_density(sim)
        value = np.where(value > 0, np.log10(value), np.nan)
        colorbar_title = 'log10(Electron Density [m⁻³])'
    elif quantity == 'hole_density':
        value = qs.get_hole_density(sim)
        value = np.where(value > 0, np.log10(value), np.nan)
        colorbar_title = 'log10(Hole Density [m⁻³])'
    elif quantity == 'temperature':
        value = qs.get_temperature(sim)
        colorbar_title = 'Temperature [K]'
    else:
        value = getattr(sim, quantity)
        colorbar_title = quantity

    for reg in sim.mesh.regions:
        region_cells = set(reg.cells)
        region_nodes = set(reg.nodes)
        # Build mesh for this region only
        if sim.mesh.cell_type == 'triangular':
            faces = []
            for i_cell, cell in enumerate(sim.mesh.cells):
                if len(cell.nodes) == 3 and i_cell in region_cells:
                    faces.append([3] + list(cell.nodes))
            if not faces:
                continue
            faces = np.hstack(faces)
            grid = pv.PolyData(points, faces)
        elif sim.mesh.cell_type == 'tetrahedron':
            cells = []
            cell_types = []
            for i_cell, cell in enumerate(sim.mesh.cells):
                if len(cell.nodes) == 4 and i_cell in region_cells:
                    cells.append([4] + list(cell.nodes))
                    cell_types.append(pv.CellType.TETRA)
            if not cells:
                continue
            cells = np.hstack(cells)
            grid = pv.UnstructuredGrid(cells, np.array(cell_types), points)
        else:
            continue
        # Mask out-of-region nodes by setting their value to np.nan
        region_value = np.array(value)
        mask = np.ones(len(points), dtype=bool)
        mask[list(region_nodes)] = False
        region_value[mask] = np.nan
        grid.point_data[quantity] = region_value

        plotter.add_mesh(grid, scalars=quantity, cmap='rainbow', show_edges=show_edges, nan_color='gray',
                        scalar_bar_args={'title': colorbar_title}, lighting=False, name=reg.name)
    plotter.render()

    if show_nodes:
        plotter.add_points(points, color='blue', point_size=6, render_points_as_spheres=True)

    if sim.mesh.mesh_type in ['2D', 'cylindrical_2D'] and sim.mesh.cell_type == 'triangular':
        # Set the camera to orthographic projection for 2D-like view
        plotter.camera.parallel_projection = True
        plotter.view_xy()
    
    plotter.show()
    

def display_quantity_plotly(sim, quantity='potential'):
    """
    Visualize the 2D or 3D mesh with color mapped to a node-based physical quantity using Plotly.
    Args:
        sim: Simulation object with mesh and quantities
        quantity: str, name of the quantity to display (e.g., 'potential', 'electron_density', etc.)
    """
    # Get node coordinates
    x, y, z = sim.mesh.get_coordinate()
    colorscale = 'Rainbow'  # Uniform colormap for all regions
    # Get node-based quantity values
    if quantity == 'potential':
        value = qs.get_potential(sim)
        colorbar_title = 'Potential [V]'
    elif quantity == 'electron_density':
        value = qs.get_electron_density(sim)
        value = np.where(value > 0, value, np.nan)  # avoid log of non-positive
        value = np.log10(value)
        colorbar_title = 'log10(Electron Density [m⁻³])'
    elif quantity == 'hole_density':
        value = qs.get_hole_density(sim)
        value = np.where(value > 0, value, np.nan)
        value = np.log10(value)
        colorbar_title = 'log10(Hole Density [m⁻³])'
    elif quantity == 'temperature':
        value = qs.get_temperature(sim)
        colorbar_title = 'Temperature [K]'
    else:
        value = getattr(sim, quantity)
        colorbar_title = quantity

    if sim.mesh.mesh_type in ['2D', 'cylindrical_2D'] and sim.mesh.cell_type == 'triangular':

        vmin, vmax = np.nanmin(value), np.nanmax(value)
        fig = go.Figure()
        # For each region, plot all triangles (cells) in that region using go.Mesh3d with z=0 for per-vertex color interpolation
        # Add a dummy Mesh3d trace for the colorbar (always visible, not linked to any region)
        fig.add_trace(go.Mesh3d(
            x=[0, 0, 0], y=[0, 0, 0], z=[0, 0, 0],
            i=[], j=[], k=[],
            intensity=[vmin, (vmin+vmax)/2, vmax],
            colorscale=colorscale,
            cmin=vmin, cmax=vmax,
            colorbar=dict(title=colorbar_title, x=-0.25, y=0.5, len=0.7, thickness=15, orientation='v'),
            showscale=True,
            opacity=0,
            hoverinfo='skip',
            name='',
            showlegend=False
        ))
        for i_region, region in enumerate(sim.mesh.regions):
            if not region.cells:
                continue
            tri_cells = [sim.mesh.cells[i_cell] for i_cell in region.cells if len(sim.mesh.cells[i_cell].nodes) == 3]
            if not tri_cells:
                continue
            # Collect all unique nodes used in this region
            region_node_set = set(n for cell in tri_cells for n in cell.nodes)
            region_node_list = sorted(region_node_set)
            node_idx_map = {n: i for i, n in enumerate(region_node_list)}
            region_x = [x[n] for n in region_node_list]
            region_y = [y[n] for n in region_node_list]
            region_z = [0 for _ in region_node_list]
            region_val = [value[n] for n in region_node_list]
            i_list, j_list, k_list = [], [], []
            for cell in tri_cells:
                n0, n1, n2 = cell.nodes
                i_list.append(node_idx_map[n0])
                j_list.append(node_idx_map[n1])
                k_list.append(node_idx_map[n2])
            hovertext = [f"{colorbar_title}: {v:.3e}" if not np.isnan(v) else "" for v in region_val]
            region_name = region.name if region.name else f'Region {i_region}'
            fig.add_trace(go.Mesh3d(
                x=region_x, y=region_y, z=region_z,
                i=i_list, j=j_list, k=k_list,
                intensity=region_val,
                colorscale=colorscale,
                cmin=vmin, cmax=vmax,
                flatshading=True,
                showscale=False,
                name=region_name,
                legendgroup=region_name,
                showlegend=True,
                opacity=1.0,
                text=hovertext,
                hoverinfo='text',
            ))
        # Add edges as black lines in 3D (z=0)
        edge_x, edge_y, edge_z = [], [], []
        for edge in sim.mesh.edges:
            n0, n1 = edge.nodes
            edge_x += [x[n0], x[n1], None]
            edge_y += [y[n0], y[n1], None]
            edge_z += [0, 0, None]
        fig.add_trace(go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines', line=dict(width=1, color='black'),
            name='Edges', showlegend=True
        ))
        # Add nodes as blue markers in 3D (z=0, hidden by default)
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=[0]*len(x),
            mode='markers', marker=dict(size=4, color='blue'),
            name='Nodes', showlegend=True, visible='legendonly'
        ))
        fig.update_layout(
            title=f"2D Mesh: {quantity}",
            scene=dict(
                xaxis_title='x',
                yaxis_title='y',
                zaxis_title='',
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=0.01),
                camera=dict(
                    eye=dict(x=0, y=0, z=2),
                    up=dict(x=0, y=1, z=0),
                    center=dict(x=0, y=0, z=0)
                ),
                xaxis=dict(
                    showgrid=False,
                    zeroline=False,
                    showbackground=False,
                    showspikes=False,
                    showline=True,
                    ticks='outside',
                    showticklabels=True,
                    title='x',
                ),
                yaxis=dict(
                    showgrid=False,
                    zeroline=False,
                    showbackground=False,
                    showspikes=False,
                    showline=True,
                    ticks='outside',
                    showticklabels=True,
                    title='y',
                ),
                zaxis=dict(
                    showgrid=False,
                    showticklabels=False,
                    title='',
                    showbackground=False
                ),
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            legend=dict(
                orientation="v",
                x=1.15, y=1,
                xanchor="left", yanchor="top",
                bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)", borderwidth=0
            ),
            width=700, height=600
        )
        fig.show()
        return

    elif sim.mesh.mesh_type == '3D' and sim.mesh.cell_type == 'tetrahedron':
        # Use a uniform rainbow colormap for all regions
        fig = go.Figure()
        vmin, vmax = np.nanmin(value), np.nanmax(value)

        # For each region, extract boundary faces and plot with region legend
        # Add a dummy Mesh3d trace for the colorbar (always visible, no faces)
        fig.add_trace(go.Mesh3d(
            x=[0, 0, 0], y=[0, 0, 0], z=[0, 0, 0],
            i=[], j=[], k=[],
            intensity=[vmin, (vmin+vmax)/2, vmax],
            colorscale=colorscale,
            cmin=vmin, cmax=vmax,
            colorbar=dict(title=colorbar_title, x=-0.25, y=0.5, len=0.7, thickness=15, orientation='v'),
            showscale=True,
            opacity=0,
            hoverinfo='skip',
            name='',
            showlegend=False
        ))

        for i_region, region in enumerate(sim.mesh.regions):
            if not region.cells:
                continue
            from collections import defaultdict
            face_count = defaultdict(int)
            for i_cell in region.cells:
                cell = sim.mesh.cells[i_cell]
                n = cell.nodes
                faces = [ (n[0], n[1], n[2]), (n[0], n[1], n[3]), (n[0], n[2], n[3]), (n[1], n[2], n[3]) ]
                for face in faces:
                    key = tuple(sorted(face))
                    face_count[key] += 1
            boundary_faces = [v for v, count in face_count.items() if count == 1]
            if not boundary_faces:
                continue
            # Vertex-duplication: create new arrays for this region's faces
            region_x, region_y, region_z, region_val = [], [], [], []
            i_list, j_list, k_list = [], [], []
            node_map = {}  # global node idx -> local idx
            next_idx = 0
            for face in boundary_faces:
                local_indices = []
                for idx in face:
                    if idx not in node_map:
                        node_map[idx] = next_idx
                        region_x.append(x[idx])
                        region_y.append(y[idx])
                        region_z.append(z[idx])
                        region_val.append(value[idx])
                        local_indices.append(next_idx)
                        next_idx += 1
                    else:
                        local_indices.append(node_map[idx])
                i_list.append(local_indices[0])
                j_list.append(local_indices[1])
                k_list.append(local_indices[2])
            region_name = region.name if region.name else f'Region {i_region}'
            legend_name = f'{region_name}'
            hovertext = [f"{colorbar_title}: {v:.3e}" if not np.isnan(v) else "" for v in region_val]
            fig.add_trace(go.Mesh3d(
                x=region_x, y=region_y, z=region_z,
                i=i_list, j=j_list, k=k_list,
                intensity=region_val,
                colorscale=colorscale,
                cmin=vmin, cmax=vmax,
                flatshading=True,
                showscale=False,
                lighting=dict(ambient=1, diffuse=0, specular=0, roughness=1, fresnel=0),
                lightposition=dict(x=0, y=0, z=100),
                name=legend_name,
                legendgroup=legend_name,
                showlegend=True,
                opacity=1.0,
                text=hovertext,
                hoverinfo='text'
            ))

        # Add edges as black lines
        edge_x, edge_y, edge_z = [], [], []
        for edge in sim.mesh.edges:
            n0, n1 = edge.nodes
            edge_x += [x[n0], x[n1], None]
            edge_y += [y[n0], y[n1], None]
            edge_z += [z[n0], z[n1], None]
        fig.add_trace(go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines', line=dict(width=1, color='black'),
            name='Edges', opacity=0.5, showlegend=True
        ))

        # Add nodes as blue markers
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers', marker=dict(size=4, color='blue'),
            name='Nodes', showlegend=True,
            visible='legendonly'
        ))

        fig.update_layout(
            title=f"3D Mesh: {quantity}",
            scene=dict(
                xaxis_title='x',
                yaxis_title='y',
                zaxis_title='z',
                aspectmode='data',
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            legend=dict(
                orientation="v",
                x=1.15, y=1,
                xanchor="left", yanchor="top",
                bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)", borderwidth=0
            )
        )
        fig.show()


def plot_profile_plotly(sim, profile='potential'):
    if sim.mesh.mesh_type!='1D':
        raise ValueError('Only 1D mesh is supported for direct plotting of quantities!')
    
    x,_,_=sim.mesh.get_coordinate()
    semi_idx = sim.node_idx_of_material(material_type='semiconductor')

    bias_txt = ''
    for electrode_name in list(sim.device.electrodes):
        if bias_txt != '':
            bias_txt = bias_txt + ', '
        bias_txt=bias_txt+'V'+electrode_name+"={:.2f}".format(sim.voltage_biases[electrode_name])+'V'

    fig = go.Figure()
    
    if profile == 'potential':
        potential=qs.get_potential(sim)
        fig.add_trace(go.Scatter(x=x, y=potential,mode='lines+markers',name='Potential'))
        yaxis_title='Potential [V]'
        title='Electric Potential Profile'

    if profile == 'dopping':
        fig.add_trace(go.Scatter(x=x[semi_idx], y=sim.NA[semi_idx], mode='lines+markers', name='Acceptor'))
        fig.add_trace(go.Scatter(x=x[semi_idx], y=sim.ND[semi_idx], mode='lines+markers', name='Donoor'))
        title='Dopping Profile'
        yaxis_title='Concentration [/m^3]'

    if profile == 'energy_band':
        Efn=qs.get_fermi_energy(sim,type='electron')
        Efp=qs.get_fermi_energy(sim,type='hole')
        Ec=qs.get_band_energy(sim,type='electron')
        Ev=qs.get_band_energy(sim,type='hole')
        fig.add_trace(go.Scatter(x=x, y=Efn, mode='lines+markers', name='Efn'))
        fig.add_trace(go.Scatter(x=x, y=Efp, mode='lines+markers', name='Efp'))
        fig.add_trace(go.Scatter(x=x, y=Ec, mode='lines+markers', name='Ec'))
        fig.add_trace(go.Scatter(x=x, y=Ev, mode='lines+markers', name='Ev'))
        title='Energy Band'
        yaxis_title='Energy [eV]'

    if profile == 'carrier_density':
        n=qs.get_electron_density(sim)
        p=qs.get_hole_density(sim)
        fig.add_trace(go.Scatter(x=x[semi_idx], y=n[semi_idx], mode='lines+markers', name='electron'))
        fig.add_trace(go.Scatter(x=x[semi_idx], y=p[semi_idx], mode='lines+markers', name='hole'))
        title='Carrier Concentration'
        yaxis_title='Concentration [/m^3]'

    if profile == 'space_charge':
        n=qs.get_electron_density(sim)
        p=qs.get_hole_density(sim)
        space_charge = -n+p+sim.ND-sim.NA
        fig.add_trace(go.Scatter(x=x[semi_idx], y=space_charge[semi_idx], mode='lines+markers', name='space_charge'))
        title='Space Charge'
        yaxis_title='Concentration [/m^3]'

    if profile == 'current_density':
        Jn_x,_,_=qs.get_current_density(sim,'electron')
        Jp_x,_,_=qs.get_current_density(sim,'hole')
        fig.add_trace(go.Scatter(x=x[semi_idx], y=Jn_x[semi_idx], mode='lines+markers', name='electron'))
        fig.add_trace(go.Scatter(x=x[semi_idx], y=Jp_x[semi_idx], mode='lines+markers', name='hole'))
        title='Current density'
        yaxis_title='Current density [A/m^2]'

    if profile == 'electric_field':
        Ex,_,_=qs.get_electric_field(sim)
        fig.add_trace(go.Scatter(x=x[semi_idx], y=Ex[semi_idx], mode='lines+markers'))
        title='Electric Field'
        yaxis_title='Electric Field [V/m]'

    if profile in ['potential','energy_band','carrier_density','current_density','electric_field']:
        fig.add_annotation(
        text=bias_txt,
        xref="paper", yref="paper",
        x=0.5, y=1.13, showarrow=False,
        font=dict(size=14, color="black"),
        align="center"
    )

    fig.update_layout(title=title,xaxis_title='x [m]',yaxis_title=yaxis_title)
    fig.show()

def plot_profile_matplotlib(sim, profile='potential'):
    if sim.mesh.mesh_type!='1D':
        raise ValueError('Only 1D mesh is supported for direct plotting of quantities!')
    
    x,_,_=sim.mesh.get_coordinate()
    semi_idx = sim.node_idx_of_material(material_type='semiconductor')

    bias_txt = ''
    for electrode_name in list(sim.device.electrodes):
        if bias_txt != '':
            bias_txt = bias_txt + ', '
        bias_txt=bias_txt+'V'+electrode_name+"={:.2f}".format(sim.voltage_biases[electrode_name])+'V'


    if profile == 'potential':
        potential=qs.get_potential(sim)
        plt.plot(x*1e6, potential, linewidth=2)
        title='Electric Potential Profile'
        yaxis_title='Potential [V]'

    if profile == 'dopping':
        plt.plot(x[semi_idx]*1e6, sim.NA[semi_idx], linewidth=2, label='Acceptor')
        plt.plot(x[semi_idx]*1e6, sim.ND[semi_idx], linewidth=2, label='Donor')
        title='Dopping Profile'
        yaxis_title='Concentration [/m^3]'

    if profile == 'energy_band':
        Efn=qs.get_fermi_energy(sim,type='electron')
        Efp=qs.get_fermi_energy(sim,type='hole')
        Ec=qs.get_band_energy(sim,type='electron')
        Ev=qs.get_band_energy(sim,type='hole')
        plt.plot(x*1e6, Efn, linewidth=2, label='Efn')
        plt.plot(x*1e6, Efp, linewidth=2, label='Efp')
        plt.plot(x*1e6, Ec, linewidth=2, label='Ec')
        plt.plot(x*1e6, Ev, linewidth=2, label='Ev')
        title='Energy Band'
        yaxis_title='Energy [eV]'

    if profile == 'carrier_density':
        n=qs.get_electron_density(sim)
        p=qs.get_hole_density(sim)
        plt.plot(x[semi_idx]*1e6, n[semi_idx], linewidth=2, label='electron')
        plt.plot(x[semi_idx]*1e6, p[semi_idx], linewidth=2, label='hole')
        title='Carrier Concentration'
        yaxis_title='Concentration [/m^3]'

    if profile == 'space_charge':
        n=qs.get_electron_density(sim)
        p=qs.get_hole_density(sim)
        space_charge = -n+p+sim.ND-sim.NA
        plt.plot(x[semi_idx]*1e6, space_charge[semi_idx], linewidth=2, label='space_charge')
        title='Space Charge'
        yaxis_title='Concentration [/m^3]'

    if profile == 'current_density':
        Jn_x,_,_=qs.get_current_density(sim,'electron')
        Jp_x,_,_=qs.get_current_density(sim,'hole')
        plt.plot(x[semi_idx]*1e6, Jn_x[semi_idx], linewidth=2, label='electron')
        plt.plot(x[semi_idx]*1e6, Jp_x[semi_idx], linewidth=2, label='hole')
        title='Current density'
        yaxis_title='Current density [A/m^2]'
    
    if profile == 'electric_field':
        Ex,_,_=qs.get_electric_field(sim)
        plt.plot(x[semi_idx]*1e6, Ex[semi_idx], linewidth=2)
        title='Electric Field'
        yaxis_title='Electric Field [V/m]'
    
    plt.title(title)
    plt.xlabel(r'Position, $x$ [$\mu$m]')
    plt.ylabel(yaxis_title)
    plt.grid(True)
    if profile in ['dopping','energy_band','carrier_density','current_density']:
        plt.legend()

    if profile in ['potential','energy_band','carrier_density','current_density','electric_field']:
        plt.gcf().text(0.5, 0.85, bias_txt, ha='center', fontsize=10)
    plt.show()

def export_vtk(sim,foldername,filename,x_scale=1,y_scale=1,z_scale=1,file_name_time_stepped_pvd=None,timestep=0):
    """
    Vectorized version of export_vtk for improved performance.
    Uses numpy array operations and bulk string formatting to reduce bottlenecks.
    """
    # x_scale, y_scale, z_scale: scale the dimension of the device to view the details better
    #                   negative scale factor could reverse the axis
    
    start_time = time.time()
    ## 20260115 paraview state 
    # for electrode_name in list(sim.device.electrodes):
    #     filename=filename+'_'+electrode_name+"_{:.2f}".format(sim.voltage_biases[electrode_name])
    filename_pvd=foldername+'/'+filename+'_main.pvd'
    
    with open(filename_pvd, "w") as f:
        f.write("<?xml version=\"1.0\"?>\n")
        f.write("<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\" compressor=\"vtkZLibDataCompressor\">\n")
        f.write("<Collection>\n")
        for i_region,region in enumerate(sim.mesh.regions):
            if len(region.cells)<=0:
                continue
            f.write(f"    <DataSet part=\"{i_region}\" file=\""+filename+f"_{i_region}.vtu\" name=\"Segment_{i_region}\"/>\n")

        f.write("  </Collection>\n")
        f.write("</VTKFile>\n")

        if file_name_time_stepped_pvd is not None:
            with open(file_name_time_stepped_pvd, "a") as f:
                for i_region,region in enumerate(sim.mesh.regions):
                    if len(region.cells)>0:
                        f.write(f"    <DataSet timestep=\"{timestep}\" part=\"{i_region}\" file=\""+filename+f"_{i_region}.vtu\" name=\"Segment_{i_region}\"/>\n")

    # Pre-compute all quantities once to avoid repeated calls
    print(f"Pre-computing quantities... ", end="", flush=True)
    quantities_start = time.time()
    
    potential = qs.get_potential(sim)
    potential = np.sign(potential) * np.maximum(np.abs(potential), 1e-30)

    # qfe = qs.get_quasi_fermi_electron(sim)
    # qfe = np.sign(qfe) * np.maximum(np.abs(qfe), 1e-30)

    # qfh = qs.get_quasi_fermi_hole(sim)
    # qfh = np.sign(qfh) * np.maximum(np.abs(qfh), 1e-30)

    n = qs.get_electron_density(sim)
    p = qs.get_hole_density(sim)

    ion = sim.ion_density

    ion_residual = sim.ion_density_residual
    
    ion_last = sim.ion_density_last_step

    mu_e = qs.get_mobility(sim, type='electron')

    Ex, Ey, Ez = qs.get_electric_field(sim)
    Ex = np.sign(Ex) * np.maximum(np.abs(Ex), 1e-30)
    Ey = np.sign(Ey) * np.maximum(np.abs(Ey), 1e-30)
    Ez = np.sign(Ez) * np.maximum(np.abs(Ez), 1e-30)

    Jx, Jy, Jz = qs.get_current_density(sim, type='electron')
    Jx = np.sign(Jx) * np.maximum(np.abs(Jx), 1e-30)
    Jy = np.sign(Jy) * np.maximum(np.abs(Jy), 1e-30)
    Jz = np.sign(Jz) * np.maximum(np.abs(Jz), 1e-30)

    Jx_h, Jy_h, Jz_h = qs.get_current_density(sim, type='hole')
    Jx_h = np.sign(Jx_h) * np.maximum(np.abs(Jx_h), 1e-30)
    Jy_h = np.sign(Jy_h) * np.maximum(np.abs(Jy_h), 1e-30)
    Jz_h = np.sign(Jz_h) * np.maximum(np.abs(Jz_h), 1e-30)

    Jxion_diffusion,Jyion_diffusion,Jzion_diffusion = qs.get_Jion_diffusion(sim, location='node')
    Jxion_diffusion = np.sign(Jxion_diffusion) * np.maximum(np.abs(Jxion_diffusion), 1e-30)
    Jyion_diffusion = np.sign(Jyion_diffusion) * np.maximum(np.abs(Jyion_diffusion), 1e-30)
    Jzion_diffusion = np.sign(Jzion_diffusion) * np.maximum(np.abs(Jzion_diffusion), 1e-30)

    Jxion_drift,Jyion_drift,Jzion_drift = qs.get_Jion_drift(sim, location='node')    
    Jxion_drift = np.sign(Jxion_drift) * np.maximum(np.abs(Jxion_drift), 1e-30)
    Jyion_drift = np.sign(Jyion_drift) * np.maximum(np.abs(Jyion_drift), 1e-30)
    Jzion_drift = np.sign(Jzion_drift) * np.maximum(np.abs(Jzion_drift), 1e-30)

    Jxion_soret,Jyion_soret,Jzion_soret = qs.get_Jion_soret(sim, location='node')
    Jxion_soret = np.sign(Jxion_soret) * np.maximum(np.abs(Jxion_soret), 1e-30)
    Jyion_soret = np.sign(Jyion_soret) * np.maximum(np.abs(Jyion_soret), 1e-30)
    Jzion_soret = np.sign(Jzion_soret) * np.maximum(np.abs(Jzion_soret), 1e-30)

    power_density = qs.get_power_density(sim)
    # power_density,power_density_node0_index = qs.get_power_density(sim, location='node')
    T = qs.get_temperature(sim)
    
    print(f"{time.time() - quantities_start:.2f}s")

    for i_region,region in enumerate(sim.mesh.regions):
        if len(region.cells)<=0:
            continue

        print(f"Processing region {i_region}... ", end="", flush=True)
        region_start = time.time()

        filename_segment=foldername+'/'+filename+f"_{i_region}.vtu"

        local_nodes_idx=np.zeros(len(sim.mesh.nodes),dtype=int)-1
        for local_i_node, i_node in enumerate(region.nodes):
            local_nodes_idx[i_node]=local_i_node

        # Convert region.nodes to numpy array for vectorized operations
        region_nodes = np.array(region.nodes)
        
        # Pre-compute all node coordinates
        node_coords = np.array([[sim.mesh.nodes[i_node].x * x_scale, 
                                sim.mesh.nodes[i_node].y * y_scale, 
                                sim.mesh.nodes[i_node].z * z_scale] 
                               for i_node in region_nodes])

        with open(filename_segment, "w") as f:
            ######################################### header information #########################################
            f.write("<?xml version=\"1.0\"?>\n")
            f.write("<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n")
            f.write(" <UnstructuredGrid>\n")
            f.write(f"  <Piece NumberOfPoints=\"{len(region.nodes)}\" NumberOfCells=\"{len(region.cells)}\">\n")

            ####################################### Point/Node information  #######################################
            f.write("   <Points>\n")
            f.write("    <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\n")
            
            # Vectorized coordinate writing
            coord_str = '\n'.join([f"{x} {y} {z}" for x, y, z in node_coords])
            f.write(coord_str + '\n')
            
            f.write("    </DataArray>\n")
            f.write("   </Points>\n")

            # Cell information - vectorized where possible
            f.write("   <Cells>\n")
            f.write("    <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n")
            
            # Vectorized connectivity writing
            connectivity_lines = []
            for i_cell in region.cells:
                cell = sim.mesh.cells[i_cell]
                conn_line = ' '.join([str(local_nodes_idx[i_node]) for i_node in cell.nodes])
                connectivity_lines.append(conn_line)
            f.write('\n'.join(connectivity_lines) + '\n')
            
            f.write("    </DataArray>\n")

            f.write("    <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n")
            
            # Vectorized offsets
            if sim.mesh.cell_type=='triangular':
                offsets = [str((i+1)*3) for i in range(len(region.cells))]
            elif sim.mesh.cell_type=='rectangular':
                offsets = [str((i+1)*4) for i in range(len(region.cells))]
            elif sim.mesh.cell_type=='hexahedron':
                offsets = [str((i+1)*8) for i in range(len(region.cells))]
            elif sim.mesh.cell_type=='tetrahedron':
                offsets = [str((i+1)*4) for i in range(len(region.cells))]
            f.write(' '.join(offsets) + '\n')
            
            f.write("    </DataArray>\n")
            
            # Figure 2 in https://docs.vtk.org/en/latest/design_documents/VTKFileFormats.html 
            f.write("    <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n")
            
            # Vectorized cell types
            if sim.mesh.cell_type=='triangular':
                cell_types = ['5'] * len(region.cells)
            elif sim.mesh.cell_type=='rectangular':
                cell_types = ['9'] * len(region.cells)
            elif sim.mesh.cell_type=='hexahedron':
                cell_types = ['12'] * len(region.cells)
            elif sim.mesh.cell_type=='tetrahedron':
                cell_types = ['10'] * len(region.cells)
            f.write(' '.join(cell_types) + '\n')
            
            f.write("    </DataArray>\n")
            f.write("   </Cells>\n")

            ######################################################   Point data  ####################################################
            f.write("   <PointData>\n")
            
            # Helper function for vectorized data writing
            def write_scalar_data(f, name, data_array, region_nodes, region_type, semiconductor_only=True):
                f.write(f"    <DataArray type=\"Float32\" Name=\"{name}\" NumberOfComponents=\"1\" format=\"ascii\">\n")
                if semiconductor_only and region_type != 'semiconductor':
                    values = ['0.0'] * len(region_nodes)
                else:
                    values = [str(data_array[i_node]) for i_node in region_nodes]
                f.write(' '.join(values) + '\n')
                f.write("    </DataArray>\n")
            
            def write_vector_data(f, name, data_arrays, region_nodes, region_type, semiconductor_only=True):
                f.write(f"    <DataArray type=\"Float32\" Name=\"{name}\" NumberOfComponents=\"3\" format=\"ascii\">\n")
                if semiconductor_only and region_type not in ['semiconductor', 'oxide']:
                    values = ['0.0 0.0 0.0'] * len(region_nodes)
                else:
                    values = [f"{data_arrays[0][i_node]} {data_arrays[1][i_node]} {data_arrays[2][i_node]}" 
                             for i_node in region_nodes]
                f.write(' '.join(values) + '\n')
                f.write("    </DataArray>\n")

            # Vectorized data writing
            write_scalar_data(f, "Acceptor Doping [log10(m^-3)]", 
                            np.log10(np.maximum(sim.NA, 1.0)), region_nodes, region.type)
            
            write_scalar_data(f, "Donor Doping [log10(m^-3)]", 
                            np.log10(np.maximum(sim.ND, 1.0)), region_nodes, region.type)

            # write_scalar_data(f, "Band Edge [eV]", 
            #                 sim.band_edge, region_nodes, region.type)
                        
            write_scalar_data(f, "Potential [V]", 
                            potential, region_nodes, region.type, semiconductor_only=False)
            
            # write_scalar_data(f, "Quasi Fermi Electron [V]", 
            #                 qfe, region_nodes, region.type, semiconductor_only=False)
            
            # write_scalar_data(f, "Quasi Fermi Hole [V]", 
            #                 qfh, region_nodes, region.type, semiconductor_only=False)
            
            write_scalar_data(f, "Electron density [log10(m^-3)]", 
                            np.log10(np.maximum(n, 1.0)), region_nodes, region.type)
            
            write_scalar_data(f, "Hole density [log10(m^-3)]", 
                            np.log10(np.maximum(p, 1.0)), region_nodes, region.type)
            
            write_scalar_data(f, "Ion density [log10(m^-3)]", 
                            np.log10(np.maximum(ion, 1.0)), region_nodes, region.type)
            
            write_scalar_data(f, "Ion density residual[log10(m^-3)]", 
                            np.log10(np.maximum(ion_residual, 1.0)), region_nodes, region.type)
            
            write_scalar_data(f, "Ion density last step [log10(m^-3)]", 
                            np.log10(np.maximum(ion_last, 1.0)), region_nodes, region.type)

            write_scalar_data(f, "Ion density max [log10(m^-3)]", 
                            np.log10(np.maximum(sim.ion_density_max_all, 1.0)), region_nodes, region.type)
            
            write_scalar_data(f, "Mobility [m^2/V/s]", 
                            mu_e, region_nodes, region.type)            
            
            # Electric field - special case for oxide regions
            f.write("    <DataArray type=\"Float32\" Name=\"Electric Field [Vm^-1]\" NumberOfComponents=\"3\" format=\"ascii\">\n")
            if region.type in ['semiconductor', 'oxide']:
                values = [f"{Ex[i_node]} {Ey[i_node]} {Ez[i_node]}" for i_node in region_nodes]
            else:
                values = ['0.0 0.0 0.0'] * len(region_nodes)
            f.write(' '.join(values) + '\n')
            f.write("    </DataArray>\n")
            
            write_vector_data(f, "Electron Current Density [Am^-2]", 
                            [Jx, Jy, Jz], region_nodes, region.type)
            
            write_vector_data(f, "Hole Current Density [Am^-2]", 
                            [Jx_h, Jy_h, Jz_h], region_nodes, region.type)

            write_vector_data(f, "Ion diffusion flux Density [Am^-2]", 
                            [Jxion_diffusion,Jyion_diffusion,Jzion_diffusion], region_nodes, region.type)

            write_vector_data(f, "Ion drift flux Density [Am^-2]", 
                            [Jxion_drift,Jyion_drift,Jzion_drift], region_nodes, region.type)

            write_vector_data(f, "Ion soret flux Density [Am^-2]", 
                            [Jxion_soret,Jyion_soret,Jzion_soret], region_nodes, region.type)

            write_scalar_data(f, "Power density [Wm^-3]", 
                            power_density, region_nodes, region.type, semiconductor_only=False)
            
            write_scalar_data(f, "Temperature [K]", 
                            T, region_nodes, region.type, semiconductor_only=False)

            f.write("   </PointData>\n")
            f.write("  </Piece>\n")
            f.write(" </UnstructuredGrid>\n")
            f.write("</VTKFile>\n")
        
        print(f"{time.time() - region_start:.2f}s")
    
    total_time = time.time() - start_time
    print(f"Total VTK export time: {total_time:.2f}s")

# def export_vtk(sim,foldername,filename,x_scale=1,y_scale=1,z_scale=1,file_name_time_stepped_pvd=None,timestep=0):
#     """
#     Vectorized version of export_vtk for improved performance.
#     Uses numpy array operations and bulk string formatting to reduce bottlenecks.
#     """
#     # x_scale, y_scale, z_scale: scale the dimension of the device to view the details better
#     #                   negative scale factor could reverse the axis
    
#     start_time = time.time()
    
#     for electrode_name in list(sim.device.electrodes):
#         filename=filename+'_'+electrode_name+"_{:.2f}".format(sim.voltage_biases[electrode_name])
#     filename_pvd=foldername+'/'+filename+'_main.pvd'
    
#     with open(filename_pvd, "w") as f:
#         f.write("<?xml version=\"1.0\"?>\n")
#         f.write("<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\" compressor=\"vtkZLibDataCompressor\">\n")
#         f.write("<Collection>\n")
#         for i_region,region in enumerate(sim.mesh.regions):
#             if len(region.cells)<=0:
#                 continue
#             f.write(f"    <DataSet part=\"{i_region}\" file=\""+filename+f"_{i_region}.vtu\" name=\"Segment_{i_region}\"/>\n")

#         f.write("  </Collection>\n")
#         f.write("</VTKFile>\n")

#         if file_name_time_stepped_pvd is not None:
#             with open(file_name_time_stepped_pvd, "a") as f:
#                 for i_region,region in enumerate(sim.mesh.regions):
#                     if len(region.cells)>0:
#                         f.write(f"    <DataSet timestep=\"{timestep}\" part=\"{i_region}\" file=\""+filename+f"_{i_region}.vtu\" name=\"Segment_{i_region}\"/>\n")

#     # Pre-compute all quantities once to avoid repeated calls
#     # print(f"Pre-computing quantities... ", end="", flush=True)
#     quantities_start = time.time()
    
#     potential = qs.get_potential(sim)
#     n = qs.get_electron_density(sim)
#     p = qs.get_hole_density(sim)
#     start = time.time()
#     Ex, Ey, Ez = qs.get_electric_field(sim)
#     # print(f"\ntime for get electric field: {time.time() - start:.2f}s")
#     start = time.time()
#     Jx, Jy, Jz = qs.get_current_density(sim, type='electron')
#     # print(f"time for get current_density: {time.time() - start:.2f}s")
#     start = time.time()
#     power_density = qs.get_power_density(sim)
#     # print(f"time for get power_density: {time.time() - start:.2f}s")
#     start = time.time()
#     T = qs.get_temperature(sim)
#     # print(f"time for get temperature: {time.time() - start:.2f}s")

#     print(f"{time.time() - quantities_start:.2f}s")

#     for i_region,region in enumerate(sim.mesh.regions):
#         if len(region.cells)<=0:
#             continue

#         filename_segment=foldername+'/'+filename+f"_{i_region}.vtu"

#         local_nodes_idx=np.zeros(len(sim.mesh.nodes),dtype=int)-1
#         for local_i_node, i_node in enumerate(region.nodes):
#             local_nodes_idx[i_node]=local_i_node

#         # Convert region.nodes to numpy array for vectorized operations
#         region_nodes = np.array(region.nodes)
        
#         # Pre-compute all node coordinates
#         node_coords = np.array([[sim.mesh.nodes[i_node].x * x_scale, 
#                                 sim.mesh.nodes[i_node].y * y_scale, 
#                                 sim.mesh.nodes[i_node].z * z_scale] 
#                                for i_node in region_nodes])

#         with open(filename_segment, "w") as f:
#             ######################################### header information #########################################
#             f.write("<?xml version=\"1.0\"?>\n")
#             f.write("<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n")
#             f.write(" <UnstructuredGrid>\n")
#             f.write(f"  <Piece NumberOfPoints=\"{len(region.nodes)}\" NumberOfCells=\"{len(region.cells)}\">\n")

#             ####################################### Point/Node information  #######################################
#             f.write("   <Points>\n")
#             f.write("    <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\n")
            
#             # Vectorized coordinate writing
#             coord_str = '\n'.join([f"{x} {y} {z}" for x, y, z in node_coords])
#             f.write(coord_str + '\n')
            
#             f.write("    </DataArray>\n")
#             f.write("   </Points>\n")

#             # Cell information - vectorized where possible
#             f.write("   <Cells>\n")
#             f.write("    <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n")
            
#             # Vectorized connectivity writing
#             connectivity_lines = []
#             for i_cell in region.cells:
#                 cell = sim.mesh.cells[i_cell]
#                 conn_line = ' '.join([str(local_nodes_idx[i_node]) for i_node in cell.nodes])
#                 connectivity_lines.append(conn_line)
#             f.write('\n'.join(connectivity_lines) + '\n')
            
#             f.write("    </DataArray>\n")

#             f.write("    <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n")
            
#             # Vectorized offsets
#             if sim.mesh.cell_type=='triangular':
#                 offsets = [str((i+1)*3) for i in range(len(region.cells))]
#             elif sim.mesh.cell_type=='rectangular':
#                 offsets = [str((i+1)*4) for i in range(len(region.cells))]
#             elif sim.mesh.cell_type=='hexahedron':
#                 offsets = [str((i+1)*8) for i in range(len(region.cells))]
#             elif sim.mesh.cell_type=='tetrahedron':
#                 offsets = [str((i+1)*4) for i in range(len(region.cells))]
#             f.write(' '.join(offsets) + '\n')
            
#             f.write("    </DataArray>\n")
            
#             # Figure 2 in https://docs.vtk.org/en/latest/design_documents/VTKFileFormats.html 
#             f.write("    <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n")
            
#             # Vectorized cell types
#             if sim.mesh.cell_type=='triangular':
#                 cell_types = ['5'] * len(region.cells)
#             elif sim.mesh.cell_type=='rectangular':
#                 cell_types = ['9'] * len(region.cells)
#             elif sim.mesh.cell_type=='hexahedron':
#                 cell_types = ['12'] * len(region.cells)
#             elif sim.mesh.cell_type=='tetrahedron':
#                 cell_types = ['10'] * len(region.cells)
#             f.write(' '.join(cell_types) + '\n')
            
#             f.write("    </DataArray>\n")
#             f.write("   </Cells>\n")

#             ######################################################   Point data  ####################################################
#             f.write("   <PointData>\n")
            
#             # Helper function for vectorized data writing
#             def write_scalar_data(f, name, data_array, region_nodes, region_type, semiconductor_only=True, scale='linear'):
#                 f.write(f"    <DataArray type=\"Float64\" Name=\"{name}\" NumberOfComponents=\"1\" format=\"ascii\">\n")
#                 if semiconductor_only and region_type != 'semiconductor':
#                     values = ['0.0'] * len(region_nodes)
#                 else:
#                     if scale == 'log':
#                         values = [str(np.log10(data_array[i_node])) for i_node in region_nodes]
#                     else:
#                         values = [str(data_array[i_node]) for i_node in region_nodes]
#                 f.write(' '.join(values) + '\n')
#                 f.write("    </DataArray>\n")
            
#             def write_vector_data(f, name, data_arrays, region_nodes, region_type, semiconductor_only=True, semiconductor_and_oxide=False):
#                 f.write(f"    <DataArray type=\"Float64\" Name=\"{name}\" NumberOfComponents=\"3\" format=\"ascii\">\n")
#                 if semiconductor_only and region_type !='semiconductor':
#                     values = ['0.0 0.0 0.0'] * len(region_nodes)
#                 elif semiconductor_and_oxide and region_type not in ['semiconductor','oxide']:
#                     values = ['0.0 0.0 0.0'] * len(region_nodes)
#                 else:
#                     values = [f"{data_arrays[0][i_node]} {data_arrays[1][i_node]} {data_arrays[2][i_node]}" 
#                              for i_node in region_nodes]
#                 f.write(' '.join(values) + '\n')
#                 f.write("    </DataArray>\n")

#             # Vectorized data writing
#             write_scalar_data(f, "Acceptor Doping [log10(m^-3)]", 
#                             sim.NA, region_nodes, region.type,scale='log')
            
#             write_scalar_data(f, "Donor Doping [log10(m^-3)]", 
#                             sim.ND, region_nodes, region.type,scale='log')
                        
#             write_scalar_data(f, "Potential [V]", 
#                             potential, region_nodes, region.type, semiconductor_only=False)
            
#             write_scalar_data(f, "Electron density [log10(m^-3)]", 
#                             n, region_nodes, region.type,scale='log')
            
#             write_scalar_data(f, "Hole density [log10(m^-3)]", 
#                             p, region_nodes, region.type,scale='log')

#             write_vector_data(f, "Electric Field [Vm^-1]", 
#                             [Ex, Ey, Ez], region_nodes, region.type, semiconductor_only=False, semiconductor_and_oxide=True)
            
#             write_vector_data(f, "Electron Current Density [Am^-2]", 
#                             [Jx, Jy, Jz], region_nodes, region.type)
            
#             write_scalar_data(f, "Power density [Wm^-3]", 
#                             power_density, region_nodes, region.type)
            
#             write_scalar_data(f, "Temperature [K]", 
#                             T, region_nodes, region.type, semiconductor_only=False)

#             f.write("   </PointData>\n")
#             f.write("  </Piece>\n")
#             f.write(" </UnstructuredGrid>\n")
#             f.write("</VTKFile>\n")
    
#     total_time = time.time() - start_time
#     # print(f"Total VTK export time: {total_time:.2f}s")

