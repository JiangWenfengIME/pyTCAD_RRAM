import numpy as np
import time
from collections import defaultdict
import plotly.graph_objs as go

class Node:
    def __init__(self,x=0,y=0,z=0):
        self.x=x      # coordinate of the node in x-axis
        self.y=y      # coordinate of the node in y-axis
        self.z=z      # coordinate of the node in z-axis

        self.edges=[]           # indices of edges that are connected to this node
        self.other_nodes=[]     # indices of nodes that are connected to this node (corresponding to self.edges)

        ## Voronoi box information to be defined in Device class after material types are defined
        self.box_volume_semi = 0.0        # voronoi box volume of semiconductor region 
        self.box_volume_semi_ox = 0.0     # voronoi box volume of semiconductor and oxide regions

class Cell:
    def __init__(self,node_idx,edges=[],region=None):
        self.nodes=node_idx          # indices of nodes of constructing this cell
        self.edges=edges             # indices of edges of this cell
        self.circumcenter=None       # circumenter coordinates of this cell
        self.region=region           # index for region, cell can only belong to one region

class Edge:
    def __init__(self,node_idx,cells=[]):
        self.nodes=node_idx          # indices of two nodes that are connected by this edge     
        self.cells=cells              # indices of cells that share this edge
        self.midpoint=None            # midpoint of this edge
        self.length=0.0               # length of the edge

        ## basic voronoi box information
        self.facet_area=[]      # area the facet between the voronoi boxes of the two nodes of the edge, for each cell shares the edge
        self.box_volume=[]      # contribution for volume of the voronoi box for each nodes of the edge,  for each cell shares the edge

        ## Voronoi box information to be defined in Device class after material types are defined
        self.facet_area_semi = 0.0    # area of facet of the voronoi box perpendicular to this edge in semiconductor region
        self.box_volume_semi = 0.0    # voronoi box volume of semiconductor region associated with this edge
        self.facet_area_semi_ox = 0.0     # area of facet of the voronoi box perpendicular to this edge in semiconductor and oxide regions
        self.box_volume_semi_ox = 0.0     # voronoi box volume of semiconductor and oxide region associated with this edge
        self.facet_area_all = 0.0     # area of facet of the voronoi box perpendicular to this edge in all regions
        self.box_volume_all = 0.0     # voronoi box volume of all regions associated with this edge

class Region:
    def __init__(self,cells,nodes=None,edges=None,name=None):
        self.name=name
        self.cells=cells             # indices of cells in this region. One cell can only belong to one region
        self.nodes=nodes             # indices of nodes in this region. Nodes can belong to multiple regions
        self.edges=edges             # indices of edges in this region. Edges can belong to multiple regions
        self.type=None               # type of the region material, semiconductor, conductor or oxide, setted in Device class
        self.material=None           # material properties of the region, setted in Device class
        self.electrode_name=None     # electrode name if this region is set to electrode, setted in Device class

class Mesh:
    def __init__(self, mesh_type='3D',cell_type='tetrahedron',filename=None,x_mesh=None,y_mesh=None,z_mesh=None,scale=1.0,symmetric_axis='x-axis'):
        self.mesh_type=mesh_type
        self.cell_type=cell_type
        if mesh_type == 'cylindrical_2D':
            self.symmetric_axis=symmetric_axis

        self.nodes=[]
        self.cells=[]
        self.edges=[]
        self.regions=[]

        reg_nums=[]
        start = time.time()
        if filename is not None:
            with open(filename, 'r') as f:
                # Reading node information from the mesh file
                line = f.readline()
                num_nodes=int(line)
                for i in range(num_nodes):
                    line = f.readline()
                    if mesh_type == '3D' and cell_type=='tetrahedron':
                        x,y,z=[float(x) for x in line.split()]
                        self.nodes.append(Node(x*scale,y*scale,z*scale))
                    elif mesh_type == '2D' and cell_type=='triangular':
                        x,y=[float(x) for x in line.split()]
                        self.nodes.append(Node(x*scale,y*scale))
                    elif mesh_type == 'cylindrical_2D' and cell_type=='triangular':
                        x,y=[float(x) for x in line.split()]
                        if self.symmetric_axis == 'x-axis' and y < 0:
                            raise ValueError('For cylindrical_2D mesh with the symmetric axis along x-axis, y coordinate should be non-negative!')
                        if self.symmetric_axis == 'y-axis' and x < 0:
                            raise ValueError('For cylindrical_2D mesh with the symmetric axis along y-axis, x coordinate should be non-negative!')
                        self.nodes.append(Node(x*scale,y*scale))  

                # Reading cell information from the mesh file
                line = f.readline()
                num_cells=int(line)
                for i_cell in range(num_cells):
                    line = f.readline()
                    if mesh_type=='3D' and cell_type=='tetrahedron':
                        # in mesh file, all indices start from 1, here we start indices from 0
                        reg_num,node0,node1,node2,node3=[int(x)-1 for x in line.split()]

                        edge0=self.check_or_add_edge(node0,node1,i_cell)
                        edge1=self.check_or_add_edge(node1,node2,i_cell)
                        edge2=self.check_or_add_edge(node2,node0,i_cell)
                        edge3=self.check_or_add_edge(node0,node3,i_cell)
                        edge4=self.check_or_add_edge(node1,node3,i_cell)
                        edge5=self.check_or_add_edge(node2,node3,i_cell)

                        new_cell=Cell([node0,node1,node2,node3],edges=[edge0,edge1,edge2,edge3,edge4,edge5],region=reg_num)
                        self.cells.append(new_cell) 

                        # TODO : process region info here

                    elif mesh_type in ['2D','cylindrical_2D'] and cell_type=='triangular':
                        # in mesh file, all indices start from 1, here we start indices from 0
                        reg_num,node0,node1,node2=[int(x)-1 for x in line.split()]

                        edge0=self.check_or_add_edge(node0,node1,i_cell)
                        edge1=self.check_or_add_edge(node1,node2,i_cell)
                        edge2=self.check_or_add_edge(node2,node0,i_cell)

                        new_cell=Cell([node0,node1,node2],edges=[edge0,edge1,edge2],region=reg_num)
                        self.cells.append(new_cell) 
                        
                    reg_nums.append(reg_num)
            end = time.time()
            print('Reading mesh file time:',end-start, 's')

            # 1. 按 reg_num 分组 cell
            region_map = defaultdict(list)
            for i_cell, reg_num in enumerate(reg_nums):
                region_map[reg_num].append(i_cell)

            # 2. 构造 Region
            for reg_num in sorted(region_map.keys()):
                cells = region_map[reg_num]
                region_nodes = set()
                region_edges = set()
                for i_cell in cells:
                    region_nodes.update(self.cells[i_cell].nodes)
                    region_edges.update(self.cells[i_cell].edges)
                self.regions.append(Region(cells, list(region_nodes), list(region_edges)))

            end = time.time()

            print('Region info pre-processing time:',end-start, 's')

            # calculate information for voronoi box discretization around nodes, edges
            start = time.time()
            self.voronoi_info()
            end = time.time()
            print('Calculating voronoi information time:',end-start, 's')

            # print basic information of the mesh
            print(mesh_type+" "+self.cell_type+" mesh loaded.")
            self.print_nodes_cell_numbers()
            
        else:
            if mesh_type=='1D':
                x_nodes=self.mesh_partition1d(x_mesh)
                for i in range(len(x_nodes)):
                    self.nodes.append(Node(x_nodes[i]))
            if mesh_type=='2D':
                self.x_nodes=self.mesh_partition1d(x_mesh)
                self.y_nodes=self.mesh_partition1d(y_mesh)
                for i in range(len(self.x_nodes)):
                    for j in range(len(self.y_nodes)):
                        self.nodes.append(Node(self.x_nodes[i],self.y_nodes[j]))
            if mesh_type=='3D':
                self.x_nodes=self.mesh_partition1d(x_mesh)
                self.y_nodes=self.mesh_partition1d(y_mesh)
                self.z_nodes=self.mesh_partition1d(z_mesh)
                for k in range(len(self.z_nodes)):
                    for i in range(len(self.x_nodes)):
                        for j in range(len(self.y_nodes)):
                            self.nodes.append(Node(self.x_nodes[i],self.y_nodes[j],self.z_nodes[k]))
            
            if mesh_type=='1D' and cell_type=='line':
                for i in range(len(x_nodes)-1):
                    if self.cell_type=='line':
                        self.cells.append(Cell(node_idx=[i,i+1],edges=[i])) 
                        self.edges.append(Edge(node_idx=[i,i+1],cells=[i])) 
                        self.nodes[i].edges.append(i)
                        self.nodes[i].other_nodes.append(i+1)
                        self.nodes[i+1].edges.append(i)
                        self.nodes[i+1].other_nodes.append(i)
            elif mesh_type in ['2D', 'cylindrical_2D'] and cell_type=='triangular':
                i_cell = 0
                for j in range(len(self.y_nodes)-1):
                    for i in range(len(self.x_nodes)-1):
                        # node of the rectangular
                        node0=i*len(self.y_nodes)+j
                        node1=i*len(self.y_nodes)+j+1
                        node2=(i+1)*len(self.y_nodes)+j
                        node3=(i+1)*len(self.y_nodes)+j+1

                        # node numbering of the rectangular
                        #              n2------n3
                        #       y ^    |       | 
                        #         |    |       |
                        #         |    n0------n1
                        #         0 ---> x             

                        if self.x_nodes[i+1]>(self.x_nodes[0]+self.x_nodes[-1])/2:
                            #              n2------n3
                            #              |  \    | 
                            #              |    \  |
                            #              n0------n1
                            # add one cell with node0, node1, node2
                            edge0=self.check_or_add_edge(node0,node1,i_cell)
                            edge1=self.check_or_add_edge(node1,node2,i_cell)
                            edge2=self.check_or_add_edge(node2,node0,i_cell)

                            new_cell=Cell([node0,node1,node2],edges=[edge0,edge1,edge2])
                            self.cells.append(new_cell)
                            i_cell+=1

                            # add one cell with node2, node3, node1
                            edge0=self.check_or_add_edge(node1,node2,i_cell)
                            edge1=self.check_or_add_edge(node2,node3,i_cell)
                            edge2=self.check_or_add_edge(node3,node1,i_cell)

                            new_cell=Cell([node2,node3,node1],edges=[edge0,edge1,edge2])
                            self.cells.append(new_cell)
                            i_cell+=1
                        else:
                            #              n2------n3
                            #              |     / | 
                            #              |  /    |
                            #              n0------n1
                            # add one cell with node0, node1, node3
                            edge0=self.check_or_add_edge(node0,node1,i_cell)
                            edge1=self.check_or_add_edge(node1,node3,i_cell)
                            edge2=self.check_or_add_edge(node3,node0,i_cell)

                            new_cell=Cell([node0,node1,node3],edges=[edge0,edge1,edge2])
                            self.cells.append(new_cell)
                            i_cell+=1

                            # add one cell with node0, node3, node2
                            edge0=self.check_or_add_edge(node0,node3,i_cell)
                            edge1=self.check_or_add_edge(node3,node2,i_cell)
                            edge2=self.check_or_add_edge(node2,node0,i_cell)

                            new_cell=Cell([node0,node3,node2],edges=[edge0,edge1,edge2])
                            self.cells.append(new_cell)
                            i_cell+=1

            else:
                raise ValueError('Only line cell type is supported for 1D mesh, only triangular cell type is supported for 2D mesh!')

            # calculate information for voronoi box discretization around nodes, edges
            start = time.time()
            self.voronoi_info()
            end = time.time()
            print('Calculating voronoi information time:',end-start, 's')
    
    def check_or_add_edge(self,node0,node1,i_cell):
        # check the edge connecting node0 and node1: 
        #   if exist
        #       1. add the cell idx to the edge.cells
        #       2. return the edge idx
        #   if not exist, 
        #       1. create a new edge and add the cell idx to the edge.cells, 
        #       2. append the edge idx to the connecting nodes
        #       3. append the other node idx to the connecting nodes
        #       4. return the idx of the new edge

        connected_edges=list(set(self.nodes[node0].edges) & set(self.nodes[node1].edges))
        if len(connected_edges)==1:
            i_edge=connected_edges[0]
            self.edges[i_edge].cells.append(i_cell)
        elif len(connected_edges)==0:
            new_edge=Edge([node0,node1],cells=[i_cell])
            self.edges.append(new_edge)

            i_edge = len(self.edges)-1
            self.nodes[node0].edges.append(i_edge)
            self.nodes[node0].other_nodes.append(node1)
            self.nodes[node1].edges.append(i_edge)
            self.nodes[node1].other_nodes.append(node0)
        return i_edge
    
    def get_coordinate(self):
        x=[]
        y=[]
        z=[]
        for node in self.nodes:
            x.append(node.x)
            y.append(node.y)
            z.append(node.z)    
        return np.array(x),np.array(y),np.array(z)

    def print_nodes_cell_numbers(self):
        print(f"Number of nodes: {len(self.nodes)}, number of edges: {len(self.edges)}, number of cells: {len(self.cells)}, number of regions: {len(self.regions)}.")

    def voronoi_info(self):
        # update circumcenter for all cells
        # circumcenter_outside_cell_nnumber = 0
        for cell in self.cells:
            if self.mesh_type=='3D' and self.cell_type=='tetrahedron':
                n0,n1,n2,n3=[self.nodes[i] for i in cell.nodes]
                cell.circumcenter=calculate_circumcentor_tetrahedron(n0,n1,n2,n3)
                # if point_outside_tetrahedron(self.cells[i_cell].circumcenter, n0, n1, n2, n3):
                #     # print(f"Warning: circumcenter of cell {i_cell} is outside the cell.")
                #     circumcenter_outside_cell_nnumber+=1
            elif self.mesh_type in ['2D','cylindrical_2D'] and self.cell_type=='triangular':
                n0,n1,n2=[self.nodes[i] for i in cell.nodes]
                cell.circumcenter=calculate_circumcentor_triangle(n0,n1,n2)
                # if not ispoint_inside_triangle(self.cells[i_cell].circumcenter, n0, n1, n2):
                #     # print(f"Warning: circumcenter of cell {i_cell} is outside the cell.")
                #     circumcenter_outside_cell_nnumber+=1
            elif self.mesh_type=='1D' and self.cell_type=='line':
                n0,n1=[self.nodes[i] for i in cell.nodes]
                cell.circumcenter=calculate_midpoint(n0,n1)
        # print("number of cells with circumcenter outside the cell:",circumcenter_outside_cell_nnumber)

        # calculate midpoint and length for all edges
        for edge in self.edges:
            n0,n1 = self.nodes[edge.nodes[0]],self.nodes[edge.nodes[1]]
            edge.midpoint=calculate_midpoint(n0,n1)
            edge.length=calculate_distance(n0,n1)
    

        # calculate information for voronoi box discretization
        for edge in self.edges:
            midpoint=edge.midpoint
            for i_cell in edge.cells:
                cell=self.cells[i_cell]
                cir_center =cell.circumcenter
                if self.mesh_type=='3D' and self.cell_type=='tetrahedron':
                    # TODO if circumcenter outside of the tetrahedron
                    # TODO facet circumcenter is calculated twice, can be optimized
                    edge_n0,edge_n1 = self.nodes[edge.nodes[0]],self.nodes[edge.nodes[1]]
                    i_other_two_nodes = [i for i in cell.nodes if i not in edge.nodes]
                    other_n0, other_n1 = self.nodes[i_other_two_nodes[0]],self.nodes[i_other_two_nodes[1]]
                    facet_cir_center0 = calculate_circumcenter_triangle_3d(edge_n0,edge_n1,other_n0)
                    facet_cir_center1 = calculate_circumcenter_triangle_3d(edge_n0,edge_n1,other_n1)
                    interface_area=1/2*(calculate_distance(cir_center,facet_cir_center0)*calculate_distance(midpoint,facet_cir_center0)+ 
                                        calculate_distance(cir_center,facet_cir_center1)*calculate_distance(midpoint,facet_cir_center1))
                    edge.facet_area.append(interface_area)
                    edge.box_volume.append(edge.length*interface_area/6)
                elif self.mesh_type in ['2D','cylindrical_2D'] and self.cell_type=='triangular':
                    edge_n0,edge_n1 = self.nodes[edge.nodes[0]],self.nodes[edge.nodes[1]]
                    # opposite_vertex = self.nodes[[i for i in cell.nodes if i not in edge.nodes][0]]
                    interface_area = calculate_distance(cir_center, midpoint)
                    # if same_side_of_line(cir_center, opposite_vertex, edge_n0, edge_n1):
                    #     interface_area = calculate_distance(cir_center, midpoint)
                    # else:
                    #     interface_area = -calculate_distance(cir_center, midpoint)
                    box_volume = (edge.length/2) * interface_area / 2
                    # if abs(box_volume)<1e-20:
                    #     continue
                    # print(box_volume)
                    edge.facet_area.append(interface_area)
                    edge.box_volume.append(box_volume)
                    
                elif self.mesh_type=='1D' and self.cell_type=='line':
                    edge.facet_area.append(1.0)
                    edge.box_volume.append(edge.length/2)
            

    def set_region_name(self,reg_num,reg_name):
        # set the name of the region for mesh from file
        self.regions[reg_num].name=reg_name

    def mesh_partition1d(self,mesh_spacing):
        nodes=[]
        for i in range(len(mesh_spacing['locations'])-1):
            location1=mesh_spacing['locations'][i]
            location2=mesh_spacing['locations'][i+1]
            spacing1=mesh_spacing['spacings'][i]
            spacing2=mesh_spacing['spacings'][i+1]
            nodes_segment=self.mesh_partition1d_segment(location1,location2,spacing1,spacing2)
            nodes.append(location1) 
            nodes=nodes+nodes_segment   
        nodes.append(location2)
        return nodes
    
    def mesh_partition1d_segment(self,x1,x2,s1,s2):
        nodes=[]
        if (s1+s2)>2*(x2-x1):
            if s1<(x2-x1) or s2<(x2-x1):
                nodes.append((x2+x1)/2)
        else:
            num_cells=int(np.ceil(2*(x2-x1)/(s2+s1)))
            if s1<=s2:
                delta_spacing=2*(x2-x1-num_cells*s1)/((num_cells-1)*num_cells)
                xx=x1
                for n in range(num_cells-1):
                    xx=xx+s1+n*delta_spacing
                    nodes.append(xx)
            else:
                delta_spacing=2*(x2-x1-num_cells*s2)/((num_cells-1)*num_cells)
                xx=x2
                for n in range(num_cells-1):
                    xx=xx-(s2+n*delta_spacing)
                    nodes.append(xx)
                nodes.sort()
        return nodes

    def set_region(self,reg_name,x_min=float('-inf'),x_max=float('inf'),y_min=float('-inf'),y_max=float('inf'),z_min=float('-inf'),z_max=float('inf')):
        # find if the region name exist 
        reg_name_exist=False
        for i_region, region in enumerate(self.regions):
            if region.name==reg_name:
                reg_name_exist=True   # if the region name exists, get the region cell numbers for updating
                region_cells=region.cells
                region_num=i_region
                break
        if reg_name_exist==False:     # if the region name does not exist, create a new region cell numbers list 
            region_num=len(self.regions)
            region_cells=[]
        for i_cell,cell in enumerate(self.cells):
            if cell.circumcenter.x>=x_min and cell.circumcenter.x<=x_max and cell.circumcenter.y>=y_min and cell.circumcenter.y<=y_max and cell.circumcenter.z>=z_min and cell.circumcenter.z<=z_max:
                if i_cell not in region_cells:
                    cell.region=region_num
                    self.cells[i_cell]=cell
                    region_cells.append(i_cell)
        if reg_name_exist:
            self.regions[region_num].cells=region_cells
        else:
            self.regions.append(Region(region_cells,name=reg_name))
        
        # update all regions 
        for i_region in range(len(self.regions)):
            region_cells=[]
            region_nodes=set([])
            region_edges=set([])
            for i_cell,cell in enumerate(self.cells):
                if i_region == cell.region:
                    region_cells.append(i_cell)
                    region_nodes = region_nodes|set(cell.nodes)
                    region_edges = region_edges|set(cell.edges)
            self.regions[i_region].cells=region_cells
            self.regions[i_region].nodes=list(region_nodes)
            self.regions[i_region].edges=list(region_edges)

    ## 202508 - interface trap
    def is_node_on_interface(self,node,reg_nums):
        if reg_nums[0]==reg_nums[1]:
            raise ValueError('The two region numbers should be different.')
        node_cell_regions=[]
        for i_edge in node.edges:
            edge = self.edges[i_edge]
            for cell in edge.cells:
                cell_obj=self.cells[cell] 
                node_cell_regions.append(cell_obj.region)
        if (reg_nums[0] in node_cell_regions) and (reg_nums[1] in node_cell_regions):
            return True
        else:
            return False
        
    def display_mesh(self, xrange=None, yrange=None, zrange=None):
        """
        Display mesh using Plotly for both 1D, 2D and 3D cases with nodes, edges, and colored regions; indices and names of regions are displayed in the legend.
        """
        import plotly.graph_objs as go
        
        # Define colors for different regions (consistent for both 2D and 3D)
        colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']

        if self.mesh_type == '1D':
            x_nodes = np.array([node.x for node in self.nodes])
            # 1D mesh display 
            fig = go.Figure()
            # Plot each cell as a colored segment by region
            for i_region, region in enumerate(self.regions):
                if len(region.cells) == 0:
                    continue
                region_color = colors[i_region % len(colors)]
                region_name = region.name if region.name else f'Region {i_region}'
                legend_name = f'{i_region}: {region_name}'
                for i_cell in region.cells:
                    cell = self.cells[i_cell]
                    n0, n1 = cell.nodes
                    x_seg = [self.nodes[n0].x, self.nodes[n1].x]
                    y_seg = [0, 0]
                    fig.add_trace(go.Scatter(
                        x=x_seg, y=y_seg,
                        mode='lines', line=dict(width=20, color=region_color),
                        name=legend_name, showlegend=(i_cell == region.cells[0]),
                        legendgroup=legend_name
                    ))

            # Plot edges as thin black lines (foreground)
            edge_x, edge_y = [], []
            for edge in self.edges:
                n0, n1 = edge.nodes
                edge_x += [self.nodes[n0].x, self.nodes[n1].x, None]
                edge_y += [0, 0, None]
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                mode='lines', line=dict(width=1, color='black'),
                name='Edges', showlegend=True)
            fig.add_trace(edge_trace)

            # Plot nodes as markers
            x_nodes = [node.x for node in self.nodes]
            y_nodes = [0 for _ in self.nodes]
            node_trace = go.Scatter(
                x=x_nodes, y=y_nodes,
                mode='markers',
                marker=dict(size=5, color='blue'),
                name='Nodes',
                showlegend=True)
            fig.add_trace(node_trace)

            fig.update_layout(
                width=800, height=200,
                xaxis=dict(showgrid=True, zeroline=False, range=xrange),
                yaxis=dict(showgrid=False, zeroline=False, visible=False),
                showlegend=True,
                legend=dict(
                    orientation="h", x=0.5, y=0.7,
                    xanchor="center", yanchor="bottom",
                    bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)", borderwidth=0
                ),
                margin=dict(l=40, r=40, t=60, b=40),
                title="1D Mesh Display"
            )
            fig.show()

        elif self.mesh_type in ['2D', 'cylindrical_2D']:
            # 2D mesh display
            fig = go.Figure()
            x_nodes = np.array([node.x for node in self.nodes])
            y_nodes = np.array([node.y for node in self.nodes])
            
            # First add colored regions in the background
            for i_region, region in enumerate(self.regions):
                if len(region.cells) == 0:
                    continue
                    
                region_color = colors[i_region % len(colors)]
                region_name = region.name if region.name else 'Unnamed Region'
                legend_name = f'{i_region}: {region_name}'
                
                # Collect all triangles for this region
                for i_cell in region.cells:     
                    cell = self.cells[i_cell]  
                    if len(cell.nodes) == 3:  # triangular cell
                        # Get coordinates of the three vertices
                        x_coords = [self.nodes[cell.nodes[i]].x for i in range(3)]
                        y_coords = [self.nodes[cell.nodes[i]].y for i in range(3)]
                        
                        # Add filled triangle without edges (background)
                        fig.add_trace(go.Scatter(
                            x=x_coords + [x_coords[0]],  # Close the triangle
                            y=y_coords + [y_coords[0]],
                            fill='toself', fillcolor=region_color,
                            opacity=1.0,  line=dict(width=0, color=region_color),
                            mode='lines', name=legend_name,
                            showlegend=(i_cell == region.cells[0]),  # Only show legend for first cell of each region
                            legendgroup=legend_name  # Group all cells of same region
                        ))
            
            # Plot all edges (foreground)
            edge_x, edge_y = [], []
            for edge in self.edges:
                n0, n1 = edge.nodes
                edge_x += [self.nodes[n0].x, self.nodes[n1].x, None]
                edge_y += [self.nodes[n0].y, self.nodes[n1].y, None]
            
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                mode='lines', line=dict(width=1.0, color='black'),
                name='Edges', showlegend=True
            )
            fig.add_trace(edge_trace)

            # Plot all nodes (foreground)
            node_trace = go.Scatter(
                x=x_nodes, y=y_nodes,
                mode='markers', marker=dict(size=5, color='blue'),
                name='Nodes', showlegend=True)
            fig.add_trace(node_trace)

            fig.update_layout(
                width=800, height=600,
                xaxis=dict(range=xrange, scaleanchor="y", scaleratio=1),
                yaxis=dict(range=yrange),
                showlegend=True,
                legend=dict(x=1.02, y=1,
                    bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)", borderwidth=0
                ),
                margin=dict(l=40, r=40, t=60, b=40),
                title="2D Mesh Display"
            )
            fig.show()
            
        elif self.mesh_type == '3D' and self.cell_type == 'tetrahedron':
            # 3D mesh display
            # Collect all node coordinates
            x = np.array([node.x for node in self.nodes])
            y = np.array([node.y for node in self.nodes])
            z = np.array([node.z for node in self.nodes])

            # Create figure
            fig = go.Figure()

            # First add colored regions in the background
            for i_region, region in enumerate(self.regions):
                if len(region.cells) == 0:
                    continue

                # --- Collect only boundary faces for this region ---
                face_count = dict()  # key: sorted tuple of 3 node indices, value: [face, count]
                for i_cell in region.cells:
                    cell = self.cells[i_cell]
                    n = cell.nodes
                    faces = [
                        (n[0], n[1], n[2]),
                        (n[0], n[1], n[3]),
                        (n[0], n[2], n[3]),
                        (n[1], n[2], n[3])
                    ]
                    for face in faces:
                        key = tuple(sorted(face))
                        if key not in face_count:
                            face_count[key] = [face, 1]
                        else:
                            face_count[key][1] += 1

                # Only keep faces that appear once (boundary faces)
                boundary_faces = [v[0] for v in face_count.values() if v[1] == 1]

                if boundary_faces:
                    i, j, k = zip(*boundary_faces)
                    region_color = colors[i_region % len(colors)]
                    region_name = region.name if region.name else 'Unnamed Region'
                    legend_name = f'{i_region}: {region_name}'

                    mesh3d = go.Mesh3d(
                        x=x, y=y, z=z,
                        i=i, j=j, k=k,
                        color=region_color, opacity=1.0,
                        name=legend_name, showscale=False, showlegend=True, flatshading=True,
                        lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0,
                            roughness=1.0, fresnel=0.0),
                        lightposition=dict(x=0, y=0, z=0)
                    )
                    fig.add_trace(mesh3d)

            # Plot all edges (foreground)
            edge_x, edge_y, edge_z = [], [], []
            for edge in self.edges:
                n0, n1 = edge.nodes
                edge_x += [self.nodes[n0].x, self.nodes[n1].x, None]
                edge_y += [self.nodes[n0].y, self.nodes[n1].y, None]
                edge_z += [self.nodes[n0].z, self.nodes[n1].z, None]
            
            edge_trace = go.Scatter3d(
                x=edge_x, y=edge_y, z=edge_z,
                mode='lines', line=dict(width=1, color='black'),
                name='Edges', showlegend=True
            )
            fig.add_trace(edge_trace)

            # Plot all nodes (foreground)
            node_trace = go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers', marker=dict(size=3, color='blue'),
                name='Nodes', showlegend=True
            )
            fig.add_trace(node_trace)

            # Update layout
            fig.update_layout(
                width=800, height=600,
                scene=dict(xaxis=dict(range=xrange), yaxis=dict(range=yrange), zaxis=dict(range=zrange),
                    aspectmode='data'),
                showlegend=True,
                legend=dict(x=1.02,y=1,
                    bgcolor="rgba(0,0,0,0)",  bordercolor="rgba(0,0,0,0)", borderwidth=0
                ),
                margin=dict(l=40, r=40, t=60, b=40),
                title="3D Mesh Display"
            )
            fig.show()
            
        else:
            raise ValueError('mesh type not supported for display here.')

    def print(self):
        print("Node number: (x, y, z), indices of edges connected to the nodes")
        for i,node in enumerate(self.nodes):
            print(f"{i}: ({node.x:.3e}, {node.y:.3e}, {node.z:.3e})", end=",(")
            for i_edge in node.edges:
                print(f"{i_edge}",end=',')
            print(')')

        print("Cell number: region, indices of nodes for this cell, indices of edges for this cell, ")
        for i_cell,cell in enumerate(self.cells):
            print(f"{i_cell}:",end=' ')
            print(f"{cell.region}",end=', (')
            for i_node in cell.nodes:
                print(f"{i_node}",end=',')
            print('),(',end='')
            for i_edge in cell.edges:
                print(f"{i_edge}",end=',')
            print('),',end='')
            cir=cell.circumcenter
            print(f"({cir.x:.3e}, {cir.y:.3e}, {cir.z:.3e})")
        
        print("Edge number: indices of nodes for this edge, indice of cells that share the edge")
        for i_edge,edge in enumerate(self.edges):
            print(f"{i_edge}:",end=' (')
            for i_node in edge.nodes:
                print(f"{i_node}",end=',')
            print('),(',end='')
            for i_cell in edge.cells:
                print(f"{i_cell}",end=',')
            print('),',end='')
            mid=edge.midpoint
            print(f"({mid.x:.3e}, {mid.y:.3e}, {mid.z:.3e})")

    
def calculate_circumcentor_triangle(n0,n1,n2):
    # calculate the circumcenter of a triangle, given the coordinates of three points of the triangle
    #   reference: https://stackoverflow.com/questions/56224824/how-do-i-find-the-circumcenter-of-the-triangle-using-python-without-external-lib
    d = 2 * (n0.x * (n1.y - n2.y) + n1.x * (n2.y - n0.y) + n2.x * (n0.y - n1.y))
    a=(n0.x * n0.x + n0.y * n0.y)
    b=(n1.x * n1.x + n1.y * n1.y)
    c=(n2.x * n2.x + n2.y * n2.y)
    ux = ( a * (n1.y - n2.y) + b * (n2.y - n0.y) + c * (n0.y - n1.y)) / d
    uy = ( a * (n2.x - n1.x) + b * (n0.x - n2.x) + c * (n1.x - n0.x)) / d
    return Node(ux, uy)

def calculate_circumcenter_triangle_3d(n0, n1, n2):
    """
    Calculate the circumcenter of a 3D triangle given three points.
    
    Args:
        n0, n1, n2: Points with attributes x, y, z.
    
    Returns:
        A point (x, y, z) representing the circumcenter.
    """
    # Midpoints of AB and AC
    mid_AB = Node((n0.x + n1.x) / 2, (n0.y + n1.y) / 2, (n0.z + n1.z) / 2)
    mid_AC = Node((n0.x + n2.x) / 2, (n0.y + n2.y) / 2, (n0.z + n2.z) / 2)

    # Direction vectors of AB and AC
    dir_AB = Node(n1.x - n0.x, n1.y - n0.y, n1.z - n0.z)
    dir_AC = Node(n2.x - n0.x, n2.y - n0.y, n2.z - n0.z)

    # Normal vector of the triangle's plane (cross product of AB and AC)
    normal = Node(
        dir_AB.y * dir_AC.z - dir_AB.z * dir_AC.y,
        dir_AB.z * dir_AC.x - dir_AB.x * dir_AC.z,
        dir_AB.x * dir_AC.y - dir_AB.y * dir_AC.x
    )

    # System of equations:
    # 1. (dir_AB) · X = (dir_AB) · mid_AB  (Perpendicular bisector plane of AB)
    # 2. (dir_AC) · X = (dir_AC) · mid_AC  (Perpendicular bisector plane of AC)
    # 3. (normal) · X = (normal) · n0      (Triangle's plane)
    M = [
        [dir_AB.x, dir_AB.y, dir_AB.z],
        [dir_AC.x, dir_AC.y, dir_AC.z],
        [normal.x, normal.y, normal.z]
    ]
    rhs = [
        dir_AB.x * mid_AB.x + dir_AB.y * mid_AB.y + dir_AB.z * mid_AB.z,
        dir_AC.x * mid_AC.x + dir_AC.y * mid_AC.y + dir_AC.z * mid_AC.z,
        normal.x * n0.x + normal.y * n0.y + normal.z * n0.z
    ]

    # Solve the system M * X = rhs
    def solve_system(M, rhs):
        n = 3
        for col in range(n):
            # Partial pivoting
            max_row = col
            for row in range(col + 1, n):
                if abs(M[row][col]) > abs(M[max_row][col]):
                    max_row = row
            M[col], M[max_row] = M[max_row], M[col]
            rhs[col], rhs[max_row] = rhs[max_row], rhs[col]
            if M[col][col] == 0:
                raise ValueError("Points are colinear; no circumcenter exists.")
            for row in range(col + 1, n):
                factor = M[row][col] / M[col][col]
                rhs[row] -= factor * rhs[col]
                for c in range(col, n):
                    M[row][c] -= factor * M[col][c]
        # Back substitution
        X = [0, 0, 0]
        for row in reversed(range(n)):
            X[row] = rhs[row]
            for col in range(row + 1, n):
                X[row] -= M[row][col] * X[col]
            X[row] /= M[row][row]
        return X

    try:
        x, y, z = solve_system(M, rhs)
    except ValueError:
        raise ValueError("No circumcenter exists (colinear points).")

    return Node(x, y, z)

def ispoint_inside_triangle(cir, a, b, c):
    (x1, y1), (x2, y2), (x3, y3) = (a.x,a.y), (b.x,b.y), (c.x,c.y)
    x, y = cir.x, cir.y
    
    # Calculate barycentric coordinates
    denominator = (y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3)
    if denominator == 0:
        return False  # The triangle is degenerate (points are colinear)
    
    a = ((y2 - y3)*(x - x3) + (x3 - x2)*(y - y3)) / denominator
    b = ((y3 - y1)*(x - x3) + (x1 - x3)*(y - y3)) / denominator
    c = 1 - a - b
    
    # Check if point is inside the triangle
    return 0 <= a <= 1 and 0 <= b <= 1 and 0 <= c <= 1

def calculate_circumcentor_tetrahedron(a,b,c,d):
    # calculate the circumcenter of a tetrahedron, given the coordinates of four points of the tetrahedron
    #   reference: https://rodolphe-vaillant.fr/entry/127/find-a-tetrahedron-circumcenter
    # ba = b - a
    ba_x, ba_y, ba_z = b.x - a.x, b.y - a.y, b.z - a.z
    # ca = c - a
    ca_x, ca_y, ca_z = c.x - a.x, c.y - a.y, c.z - a.z
    # da = d - a
    da_x, da_y, da_z = d.x - a.x, d.y - a.y, d.z - a.z
 
    # Squares of lengths of the edges incident to 'a'.
    len_ba = ba_x * ba_x + ba_y * ba_y + ba_z * ba_z
    len_ca = ca_x * ca_x + ca_y * ca_y + ca_z * ca_z
    len_da = da_x * da_x + da_y * da_y + da_z * da_z
 
    # Cross products of these edges.
    # c cross d
    cross_cd_x = ca_y * da_z - da_y * ca_z
    cross_cd_y = ca_z * da_x - da_z * ca_x
    cross_cd_z = ca_x * da_y - da_x * ca_y
    # d cross b
    cross_db_x = da_y * ba_z - ba_y * da_z
    cross_db_y = da_z * ba_x - ba_z * da_x
    cross_db_z = da_x * ba_y - ba_x * da_y
    # b cross c
    cross_bc_x = ba_y * ca_z - ca_y * ba_z
    cross_bc_y = ba_z * ca_x - ca_z * ba_x
    cross_bc_z = ba_x * ca_y - ca_x * ba_y
 
    # Calculate the denominator of the formula.
    denominator = 0.5 / (ba_x * cross_cd_x + ba_y * cross_cd_y + ba_z * cross_cd_z)
 
    # Calculate offset (from 'a') of circumcenter.
    circ_x = (len_ba * cross_cd_x + len_ca * cross_db_x + len_da * cross_bc_x) * denominator
    circ_y = (len_ba * cross_cd_y + len_ca * cross_db_y + len_da * cross_bc_y) * denominator
    circ_z = (len_ba * cross_cd_z + len_ca * cross_db_z + len_da * cross_bc_z) * denominator
    return Node(a.x+circ_x, a.y+circ_y, a.z+circ_z)

def point_outside_tetrahedron(point, a,b,c,d):
    """ Check if a point is outside a tetrahedron using barycentric coordinates in 3D. """
    (A, B, C, D) = np.array([a.x,a.y,a.z]), np.array([b.x,b.y,b.z]), np.array([c.x,c.y,c.z]), np.array([d.x,d.y,d.z])
    P = np.array([point.x,point.y,point.z])
    
    def signed_volume(a, b, c, d):
        """Calculate signed volume of tetrahedron formed by points a,b,c,d"""
        return (1/6) * np.dot(np.cross(np.subtract(b,a), np.subtract(c,a)), np.subtract(d,a))
    
    # Calculate volumes
    v0 = signed_volume(A, B, C, D)
    v1 = signed_volume(P, B, C, D)
    v2 = signed_volume(A, P, C, D)
    v3 = signed_volume(A, B, P, D)
    v4 = signed_volume(A, B, C, P)
    
    # Check if all same sign (inside) or not (outside)
    has_neg = (v1 < 0) or (v2 < 0) or (v3 < 0) or (v4 < 0)
    has_pos = (v1 > 0) or (v2 > 0) or (v3 > 0) or (v4 > 0)
    
    # Point is outside if volumes don't have same sign as original
    if (v0 > 0 and has_neg) or (v0 < 0 and has_pos):
        return True
    return False

def same_side_of_line(a, b, n0, n1):
    # check if point a and b are on the same side of line n0-n1
    cross_a = (n1.x - n0.x) * (a.y - n0.y) - (n1.y - n0.y) * (a.x - n0.x)
    cross_b = (n1.x - n0.x) * (b.y - n0.y) - (n1.y - n0.y) * (b.x - n0.x)
    # if cross_a == 0 or cross_b == 0:  At least one point lies on the line 
    return (cross_a * cross_b) >= 0

def calculate_distance(n0,n1):
    return np.sqrt((n0.x-n1.x)**2+(n0.y-n1.y)**2+(n0.z-n1.z)**2)

def calculate_midpoint(n0,n1):
    # calculate the midpoint of a line segment
    return Node((n0.x+n1.x)/2, (n0.y+n1.y)/2, (n0.z+n1.z)/2)
