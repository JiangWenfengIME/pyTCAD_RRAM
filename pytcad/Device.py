import numpy as np
import plotly.graph_objs as go

class Device:
    def __init__(self,mesh, area=1e-12, width=1e-6):
        self.mesh=mesh
        if mesh.mesh_type == '1D':
            self.area=area
        elif mesh.mesh_type == '2D':
            self.width=width

        # electrode names and associated region numbers
        self.electrodes={}

        self.dopping_donor=np.zeros(len(self.mesh.nodes))+1e-30
        self.dopping_acceptor=np.zeros(len(self.mesh.nodes))+1e-30
        self.ion_density=np.zeros(len(self.mesh.nodes))+1e-30
        # self.ion_density_initial = np.zeros(len(self.mesh.nodes))+1e-30

        self.residual_ion_percentage=0.01  # the percentage of the residual ion density after reset compared to the initial ion density before forming, which is used to simulate the forming effect in RRAM device, where the ion density after reset is usually much lower than the initial ion density before forming, but it is not zero.
        self.ion_density_residual=np.zeros(len(self.mesh.nodes))+1e-30
        self.max_residual_ion_value=1e22
        self.max_ion_density_residual=np.zeros(len(self.mesh.nodes))+1e-30

        self.thermal_boundary_node_idx = []

        self.ion_boundary_idx=[]

        self.ref_material=None  # reference material for semiconductor region

        self.voronoi_info_initialized = False 

    def find_region_number(self,reg_name):
        reg_name_exist=False
        for i_region, region in enumerate(self.mesh.regions):
            if region.name==reg_name:
                reg_name_exist=True   # if the region name exists
                reg_num=i_region
                break
        if reg_name_exist==False:     
            raise ValueError("Region name not exist.")
        return reg_num
    
    def set_material(self,reg_name=None,type=None,material=None):
        reg_num=self.find_region_number(reg_name)
        
        self.mesh.regions[reg_num].type=type
        self.mesh.regions[reg_num].material=material
        
        if type=='semiconductor':
            self.ref_material=material  # using the last semicondcutor setting material for refrence materials; used for multiple semiconductor materials in one device  

    def set_electrode(self,reg_name=None,type=None,material=None,electrode_name=None):
        reg_num=self.find_region_number(reg_name)
        
        if electrode_name is not None:
            if (type is None) or (type=='conductor'):
                self.mesh.regions[reg_num].type='conductor'
            else:
                raise ValueError("Only conductor region can be set to electrode.")
            
            self.electrodes[electrode_name]=reg_num
            self.mesh.regions[reg_num].electrode_name=electrode_name

    def set_uniform_doping(self,reg_name,conc,type):
        reg_num=self.find_region_number(reg_name)
        
        if self.mesh.regions[reg_num].type != "semiconductor":
            raise ValueError("Only semiconductor region can be dopped.")
        
        # cell_idx = self.mesh.regions[reg_num].cells
        # if type=="donor":
        #     self.dopping_donor[cell_idx]+=conc
        # if type=="acceptor":
        #     self.dopping_acceptor[cell_idx]+=conc

        node_idx = self.mesh.regions[reg_num].nodes
        if type=="donor":
            self.dopping_donor[node_idx]+=conc
        if type=="acceptor":
            self.dopping_acceptor[node_idx]+=conc
    
    def set_uniform_ion(self,reg_name,conc,type):
        reg_num=self.find_region_number(reg_name)
        
        if self.mesh.regions[reg_num].type != "semiconductor":
            raise ValueError("Only semiconductor region can be dopped.")
        
        # cell_idx = self.mesh.regions[reg_num].cells
        # if type=="donor":
        #     self.dopping_donor[cell_idx]+=conc
        # if type=="acceptor":
        #     self.dopping_acceptor[cell_idx]+=conc

        node_idx = self.mesh.regions[reg_num].nodes
        if type=="donor":
            self.ion_density[node_idx]+=conc
        if type=="acceptor":
            self.ion_density[node_idx]+=conc
        
        self.ion_density = self.ion_density*(1-self.residual_ion_percentage)
        self.ion_density_residual = self.ion_density * self.residual_ion_percentage  # set the residual ion density to be 10% of the initial ion density, which is used to simulate the forming effect in RRAM device, where the ion density after reset is usually much lower than the initial ion density before forming, but it is not zero.
    
    def set_gaussian_doping(self,reg_name,conc,type,char,x_min,x_max,y_peak=0):
        reg_num=self.find_region_number(reg_name)
        
        if self.mesh.regions[reg_num].type != "semiconductor":
            raise ValueError("Only semiconductor region can be dopped.")
        
        for i_node in self.mesh.regions[reg_num].nodes:
            node=self.mesh.nodes[i_node]
            if node.x >= x_min and node.x <= x_max:
                length_x=0
            else:
                length_x=min(abs(x_min-node.x),abs(x_max-node.x))
            length_y=abs(y_peak-node.y)

            length=np.sqrt(length_x**2+length_y**2)
            dopping_conc=np.exp(-length**2/(char**2))*conc

            if type=="donor":
                self.dopping_donor[i_node]+=dopping_conc
            if type=="acceptor":
                self.dopping_acceptor[i_node]+=dopping_conc
    
    def set_gaussian_ion(self,reg_name,conc,type,char,x_min,x_max,y_peak=0):
        reg_num=self.find_region_number(reg_name)
        
        if self.mesh.regions[reg_num].type != "semiconductor":
            raise ValueError("Only semiconductor region can be dopped.")
        
        for i_node in self.mesh.regions[reg_num].nodes:
            node=self.mesh.nodes[i_node]
            if node.x >= x_min and node.x <= x_max:
                length_x=0
            else:
                length_x=min(abs(x_min-node.x),abs(x_max-node.x))
            length_y=abs(y_peak-node.y)

            length=np.sqrt(length_x**2+length_y**2)
            dopping_conc=np.exp(-length**2/(char**2))*conc

            if type=="donor":
                self.ion_density[i_node]+=dopping_conc
            if type=="acceptor":
                self.ion_density[i_node]+=dopping_conc
        
        self.ion_density = self.ion_density*(1-self.residual_ion_percentage)
        self.ion_density_residual = self.ion_density * self.residual_ion_percentage # set the residual ion density to be 10% of the initial ion density, which is used to simulate the forming effect in RRAM device, where the ion density after reset is usually much lower than the initial ion density before forming, but it is not zero.

    def set_gaussian_ion_strip(self, reg_name, conc, type, char, x_peak=0):
        reg_num = self.find_region_number(reg_name)
        
        if self.mesh.regions[reg_num].type != "semiconductor":
            raise ValueError("Only semiconductor region can be dopped.")
        
        for i_node in self.mesh.regions[reg_num].nodes:
            node = self.mesh.nodes[i_node]
            
            dist_x = abs(node.x - x_peak)
            
            # 高斯分布计算: conc * exp(-(dx^2 / char^2))
            dopping_conc = conc * np.exp(-(dist_x**2) / (char**2))
            
            if type=="donor":
                self.ion_density[i_node]+=dopping_conc
            if type=="acceptor":
                self.ion_density[i_node]+=dopping_conc
        
        self.ion_density = self.ion_density*(1-self.residual_ion_percentage)
        self.ion_density_residual = self.ion_density * self.residual_ion_percentage # set the residual ion density to be 10% of the initial ion density, which is used to simulate the forming effect in RRAM device, where the ion density after reset is usually much lower than the initial ion density before forming, but it is not zero.

    def set_gaussian_ion_area_strip(self, reg_name, conc, type, char, x_peak_min=0, x_peak_max=0):
        reg_num = self.find_region_number(reg_name)
        
        if self.mesh.regions[reg_num].type != "semiconductor":
            raise ValueError("Only semiconductor region can be dopped.")
        
        for i_node in self.mesh.regions[reg_num].nodes:
            node = self.mesh.nodes[i_node]
            if node.x <= x_peak_max :
                if type=="donor":
                    self.ion_density[i_node]+=conc
                if type=="acceptor":
                    self.ion_density[i_node]+=conc
            else:          
                dist_x = abs(node.x - x_peak_max)               
                # 高斯分布计算: conc * exp(-(dx^2 / char^2))
                dopping_conc = conc * np.exp(-(dist_x**2) / (char**2))
                
                if type=="donor":
                    self.ion_density[i_node]+=dopping_conc
                if type=="acceptor":
                    self.ion_density[i_node]+=dopping_conc
        
        self.ion_density = self.ion_density*(1-self.residual_ion_percentage)
        self.ion_density_residual = self.ion_density * self.residual_ion_percentage # set the residual ion density to be 10% of the initial ion density, which is used to simulate the forming effect in RRAM device, where the ion density after reset is usually much lower than the initial ion density before forming, but it is not zero.


    def set_ion_boundary(self,reg_name,conc,type,x_min,x_max,y_min,y_max):
        reg_num=self.find_region_number(reg_name)
        
        if self.mesh.regions[reg_num].type != "semiconductor":
            raise ValueError("Only semiconductor region can be dopped.")
        
        for i_node in self.mesh.regions[reg_num].nodes:
            node=self.mesh.nodes[i_node]
            doping = 0
            if node.x >= x_min and node.x <= x_max and node.y >= y_min and node.y <= y_max:
                doping = conc
                self.ion_boundary_idx.append(i_node)
                if type=="donor":
                    self.ion_density[i_node]=doping*(1-self.residual_ion_percentage)
                if type=="acceptor":
                    self.ion_density[i_node]=doping*(1-self.residual_ion_percentage)

    def set_ion_boundary_region_interface(self,reg_name_1,reg_name_2,conc,type):
        reg_num_1=self.find_region_number(reg_name_1)
        reg_num_2=self.find_region_number(reg_name_2)
        
        if (self.mesh.regions[reg_num_1].type != "semiconductor") and (self.mesh.regions[reg_num_2].type != "semiconductor"):
            raise ValueError("No semiconductor regions be selected.")
        
        node_index_1 = set(self.mesh.regions[reg_num_1].nodes)
        node_index_2 = set(self.mesh.regions[reg_num_2].nodes)
        intersection_node_idx = list(set(node_index_1) & set(node_index_2))

        for i_node in intersection_node_idx:
            doping = conc
            self.ion_boundary_idx.append(i_node)
            if type=="donor":
                self.ion_density[i_node]=doping*(1-self.residual_ion_percentage)
            if type=="acceptor":
                self.ion_density[i_node]=doping*(1-self.residual_ion_percentage)

    def set_ion_boundary_area(self,reg_name,x_min,x_max,y_min,y_max):
        reg_num=self.find_region_number(reg_name)
        
        if self.mesh.regions[reg_num].type != "semiconductor":
            raise ValueError("Only semiconductor region can be dopped.")
        
        for i_node in self.mesh.regions[reg_num].nodes:
            node=self.mesh.nodes[i_node]
            if node.x >= x_min and node.x <= x_max and node.y >= y_min and node.y <= y_max:
                self.ion_boundary_idx.append(i_node)


    def dopping_on_node(self,type='acceptor'):
        # convert the dopping variable stored in cells to to variable stored in nodes 

        dopping_log=np.zeros(len(self.mesh.nodes))

        if type == 'acceptor':
            dopping_cell = self.dopping_acceptor
        else:
            dopping_cell = self.dopping_donor

        for region in self.mesh.regions:
            if region.type != 'semiconductor':
                continue
            for i_cell in region.cells:
                cell=self.mesh.cells[i_cell]
                dopping_log[cell.nodes] = np.log10(dopping_cell[i_cell])
        
        dopping = 10**dopping_log

        return dopping

    def set_thermal_boundary(self,x_min,x_max,y_min,y_max):
        for i_node,node in enumerate(self.mesh.nodes):
            if node.x >= x_min and node.x <= x_max and node.y >= y_min and node.y <= y_max:
                self.thermal_boundary_node_idx.append(i_node)

    ## 20260310 ion initial profile
    def set_grf_points(self,reg_name):
        reg_num=self.find_region_number(reg_name)
        node_idx = self.mesh.regions[reg_num].nodes
        reg_band_edge_points_list=[]
        for idx in node_idx:
            reg_band_edge_points_list.append([self.mesh.nodes[idx].x,
                                        self.mesh.nodes[idx].y,
                                        self.mesh.nodes[idx].z])
            # self.band_edge_points[idx]=[self.mesh.nodes[idx].x,
            #                             self.mesh.nodes[idx].y,
            #                             self.mesh.nodes[idx].z]
        reg_band_edge_points=np.array(reg_band_edge_points_list)
        return reg_band_edge_points

    # def set_grf_ion_initial_value(self,grf_result,reg_name,x_min=0, x_max=0,y_min=0,y_max=0):
    #     reg_num=self.find_region_number(reg_name)
    #     node_idx = self.mesh.regions[reg_num].nodes
    #     # self.band_edge = grf_result
    #     for i in range(len(node_idx)):
    #         node=self.mesh.nodes[node_idx[i]]
    #         if node.x >= x_min and node.x <= x_max and node.y >= y_min and node.y <= y_max:
    #             value = 0
    #             if grf_result[i] < 24.5:
    #                 value = 1
    #             else:
    #                 value = grf_result[i]
    #             self.ion_density[node_idx[i]] = np.power(10,value)

    def set_grf_ion_initial_value(self,grf_result,reg_name,x_min=0, x_max=0,y_min=0,y_max=0):
        reg_num=self.find_region_number(reg_name)
        node_idx = self.mesh.regions[reg_num].nodes
        # self.band_edge = grf_result
        for i in range(len(node_idx)):
            node=self.mesh.nodes[node_idx[i]]
            if node.x >= x_min and node.x <= x_max and node.y >= y_min and node.y <= y_max:
                value = 0
                value = grf_result[i]
                self.ion_density[node_idx[i]] = np.power(10,value)
                # self.ion_density[node_idx[i]] = np.power(10,grf_result[i])
                # self.ion_density_initial[node_idx[i]] = np.power(10,grf_result[i])

    def set_grf_doping(self,grf_result,reg_name,x_min=0, x_max=0,y_min=0,y_max=0):
        reg_num=self.find_region_number(reg_name)
        node_idx = self.mesh.regions[reg_num].nodes
        # self.band_edge = grf_result
        for i in range(len(node_idx)):
            node=self.mesh.nodes[node_idx[i]]
            if node.x >= x_min and node.x <= x_max and node.y >= y_min and node.y <= y_max:
                self.dopping_donor[node_idx[i]] = np.power(10,grf_result[i])

    def voronoi_info(self):
        if self.voronoi_info_initialized == False:
            self.voronoi_info_initialized = True
            # self.mesh.voronoi_info()
            for edge in self.mesh.edges:
                for i_cell, facet_area, box_volume in zip(edge.cells, edge.facet_area, edge.box_volume): 
                    reg_num = self.mesh.cells[i_cell].region 
                    if reg_num is None:
                        continue
                    region = self.mesh.regions[reg_num]
                    if region.type == 'conductor' or region.type == 'oxide' or region.type == 'semiconductor':
                        edge.facet_area_all += facet_area
                        edge.box_volume_all += box_volume
                    if region.type == 'oxide' or region.type == 'semiconductor':
                        edge.facet_area_semi_ox += facet_area
                        edge.box_volume_semi_ox += box_volume
                    if region.type == 'semiconductor':
                        edge.facet_area_semi += facet_area
                        edge.box_volume_semi += box_volume
            for node in self.mesh.nodes:
                for i_edge in node.edges:
                    edge = self.mesh.edges[i_edge]
                    node.box_volume_semi += edge.box_volume_semi
                    node.box_volume_semi_ox += edge.box_volume_semi_ox
        
    # 
    # export vtk files to see internal variables
    #     the vtk file can be opened by the online tool: https://www.simcapsule.cn/app/clouddesktop/main.html#/ -> "CAE Post"
    #        (TODO: maybe to see the vtk results in jupyternotebook)
    #        https://docs.vtk.org/en/latest/ 
    #        References: https://dash.plotly.com/vtk/structure 
    #               

    def export_vtk(self,foldername,filename,x_scale=1,y_scale=1,z_scale=1):
        # x_scale, y_scale: scale the dimension of the device to view the detials better
        #                   negative scale factor could reverse the x-axis or y-axis 
        filename_pvd=foldername+'/'+filename+'_main.pvd'
        with open(filename_pvd, "w") as f:
            f.write("<?xml version=\"1.0\"?>\n")
            f.write("<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\" compressor=\"vtkZLibDataCompressor\">\n")
            f.write("<Collection>\n")
            for i_region,region in enumerate(self.mesh.regions):
                if len(region.cells)>0:
                    f.write(f"    <DataSet part=\"{i_region}\" file=\""+filename+f"_{i_region}.vtu\" name=\"{region.name}\"/>\n")

            f.write("  </Collection>\n")
            f.write("</VTKFile>\n")
        for i_region,region in enumerate(self.mesh.regions):

            if len(region.cells)<=0:
                continue

            filename_sigment=foldername+'/'+filename+f"_{i_region}.vtu"

            # new_nodes_numbers=[]
            # num_new_node=0

            # for i_node,node in enumerate(self.mesh.nodes):
            #     if i_node in region.nodes:
            #         new_nodes_numbers.append(num_new_node)
            #         num_new_node=num_new_node+1
            #     else:
            #         new_nodes_numbers.append(None)
            local_nodes_idx=np.zeros(len(self.mesh.nodes),dtype=int)-1
            for local_i_node, i_node in enumerate(region.nodes):
                local_nodes_idx[i_node]=local_i_node
                    
            with open(filename_sigment, "w") as f:
                #############################  header information     #################################
                f.write("<?xml version=\"1.0\"?>\n")
                f.write("<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n")
                f.write(" <UnstructuredGrid>\n")

                f.write(f"  <Piece NumberOfPoints=\"{len(region.nodes)}\" NumberOfCells=\"{len(region.cells)}\">\n")

                ############################# Point/Node information  ###############################
                f.write("   <Points>\n")
                f.write("    <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\n");
                # for i_node,node in enumerate(self.mesh.nodes):
                #     if new_nodes_numbers[i_node] is not None:
                #         f.write(str(node.x*x_scale)+" "+str(node.y*y_scale)+" "+str(node.z*z_scale)+'\n')

                for i_node in region.nodes:
                    node = self.mesh.nodes[i_node]
                    f.write(str(node.x*x_scale)+" "+str(node.y*y_scale)+" "+str(node.z*z_scale)+'\n')
                f.write("\n")
                f.write("    </DataArray>\n")
                f.write("   </Points>\n")

                ############################# Cell information       ################################
                f.write("   <Cells>\n")
                f.write("    <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n")
                for i_cell in region.cells:
                    cell=self.mesh.cells[i_cell]
                    for i_node in cell.nodes:
                        f.write(str(local_nodes_idx[i_node])+" ")
                    f.write("\n")
                f.write("\n")
                f.write("    </DataArray>\n")

                # TODO: what is the "offset" means?
                f.write("    <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n")
                for i in range(len(region.cells)):
                    if self.mesh.cell_type=='triangular':
                        f.write(str((i+1)*3)+" ")
                    elif self.mesh.cell_type=='rectangular':
                        f.write(str((i+1)*4)+" ")
                    elif self.mesh.cell_type=='hexahedron':
                        f.write(str((i+1)*8)+" ")
                    elif self.mesh.cell_type=='tetrahedron':
                        f.write(str((i+1)*4)+" ")
                f.write("\n")
                f.write("    </DataArray>\n")
                
                # TODO: what is the "types" means?
                f.write("    <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n")
                for cell in region.cells:
                    if self.mesh.cell_type=='triangular':
                        f.write('5 ')
                    elif self.mesh.cell_type=='rectangular':
                        f.write('9 ')
                    elif self.mesh.cell_type=='hexahedron':
                        f.write('12 ')
                    elif self.mesh.cell_type=='tetrahedron':
                        f.write('10 ')
                f.write("\n")
                f.write("    </DataArray>\n")

                f.write("   </Cells>\n") # Cell inforamtion end

                ############################# Cell data  ################################
                f.write("   <PointData>\n")

                f.write("    <DataArray type=\"Int32\" Name=\"Region ID\" NumberOfComponents=\"1\" format=\"ascii\">\n")
                for i_cell in region.cells:
                    cell=self.mesh.cells[i_cell]
                    f.write(str(cell.region)+' ')
                f.write("\n")
                f.write("    </DataArray>\n")

                f.write("    <DataArray type=\"Float32\" Name=\"Acceptor Doping [log10(m^-3)]\" NumberOfComponents=\"1\" format=\"ascii\">\n")
                if region.type=='semiconductor':
                    for i_node in region.nodes:
                        f.write(str(np.log10(self.dopping_acceptor[i_node]))+' ')
                else:
                    for i_node in region.nodes:
                        f.write(str(0.0)+' ')
                f.write("\n")
                f.write("    </DataArray>\n")

                f.write("    <DataArray type=\"Float32\" Name=\"Donor Doping [log10(m^-3)]\" NumberOfComponents=\"1\" format=\"ascii\">\n")
                if region.type=='semiconductor':
                    for i_node in region.nodes:
                        f.write(str(np.log10(self.dopping_donor[i_node]))+' ')
                else:
                    for i_node in region.nodes:
                        f.write(str(0.0)+' ')
                f.write("\n")
                f.write("    </DataArray>\n")

                f.write("    <DataArray type=\"Float32\" Name=\"Ion Density [log10(m^-3)]\" NumberOfComponents=\"1\" format=\"ascii\">\n")
                if region.type=='semiconductor':
                    for i_node in region.nodes:
                            f.write(str(np.log10(self.ion_density[i_node]))+' ')
                else:
                    for i_node in region.nodes:
                        f.write(str(0.0)+' ')                
                f.write("\n")
                f.write("    </DataArray>\n")

                f.write("   </PointData>\n")
                
                ########################### end of the vtk file  ############################
                f.write("  </Piece>\n")
                f.write(" </UnstructuredGrid>\n")
                f.write("</VTKFile>")
                f.write("\n")
    
    