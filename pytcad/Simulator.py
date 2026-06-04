import numpy as np
import scipy as sp
from scipy.sparse.linalg import spsolve as spsolve
# from pytcad.Assemble import assemble
from pytcad.Assemble_vectorized import assemble
import pytcad.Physics as phys
import pytcad.Quantities as qs
import time
import copy as copy

class Simulator:
    def __init__(self,device,conf):
        self.device=device
        self.device.voronoi_info()
        self.mesh=device.mesh
        self.conf=conf

        # store the voltage biases and built-in potentials for each electrode
        # use the electrode name as the key
        self.voltage_biases={}
        self.electrode_built_in_potential={}
        for electrode_name in list(self.device.electrodes):
            self.voltage_biases[electrode_name]=0.0
            self.electrode_built_in_potential[electrode_name]=0.0

        # convert the dopping variable stored in cells in device to to variable stored in nodes
        self.ion_density = device.ion_density.copy()
        self.ion_density_residual = device.ion_density_residual.copy()
        # self.ion_density_initial = device.ion_density_initial.copy()
        # self.ion_density_residual = 0.0
        self.max_ion_density_residual = device.max_ion_density_residual.copy()+device.max_residual_ion_value

        self.ion_density_max = np.zeros(len(self.mesh.nodes))+1e-30
        self.ion_density_max_all = np.zeros(len(self.mesh.nodes))+1e-30

        # link ion density to device ion doping on nodes
        self.ND=device.dopping_donor + 2*(self.ion_density + self.ion_density_residual)
        self.NA=device.dopping_acceptor

        self.time_step = 0.001
        self.ion_density_last_step = None
        # self.ion_density_last_step = copy.deepcopy(self.ion_density)
        self.ion_boundary_idx=self.device.ion_boundary_idx

        self.ref_material=self.device.ref_material

        # initialize unknowns that can be solved
        # solve of some variables can be turned off by setting in conf
        #    for instance, conf['with_hole']=False will not solve hole density
        #                  conf['with_temperature']=False will not solve temperature
        self.quans=[]
        potential=self.initialize_potential()
        quan={'unknown':True,
            'name':'potential',
            'value':potential,
            'l2_norm':[],
            'vectrize_indices': None
            }
        self.quans.append(quan)

        quan={'unknown':self.conf['with_electrode_voltage'],
            'name':'electrode_voltage',
            'value':np.zeros(len(self.mesh.nodes)),
            'l2_norm':[],
            'vectrize_indices': None}
        self.quans.append(quan)

        n=self.initialize_concentration(type='electron')
        quan={'unknown':self.conf['with_electron'],
            'name':'electron_density',
            'value':n,
            'l2_norm':[],
            'vectrize_indices': None
            }
        self.quans.append(quan)

        p=self.initialize_concentration(type='hole')
        quan={'unknown':self.conf['with_hole'],
            'name':'hole_density',
            'value':p,
            'l2_norm':[],
            'vectrize_indices': None
            }
        self.quans.append(quan)

        T=self.initialize_temperature()
        quan={'unknown':self.conf['with_temperature'],
            'name':'temperature',
            'value':T,
            'l2_norm':[],
            'vectrize_indices': None
            }
        self.quans.append(quan)


        ion_density=self.ion_density
        quan={'unknown':self.conf['with_ion'],
            'name':'ion_density',
            'value':ion_density,
            'l2_norm':[],
            'vectrize_indices': None
            }
        self.quans.append(quan)

        # map the unknowns to the matrix A and vector b
        # indexing information for potential, electron density, hole density, temperature
        self.mapping_info, self.num_unknowns, self.num_unknowns_per_quan=self.mapping()

        
        if self.conf['method']=='newton':
            # scale the coefficients of carrier density in Matrix A for Newton iterations 
            self.scaling_factor=np.ones(self.num_unknowns)
            for quan in self.quans:
                map_idx,_,_ = self.mapping_info[quan['name']]
                if quan['name'] == 'electron_density' or quan['name'] == 'hole_density' or quan['name'] == 'ion_density':
                    self.scaling_factor[map_idx]=1e16
        
        # indexing infomation for ion density drift-diffusion equation
        # self.mapping_info_ion, self.num_unknowns_ion=self.mapping_ion()

    def mapping(self):
        mapping_info={}
        num_unknowns_per_quan={}
        num_unknowns=0
        idx=0
        for quan in self.quans:
            mapping_idx=[]      # index of unknowns in the Matrix A
            mapping_i_node=[]   # index of nodes in the mesh conresponding to the index of unknowns
            mapping_idx_of_node=[] # index of unknowns in the Matrix with the order of index of nodes in the mesh, if the node is not unknowns, set None
            if self.conf['method'] == 'gummel':
                idx=0
            if quan['name']=='potential':
                non_electrode_node_idx=list((set(self.node_idx_of_material('semiconductor'))|set(self.node_idx_of_material('oxide')))-set(self.node_idx_of_material('conductor')))
                for i_node,node in enumerate(self.mesh.nodes):
                    if i_node in non_electrode_node_idx:
                        mapping_idx.append(idx)
                        mapping_i_node.append(i_node)
                        mapping_idx_of_node.append(idx)
                        idx=idx+1
                    else:
                        mapping_idx_of_node.append(None)

            if quan['name']=='electrode_voltage' and quan['unknown']:
                # for electrode connected with external compliance circuit, we need to include the nodes in the intersection boundary with semiconductor to calculate the contact current and update the voltage bias on the electrode
                contact_name = self.conf['compliance_electrode']

                electrode_region=self.mesh.regions[self.device.electrodes[contact_name]]
                electrode_node_idx = set(electrode_region.nodes)
                semi_node_idx = self.node_idx_of_material('semiconductor')
                intersection_node_idx = list(set(electrode_node_idx) & set(semi_node_idx))

                # all nodes in the intersection boundary will have the same potential changes (biased by built-in potential), so we only need to include one of them in the unknowns, and use the same index in the matrix A to solve the potential at these nodes.
                mapping_idx.append(idx)
                for i_node,node in enumerate(self.mesh.nodes):
                    if i_node in intersection_node_idx:
                        mapping_i_node.append(i_node)
                        mapping_idx_of_node.append(idx)
                    else:
                        mapping_idx_of_node.append(None)
                idx=idx+1

            if quan['name']=='electron_density' and quan['unknown']:
                #202506-jwf, include semiconductor and oxide inface, except semiconductor and conductor interface 
                pure_semi_node_idx=list(set(self.node_idx_of_material('semiconductor'))-set(self.node_idx_of_material('conductor')))
                for i_node,node in enumerate(self.mesh.nodes):
                    if i_node in pure_semi_node_idx:
                        mapping_idx.append(idx)
                        mapping_i_node.append(i_node)
                        mapping_idx_of_node.append(idx)
                        idx=idx+1
                    else:
                        mapping_idx_of_node.append(None)
            if quan['name']=='hole_density'  and quan['unknown']:
                pure_semi_node_idx=list(set(self.node_idx_of_material('semiconductor'))-set(self.node_idx_of_material('conductor')))
                for i_node,cell in enumerate(self.mesh.nodes):
                    if i_node in pure_semi_node_idx:
                        mapping_idx.append(idx)
                        mapping_i_node.append(i_node)
                        mapping_idx_of_node.append(idx)
                        idx=idx+1
                    else:
                        mapping_idx_of_node.append(None)

            if quan['name']=='temperature' and quan['unknown']:
                if not self.conf['with_conductor_temperature']:
                    non_electrode_node_idx=list((set(self.node_idx_of_material('semiconductor'))|set(self.node_idx_of_material('oxide')))-set(self.node_idx_of_material('conductor')))
                    for i_node,node in enumerate(self.mesh.nodes):
                        if i_node in non_electrode_node_idx:
                            mapping_idx.append(idx)
                            mapping_i_node.append(i_node)
                            mapping_idx_of_node.append(idx)
                            idx=idx+1
                        else:
                            mapping_idx_of_node.append(None)
                else:
                    all_temperature_node_idx=list((set(self.node_idx_of_material('semiconductor'))|set(self.node_idx_of_material('oxide')))|set(self.node_idx_of_material('conductor'))-set(self.device.thermal_boundary_node_idx))
                    for i_node,node in enumerate(self.mesh.nodes):
                        if i_node in all_temperature_node_idx:
                            mapping_idx.append(idx)
                            mapping_i_node.append(i_node)
                            mapping_idx_of_node.append(idx)
                            idx=idx+1
                        else:
                            mapping_idx_of_node.append(None)
            if quan['name']=='ion_density' and quan['unknown']:
                 ## only pure semiconductor nodes, except conductor interface
                # ion_node_idx=list(set(self.node_idx_of_material('semiconductor'))-set(self.node_idx_of_material('conductor')))

                # ion_node_idx=list(set(self.node_idx_of_material('semiconductor'))-set(self.mesh.regions[2].nodes))  # exclude the nodes in region 2 (the top electrode/hfox interface)
                ion_node_idx=list(set(self.node_idx_of_material('semiconductor'))) 

                for i_node,node in enumerate(self.mesh.nodes) :
                    # if i_node in ion_node_idx :
                    if (i_node in ion_node_idx) and (i_node not in self.ion_boundary_idx):
                        mapping_idx.append(idx)
                        mapping_i_node.append(i_node)
                        mapping_idx_of_node.append(idx)
                        idx=idx+1
                    else:
                        mapping_idx_of_node.append(None)
            mapping_info[quan['name']]=[mapping_idx,mapping_i_node,mapping_idx_of_node]  
            num_unknowns_per_quan[quan['name']]=len(mapping_idx)
            num_unknowns+=len(mapping_idx)
        return mapping_info, num_unknowns, num_unknowns_per_quan
    
    def mapping_ion(self):
        idx=0
        mapping_idx=[]      # index of unknowns in the Matrix A
        mapping_i_node=[]   # index of nodes in the mesh conresponding to the index of unknowns
        mapping_idx_of_node=[] # index of unknowns in the Matrix with the order of index of nodes in the mesh, if the node do not have index, set None
        
        ## only pure semiconductor nodes, except conductor interface
        ion_node_idx=list(set(self.node_idx_of_material('semiconductor'))-set(self.node_idx_of_material('conductor')))

        ## all semiconductor nodes, include semiconductor and oxide interface
        # works for reset
        # ion_node_idx = self.node_idx_of_material('semiconductor')

        ## exclude the nodes in region 2 (the top electrode/hfox interface)
        # for set
        # ion_node_idx = list(set(self.node_idx_of_material('semiconductor'))-set(self.mesh.regions[2].nodes))  
        for i_node,node in enumerate(self.mesh.nodes):
            if i_node in ion_node_idx:
                mapping_idx.append(idx)
                mapping_i_node.append(i_node)
                mapping_idx_of_node.append(idx)
                idx=idx+1
            else:
                mapping_idx_of_node.append(None)
            
        mapping_info=[mapping_idx,mapping_i_node,mapping_idx_of_node]  
        num_unknowns=len(mapping_idx)
        return mapping_info, num_unknowns
    
    def node_idx_of_material(self,material_type='semiconductor'):
        node_idx=[]
        for region in self.mesh.regions:
            if region.type==material_type:
                node_idx=list(set(node_idx)|set(region.nodes))
        return node_idx
    
    def initialize_potential(self):
        built_in_all=self.built_in_potential()
        potential=built_in_all.copy()
        return potential
    
    def initialize_concentration(self,type='electron'):
        quan_value=np.ones(len(self.mesh.nodes))
        for reg_num,region in enumerate(self.mesh.regions):
            if region.type=='semiconductor':
                node_idx=region.nodes
                quan_value[node_idx]=phys.equilibrium_concentration(self.ND[node_idx],self.NA[node_idx],self.mesh.regions[reg_num].material,type=type)
        return quan_value


    def initialize_temperature(self):
        quan_value=np.ones(len(self.mesh.nodes))*phys.constant['T0']         
        return quan_value
    
    def set_contact_voltage(self,contact_name='drain',voltage=0.1):
        # set the new voltage 
        self.voltage_biases[contact_name]=voltage
        
        # update Poisson boundary conditions
        potential=qs.get_potential(self)
        # for reg_num,region in enumerate(self.mesh.regions):
        #     node_idx=region.nodes
        #     if region.type=='conductor':
        #         vapplied = self.voltage_biases[self.mesh.regions[reg_num].electrode_name]
        #         built_in = self.electrode_built_in_potential[self.mesh.regions[reg_num].electrode_name]
        #         potential[node_idx]=vapplied+built_in
        built_in = self.built_in_potential().copy()
        for reg_num,region in enumerate(self.mesh.regions):
            if region.type=='conductor':
                node_idx=region.nodes
                vapplied = self.voltage_biases[self.mesh.regions[reg_num].electrode_name]
                potential[node_idx]=vapplied+built_in[node_idx]
        return potential

    # def set_contact_voltage(self,contact_name='drain',voltage=0.1):
    #     # set the new voltage 
    #     self.voltage_biases[contact_name]=voltage
        
    #     # update Poisson boundary conditions
    #     potential=qs.get_potential(self)
    #     # for reg_num,region in enumerate(self.mesh.regions):
    #     #     node_idx=region.nodes
    #     #     if region.type=='conductor':
    #     #         vapplied = self.voltage_biases[self.mesh.regions[reg_num].electrode_name]
    #     #         built_in = self.electrode_built_in_potential[self.mesh.regions[reg_num].electrode_name]
    #     #         potential[node_idx]=vapplied+built_in
    #     built_in = self.built_in_potential().copy()
    #     for reg_num,region in enumerate(self.mesh.regions):
    #         if region.type=='conductor':
    #             node_idx=region.nodes
    #             vapplied = 0
    #             if self.mesh.regions[reg_num].electrode_name is None:
    #                 shared_nodes_idx = []
    #                 for reg_num2,region2 in enumerate(self.mesh.regions):
    #                     if region2.type=='conductor':
    #                         shared_nodes_idx=list(set(node_idx).intersection(set(region2.nodes))|set(shared_nodes_idx))
    #                     if len(shared_nodes_idx)>0:
    #                         vapplied = self.voltage_biases[self.mesh.regions[reg_num2].electrode_name]
    #             else:
    #                 vapplied = self.voltage_biases[self.mesh.regions[reg_num].electrode_name]
    #             potential[node_idx]=vapplied+built_in[node_idx]
    #     return potential
 
    def run(self):
        # if self.conf['with_hysteresis']:
        #     self.last_heiman_trap_occupancy[:] = self.heiman_trap_occupancy

        for iter in range(self.conf['max_iter']):
            if iter==0:
                print('')
                for electrode_name in list(self.device.electrodes):
                    print('V'+electrode_name+f'={self.voltage_biases[electrode_name]}',end=' ')
                print('')
                if self.conf['show_details']:
                    print("Number of nodes:", len(self.device.mesh.nodes))
                    for quan in self.quans:
                        if not quan['name'] == 'ion_density':
                            print("Number of unknowns for "+quan['name']+f": {self.num_unknowns_per_quan[quan['name']]}")
                        else:
                            print("Number of unknowns for "+quan['name']+f": {self.num_unknowns_ion}")
                    
                print("iter number",end='')
                for quan in self.quans:
                    if quan['unknown'] and quan['name']!='ion_density':
                        print(" || L2 Norm of "+quan['name'],end='')
                print(' || time ')

            start_iter = time.time()
            
            if self.conf['method']=='newton':
                A=sp.sparse.lil_array((self.num_unknowns,self.num_unknowns))
                b=np.zeros((self.num_unknowns))

                for quan in self.quans:
                    if quan['unknown'] == False or quan['name']=='ion_density':
                        continue
                    start = time.time()
                    [A,b]=assemble(self,quan,A,b)
                    # print('assemble '+quan['name']+' time ', time.time()-start, 's')
                # start = time.time()
                dQ=self.solve(A,b)
                # print('solve matrix time ', time.time()-start, 's')
                # self.residual = np.array(A @ dQ)-b.reshape(-1)
                
                
                for quan in self.quans:
                    if quan['unknown']==False or quan['name']=='ion_density':
                        continue
                    self.update(quan,dQ)
                
            else:
                for quan in self.quans:
                    if quan['unknown'] == False:
                        continue
                    num_unknowns=self.num_unknowns_per_quan[quan['name']]
                    A=sp.sparse.lil_array((num_unknowns,num_unknowns))
                    b=np.zeros((num_unknowns,1))

                    [A,b]=assemble(self,quan,A,b)

                    # start = time.time()
                    dQ=self.solve(A,b)
                    
                    self.update(quan,dQ)
                

            resolution_reached=True
            for quan in self.quans:    
                if quan['unknown'] == False or quan['name']=='ion_density':
                    continue 
                if quan['l2_norm'][-1]>self.conf['l2_resolution']:
                    resolution_reached=False
            # resolution_reached=True
            # for quan in self.quans:    
            #     if (quan['name'] == 'potential') and (quan['l2_norm'][-1]>self.conf['l2_resolution']):
            #         resolution_reached=False

            if self.conf['show_details']:
                print(iter+1,end='')
                for quan in self.quans:
                    if quan['unknown'] == False or quan['name']=='ion_density':
                        continue
                    print(f' || {quan["l2_norm"][-1]:.4e}', end='')
                end_iter = time.time()
                print(' || ', end_iter-start_iter, end='s')
                print('')

            if resolution_reached:
                converged = True
                return converged
                
        if self.conf['show_details']==False:
            print(iter+1, end='')
            for quan in self.quans:
                if quan['unknown'] and quan['name']!='ion_density':
                    print(f" || {quan['l2_norm'][-1]:.4e}", end='')
            end_iter = time.time()
            print(' || ', end_iter-start_iter, end='s')
            print('')

        self.iter=iter+1

        converged = False
        return converged
    
    def run_all(self, update_ion = True):
        
        self.ion_density_last_step = copy.deepcopy(self.ion_density)
        # if self.conf['with_hysteresis']:
        #     self.last_heiman_trap_occupancy[:] = self.heiman_trap_occupancy
        converged = False
        for iter in range(self.conf['max_iter']):
            if iter==0:
                print('')
                for electrode_name in list(self.device.electrodes):
                    print('V'+electrode_name+f'={self.voltage_biases[electrode_name]}',end=' ')
                print('')
                if self.conf['show_details']:
                    print("Number of nodes:", len(self.device.mesh.nodes))
                    for quan in self.quans:
                        print("Number of unknowns for "+quan['name']+f": {self.num_unknowns_per_quan[quan['name']]}")
                    
                print("iter number",end='')
                for quan in self.quans:
                    if quan['unknown']:
                        print(" || L2 Norm of "+quan['name'],end='')
                print(' || time ')

            start_iter = time.time()
            
            if self.conf['method']=='newton':
                A=sp.sparse.lil_array((self.num_unknowns,self.num_unknowns))
                b=np.zeros((self.num_unknowns))

                for quan in self.quans:
                    if quan['unknown'] == False:
                        continue
                    start = time.time()
                    [A,b]=assemble(self,quan,A,b)
                    # print('assemble '+quan['name']+' time ', time.time()-start, 's')
                # start = time.time()
                dQ=self.solve(A,b)
                # print('solve matrix time ', time.time()-start, 's')
                # self.residual = np.array(A @ dQ)-b.reshape(-1)
                
                
                for quan in self.quans:
                    if quan['unknown']==False:
                        continue
                    if quan['name']=='ion_density' and update_ion==False:
                        continue
                    self.update(quan,dQ) 

            # if self.conf['compliance']:
            #     Jn, Jp = qs.get_contact_current(self,contact_name='TE')
            #     print(f'Total contact current TE: {Jn+Jp} A')

            #     vte = self.voltage_biases['TE']
            #     vapplied = self.vapplied
            #     vtargert = (vapplied - abs(Jn+Jp)*self.conf['serial_resistance'])
            #     dv = vtargert - vte
            #     vte_new = vte + self.conf['damping']*dv
            #     # vte_new = vte + 0.3*dv
            #     print(f'Voltage across the device: {vte_new} V')
            #     self.set_contact_voltage('TE', vte_new)
                

            resolution_reached=True
            for quan in self.quans:    
                if quan['unknown'] == False:
                    continue 
                if quan['l2_norm'][-1]>self.conf['l2_resolution']:
                    resolution_reached=False
                if np.isnan(quan['l2_norm'][-1]):
                    resolution_reached=False
                    raise ValueError(f'L2 norm of {quan["name"]} is NaN, which may be caused by divergence.')
            # resolution_reached=True
            # for quan in self.quans:    
            #     if (quan['name'] == 'potential') and (quan['l2_norm'][-1]>self.conf['l2_resolution']):
            #         resolution_reached=False

            if self.conf['show_details']:
                print(iter+1,end='')
                for quan in self.quans:
                    if quan['unknown'] == False:
                        continue
                    print(f' || {quan["l2_norm"][-1]:.4e}', end='')
                end_iter = time.time()
                print(' || ', end_iter-start_iter, end='s')
                print('')

            if resolution_reached:
                converged = True
                break
                
        if self.conf['show_details']==False:
            print(iter+1, end='')
            for quan in self.quans:
                if quan['unknown'] and quan['name']!='ion_density':
                    print(f" || {quan['l2_norm'][-1]:.4e}", end='')
            end_iter = time.time()
            print(' || ', end_iter-start_iter, end='s')
            print('')

        self.iter=iter+1

        return converged

    def run_ion(self):
        for iter in range(self.conf['max_iter']):
            if iter==0:
                    
                print("iter number",end='')
                for quan in self.quans:
                    if quan['name']=='ion_density' and quan['unknown']:
                        print(" || L2 Norm of "+quan['name'],end='')
                print(' || time ')

            start_iter = time.time()

            for quan in self.quans:
                if quan['unknown'] and quan['name']=='ion_density':
                    A = sp.sparse.lil_array((self.num_unknowns_ion,self.num_unknowns_ion))
                    b = np.zeros((self.num_unknowns_ion))
                    start = time.time()
                    [A,b]=assemble(self,quan,A,b)
                    # print('assemble '+quan['name']+' time ', time.time()-start, 's')

                    dQ=self.solve_ion(A,b)
                    self.update_ion(quan,dQ)

            resolution_reached=True
            for quan in self.quans:    
                if quan['name']!='ion_density' or quan['unknown']==False:
                    continue 
                if quan['l2_norm'][-1]>self.conf['l2_resolution']:
                    resolution_reached=False

            if self.conf['show_details']:
                print(iter+1,end='')
                for quan in self.quans:
                    if quan['name']!='ion_density' or quan['unknown'] == False:
                        continue
                    print(f' || {quan["l2_norm"][-1]:.4e}', end='')
                end_iter = time.time()
                print(' || ', end_iter-start_iter, end='s')
                print('')

            if resolution_reached:
                break
            
        if self.conf['show_details']==False:
            print(iter+1, end='')
            for quan in self.quans:
                if quan['unknown'] and quan['name']=='ion_density':
                    print(f" || {quan['l2_norm'][-1]:.4e}", end='')
            end_iter = time.time()
            print(' || ', end_iter-start_iter, end='s')
            print('')

        self.ion_density_last_step = self.ion_density.copy()
    
    def solve(self,A,b):
        if self.conf['method']=='newton':
            A=A*np.broadcast_to(np.array(self.scaling_factor).T,A.shape)    

        A=A.tocsr()        
        dQ=spsolve(A,b)

        if self.conf['method']=='newton':
            dQ=dQ*np.array(self.scaling_factor)
        return dQ
    
    def solve_ion(self,A,b):
        A=A.tocsr()        
        dQ=spsolve(A,b)
        return dQ
    
    def update(self,quan,dQ):
        damping=self.conf['damping']
        
        Q=quan['value']
        
        damping_type='linear'
        if quan['name'] == 'electron_density' or quan['name'] == 'hole_density' or quan['name'] == 'ion_density':
            damping_type='log'

        map_idx,map_i_node,_=self.mapping_info[quan['name']]
        if quan['name'] == 'electrode_voltage':
            damping = self.conf['damping_electrode_voltage']

        if damping_type=='linear':
            Q[map_i_node]=Q[map_i_node]+damping*dQ[map_idx]
            quan['l2_norm'].append(np.linalg.norm(dQ[map_idx])/len(dQ[map_idx]))
                
        elif damping_type=='log':
            dQ_Q=dQ[map_idx]/Q[map_i_node]
            scale=1+dQ_Q
            scale[scale<1e-1]=1e-1
            scale[scale>1e1]=1e1
            Q[map_i_node]=Q[map_i_node]*(scale**damping)
            # Q[Q<1e-31]=1e-31
            quan['l2_norm'].append(np.linalg.norm(np.abs(dQ_Q))/len(dQ_Q))
            # quan['l2_norm'].append(np.linalg.norm(np.log10(scale))/len(dQ_Q))

        # if quan['name']=='temperature':
        #     T=Q
        #     T[T<phys.constant['T0']]=phys.constant['T0']
        #     T[T>1000]=1000
        #     Q=T
        if quan['name'] == 'ion_density':
            self.ion_density = Q.copy()
            # self.ND = 2*self.ion_density

            # Forming ion residual
            self.ion_density_residual = np.maximum(self.ion_density*self.device.residual_ion_percentage, self.ion_density_residual)
            self.ion_density_residual = np.minimum(self.ion_density_residual, self.max_ion_density_residual)
            # self.ion_density_max = np.maximum(self.ion_density, self.ion_density_max)
            self.ion_density_max_all = np.maximum(self.ion_density, self.ion_density_max_all)
            # self.ion_density_max = np.maximum(self.ion_density, self.ion_density_max).clip(max=1e24)
            self.ion_density_max = np.maximum(self.ion_density, self.ion_density_max)
            self.ND = self.device.dopping_donor + 2*(self.ion_density + self.ion_density_residual)

            ## Forming Ea
            # self.ion_density = np.maximum(self.ion_density_initial*0.1, self.ion_density)  ##failed

            # self.ion_density_max = np.maximum(self.ion_density, self.ion_density_max)
            # self.ND = self.device.dopping_donor + 2*self.ion_density

            # update the equilibrim carrier concentration for semicondcutor/conductor interface nodes
            inter_node_idx = list(set(self.node_idx_of_material('conductor')) & set(self.node_idx_of_material('semiconductor')))
            eqi_n = self.initialize_concentration('electron')
            n = qs.get_electron_density(self)
            n[inter_node_idx] = eqi_n[inter_node_idx]

            self.set_contact_voltage('BE',0.0) # update the voltage bias on the bottom electrode to make sure it is grounded, which is important  when ion density changes a lot and cause large change of built-in potential

        if quan['name']=='electrode_voltage':
            v_electrode = np.mean(Q[map_i_node])
            if self.conf.get('compliance_voltage_neg',False):
                if v_electrode < self.vapplied:
                    v_electrode = self.vapplied
                if v_electrode > 0.0:
                    v_electrode = 0.0                
            else:
                if v_electrode>self.vapplied:
                    v_electrode = self.vapplied
                if v_electrode<0.0:
                    v_electrode = 0.0
            self.set_contact_voltage(contact_name=self.conf['compliance_electrode'], voltage=v_electrode)
            if self.conf['compliance_electrode_near'] is not None:
                self.set_contact_voltage(contact_name=self.conf['compliance_electrode_near'], voltage=v_electrode)
    # def update_applied_voltage(self):
    #     Jn,Jp = qs.get_contact_current(self,contact_name='TE')
    #     current = abs(Jn+Jp)
    #     v_te = self.voltage_biases['TE']
    #     resistance = v_te/current
    #     serial_resistor = self.device.serial_resistor
    #     if self.conf['compliance']:
    #         v_te = self.v*resistance/(resistance+serial_resistor)
    #         self.set_contact_voltage('TE',v_te)
    #     print(f'Applied voltage updated to {v_te:.4f} V due to serial resistor drop.')
    
    def update_ion(self,quan,dQ):
        damping=self.conf['damping_ion']
        Q=quan['value']
        
        map_idx,map_i_node,_=self.mapping_info_ion
        dQ_Q=dQ[map_idx]/Q[map_i_node]
        scale=1+dQ_Q
        scale[scale<1e-5]=1e-5
        Q[map_i_node]=Q[map_i_node]*(scale**damping)
        quan['l2_norm'].append(np.linalg.norm(np.abs(dQ_Q))/len(dQ_Q))

        self.ion_density = Q.copy()
        self.ND = 2*self.ion_density


    def get_contact_current(self,contact_name):
        return qs.get_contact_current(self,contact_name)

    def built_in_potential(self):
        # calculate the built-in potential on each node
        # obtain the built-in potential for electrode and store them in self.electrode_built_in_potential for later use when set contact voltage
        
        potential=np.zeros(len(self.mesh.nodes))         # potential on each node

        for reg_num,region in enumerate(self.mesh.regions):
            #  for conductor region, built-in potential is the average built-in potential of the shared nodes with semiconductor or oxide regions
            if region.type=='conductor':
                node_idx=region.nodes
                # shared_nodes_idx = []
                # for region2 in self.mesh.regions:
                #     if region2.type=='semiconductor':
                #         shared_nodes_idx=list(set(node_idx).intersection(set(region2.nodes))|set(shared_nodes_idx))
                # if len(shared_nodes_idx)>0:
                #     built_in=potential[shared_nodes_idx].mean()
                # else:
                #     built_in=0
                #     for region2 in self.mesh.regions:
                #         if region2.type=='oxide':
                #             shared_nodes_idx=list(set(node_idx).intersection(set(region2.nodes))|set(shared_nodes_idx))
                #     if len(shared_nodes_idx)>0:
                #         built_in=potential[shared_nodes_idx].mean()
                #     else:
                #         raise ValueError('Conductor region should be in contact with semiconductor or oxide region to define built-in potential!')
                potential[node_idx]=0.0
                # self.electrode_built_in_potential[region.electrode_name]=built_in
    
        for reg_num,region in enumerate(self.mesh.regions):
            #   for semiconductor region, built-in potential is calculated based on doping and material properties
            if region.type=='semiconductor':
                node_idx=region.nodes
                built_in=phys.built_in_potential(self.ND[node_idx],self.NA[node_idx],self.mesh.regions[reg_num].material,self.ref_material)
                potential[node_idx]=built_in

        for reg_num,region in enumerate(self.mesh.regions):
            #   for oxide region, built-in potential is the average built-in potential of the shared nodes with semiconductor regions
            if region.type=='oxide':
                node_idx=region.nodes
                shared_nodes_idx = []
                for region2 in self.mesh.regions:
                    if region2.type=='semiconductor':
                        shared_nodes_idx=list(set(node_idx).intersection(set(region2.nodes))|set(shared_nodes_idx))
                if len(shared_nodes_idx)>0:
                    built_in=potential[shared_nodes_idx].mean()
                else:
                    raise ValueError('Oxide region should be in contact with semiconductor region to define built-in potential!')
                
                pure_oxide_node_idx = list(set(node_idx)-set(self.node_idx_of_material('semiconductor')))
                potential[pure_oxide_node_idx]=built_in

        # for reg_num,region in enumerate(self.mesh.regions):
        #     #  for conductor region, built-in potential is the average built-in potential of the shared nodes with semiconductor or oxide regions
        #     if region.type=='conductor':
        #         node_idx=region.nodes
        #         shared_nodes_idx = []
        #         for region2 in self.mesh.regions:
        #             if region2.type=='semiconductor':
        #                 shared_nodes_idx=list(set(node_idx).intersection(set(region2.nodes))|set(shared_nodes_idx))
        #         if len(shared_nodes_idx)>0:
        #             built_in=potential[shared_nodes_idx].mean()
        #         else:
        #             built_in=0
        #             for region2 in self.mesh.regions:
        #                 if region2.type=='oxide':
        #                     shared_nodes_idx=list(set(node_idx).intersection(set(region2.nodes))|set(shared_nodes_idx))
        #             if len(shared_nodes_idx)>0:
        #                 built_in=potential[shared_nodes_idx].mean()
        #             else:
        #                 raise ValueError('Conductor region should be in contact with semiconductor or oxide region to define built-in potential!')
        #         potential[node_idx]=built_in
        #         self.electrode_built_in_potential[region.electrode_name]=built_in
        return potential


    def set_ion_density_tracing(self):
        self.ion_density_max = copy.deepcopy(self.ion_density)