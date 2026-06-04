import numpy as np
import scipy as sp
from pytcad.Physics import constant as cons
import pytcad.Physics as phys
import pytcad.Quantities as qs
import time

def assemble(sim,quan,A,b):
    if quan['name'] == 'potential':
        [A,b]=assemble_possion(sim,quan,A,b)
    elif quan['name'] == 'electron_density' or quan['name'] == 'hole_density':
        [A,b]=assemble_dd(sim,quan,A,b)
    elif quan['name'] == 'electrode_voltage':
        [A,b]=assemble_electrode_voltage(sim,quan,A,b)
    elif quan['name'] == 'temperature':
        [A,b]=assemble_heat(sim,quan,A,b)  
    elif quan['name'] == 'ion_density':
        [A,b]=assemble_ion(sim,quan,A,b)
    return A,b  

def assemble_possion(sim,quan,A,b):

    # start = time.time()
    potential=qs.get_potential(sim)
    n=qs.get_electron_density(sim)
    p=qs.get_hole_density(sim)
    T=qs.get_temperature(sim)
    epsilon=qs.get_epsilon(sim)  # epsilon is evaluated on edges
    # print(' time1 ', time.time() -start, 's')
    if quan['vectrize_indices'] is None:
        # print(' time1 ', time.time() -start, 's')
        # start = time.time()
        map_idx,map_i_node,map_idx_of_node=sim.mapping_info[quan['name']]
        _,map_i_node_carrier_e,map_idx_of_node_e=sim.mapping_info['electron_density']
        _,map_i_node_carrier_h,map_idx_of_node_h=sim.mapping_info['hole_density']

        i_indices = map_idx
        i_nodes = map_i_node
        i_node_cp_eh = []  # nodes that are coupled with carrier density
        i_indice_cp_eh = []
        j_indice_cp_e = []
        j_indice_cp_h = []
        for i_idx,i_node in zip(map_idx,map_i_node):
            if sim.conf['method']=='newton':      
                if (i_node in map_i_node_carrier_e) or (i_node in map_i_node_carrier_h):
                    i_node_cp_eh.append(i_node)
                    i_indice_cp_eh.append(i_idx)
                if sim.conf['with_electron'] and (i_node in map_i_node_carrier_e):
                    j_idx=map_idx_of_node_e[i_node]
                    j_indice_cp_e.append(j_idx)
                if sim.conf['with_hole'] and (i_node in map_i_node_carrier_h):
                    j_idx=map_idx_of_node_h[i_node]
                    j_indice_cp_h.append(j_idx)
        box_volume_nodes = []

        i_indices_non_diagonal = []
        j_indices_non_diagonal = []
        eps_non_diagonal = []
        facet_area_non_diagonal = []
        edge_lengths_non_diagonal = []
        i_indices_diagonal = []
        j_indices_diagonal = []
        i_nodes_diagonal = []
        j_nodes_diagonal = []
        eps_diagonal = []
        facet_area_diagonal = []
        edge_lengths_diagonal = []
        
        i_idx_diagonal = 0
        for i_idx,i_node in zip(map_idx,map_i_node): # loop for all nodes
            node=sim.mesh.nodes[i_node]

            j_idx_diagonal = 0
            for i_edge, i_other_node in zip(node.edges, node.other_nodes):
                edge = sim.mesh.edges[i_edge]
                eps=epsilon[i_edge]*cons['Epsilon0']
                facet_area = edge.facet_area_semi_ox
                
                if i_other_node in map_i_node:
                    j_idx=map_idx_of_node[i_other_node]
                    i_indices_non_diagonal.append(i_idx)
                    j_indices_non_diagonal.append(j_idx)
                    eps_non_diagonal.append(eps)
                    facet_area_non_diagonal.append(facet_area)
                    edge_lengths_non_diagonal.append(edge.length)
                
                # i_indices_diagonal.append(i_idx)
                i_indices_diagonal.append(i_idx_diagonal)
                j_indices_diagonal.append(j_idx_diagonal) # this index is faked, becasue acctual j_idx might not exist for the case that the other node is not in the potential mappping; we just need the summation over all j indices to fill in the diagonal position
                j_idx_diagonal += 1  # Increment for diagonal index
                eps_diagonal.append(eps)
                facet_area_diagonal.append(facet_area)
                edge_lengths_diagonal.append(edge.length)
                i_nodes_diagonal.append(i_node)
                j_nodes_diagonal.append(i_other_node)
            i_idx_diagonal += 1
            if (i_node in map_i_node_carrier_e) or (i_node in map_i_node_carrier_h):
                box_volume_nodes.append(node.box_volume_semi)

        box_volume_nodes = np.array(box_volume_nodes)
        eps_non_diagonal = np.array(eps_non_diagonal)
        facet_area_non_diagonal = np.array(facet_area_non_diagonal)
        edge_lengths_non_diagonal = np.array(edge_lengths_non_diagonal)
        eps_diagonal = np.array(eps_diagonal)
        facet_area_diagonal = np.array(facet_area_diagonal)
        edge_lengths_diagonal = np.array(edge_lengths_diagonal)
        # print(' time1 ', time.time()-start, 's')
        quan['vectrize_indices'] = [i_indices, i_nodes, i_node_cp_eh, i_indice_cp_eh, j_indice_cp_e, j_indice_cp_h, box_volume_nodes,
                                 i_indices_non_diagonal, j_indices_non_diagonal,
                                 eps_non_diagonal, facet_area_non_diagonal,
                                 edge_lengths_non_diagonal, i_indices_diagonal,
                                 j_indices_diagonal, i_nodes_diagonal, j_nodes_diagonal,
                                 eps_diagonal, facet_area_diagonal, edge_lengths_diagonal]
    else:
        (i_indices, i_nodes, i_node_cp_eh, i_indice_cp_eh, j_indice_cp_e, j_indice_cp_h, box_volume_nodes,
         i_indices_non_diagonal, j_indices_non_diagonal, eps_non_diagonal, 
         facet_area_non_diagonal, edge_lengths_non_diagonal, i_indices_diagonal, 
         j_indices_diagonal, i_nodes_diagonal, j_nodes_diagonal, eps_diagonal, 
         facet_area_diagonal, edge_lengths_diagonal) = quan['vectrize_indices']

    # start = time.time()
    charge_density=cons['q']*(-1.0*n[i_node_cp_eh]+1.0*p[i_node_cp_eh]+1.0*sim.ND[i_node_cp_eh]-1.0*sim.NA[i_node_cp_eh]) 
    b[i_indice_cp_eh] = -(charge_density * box_volume_nodes)

    # gummel contribution
    if sim.conf['method']=='gummel':
        gummel_contrib = box_volume_nodes*cons['q']*(n[i_node_cp_eh]+p[i_node_cp_eh])/phys.thermal_potential(T[i_node_cp_eh])
        A[i_indice_cp_eh,i_indice_cp_eh] = -gummel_contrib
    else:
        A[i_indice_cp_eh,i_indice_cp_eh] = 0.0
    # assemble associated items in newton iteration
    if sim.conf['method']=='newton':      
        if sim.conf['with_electron'] and len(j_indice_cp_e) > 0:
            A[i_indice_cp_eh,j_indice_cp_e] = -cons['q']*box_volume_nodes
        if sim.conf['with_hole'] and len(j_indice_cp_h) > 0:
            A[i_indice_cp_eh,j_indice_cp_h] = cons['q']*box_volume_nodes

    A[i_indices_non_diagonal,j_indices_non_diagonal]=eps_non_diagonal*facet_area_non_diagonal/edge_lengths_non_diagonal

    A[i_indices,i_indices] -= sum_over_i_indices(i_indices_diagonal, j_indices_diagonal,  
                            eps_diagonal*facet_area_diagonal/edge_lengths_diagonal)
    b[i_indices] -= sum_over_i_indices(i_indices_diagonal, j_indices_diagonal,  
                            eps_diagonal*facet_area_diagonal/edge_lengths_diagonal*(potential[j_nodes_diagonal]-potential[i_nodes_diagonal]))
    
    # if sim.conf['method']=='newton' and sim.conf['with_electrode_voltage']:
    #     # Add coupling for electrode voltage in Poisson's equation for newton method
    #     map_idx, map_i_node, map_idx_of_node = sim.mapping_info[quan['name']]
    #     _, map_i_node_electrode, map_idx_of_node_electrode = sim.mapping_info['electrode_voltage']
    #     for i_idx, i_node in zip(map_idx, map_i_node):
    #         node = sim.mesh.nodes[i_node]
    #         for i_edge, i_other_node in zip(node.edges, node.other_nodes):
    #             edge = sim.mesh.edges[i_edge]
    #             if i_other_node in map_i_node_electrode:
    #                 j_idx_electrode = map_idx_of_node_electrode[i_other_node]
    #                 A[i_idx, j_idx_electrode] += epsilon[i_edge]*cons['Epsilon0']*edge.facet_area_semi_ox/edge.length  # Electrode voltage contribution to Poisson's equation

    # the coupling with ion density for newton method
    if sim.conf['method']=='newton' and sim.conf['with_ion']:
        _,map_i_node_ion,map_idx_of_node_ion=sim.mapping_info['ion_density']
        for i_idx, i_node in zip(i_indices, i_nodes):
            node = sim.mesh.nodes[i_node]
            if i_node in map_i_node_ion:
                j_idx_ion = map_idx_of_node_ion[i_node]
                A[i_idx, j_idx_ion] = 2.0 * cons['q'] * node.box_volume_semi  # Ion contribution to Poisson's equation

    # print(' time2 ', time.time()-start, 's')
    return (A,b)

def sum_over_i_indices(i_indices,j_indices,value_list):
    A_dummy = sp.sparse.lil_array((max(i_indices)+1,max(j_indices)+1))
    A_dummy[i_indices,j_indices] = value_list
    return A_dummy.sum(axis=1)

def assemble_electrode_voltage(sim, quan, A, b):
    # start = time.time()
    # print('Assembling for electrode voltage...')
    potential = qs.get_potential(sim)
    n=qs.get_electron_density(sim)
    p=qs.get_hole_density(sim)
    T=qs.get_temperature(sim)
    mobility_e=qs.get_mobility(sim,'electron')
    mobility_h=qs.get_mobility(sim,'hole')
    built_in = sim.built_in_potential()

    map_idx, map_i_node, _ = sim.mapping_info[quan['name']]
    semi_node_idx = sim.node_idx_of_material('semiconductor')
    _, map_i_node_p, map_idx_of_node_p = sim.mapping_info['potential']
    _, map_i_node_e, map_idx_of_node_e = sim.mapping_info['electron_density']
    _, map_i_node_h, map_idx_of_node_h = sim.mapping_info['hole_density']

    if sim.conf['compliance']:
        i_idx = map_idx[0]  # all nodes for setting electrode voltage as single unknown
        Idevice = 0.0 

        for i_node in map_i_node:
            node = sim.mesh.nodes[i_node]
            for i_edge, i_other_node in zip(node.edges, node.other_nodes):
                if (i_other_node not in semi_node_idx) or (i_other_node in map_i_node):
                    continue

                edge = sim.mesh.edges[i_edge]
                if sim.mesh.mesh_type=='1D':
                    facet_area = edge.facet_area_semi * sim.device.area
                elif sim.mesh.mesh_type=='2D':
                    facet_area = edge.facet_area_semi * sim.device.width
                elif sim.mesh.mesh_type=='3D':
                    facet_area = edge.facet_area_semi
                elif sim.mesh.mesh_type=='cylindrical_2D':
                    midpoint = edge.midpoint
                    if sim.mesh.symmetric_axis=='x-axis':
                        facet_area = edge.facet_area_semi * (midpoint.y)*2*np.pi
                    elif sim.mesh.symmetric_axis=='y-axis':
                        facet_area = edge.facet_area_semi * (midpoint.x)*2*np.pi
                v_i, v_j = potential[i_node], potential[i_other_node]
                n_i, n_j = n[i_node], n[i_other_node]
                p_i, p_j = p[i_node], p[i_other_node]
                T_i, T_j = T[i_node], T[i_other_node]
                Tij = (T_i + T_j) / 2
                mu_e = mobility_e[i_edge]
                mu_h = mobility_h[i_edge]

                if sim.conf['with_electron']:
                    Jij = phys.scharfetter_gummel(n_i, n_j, v_i, v_j, edge.length, mu_e, Tij, type='electron')

                    dJij_dvi = phys.scharfetter_gummel_dVi(n_i, n_j, v_i, v_j, edge.length, mu_e, Tij, type='electron')
                    dJij_dvj = phys.scharfetter_gummel_dVj(n_i, n_j, v_i, v_j, edge.length, mu_e, Tij, type='electron')

                    dJij_dnj = phys.scharfetter_gummel_j(v_i, v_j, edge.length, mu_e, Tij, type='electron')
                    
                    Idevice += Jij * facet_area
                    A[i_idx, i_idx] += dJij_dvi * facet_area 
                    if (i_other_node in map_i_node_p):
                        j_idx_p = map_idx_of_node_p[i_other_node]
                        A[i_idx, j_idx_p] += dJij_dvj * facet_area 
                    if (i_other_node in map_i_node_e):
                        j_idx_e = map_idx_of_node_e[i_other_node]
                        A[i_idx, j_idx_e] += dJij_dnj * facet_area 

        v_electrode = np.mean(quan['value'][map_i_node])  
        Vc = sim.vapplied - v_electrode

        if sim.conf['compliance_method'] == 'serial_resistor':
            # 1: compliance by serial resistor
            serial_resistance = sim.conf.get('serial_resistance', 0.0)
            ic = Vc/serial_resistance
            dic_dvc = 1.0/serial_resistance
        elif sim.conf['compliance_method'] == 'ideal':
            # 2: ideal compliance
            Ic = sim.conf.get('Ic', 0.0)

            alpha = 1.0
            ic = Ic*(1-np.exp(-alpha*Vc))
            dic_dvc = Ic *(-np.exp(-alpha*Vc)) * (-alpha)
        elif sim.conf['compliance_method'] == 'iv_func':
            func = sim.conf.get('func', 0.0)
            ic, dic_dvc = func(Vc)
        else:
            raise ValueError('Compliance method not defined!')
        
        A[i_idx, i_idx] += dic_dvc
        b[i_idx] -= Idevice - ic

    else:
        for i_idx, i_node in zip(map_idx, map_i_node):
            A[i_idx, i_idx] = 1.0
            b[i_idx] = 0.0
    # print(' time ', time.time() -start, 's')
    return (A, b)

def assemble_dd(sim, quan, A, b):    

    # start = time.time()
    T = qs.get_temperature(sim)
    R, dR_dn, dR_dp = qs.get_recombination(sim)

    # Determine carrier type and get mobility
    if quan['name'] == 'electron_density':
        carrier_type = 'electron'
        carrier_conc = qs.get_electron_density(sim)
    elif quan['name'] == 'hole_density':
        carrier_type = 'hole'
        carrier_conc = qs.get_hole_density(sim)
    
    if sim.conf.get('uniform_band', True):
        # (default) for uniform band structure
        potential = qs.get_potential(sim)  
    else:
        # for non-uniform band structure, the flux is calculated based on effective potential
        potential = qs.get_effective_potential(sim,type=carrier_type)


    mobility = qs.get_mobility(sim, carrier_type) # mobility is evaluated on edges
    # print(' time1 ', time.time() -start, 's')

    if quan['vectrize_indices'] is None:
        map_idx, map_i_node, map_idx_of_node = sim.mapping_info[quan['name']]
        _, _, map_idx_of_node_p = sim.mapping_info['potential']
        
        i_indices = map_idx
        i_nodes = map_i_node
        j_indice_potential = []
        
        # For newton coupling with potential
        for i_node in map_i_node:
            if sim.conf['method'] == 'newton':
                j_idx_p = map_idx_of_node_p[i_node]
                j_indice_potential.append(j_idx_p)
        
        # Store edge-related information for vectorization
        i_indices_non_diagonal = []
        j_indices_non_diagonal = []
        i_edges_non_diagonal = []
        i_nodes_non_diagonal = []
        j_nodes_non_diagonal = []
        facet_areas_non_diagonal = []
        edge_lengths_non_diagonal = []
        
        i_indices_diagonal = []
        j_indices_diagonal = []
        i_edges_diagonal = []
        i_nodes_diagonal = []
        j_nodes_diagonal = []
        facet_areas_diagonal = []
        edge_lengths_diagonal = []
        box_volumes_nodes = []
        
        # For newton potential coupling
        i_indices_newton_p = []
        j_indices_newton_p = []
        i_edges_newton_p = []
        i_nodes_newton_p = []
        j_nodes_newton_p = []
        facet_areas_newton_p = []
        edge_lengths_newton_p = []
        
        for i_idx, i_node in zip(map_idx, map_i_node):
            node = sim.mesh.nodes[i_node]
            j_idx_diagonal = 0
            
            for i_edge,i_other_node in zip(node.edges, node.other_nodes):
                edge = sim.mesh.edges[i_edge]
                facet_area = edge.facet_area_semi

                if i_other_node in map_i_node:
                    j_idx = map_idx_of_node[i_other_node]
                    i_indices_non_diagonal.append(i_idx)
                    j_indices_non_diagonal.append(j_idx)
                    i_edges_non_diagonal.append(i_edge)
                    i_nodes_non_diagonal.append(i_node)
                    j_nodes_non_diagonal.append(i_other_node)
                    facet_areas_non_diagonal.append(facet_area)
                    edge_lengths_non_diagonal.append(edge.length)
                
                # For diagonal accumulation
                i_indices_diagonal.append(i_idx)
                j_indices_diagonal.append(j_idx_diagonal)
                i_edges_diagonal.append(i_edge)
                i_nodes_diagonal.append(i_node)
                j_nodes_diagonal.append(i_other_node)
                facet_areas_diagonal.append(facet_area)
                edge_lengths_diagonal.append(edge.length)
                j_idx_diagonal += 1
                
                # For newton potential coupling
                if sim.conf['method'] == 'newton' and i_other_node in map_i_node:
                    j_idx_p = map_idx_of_node_p[i_other_node]
                    i_indices_newton_p.append(i_idx)
                    j_indices_newton_p.append(j_idx_p)
                    i_edges_newton_p.append(i_edge)
                    i_nodes_newton_p.append(i_node)
                    j_nodes_newton_p.append(i_other_node)
                    facet_areas_newton_p.append(facet_area)
                    edge_lengths_newton_p.append(edge.length)
            box_volumes_nodes.append(node.box_volume_semi)
        
        # Convert to numpy arrays
        facet_areas_non_diagonal = np.array(facet_areas_non_diagonal)
        edge_lengths_non_diagonal = np.array(edge_lengths_non_diagonal)
        
        facet_areas_diagonal = np.array(facet_areas_diagonal)
        edge_lengths_diagonal = np.array(edge_lengths_diagonal)
        box_volumes_nodes = np.array(box_volumes_nodes)
        
        edge_lengths_newton_p = np.array(edge_lengths_newton_p) 
        facet_areas_newton_p = np.array(facet_areas_newton_p) 
        
        quan['vectrize_indices'] = [
            i_indices, i_nodes, j_indice_potential,
            i_indices_non_diagonal, j_indices_non_diagonal,
            i_edges_non_diagonal, i_nodes_non_diagonal, j_nodes_non_diagonal, facet_areas_non_diagonal, edge_lengths_non_diagonal,
            i_indices_diagonal, j_indices_diagonal,
            i_edges_diagonal, i_nodes_diagonal, j_nodes_diagonal, facet_areas_diagonal, edge_lengths_diagonal,
            box_volumes_nodes,
            i_indices_newton_p, j_indices_newton_p,
            i_edges_newton_p, i_nodes_newton_p, j_nodes_newton_p, facet_areas_newton_p, edge_lengths_newton_p
        ]
    
    # Unpack vectorized indices
    (i_indices, i_nodes, j_indice_potential,
     i_indices_non_diagonal, j_indices_non_diagonal,
     i_edges_non_diagonal, i_nodes_non_diagonal, j_nodes_non_diagonal, facet_areas_non_diagonal, edge_lengths_non_diagonal,
     i_indices_diagonal, j_indices_diagonal,
     i_edges_diagonal, i_nodes_diagonal, j_nodes_diagonal, facet_areas_diagonal, edge_lengths_diagonal,
     box_volumes_nodes,
     i_indices_newton_p, j_indices_newton_p,
     i_edges_newton_p, i_nodes_newton_p, j_nodes_newton_p, facet_areas_newton_p, edge_lengths_newton_p) = quan['vectrize_indices']
    
    # Vectorized computation of Scharfetter-Gummel coefficients for non-diagonal terms
    if len(i_edges_non_diagonal) > 0:
        mu_edges_nd = mobility[i_edges_non_diagonal]
        v_i_nd = potential[i_nodes_non_diagonal]
        v_j_nd = potential[j_nodes_non_diagonal]
        temp_nd = (T[i_nodes_non_diagonal] + T[j_nodes_non_diagonal]) / 2
        
        # Vectorized Scharfetter-Gummel coefficients
        sg_j_nd = phys.scharfetter_gummel_j(v_i_nd, v_j_nd, edge_lengths_non_diagonal, mu_edges_nd, temp_nd, type=carrier_type)

        # Assemble non-diagonal terms
        A[i_indices_non_diagonal, j_indices_non_diagonal] = sg_j_nd * facet_areas_non_diagonal
    
    # Vectorized computation for diagonal terms and RHS
    mu_edges_d = mobility[i_edges_diagonal]
    v_i_d = potential[i_nodes_diagonal]
    v_j_d = potential[j_nodes_diagonal]
    c_i_d = carrier_conc[i_nodes_diagonal]
    c_j_d = carrier_conc[j_nodes_diagonal]
    temp_d = (T[i_nodes_diagonal] + T[j_nodes_diagonal]) / 2
    
    # Vectorized Scharfetter-Gummel coefficients
    sg_i_d = phys.scharfetter_gummel_i(v_i_d, v_j_d, edge_lengths_diagonal, mu_edges_d, temp_d, type=carrier_type)
    sg_j_d = phys.scharfetter_gummel_j(v_i_d, v_j_d, edge_lengths_diagonal, mu_edges_d, temp_d, type=carrier_type)

    # Assemble diagonal terms using sparse matrix accumulation
    A_dummy = sp.sparse.lil_array((sim.num_unknowns, sim.num_unknowns))
    A_dummy[i_indices_diagonal, j_indices_diagonal] = sg_i_d * facet_areas_diagonal
    A[i_indices, i_indices] = A_dummy.sum(axis=1)[i_indices]
    
    # Assemble RHS terms
    b_dummy = sp.sparse.lil_array((sim.num_unknowns, sim.num_unknowns))
    b_dummy[i_indices_diagonal, j_indices_diagonal] = -(c_i_d * sg_i_d + c_j_d * sg_j_d) * facet_areas_diagonal
    b[i_indices] = b_dummy.sum(axis=1)[i_indices]
    
    R[i_nodes]

    # Add recombination terms
    if carrier_type == 'electron':
        b[i_indices] += cons['q'] * R[i_nodes] * box_volumes_nodes
    elif carrier_type == 'hole':
        b[i_indices] -= cons['q'] * R[i_nodes] * box_volumes_nodes

    # Newton method coupling with potential
    if sim.conf['method'] == 'newton':
        # Add potential coupling for current nodes
        if len(j_indice_potential) > 0:
            
            # Compute diagonal potential coupling
            sg_dvi_diag = phys.scharfetter_gummel_dVi(c_i_d, c_j_d, v_i_d, v_j_d, edge_lengths_diagonal, mu_edges_d, temp_d, type=carrier_type)

            A_dummy_newton = sp.sparse.lil_array((sim.num_unknowns, sim.num_unknowns))
            A_dummy_newton[i_indices_diagonal, j_indices_diagonal] = sg_dvi_diag * facet_areas_diagonal
            A[i_indices, j_indice_potential] = A_dummy_newton.sum(axis=1)[i_indices]
        
        # Add off-diagonal potential coupling
        if len(i_edges_newton_p) > 0:
            mu_edges_np = mobility[i_edges_newton_p]
            v_i_np = potential[i_nodes_newton_p]
            v_j_np = potential[j_nodes_newton_p]
            c_i_np = carrier_conc[i_nodes_newton_p]
            c_j_np = carrier_conc[j_nodes_newton_p]
            temp_np = (T[i_nodes_newton_p] + T[j_nodes_newton_p]) / 2

            sg_dvi_np = phys.scharfetter_gummel_dVi(c_i_np, c_j_np, v_i_np, v_j_np, edge_lengths_newton_p, mu_edges_np, temp_np, type=carrier_type)

            A[i_indices_newton_p, j_indices_newton_p] = (-sg_dvi_np) * facet_areas_newton_p

    # if sim.conf['method']=='newton' and sim.conf['with_electrode_voltage']:
    #     # Add coupling for electrode voltage in carrier continuity equation for newton method
    #     map_idx, map_i_node, map_idx_of_node = sim.mapping_info[quan['name']]
    #     _, map_i_node_electrode, map_idx_of_node_electrode = sim.mapping_info['electrode_voltage']
    #     for i_idx, i_node in zip(map_idx, map_i_node):
    #         node = sim.mesh.nodes[i_node]
    #         for i_edge, i_other_node in zip(node.edges, node.other_nodes):
    #             edge = sim.mesh.edges[i_edge]
    #             if i_other_node in map_i_node_electrode:
    #                 v_i, v_j = potential[i_node], potential[i_other_node]
    #                 c_i, c_j = carrier_conc[i_node], carrier_conc[i_other_node]
    #                 T_i, T_j = T[i_node], T[i_other_node]
    #                 temp = (T_i + T_j) / 2
    #                 mu_edge = mobility[i_edge]
    #                 sg_dvj = phys.scharfetter_gummel_dVj(c_i, c_j, v_i, v_j, edge.length, mu_edge, temp, type=carrier_type)

    #                 j_idx_electrode = map_idx_of_node_electrode[i_other_node]
    #                 A[i_idx, j_idx_electrode] += sg_dvj * edge.facet_area_semi  # Electrode voltage contribution to carrier continuity equation

    return (A, b)


def assemble_heat(sim,quan,A,b):
    # Assemble for heat eaquation
    T=qs.get_temperature(sim)
    potential = qs.get_potential(sim)
    n=qs.get_electron_density(sim)
    p=qs.get_hole_density(sim)
    # power_density=qs.get_power_density(sim)
    mobility_e=qs.get_mobility(sim,'electron')
    mobility_h=qs.get_mobility(sim,'hole')

    kappa=qs.get_kappa(sim)

    if quan['vectrize_indices'] is None:
        map_idx,map_i_node,map_idx_of_node=sim.mapping_info[quan['name']]

        i_indices = map_idx
        i_nodes = map_i_node

        i_indices_diagonal = []
        j_indices_diagonal = []
        i_edges_diagonal = []
        edge_facet_areas_diagonal = []
        edge_lengths_diagonal = []
        i_nodes_diagonal = []
        j_nodes_diagonal = []

        i_indices_non_diagonal = []
        j_indices_non_diagonal = []
        i_edges_non_diagonal = []
        edge_facet_areas_non_diagonal = []
        edge_lengths_non_diagonal = []

        i_idx_diagonal = 0
        for i_idx,i_node in zip(map_idx,map_i_node):
            node = sim.mesh.nodes[i_node]
            j_idx_diagonal = 0
            for i_edge, i_other_node in zip(node.edges, node.other_nodes):
                edge = sim.mesh.edges[i_edge]
                if sim.conf['with_conductor_temperature']:
                    facet_area = edge.facet_area_all
                else:
                    facet_area = edge.facet_area_semi_ox
                i_indices_diagonal.append(i_idx_diagonal)
                j_indices_diagonal.append(j_idx_diagonal) # this index is faked, becasue acctual j_idx might not exist for the case that the other node is not in the potential mappping; we just need the summation over all j indices to fill in the diagonal position
                j_idx_diagonal += 1
                i_edges_diagonal.append(i_edge)
                edge_facet_areas_diagonal.append(facet_area)
                edge_lengths_diagonal.append(edge.length)
                i_nodes_diagonal.append(i_node)
                j_nodes_diagonal.append(i_other_node)

                if i_other_node in map_i_node:
                    j_idx = map_idx_of_node[i_other_node]
                    i_indices_non_diagonal.append(i_idx)
                    j_indices_non_diagonal.append(j_idx)
                    i_edges_non_diagonal.append(i_edge)
                    edge_facet_areas_non_diagonal.append(facet_area)
                    edge_lengths_non_diagonal.append(edge.length)
            i_idx_diagonal += 1

        edge_facet_areas_diagonal = np.array(edge_facet_areas_diagonal)
        edge_lengths_diagonal =  np.array(edge_lengths_diagonal)

        edge_facet_areas_non_diagonal = np.array(edge_facet_areas_non_diagonal)
        edge_lengths_non_diagonal = np.array(edge_lengths_non_diagonal)

        # indices for heat generation term vectorization
        semi_node_idx = sim.node_idx_of_material('semiconductor')
        ii_indices = []
        i_indices_q = []
        j_indices_q = []
        i_indices_q_non_diagonal = []
        j_indices_q_non_diagonal = []
        i_nodes_q = []
        j_nodes_q = []
        i_edges_q = []
        edge_lengths_q = []
        edge_box_volumes_q = []

        i_idx_q = 0
        for i_idx,i_node in zip(map_idx,map_i_node): # loop for all nodes
            node=sim.mesh.nodes[i_node]
            if (i_node in semi_node_idx):
                ii_indices.append(i_idx)
                j_idx_q = 0
                for i_edge,i_other_node in zip(node.edges, node.other_nodes):
                    edge = sim.mesh.edges[i_edge]
                    if (i_other_node in semi_node_idx) and (i_other_node in map_i_node):
                        j_idx=map_idx_of_node[i_other_node]
                        i_indices_q.append(i_idx_q)
                        j_indices_q.append(j_idx_q)
                        j_idx_q += 1

                        i_indices_q_non_diagonal.append(i_idx)
                        j_indices_q_non_diagonal.append(j_idx)

                        i_nodes_q.append(i_node)
                        j_nodes_q.append(i_other_node)
                        i_edges_q.append(i_edge)
                        edge_lengths_q.append(edge.length)
                        edge_box_volumes_q.append(edge.box_volume_semi)
                i_idx_q += 1

        edge_lengths_q = np.array(edge_lengths_q)
        edge_box_volumes_q = np.array(edge_box_volumes_q)

        quan['vectrize_indices'] = [i_indices, i_nodes,
                                    i_indices_diagonal, j_indices_diagonal, i_edges_diagonal, edge_facet_areas_diagonal, edge_lengths_diagonal, i_nodes_diagonal, j_nodes_diagonal,
                                    i_indices_non_diagonal, j_indices_non_diagonal, i_edges_non_diagonal, edge_facet_areas_non_diagonal, edge_lengths_non_diagonal,
                                    ii_indices, i_indices_q, j_indices_q, i_indices_q_non_diagonal, j_indices_q_non_diagonal, i_nodes_q, j_nodes_q, i_edges_q, edge_lengths_q, edge_box_volumes_q]
    else:
        (i_indices, i_nodes,
         i_indices_diagonal, j_indices_diagonal, i_edges_diagonal, edge_facet_areas_diagonal, edge_lengths_diagonal, i_nodes_diagonal, j_nodes_diagonal,
         i_indices_non_diagonal, j_indices_non_diagonal, i_edges_non_diagonal, edge_facet_areas_non_diagonal, edge_lengths_non_diagonal,
         ii_indices, i_indices_q, j_indices_q, i_indices_q_non_diagonal, j_indices_q_non_diagonal, i_nodes_q, j_nodes_q, i_edges_q, edge_lengths_q, edge_box_volumes_q) = quan['vectrize_indices']

    # thermal conduction term assemble
    kappa_edges_diagonal = kappa[i_edges_diagonal]
    Tj_diagonal = T[j_nodes_diagonal]
    Ti_diagonal = T[i_nodes_diagonal]

    A[i_indices,i_indices] -= sum_over_i_indices(i_indices_diagonal, j_indices_diagonal,  kappa_edges_diagonal*edge_facet_areas_diagonal/edge_lengths_diagonal)
    b[i_indices] -= sum_over_i_indices(i_indices_diagonal, j_indices_diagonal, kappa_edges_diagonal*edge_facet_areas_diagonal/edge_lengths_diagonal*(Tj_diagonal-Ti_diagonal))

    kappa_edges_non_diagonal = kappa[i_edges_non_diagonal]
    A[i_indices_non_diagonal,j_indices_non_diagonal]=kappa_edges_non_diagonal*edge_facet_areas_non_diagonal/edge_lengths_non_diagonal

        
    # heating term assemble    
    v_i, v_j = potential[i_nodes_q], potential[j_nodes_q]
    n_i, n_j = n[i_nodes_q], n[j_nodes_q]
    p_i, p_j = p[i_nodes_q], p[j_nodes_q]
    Tij = (T[i_nodes_q] + T[j_nodes_q])/2
    mu_e = mobility_e[i_edges_q]
    mu_h = mobility_h[i_edges_q]

    if sim.conf['with_electron']:
        Je = phys.scharfetter_gummel(n_i,n_j,v_i,v_j,edge_lengths_q,mu_e,Tij,type='electron')
        dJe_dTi = phys.scharfetter_gummel_dTi(n_i,n_j,v_i,v_j,edge_lengths_q,mu_e,Tij,type='electron')
        dJe_dTj = phys.scharfetter_gummel_dTj(n_i,n_j,v_i,v_j,edge_lengths_q,mu_e,Tij,type='electron')
    else:
        Je = 0.0
        dJe_dTi = 0.0
        dJe_dTj = 0.0
    if sim.conf['with_hole']:
        Jh = phys.scharfetter_gummel(p_i,p_j,v_i,v_j,edge_lengths_q,mu_h,Tij,type='hole')
        dJh_dTi = phys.scharfetter_gummel_dTi(p_i,p_j,v_i,v_j,edge_lengths_q,mu_h,Tij,type='hole')
        dJh_dTj = phys.scharfetter_gummel_dTj(p_i,p_j,v_i,v_j,edge_lengths_q,mu_h,Tij,type='hole')
    else:
        Jh = 0.0
        dJh_dTi = 0.0
        dJh_dTj = 0.0
    
    E_edges = - (v_j - v_i)/edge_lengths_q
    Q_edges = (Je + Jh) * E_edges  # power density on edges
    dQ_dTi = (dJe_dTi + dJh_dTi) * E_edges
    dQ_dTj = (dJe_dTj + dJh_dTj) * E_edges
    A[ii_indices,ii_indices] += sum_over_i_indices(i_indices_q, j_indices_q, dQ_dTi*edge_box_volumes_q)
    b[ii_indices] -= sum_over_i_indices(i_indices_q, j_indices_q, Q_edges * edge_box_volumes_q)

    A[i_indices_q_non_diagonal,j_indices_q_non_diagonal] +=  dQ_dTj * edge_box_volumes_q

    # print(' time2 ', time.time()-start, 's')
    return (A,b)

def assemble_ion(sim,quan,A,b):
    potential = qs.get_potential(sim)
    D0=qs.get_D0(sim)   # defusion coefficient pre-factor on edges
    Ea=qs.get_Ea(sim)   # activation energy on edges
    T = qs.get_temperature(sim)
    # R = qs.get_ion_recombination(sim,T)
    R = np.zeros(len(sim.mesh.nodes))  # no recombination for ion
    ion_last_step = sim.ion_density_last_step
    time_step=sim.time_step

    c_ion = quan['value']  # ion charge

    if quan['vectrize_indices'] is None:

        map_idx,map_i_node,map_idx_of_node=sim.mapping_info[quan['name']]
        _, map_i_node_p, map_idx_of_node_p = sim.mapping_info['potential']

        i_indices = map_idx
        i_nodes = map_i_node

        # indices and related info for coupling with ion concentration
        box_volume_nodes = []
        i_indices_diagonal = []
        j_indices_diagonal = []
        i_edges_diagonal = []
        edge_facet_areas_diagonal = []
        edge_lengths_diagonal = []
        i_nodes_diagonal = []
        j_nodes_diagonal = []

        i_indices_non_diagonal = []
        j_indices_non_diagonal = []
        i_edges_non_diagonal = []
        edge_facet_areas_non_diagonal = []
        edge_lengths_non_diagonal = []
        i_nodes_non_diagonal = []
        j_nodes_non_diagonal = []

        i_idx_diagonal = 0
        for i_idx,i_node in zip(map_idx,map_i_node): # loop for all nodes
            node=sim.mesh.nodes[i_node]
            box_volume_nodes.append(node.box_volume_semi)

            j_idx_diagonal = 0
            for i_edge,i_other_node in zip(node.edges, node.other_nodes):
                edge = sim.mesh.edges[i_edge]

                i_indices_diagonal.append(i_idx_diagonal)
                j_indices_diagonal.append(j_idx_diagonal) # this index is faked, becasue acctual j_idx might not exist for the case that the other node is not in the potential mappping; we just need the summation over all j indices to fill in the diagonal position
                j_idx_diagonal += 1

                i_edges_diagonal.append(i_edge)
                edge_facet_areas_diagonal.append(edge.facet_area_semi)
                edge_lengths_diagonal.append(edge.length)
                i_nodes_diagonal.append(i_node)
                j_nodes_diagonal.append(i_other_node)
                if i_other_node in map_i_node:
                    j_idx = map_idx_of_node[i_other_node]
                    i_indices_non_diagonal.append(i_idx)
                    j_indices_non_diagonal.append(j_idx)
                    i_edges_non_diagonal.append(i_edge)
                    edge_facet_areas_non_diagonal.append(edge.facet_area_semi)
                    edge_lengths_non_diagonal.append(edge.length)
                    i_nodes_non_diagonal.append(i_node)
                    j_nodes_non_diagonal.append(i_other_node)
            i_idx_diagonal += 1
        box_volume_nodes = np.array(box_volume_nodes)

        # indices and related info for coupling with potential for newton method
        i_indices_p = []
        j_indices_p = []
        i_indices_diagonal_p = []
        j_indices_diagonal_p = []
        i_edges_diagonal_p = []
        edge_facet_areas_diagonal_p = []
        edge_lengths_diagonal_p = []
        i_nodes_diagonal_p = []
        j_nodes_diagonal_p = []

        i_indices_non_diagonal_p = []
        j_indices_non_diagonal_p = []
        i_edges_non_diagonal_p = []
        edge_facet_areas_non_diagonal_p = []
        edge_lengths_non_diagonal_p = []
        i_nodes_non_diagonal_p = []
        j_nodes_non_diagonal_p = []

        i_idx_diagonal_p = 0
        for i_idx,i_node in zip(map_idx,map_i_node): # loop for all nodes
            node=sim.mesh.nodes[i_node]
            if i_node in map_i_node_p:
                j_idx_p = map_idx_of_node_p[i_node]
                i_indices_p.append(i_idx)
                j_indices_p.append(j_idx_p)

            j_idx_diagonal_p = 0
            for i_edge,i_other_node in zip(node.edges, node.other_nodes):
                edge = sim.mesh.edges[i_edge]
                if i_node in map_i_node_p:
                    i_indices_diagonal_p.append(i_idx_diagonal_p)
                    j_indices_diagonal_p.append(j_idx_diagonal_p) # this index is faked, becasue acctual j_idx might not exist for the case that the other node is not in the potential mappping; we just need the summation over all j indices to fill in the diagonal position
                    j_idx_diagonal_p += 1

                    i_edges_diagonal_p.append(i_edge)
                    edge_facet_areas_diagonal_p.append(edge.facet_area_semi)
                    edge_lengths_diagonal_p.append(edge.length)
                    i_nodes_diagonal_p.append(i_node)
                    j_nodes_diagonal_p.append(i_other_node)

                if i_other_node in map_i_node_p:
                    j_idx_p = map_idx_of_node_p[i_other_node]
                    i_indices_non_diagonal_p.append(i_idx)
                    j_indices_non_diagonal_p.append(j_idx_p)

                    i_edges_non_diagonal_p.append(i_edge)
                    edge_facet_areas_non_diagonal_p.append(edge.facet_area_semi)
                    edge_lengths_non_diagonal_p.append(edge.length)
                    i_nodes_non_diagonal_p.append(i_node)
                    j_nodes_non_diagonal_p.append(i_other_node)
            if i_node in map_i_node_p:
                i_idx_diagonal_p += 1

        quan['vectrize_indices'] = [i_indices, i_nodes,
                                    i_indices_diagonal, j_indices_diagonal, i_edges_diagonal, edge_facet_areas_diagonal, edge_lengths_diagonal, i_nodes_diagonal, j_nodes_diagonal,
                                    i_indices_non_diagonal, j_indices_non_diagonal, i_edges_non_diagonal, edge_facet_areas_non_diagonal, edge_lengths_non_diagonal, i_nodes_non_diagonal, j_nodes_non_diagonal,
                                    box_volume_nodes,
                                    i_indices_p, j_indices_p,
                                    i_indices_diagonal_p, j_indices_diagonal_p, i_edges_diagonal_p, edge_facet_areas_diagonal_p, edge_lengths_diagonal_p, i_nodes_diagonal_p, j_nodes_diagonal_p,
                                    i_indices_non_diagonal_p, j_indices_non_diagonal_p, i_edges_non_diagonal_p, edge_facet_areas_non_diagonal_p, edge_lengths_non_diagonal_p, i_nodes_non_diagonal_p, j_nodes_non_diagonal_p]
    else:
        (i_indices, i_nodes,
         i_indices_diagonal, j_indices_diagonal, i_edges_diagonal, edge_facet_areas_diagonal, edge_lengths_diagonal, i_nodes_diagonal, j_nodes_diagonal,
         i_indices_non_diagonal, j_indices_non_diagonal, i_edges_non_diagonal, edge_facet_areas_non_diagonal, edge_lengths_non_diagonal, i_nodes_non_diagonal, j_nodes_non_diagonal,
         box_volume_nodes,
         i_indices_p, j_indices_p,
         i_indices_diagonal_p, j_indices_diagonal_p, i_edges_diagonal_p, edge_facet_areas_diagonal_p, edge_lengths_diagonal_p, i_nodes_diagonal_p, j_nodes_diagonal_p,
         i_indices_non_diagonal_p, j_indices_non_diagonal_p, i_edges_non_diagonal_p, edge_facet_areas_non_diagonal_p, edge_lengths_non_diagonal_p, i_nodes_non_diagonal_p, j_nodes_non_diagonal_p) = quan['vectrize_indices']
        
    # assemble for diagonal terms and rhs
    v_i = potential[i_nodes_diagonal]
    v_j = potential[j_nodes_diagonal]
    T_i = T[i_nodes_diagonal]
    T_j = T[j_nodes_diagonal]
    c_i = c_ion[i_nodes_diagonal]
    c_j = c_ion[j_nodes_diagonal]
    Tij = (T_i + T_j)/2

    Delta = 2.0*cons['q']/(cons['k']*Tij)*(v_j - v_i) - Ea[i_edges_diagonal]/(cons['k']*Tij**2) * (T_j - T_i)
    D_edge = D0[i_edges_diagonal] * np.exp(-Ea[i_edges_diagonal]/(cons['k']*Tij))
    dJij_dci =  D_edge / edge_lengths_diagonal * phys.Bern(Delta)
    dJij_dcj = -D_edge / edge_lengths_diagonal * phys.Bern(-Delta)
    Jij = c_i * dJij_dci + c_j * dJij_dcj

    A[i_indices,i_indices] += sum_over_i_indices(i_indices_diagonal, j_indices_diagonal, dJij_dci * edge_facet_areas_diagonal) 
    A[i_indices,i_indices] += 1/time_step * box_volume_nodes

    
    b[i_indices] -= sum_over_i_indices(i_indices_diagonal, j_indices_diagonal, Jij * edge_facet_areas_diagonal)
    b[i_indices] -= (R[i_nodes] + (c_ion[i_nodes] - ion_last_step[i_nodes]) / time_step) * box_volume_nodes

    # assemble for non-diagonal terms
    v_i = potential[i_nodes_non_diagonal]
    v_j = potential[j_nodes_non_diagonal]
    T_i = T[i_nodes_non_diagonal]
    T_j = T[j_nodes_non_diagonal]
    c_i = c_ion[i_nodes_non_diagonal]
    c_j = c_ion[j_nodes_non_diagonal]
    Tij = (T_i + T_j)/2 

    Delta = 2.0*cons['q']/(cons['k']*Tij)*(v_j - v_i) - Ea[i_edges_non_diagonal]/(cons['k']*Tij**2) * (T_j - T_i)
    D_edge = D0[i_edges_non_diagonal] * np.exp(-Ea[i_edges_non_diagonal]/(cons['k']*Tij))
    dJij_dcj = -D_edge / edge_lengths_non_diagonal * phys.Bern(-Delta)
    A[i_indices_non_diagonal,j_indices_non_diagonal]=dJij_dcj * edge_facet_areas_non_diagonal

    
    # assemble for potential coupling terms for newton method
    if sim.conf['method'] == 'newton':
        # diagonal potential coupling
        v_i = potential[i_nodes_diagonal_p]
        v_j = potential[j_nodes_diagonal_p]
        T_i = T[i_nodes_diagonal_p]
        T_j = T[j_nodes_diagonal_p]
        c_i = c_ion[i_nodes_diagonal_p]
        c_j = c_ion[j_nodes_diagonal_p]
        Tij = (T_i + T_j)/2

        Delta = 2.0*cons['q']/(cons['k']*Tij)*(v_j - v_i) - Ea[i_edges_diagonal_p]/(cons['k']*Tij**2) * (T_j - T_i)
        D_edge = D0[i_edges_diagonal_p] * np.exp(-Ea[i_edges_diagonal_p]/(cons['k']*Tij))
        dJij_dvi = - D_edge/edge_lengths_diagonal_p * (c_i * phys.Bern_dx(Delta) + c_j * phys.Bern_dx(-Delta)) * 2.0*cons['q']/(cons['k']*Tij) 
        
        A[i_indices_p, j_indices_p] = sum_over_i_indices(i_indices_diagonal_p, j_indices_diagonal_p, dJij_dvi * edge_facet_areas_diagonal_p)

        # non-diagonal potential coupling
        v_i = potential[i_nodes_non_diagonal_p]
        v_j = potential[j_nodes_non_diagonal_p]
        T_i = T[i_nodes_non_diagonal_p]
        T_j = T[j_nodes_non_diagonal_p]
        c_i = c_ion[i_nodes_non_diagonal_p]
        c_j = c_ion[j_nodes_non_diagonal_p]
        Tij = (T_i + T_j)/2 

        Delta = 2.0*cons['q']/(cons['k']*Tij)*(v_j - v_i) - Ea[i_edges_non_diagonal_p]/(cons['k']*Tij**2) * (T_j - T_i)
        D_edge = D0[i_edges_non_diagonal_p] * np.exp(-Ea[i_edges_non_diagonal_p]/(cons['k']*Tij))
        dJij_dvj = D_edge/edge_lengths_non_diagonal_p * (c_i * phys.Bern_dx(Delta) + c_j * phys.Bern_dx(-Delta)) * 2.0*cons['q']/(cons['k']*Tij)

        A[i_indices_non_diagonal_p, j_indices_non_diagonal_p] = dJij_dvj * edge_facet_areas_non_diagonal_p


    return (A,b)