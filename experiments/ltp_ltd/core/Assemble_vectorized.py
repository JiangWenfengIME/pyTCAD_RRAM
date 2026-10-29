import numpy as np
import scipy as sp
from experiments.ltp_ltd.core.Physics import constant as cons
import experiments.ltp_ltd.core.Physics as phys
import experiments.ltp_ltd.core.Quantities as qs
import time

def assemble(sim,quan,A,b):
    if quan['name'] == 'potential':
        [A,b]=assemble_possion(sim,quan,A,b)
    elif quan['name'] == 'electron_density' or quan['name'] == 'hole_density':
        [A,b]=assemble_dd(sim,quan,A,b)
    elif quan['name'] == 'electrode_voltage':
        [A,b]=assemble_electrode_voltage(sim,quan,A,b)
    elif quan['name'] == 'temperature':
        # [A,b]=assemble_heat(sim,quan,A,b)  
        [A,b]=assemble_heat_trans(sim,quan,A,b)  
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
                box_volume_nodes.append(node.box_volume_semi_ox)

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
    
    if sim.conf['method']=='newton' and sim.conf['with_electrode_voltage']:
        # Add coupling for electrode voltage in Poisson's equation for newton method
        map_idx, map_i_node, map_idx_of_node = sim.mapping_info[quan['name']]
        _, map_i_node_electrode, map_idx_of_node_electrode = sim.mapping_info['electrode_voltage']
        for i_idx, i_node in zip(map_idx, map_i_node):
            node = sim.mesh.nodes[i_node]
            for i_edge, i_other_node in zip(node.edges, node.other_nodes):
                edge = sim.mesh.edges[i_edge]
                if i_other_node in map_i_node_electrode:
                    j_idx_electrode = map_idx_of_node_electrode[i_other_node]
                    A[i_idx, j_idx_electrode] += epsilon[i_edge]*cons['Epsilon0']*edge.facet_area_semi_ox/edge.length  # Electrode voltage contribution to Poisson's equation

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
    # np.savetxt('b[i_indices]_woR.txt',b[i_indices])
    # R[i_nodes]
    # print(R[i_nodes])
    # Add recombination terms
    if carrier_type == 'electron':
        # b[i_indices] += cons['q'] * R[i_nodes] * box_volumes_nodes
        # 20260915 q has been Simplified in scharfetter_gummel
        b[i_indices] +=  R[i_nodes] * box_volumes_nodes
        # np.savetxt('R.txt',R[i_nodes])
        # np.savetxt('b[i_indices].txt',b[i_indices])
    elif carrier_type == 'hole':
        # b[i_indices] -= cons['q'] * R[i_nodes] * box_volumes_nodes
        # 20260915 q has been Simplified in scharfetter_gummel
        b[i_indices] -=  R[i_nodes] * box_volumes_nodes

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

    if sim.conf['method']=='newton' and sim.conf['with_electrode_voltage']:
        # Add coupling for electrode voltage in carrier continuity equation for newton method
        map_idx, map_i_node, map_idx_of_node = sim.mapping_info[quan['name']]
        _, map_i_node_electrode, map_idx_of_node_electrode = sim.mapping_info['electrode_voltage']
        for i_idx, i_node in zip(map_idx, map_i_node):
            node = sim.mesh.nodes[i_node]
            for i_edge, i_other_node in zip(node.edges, node.other_nodes):
                edge = sim.mesh.edges[i_edge]
                if i_other_node in map_i_node_electrode:
                    v_i, v_j = potential[i_node], potential[i_other_node]
                    c_i, c_j = carrier_conc[i_node], carrier_conc[i_other_node]
                    T_i, T_j = T[i_node], T[i_other_node]
                    temp = (T_i + T_j) / 2
                    mu_edge = mobility[i_edge]
                    sg_dvj = phys.scharfetter_gummel_dVj(c_i, c_j, v_i, v_j, edge.length, mu_edge, temp, type=carrier_type)

                    j_idx_electrode = map_idx_of_node_electrode[i_other_node]
                    A[i_idx, j_idx_electrode] += sg_dvj * edge.facet_area_semi  # Electrode voltage contribution to carrier continuity equation

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

    # kappa=qs.get_kappa(sim)
    ## 20260309
    kappa, alpha = qs.get_kappa_and_alpha(sim)

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
         
        # 20260309
        i_nodes_non_diag = []
        j_nodes_non_diag = []

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

                    ## 20260309
                    i_nodes_non_diag.append(i_node)
                    j_nodes_non_diag.append(i_other_node)

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
                                    i_nodes_non_diag, j_nodes_non_diag, ## 20260309
                                    i_indices_non_diagonal, j_indices_non_diagonal, i_edges_non_diagonal, edge_facet_areas_non_diagonal, edge_lengths_non_diagonal,
                                    ii_indices, i_indices_q, j_indices_q, i_indices_q_non_diagonal, j_indices_q_non_diagonal, i_nodes_q, j_nodes_q, i_edges_q, edge_lengths_q, edge_box_volumes_q]
    else:
        (i_indices, i_nodes,
         i_indices_diagonal, j_indices_diagonal, i_edges_diagonal, edge_facet_areas_diagonal, edge_lengths_diagonal, i_nodes_diagonal, j_nodes_diagonal,
         i_nodes_non_diag, j_nodes_non_diag, ## 20260309
         i_indices_non_diagonal, j_indices_non_diagonal, i_edges_non_diagonal, edge_facet_areas_non_diagonal, edge_lengths_non_diagonal,
         ii_indices, i_indices_q, j_indices_q, i_indices_q_non_diagonal, j_indices_q_non_diagonal, i_nodes_q, j_nodes_q, i_edges_q, edge_lengths_q, edge_box_volumes_q) = quan['vectrize_indices']

    # thermal conduction term assemble
    # kappa_edges_diagonal = kappa[i_edges_diagonal]  

    ## 20260309
    temp_d = (T[j_nodes_diagonal] + T[i_nodes_diagonal]) / 2
    kappa_edges_diagonal = kappa[i_edges_diagonal] * (1 + alpha[i_edges_diagonal] * (temp_d - 300))

    Tj_diagonal = T[j_nodes_diagonal]
    Ti_diagonal = T[i_nodes_diagonal]

    A[i_indices,i_indices] -= sum_over_i_indices(i_indices_diagonal, j_indices_diagonal,  kappa_edges_diagonal*edge_facet_areas_diagonal/edge_lengths_diagonal)
    b[i_indices] -= sum_over_i_indices(i_indices_diagonal, j_indices_diagonal, kappa_edges_diagonal*edge_facet_areas_diagonal/edge_lengths_diagonal*(Tj_diagonal-Ti_diagonal))

    # kappa_edges_non_diagonal = kappa[i_edges_non_diagonal]

    ## 20260309
    temp_nd = (T[i_nodes_non_diag] + T[j_nodes_non_diag]) / 2
    kappa_edges_non_diagonal = kappa[i_edges_non_diagonal] * (1 + alpha[i_edges_non_diagonal] * (temp_nd - 300))    

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


def assemble_heat_transient(sim,quan,A,b):
    # Assemble for heat eaquation: rho*Cp*dT/dt - div(kappa*grad(T)) - |Q| = 0
    T=qs.get_temperature(sim)
    potential = qs.get_potential(sim)
    n=qs.get_electron_density(sim)
    p=qs.get_hole_density(sim)
    mobility_e=qs.get_mobility(sim,'electron')
    mobility_h=qs.get_mobility(sim,'hole')

    kappa, alpha = qs.get_kappa_and_alpha(sim)

    # ===== 20260xxx 瞬态项所需物理量 =====
    # TODO: 按实际接口名调整，例如 qs.get_rho_Cp(sim) 或
    # 由材料密度*比热组合而来的按节点数组 (shape = num_nodes)
    with_transient = sim.conf.get('heat_transient')
    if with_transient:
        rho_cp = qs.get_rho_cp(sim)                      # 每节点 rho*Cp，需按实际API调整
        T_last = sim.temperature_last_step        # 上一时间步温度，需按实际API调整
        dt = sim.time_step                    # 热传导时间步长，需按实际API调整

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
        # 20260xxx 瞬态项：每条边分摊给 i_node 的 box_volume
        edge_box_volumes_diagonal = []

        i_nodes_non_diag = []
        j_nodes_non_diag = []

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
                    box_volume = edge.box_volume_all        # 20260xxx 需与 facet_area 对应的 box_volume 属性
                else:
                    facet_area = edge.facet_area_semi_ox
                    box_volume = edge.box_volume_semi_ox     # 20260xxx

                i_indices_diagonal.append(i_idx_diagonal)
                j_indices_diagonal.append(j_idx_diagonal)
                j_idx_diagonal += 1
                i_edges_diagonal.append(i_edge)
                edge_facet_areas_diagonal.append(facet_area)
                edge_lengths_diagonal.append(edge.length)
                i_nodes_diagonal.append(i_node)
                j_nodes_diagonal.append(i_other_node)
                edge_box_volumes_diagonal.append(box_volume)  # 20260xxx

                if i_other_node in map_i_node:
                    j_idx = map_idx_of_node[i_other_node]
                    i_indices_non_diagonal.append(i_idx)
                    j_indices_non_diagonal.append(j_idx)
                    i_edges_non_diagonal.append(i_edge)
                    edge_facet_areas_non_diagonal.append(facet_area)
                    edge_lengths_non_diagonal.append(edge.length)

                    i_nodes_non_diag.append(i_node)
                    j_nodes_non_diag.append(i_other_node)

            i_idx_diagonal += 1

        edge_facet_areas_diagonal = np.array(edge_facet_areas_diagonal)
        edge_lengths_diagonal =  np.array(edge_lengths_diagonal)
        edge_box_volumes_diagonal = np.array(edge_box_volumes_diagonal)  # 20260xxx

        edge_facet_areas_non_diagonal = np.array(edge_facet_areas_non_diagonal)
        edge_lengths_non_diagonal = np.array(edge_lengths_non_diagonal)

        # 20260xxx 瞬态项：每个节点的总 box_volume（只依赖网格几何，预处理阶段算一次即可）
        box_volume_nodes = sum_over_i_indices(i_indices_diagonal, j_indices_diagonal,
                                               edge_box_volumes_diagonal)

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
                                    i_nodes_non_diag, j_nodes_non_diag,
                                    i_indices_non_diagonal, j_indices_non_diagonal, i_edges_non_diagonal, edge_facet_areas_non_diagonal, edge_lengths_non_diagonal,
                                    ii_indices, i_indices_q, j_indices_q, i_indices_q_non_diagonal, j_indices_q_non_diagonal, i_nodes_q, j_nodes_q, i_edges_q, edge_lengths_q, edge_box_volumes_q,
                                    box_volume_nodes]  # 20260xxx 追加缓存
    else:
        (i_indices, i_nodes,
         i_indices_diagonal, j_indices_diagonal, i_edges_diagonal, edge_facet_areas_diagonal, edge_lengths_diagonal, i_nodes_diagonal, j_nodes_diagonal,
         i_nodes_non_diag, j_nodes_non_diag,
         i_indices_non_diagonal, j_indices_non_diagonal, i_edges_non_diagonal, edge_facet_areas_non_diagonal, edge_lengths_non_diagonal,
         ii_indices, i_indices_q, j_indices_q, i_indices_q_non_diagonal, j_indices_q_non_diagonal, i_nodes_q, j_nodes_q, i_edges_q, edge_lengths_q, edge_box_volumes_q,
         box_volume_nodes) = quan['vectrize_indices']   # 20260xxx 解包

    # thermal conduction term assemble
    temp_d = (T[j_nodes_diagonal] + T[i_nodes_diagonal]) / 2
    kappa_edges_diagonal = kappa[i_edges_diagonal] * (1 + alpha[i_edges_diagonal] * (temp_d - 300))

    Tj_diagonal = T[j_nodes_diagonal]
    Ti_diagonal = T[i_nodes_diagonal]

    A[i_indices,i_indices] -= sum_over_i_indices(i_indices_diagonal, j_indices_diagonal,  kappa_edges_diagonal*edge_facet_areas_diagonal/edge_lengths_diagonal)
    b[i_indices] -= sum_over_i_indices(i_indices_diagonal, j_indices_diagonal, kappa_edges_diagonal*edge_facet_areas_diagonal/edge_lengths_diagonal*(Tj_diagonal-Ti_diagonal))

    temp_nd = (T[i_nodes_non_diag] + T[j_nodes_non_diag]) / 2
    kappa_edges_non_diagonal = kappa[i_edges_non_diagonal] * (1 + alpha[i_edges_non_diagonal] * (temp_nd - 300))

    A[i_indices_non_diagonal,j_indices_non_diagonal]=kappa_edges_non_diagonal*edge_facet_areas_non_diagonal/edge_lengths_non_diagonal

    # ===== 20260xxx 瞬态项装配：rho*Cp*dT/dt =====
    # 残差 F_i^trans = rho_cp_i * box_volume_i * (T_i - T_i_last) / dt
    # 按现有约定 A = -dF/dT, b = +F
    if with_transient:
        rho_cp_nodes = rho_cp[i_nodes]           # i_nodes 与 i_indices 一一对应，顺序一致
        T_i = T[i_nodes]
        T_i_last = T_last[i_nodes]

        cap_coeff = rho_cp_nodes * box_volume_nodes / dt   # 标量数组，shape=(N,)

        A[i_indices, i_indices] -= cap_coeff
        b[i_indices] += cap_coeff * (T_i - T_i_last)

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
    Q_edges = (Je + Jh) * E_edges
    dQ_dTi = (dJe_dTi + dJh_dTi) * E_edges
    dQ_dTj = (dJe_dTj + dJh_dTj) * E_edges
    A[ii_indices,ii_indices] += sum_over_i_indices(i_indices_q, j_indices_q, dQ_dTi*edge_box_volumes_q)
    b[ii_indices] -= sum_over_i_indices(i_indices_q, j_indices_q, Q_edges * edge_box_volumes_q)

    A[i_indices_q_non_diagonal,j_indices_q_non_diagonal] +=  dQ_dTj * edge_box_volumes_q

    return (A,b)


def assemble_heat_trans(sim,quan,A,b):
    # Assemble for heat eaquation: rho*Cp*dT/dt - div(kappa*grad(T)) - |Q| = 0
    T=qs.get_temperature(sim)
    potential = qs.get_potential(sim)
    n=qs.get_electron_density(sim)
    p=qs.get_hole_density(sim)
    mobility_e=qs.get_mobility(sim,'electron')
    mobility_h=qs.get_mobility(sim,'hole')

    kappa, kappa_node, alpha = qs.get_kappa_and_alpha(sim)

    # kappa_node = qs.get_kappa_node(sim)

    # ===== 202608 瞬态项所需物理量 =====
    # TODO: 按实际接口名调整，例如 qs.get_rho_Cp(sim) 或
    # 由材料密度*比热组合而来的按节点数组 (shape = num_nodes)
    with_transient = sim.conf.get('heat_transient',False)
    # with_transient = True
    if with_transient:
        rho_cp = qs.get_rho_cp(sim)                      # 每节点 rho*Cp，需按实际API调整
        T_last = sim.temperature_last_step        # 上一时间步温度，需按实际API调整
        dt = sim.time_step                    # 热传导时间步长，需按实际API调整

    # ===== 20260908 Robin 边界所需物理量 =====
    with_robin = sim.conf.get('heat_robin_boundary', False)
    if with_robin:
        T_ambient = sim.conf.get('heat_robin_T_ambient', 300.0)
        L_ext_default = sim.conf.get('heat_robin_L_ext', 1e-6)
        # L_ext_by_material = sim.conf.get('heat_robin_L_ext_by_material', {})

    if quan['vectrize_indices'] is None:
        map_idx,map_i_node,map_idx_of_node=sim.mapping_info[quan['name']]
        robin_node_idx = sim.node_idx_of_thremal_robin_boundary

        i_indices = map_idx
        i_nodes = map_i_node

        i_indices_diagonal = []
        j_indices_diagonal = []
        i_edges_diagonal = []
        edge_facet_areas_diagonal = []
        edge_lengths_diagonal = []
        i_nodes_diagonal = []
        j_nodes_diagonal = []
        # 202608 瞬态项：每个节点的 box_volume，直接取节点属性（参考 Poisson 写法），
        # 不再需要按边累加 + sum_over_i_indices
        box_volume_nodes = []

        i_nodes_non_diag = []
        j_nodes_non_diag = []

        i_indices_non_diagonal = []
        j_indices_non_diagonal = []
        i_edges_non_diagonal = []
        edge_facet_areas_non_diagonal = []
        edge_lengths_non_diagonal = []

        # ===== 20260908 Robin 边界：与 box_volume_nodes 同一循环内一并收集 =====
        i_indices_robin = []
        i_nodes_robin = []
        h_eff_robin = []
        area_robin = []

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
                j_indices_diagonal.append(j_idx_diagonal)
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

                    i_nodes_non_diag.append(i_node)
                    j_nodes_non_diag.append(i_other_node)

            i_idx_diagonal += 1

            # 202608 瞬态项：直接取节点的 box_volume 属性（与 Poisson 写法一致）
            if sim.conf['with_conductor_temperature']:
                box_volume_nodes.append(node.box_volume_all)
            else:
                box_volume_nodes.append(node.box_volume_semi_ox)

            # ===== 20260908 Robin 边界：在同一循环内直接判断并收集该节点 =====
            if with_robin:
                boundary_area = node.boundary_facet_area
                if boundary_area > 0.0:
                    # TODO: 如果只想在特定材料/特定接触区域施加 Robin 边界，
                    # 而不是对所有暴露在外部边界的节点都施加，在这里加筛选，例如：
                    # material = sim.node_materials[i_node]
                    # if material not in sim.device.robin_boundary_materials:
                    #     continue（此处用 continue 会跳过 box_volume_nodes 之外内容，
                    #     实际写法应改成 if 包裹整个 append 块，而不是 continue）
                    kappa_node_i = kappa_node[i_node] 
                    # print("kappa_node 在 robin 节点上的取值:", kappa_node_i) 返回全0值，更改get函数
                    kappa_alpha_i = kappa_node_i * (1 + 0.01 * (T[i_node] - 300))

                    # L_ext = L_ext_by_material.get(material, L_ext_default)
                    L_ext = L_ext_default
                    if L_ext is not None and L_ext > 0.0 and (i_node in robin_node_idx):
                        i_indices_robin.append(i_idx)
                        i_nodes_robin.append(i_node)
                        h_eff_robin.append(kappa_alpha_i / L_ext)
                        area_robin.append(boundary_area)
                    # 若该材料未配置 L_ext，则跳过，不对该节点施加 Robin 边界

        edge_facet_areas_diagonal = np.array(edge_facet_areas_diagonal)
        edge_lengths_diagonal =  np.array(edge_lengths_diagonal)
        box_volume_nodes = np.array(box_volume_nodes)  # 20260xxx，顺序与 i_indices/i_nodes 一致

        edge_facet_areas_non_diagonal = np.array(edge_facet_areas_non_diagonal)
        edge_lengths_non_diagonal = np.array(edge_lengths_non_diagonal)

        # 20260908 Robin 边界：转换为数组（若 with_robin=False，均为空数组）
        i_indices_robin = np.array(i_indices_robin, dtype=np.int64)
        i_nodes_robin = np.array(i_nodes_robin, dtype=np.int64)
        h_eff_robin = np.array(h_eff_robin, dtype=np.float64)
        area_robin = np.array(area_robin, dtype=np.float64)

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
                                    i_nodes_non_diag, j_nodes_non_diag,
                                    i_indices_non_diagonal, j_indices_non_diagonal, i_edges_non_diagonal, edge_facet_areas_non_diagonal, edge_lengths_non_diagonal,
                                    ii_indices, i_indices_q, j_indices_q, i_indices_q_non_diagonal, j_indices_q_non_diagonal, i_nodes_q, j_nodes_q, i_edges_q, edge_lengths_q, edge_box_volumes_q,
                                    box_volume_nodes,
                                    i_indices_robin, i_nodes_robin, h_eff_robin, area_robin]  # 20260908 追加缓存
    else:
        (i_indices, i_nodes,
         i_indices_diagonal, j_indices_diagonal, i_edges_diagonal, edge_facet_areas_diagonal, edge_lengths_diagonal, i_nodes_diagonal, j_nodes_diagonal,
         i_nodes_non_diag, j_nodes_non_diag,
         i_indices_non_diagonal, j_indices_non_diagonal, i_edges_non_diagonal, edge_facet_areas_non_diagonal, edge_lengths_non_diagonal,
         ii_indices, i_indices_q, j_indices_q, i_indices_q_non_diagonal, j_indices_q_non_diagonal, i_nodes_q, j_nodes_q, i_edges_q, edge_lengths_q, edge_box_volumes_q,
         box_volume_nodes,
         i_indices_robin, i_nodes_robin, h_eff_robin, area_robin) = quan['vectrize_indices']   # 20260xxx 解包

    # thermal conduction term assemble
    temp_d = (T[j_nodes_diagonal] + T[i_nodes_diagonal]) / 2
    kappa_edges_diagonal = kappa[i_edges_diagonal] * (1 + alpha[i_edges_diagonal] * (temp_d - 300))

    Tj_diagonal = T[j_nodes_diagonal]
    Ti_diagonal = T[i_nodes_diagonal]

    A[i_indices,i_indices] -= sum_over_i_indices(i_indices_diagonal, j_indices_diagonal,  kappa_edges_diagonal*edge_facet_areas_diagonal/edge_lengths_diagonal)
    b[i_indices] -= sum_over_i_indices(i_indices_diagonal, j_indices_diagonal, kappa_edges_diagonal*edge_facet_areas_diagonal/edge_lengths_diagonal*(Tj_diagonal-Ti_diagonal))

    temp_nd = (T[i_nodes_non_diag] + T[j_nodes_non_diag]) / 2
    kappa_edges_non_diagonal = kappa[i_edges_non_diagonal] * (1 + alpha[i_edges_non_diagonal] * (temp_nd - 300))

    A[i_indices_non_diagonal,j_indices_non_diagonal]=kappa_edges_non_diagonal*edge_facet_areas_non_diagonal/edge_lengths_non_diagonal

    # ===== 20260908 Robin（对流型）边界条件装配 =====
    # -kappa*dT/dn|_boundary = h_eff*(T_boundary - T_ambient)
    # 残差 F_i^Robin = h_eff_i * A_boundary_i * (T_i - T_ambient)
    # 按 A = -dF/dT, b = +F 的约定；h_eff_robin/area_robin 已在预处理阶段
    # 按材料 kappa 和 L_ext 算好，此处只做向量化数值计算
    # 相当于额外增加一项，当前节点在原来位置也正常计算，更新也会更新在原始正常计算节点上
    if with_robin and len(i_indices_robin) > 0:
        T_boundary = T[i_nodes_robin]
        robin_coeff = h_eff_robin * area_robin  # shape=(N_robin,)，预处理阶段已算好，无需再乘

        A[i_indices_robin, i_indices_robin] -= robin_coeff
        b[i_indices_robin] += robin_coeff * (T_boundary - T_ambient)


    # ===== 20260xxx 瞬态项装配：rho*Cp*dT/dt =====
    # 残差 F_i^trans = rho_cp_i * box_volume_i * (T_i - T_i_last) / dt
    # 按现有约定 A = -dF/dT, b = +F
    if with_transient:
        rho_cp_nodes = rho_cp[i_nodes]           # i_nodes 与 i_indices 一一对应，顺序一致
        T_i = T[i_nodes]
        T_i_last = T_last[i_nodes]

        cap_coeff = rho_cp_nodes * box_volume_nodes / dt   # 标量数组，shape=(N,)

        A[i_indices, i_indices] -= cap_coeff
        b[i_indices] += cap_coeff * (T_i - T_i_last)

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
    Q_edges = (Je + Jh) * E_edges
    dQ_dTi = (dJe_dTi + dJh_dTi) * E_edges
    dQ_dTj = (dJe_dTj + dJh_dTj) * E_edges
    A[ii_indices,ii_indices] += sum_over_i_indices(i_indices_q, j_indices_q, dQ_dTi*edge_box_volumes_q)
    b[ii_indices] -= sum_over_i_indices(i_indices_q, j_indices_q, Q_edges * edge_box_volumes_q)

    A[i_indices_q_non_diagonal,j_indices_q_non_diagonal] +=  dQ_dTj * edge_box_volumes_q

    return (A,b)


# def assemble_heat(sim, quan, A, b):
#     """
#     Heat 向量化组装
#     """
#     # 获取物理量
#     T = qs.get_temperature(sim)
#     n = qs.get_electron_density(sim)
#     p = qs.get_hole_density(sim)
#     potential = qs.get_potential(sim)
#     mobility_e=qs.get_mobility(sim,'electron',location='edge')
#     mobility_h=qs.get_mobility(sim,'hole',location='edge')
#     power_density, power_density_node0_index = qs.get_power_density(sim, location='edge')
#     # kappa = qs.get_kappa(sim)
#     kappa, alpha = qs.get_kappa_and_alpha(sim)
    
#     # ===== 预处理阶段 =====
#     if quan.get('vectorize_indices_heat') is None:
#         map_idx, map_i_node, map_idx_of_node = sim.mapping_info[quan['name']]

#         i_indices_non_diag = []
#         j_indices_non_diag = []
#         i_edges_non_diag = []
#         i_nodes_non_diag = []
#         j_nodes_non_diag = []
#         edge_lengths_non_diag = []
#         interface_areas_non_diag = []
#         box_volumes_non_diag = []
#         power_node0_flags_non_diag = []
        
#         i_indices_diag = []
#         j_indices_diag = []
#         i_edges_diag = []
#         i_nodes_diag = []
#         j_nodes_diag = []
#         edge_lengths_diag = []
#         interface_areas_diag = []
#         box_volumes_diag = []
#         power_node0_flags_diag = []
        
#         i_indices = map_idx
#         i_nodes = map_i_node
        
#         # ✓ 关键修复：使用连续的顺序索引
#         i_idx_sequential = 0  # 顺序索引，从 0 开始
        
#         for i_idx, i_node in zip(map_idx, map_i_node):
#             node = sim.mesh.nodes[i_node]
#             j_idx_diag = 0
            
#             for i_edge, i_other_node in zip(node.edges, node.other_nodes):
#                 edge = sim.mesh.edges[i_edge]
                
#                 if sim.conf['with_conductor_temperature']:
#                     facet_area = edge.facet_area_all
#                 else:
#                     facet_area = edge.facet_area_semi_ox       
#                 power_node0_flag = 1 if power_density_node0_index[i_edge] == i_node else 0
                
#                 # 非对角元素
#                 if i_other_node in map_i_node:
#                     j_idx = map_idx_of_node[i_other_node]
#                     i_indices_non_diag.append(i_idx)
#                     j_indices_non_diag.append(j_idx)
#                     i_edges_non_diag.append(i_edge)
#                     i_nodes_non_diag.append(i_node)
#                     j_nodes_non_diag.append(i_other_node)
#                     edge_lengths_non_diag.append(edge.length)
#                     # interface_areas_non_diag.append(edge.interface_area_semi_ox)
#                     # box_volumes_non_diag.append(edge.box_volume_semi_ox)
#                     interface_areas_non_diag.append(facet_area)
#                     box_volumes_non_diag.append(edge.box_volume_all)
#                     power_node0_flags_non_diag.append(power_node0_flag)
                
#                 # ✓ 对角元素：使用顺序索引
#                 i_indices_diag.append(i_idx_sequential)  # ← 改为顺序索引
#                 j_indices_diag.append(j_idx_diag)
#                 i_edges_diag.append(i_edge)
#                 i_nodes_diag.append(i_node)
#                 j_nodes_diag.append(i_other_node)
#                 edge_lengths_diag.append(edge.length)
#                 interface_areas_diag.append(facet_area)
#                 box_volumes_diag.append(edge.box_volume_all)
#                 power_node0_flags_diag.append(power_node0_flag)
                
#                 j_idx_diag += 1
            
#             i_idx_sequential += 1  # ← 每个节点后递增
        
#         # 转换为 NumPy 数组
#         vectorize_indices_heat = {
#             'i_indices': i_indices,
#             'i_nodes': i_nodes,
#             'i_indices_non_diag': np.array(i_indices_non_diag, dtype=np.int32),
#             'j_indices_non_diag': np.array(j_indices_non_diag, dtype=np.int32),
#             'i_edges_non_diag': np.array(i_edges_non_diag, dtype=np.int32),
#             'i_nodes_non_diag': np.array(i_nodes_non_diag, dtype=np.int32),
#             'j_nodes_non_diag': np.array(j_nodes_non_diag, dtype=np.int32),
#             'edge_lengths_non_diag': np.array(edge_lengths_non_diag, dtype=np.float64),
#             'interface_areas_non_diag': np.array(interface_areas_non_diag, dtype=np.float64),
#             'box_volumes_non_diag': np.array(box_volumes_non_diag, dtype=np.float64),
#             'power_node0_flags_non_diag': np.array(power_node0_flags_non_diag, dtype=np.int32),
#             'i_indices_diag': np.array(i_indices_diag, dtype=np.int32),
#             'j_indices_diag': np.array(j_indices_diag, dtype=np.int32),
#             'i_edges_diag': np.array(i_edges_diag, dtype=np.int32),
#             'i_nodes_diag': np.array(i_nodes_diag, dtype=np.int32),
#             'j_nodes_diag': np.array(j_nodes_diag, dtype=np.int32),
#             'edge_lengths_diag': np.array(edge_lengths_diag, dtype=np.float64),
#             'interface_areas_diag': np.array(interface_areas_diag, dtype=np.float64),
#             'box_volumes_diag': np.array(box_volumes_diag, dtype=np.float64),
#             'power_node0_flags_diag': np.array(power_node0_flags_diag, dtype=np.int32),
#         }
        
#         quan['vectorize_indices_heat'] = vectorize_indices_heat
    
#     else:
#         vectorize_indices_heat = quan['vectorize_indices_heat']
    
#     # 解包
#     i_indices = vectorize_indices_heat['i_indices']
#     i_nodes = vectorize_indices_heat['i_nodes']
#     i_indices_non_diag = vectorize_indices_heat['i_indices_non_diag']
#     j_indices_non_diag = vectorize_indices_heat['j_indices_non_diag']
#     i_edges_non_diag = vectorize_indices_heat['i_edges_non_diag']
#     i_nodes_non_diag = vectorize_indices_heat['i_nodes_non_diag']
#     j_nodes_non_diag = vectorize_indices_heat['j_nodes_non_diag']
#     edge_lengths_non_diag = vectorize_indices_heat['edge_lengths_non_diag']
#     interface_areas_non_diag = vectorize_indices_heat['interface_areas_non_diag']
#     box_volumes_non_diag = vectorize_indices_heat['box_volumes_non_diag']
#     power_node0_flags_non_diag = vectorize_indices_heat['power_node0_flags_non_diag']
#     i_indices_diag = vectorize_indices_heat['i_indices_diag']
#     j_indices_diag = vectorize_indices_heat['j_indices_diag']
#     i_edges_diag = vectorize_indices_heat['i_edges_diag']
#     i_nodes_diag = vectorize_indices_heat['i_nodes_diag']
#     j_nodes_diag = vectorize_indices_heat['j_nodes_diag']
#     edge_lengths_diag = vectorize_indices_heat['edge_lengths_diag']
#     interface_areas_diag = vectorize_indices_heat['interface_areas_diag']
#     box_volumes_diag = vectorize_indices_heat['box_volumes_diag']
#     power_node0_flags_diag = vectorize_indices_heat['power_node0_flags_diag']
    
#     # ===== 向量化计算 =====
    
#     # 非对角元素
#     if len(i_edges_non_diag) > 0:
#         ## origin
#         # kappa_edges_nd = kappa[i_edges_non_diag] * (1 + 0.01 * (T[i_nodes_non_diag] - 300))
#         # v_i_nd = potential[i_nodes_non_diag]
#         # v_j_nd = potential[j_nodes_non_diag]
#         # temp_nd = (T[i_nodes_non_diag] + T[j_nodes_non_diag]) / 2

#         ## 20260129
#         temp_nd = (T[i_nodes_non_diag] + T[j_nodes_non_diag]) / 2
#         kappa_edges_nd = kappa[i_edges_non_diag] * (1 + alpha[i_edges_non_diag] * (temp_nd - 300))
#         v_i_nd = potential[i_nodes_non_diag]
#         v_j_nd = potential[j_nodes_non_diag]

        
#         power_density_nd = np.where(
#             power_node0_flags_non_diag == 1,
#             power_density[i_edges_non_diag, 0],
#             power_density[i_edges_non_diag, 1]
#         )
#         sign_Q_nd = np.sign(power_density_nd)
#         sign_Q_nd = np.where(sign_Q_nd == 0, 1.0, sign_Q_nd)
        
#         A_non_diag_base = -kappa_edges_nd * interface_areas_non_diag / edge_lengths_non_diag
        
#         A_non_diag_e = 0.0
#         if sim.conf['with_electron']:
#             sg_dTj_e = phys.scharfetter_gummel_dTj(
#                 n[i_nodes_non_diag], n[j_nodes_non_diag],
#                 v_i_nd, v_j_nd, edge_lengths_non_diag,
#                 mobility_e[i_edges_non_diag], temp_nd, type='electron'
#             )
#             A_non_diag_e = -sign_Q_nd * box_volumes_non_diag * \
#                           ((v_i_nd - v_j_nd) / edge_lengths_non_diag) * sg_dTj_e
        
#         A_non_diag_h = 0.0
#         if sim.conf['with_hole']:
#             sg_dTj_h = phys.scharfetter_gummel_dTj(
#                 p[i_nodes_non_diag], p[j_nodes_non_diag],
#                 v_i_nd, v_j_nd, edge_lengths_non_diag,
#                 mobility_h[i_edges_non_diag], temp_nd, type='hole'
#             )
#             A_non_diag_h = -sign_Q_nd * box_volumes_non_diag * \
#                           ((v_i_nd - v_j_nd) / edge_lengths_non_diag) * sg_dTj_h
        
#         A[i_indices_non_diag, j_indices_non_diag] = A_non_diag_base + A_non_diag_e + A_non_diag_h
    
#     # 对角元素和右端项
#     ## origin
#     # kappa_edges_d = kappa[i_edges_diag] * (1 + 0.01 * (T[i_nodes_diag] - 300))
#     # v_i_d = potential[i_nodes_diag]
#     # v_j_d = potential[j_nodes_diag]
#     # temp_d = (T[i_nodes_diag] + T[j_nodes_diag]) / 2
#     ## 20260129
#     temp_d = (T[i_nodes_diag] + T[j_nodes_diag]) / 2
#     kappa_edges_d = kappa[i_edges_diag] * (1 + alpha[i_edges_diag] * (temp_d - 300))
#     v_i_d = potential[i_nodes_diag]
#     v_j_d = potential[j_nodes_diag]

    
#     power_density_d = np.where(
#         power_node0_flags_diag == 1,
#         power_density[i_edges_diag, 0],
#         power_density[i_edges_diag, 1]
#     )
#     sign_Q_d = np.sign(power_density_d)
#     sign_Q_d = np.where(sign_Q_d == 0, 1.0, sign_Q_d)
    
#     A_diag_base = kappa_edges_d * interface_areas_diag / edge_lengths_diag
    
#     A_diag_e = 0.0
#     if sim.conf['with_electron']:
#         sg_dTi_e = phys.scharfetter_gummel_dTi(
#             n[i_nodes_diag], n[j_nodes_diag],
#             v_i_d, v_j_d, edge_lengths_diag,
#             mobility_e[i_edges_diag], temp_d, type='electron'
#         )
#         A_diag_e = -sign_Q_d * box_volumes_diag * \
#                    ((v_i_d - v_j_d) / edge_lengths_diag) * sg_dTi_e
    
#     A_diag_h = 0.0
#     if sim.conf['with_hole']:
#         sg_dTi_h = phys.scharfetter_gummel_dTi(
#             p[i_nodes_diag], p[j_nodes_diag],
#             v_i_d, v_j_d, edge_lengths_diag,
#             mobility_h[i_edges_diag], temp_d, type='hole'
#         )
#         A_diag_h = -sign_Q_d * box_volumes_diag * \
#                    ((v_i_d - v_j_d) / edge_lengths_diag) * sg_dTi_h
    
#     # ✓ 累加对角元素（现在 i_indices_diag 是连续的 0, 1, 2, ...）
#     A_diag_total = A_diag_base + A_diag_e + A_diag_h
#     diag_values = sum_over_i_indices(i_indices_diag, j_indices_diag, A_diag_total)
#     A[i_indices, i_indices] += diag_values
    
#     # ✓ 右端项
#     b_heat_cond = kappa_edges_d * interface_areas_diag / edge_lengths_diag * \
#                   (T[j_nodes_diag] - T[i_nodes_diag])
#     b_power = np.abs(power_density_d) * box_volumes_diag
#     b_total = b_heat_cond + b_power
#     b_values = sum_over_i_indices(i_indices_diag, j_indices_diag, b_total)
#     b[i_indices] += b_values
    
#     return (A, b)


## assemble ion transport 20250922

def assemble_ion(sim,quan,A,b):
    potential = qs.get_potential(sim)
    D0=qs.get_D0(sim)   # defusion coefficient pre-factor on edges
    Ea=qs.get_Ea(sim)   # activation energy on edges
    T = qs.get_temperature(sim)
    # R = qs.get_ion_recombination(sim,T)
    # R = np.zeros(len(sim.mesh.nodes))  # no recombination for ion
    R = qs.get_ion_recombination(sim, T)
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

    if sim.conf['with_soret']:
        Delta = 2.0*cons['q']/(cons['k']*Tij)*(v_j - v_i) - Ea[i_edges_diagonal]/(cons['k']*Tij**2) * (T_j - T_i)
    else:
        Delta = 2.0*cons['q']/(cons['k']*Tij)*(v_j - v_i)
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

    if sim.conf['with_soret']:
        Delta = 2.0*cons['q']/(cons['k']*Tij)*(v_j - v_i) - Ea[i_edges_non_diagonal]/(cons['k']*Tij**2) * (T_j - T_i)
    else:
        Delta = 2.0*cons['q']/(cons['k']*Tij)*(v_j - v_i)
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

        if sim.conf['with_soret']:
            Delta = 2.0*cons['q']/(cons['k']*Tij)*(v_j - v_i) - Ea[i_edges_diagonal_p]/(cons['k']*Tij**2) * (T_j - T_i)
        else:
            Delta = 2.0*cons['q']/(cons['k']*Tij)*(v_j - v_i)
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

        if sim.conf['with_soret']:
            Delta = 2.0*cons['q']/(cons['k']*Tij)*(v_j - v_i) - Ea[i_edges_non_diagonal_p]/(cons['k']*Tij**2) * (T_j - T_i)
        else:
            Delta = 2.0*cons['q']/(cons['k']*Tij)*(v_j - v_i) 
        D_edge = D0[i_edges_non_diagonal_p] * np.exp(-Ea[i_edges_non_diagonal_p]/(cons['k']*Tij))
        dJij_dvj = D_edge/edge_lengths_non_diagonal_p * (c_i * phys.Bern_dx(Delta) + c_j * phys.Bern_dx(-Delta)) * 2.0*cons['q']/(cons['k']*Tij)

        A[i_indices_non_diagonal_p, j_indices_non_diagonal_p] = dJij_dvj * edge_facet_areas_non_diagonal_p


    return (A,b)