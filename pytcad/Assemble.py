from pytcad.Physics import constant as cons
import numpy as np
import pytcad.Physics as phys
import pytcad.Quantities as qs
import time

def assemble(sim,quan,A,b):
    if quan['name'] == 'potential':
        [A,b]=assemble_possion(sim,quan,A,b)
    elif quan['name'] == 'electron_density' or quan['name'] == 'hole_density':
        [A,b]=assemble_dd(sim,quan,A,b)
    elif quan['name'] == 'temperature':
        [A,b]=assemble_heat(sim,quan,A,b)  
    return A,b  

def assemble_possion(sim,quan,A,b):
    
    # start = time.time()
    potential=qs.get_potential(sim)
    n=qs.get_electron_density(sim)
    p=qs.get_hole_density(sim)
    T=qs.get_temperature(sim)
    epsilon=qs.get_epsilon(sim)
    
    # print(' time1 ', time.time() -start, 's')
    map_idx,map_i_node,map_idx_of_node=sim.mapping_info[quan['name']]
    _,map_i_node_carrier_e,map_idx_of_node_e=sim.mapping_info['electron_density']
    _,map_i_node_carrier_h,map_idx_of_node_h=sim.mapping_info['hole_density']

    # start = time.time()
    for i_idx,i_node in zip(map_idx,map_i_node): # loop for all nodes
        node=sim.mesh.nodes[i_node]
        charge_density=cons['q']*(-1.0*n[i_node]+1.0*p[i_node]+1.0*sim.ND[i_node]-1.0*sim.NA[i_node])

        A_ii=0
        b_i=0
        for i_edge,i_other_node in zip(node.edges,node.other_nodes):
            edge = sim.mesh.edges[i_edge]
            eps=epsilon[i_edge]*cons['Epsilon0']
            facet_area = edge.facet_area_semi_ox

            if i_other_node in map_i_node:
                j_idx=map_idx_of_node[i_other_node]
                A[i_idx,j_idx]=eps*facet_area/edge.length

            A_ii-=eps*facet_area/edge.length
            b_i-=eps*facet_area/edge.length*(potential[i_other_node]-potential[i_node])

        b_i-=charge_density*node.box_volume_semi
        # gummel contribution
        if sim.conf['method']=='gummel':                           
            A_ii-=node.box_volume_semi*cons['q']*(n[i_node]+p[i_node])/phys.thermal_potential(T[i_node])
        # assemble associated items in newton iteration
        if sim.conf['method']=='newton':      
            if sim.conf['with_electron'] and (i_node in map_i_node_carrier_e):
                j_idx=map_idx_of_node_e[i_node]
                A[i_idx,j_idx]+=-cons['q']*node.box_volume_semi
            if sim.conf['with_hole'] and (i_node in map_i_node_carrier_h):
                j_idx=map_idx_of_node_h[i_node]
                A[i_idx,j_idx]+=cons['q']*node.box_volume_semi

        A[i_idx,i_idx]+=A_ii
        b[i_idx]+=b_i

    # print(' time2 ', time.time()-start, 's')
    return (A,b)

def assemble_dd(sim, quan, A, b):    

    T = qs.get_temperature(sim)
    R,dR_dn,dR_dp = qs.get_recombination(sim)

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

    mobility = qs.get_mobility(sim, carrier_type)
    
    map_idx, map_i_node, map_idx_of_node = sim.mapping_info[quan['name']]
    _, _, map_idx_of_node_p = sim.mapping_info['potential']

    _,map_i_node_carrier_e,map_idx_of_node_e=sim.mapping_info['electron_density']
    _,map_i_node_carrier_h,map_idx_of_node_h=sim.mapping_info['hole_density']
    
    for i_idx, i_node in zip(map_idx, map_i_node):
        node = sim.mesh.nodes[i_node]
        v_i = potential[i_node]
        c_i = carrier_conc[i_node]

        A_ii = 0.0 
        A_iip = 0.0
        b_i = 0.0
        for i_edge,i_other_node in zip(node.edges,node.other_nodes):
            edge = sim.mesh.edges[i_edge]
            facet_area = edge.facet_area_semi
            
            mu_edge = mobility[i_edge]
            v_j = potential[i_other_node]
            c_j = carrier_conc[i_other_node]
            temp = (T[i_node] + T[i_other_node]) / 2

            sg_j = phys.scharfetter_gummel_j(v_i, v_j, edge.length, mu_edge, temp, type=carrier_type)
            sg_i = phys.scharfetter_gummel_i(v_i, v_j, edge.length, mu_edge, temp, type=carrier_type)
            
            if i_other_node in map_i_node:
                j_idx = map_idx_of_node[i_other_node]
                A[i_idx, j_idx] = sg_j * facet_area
            
            A_ii += sg_i * facet_area
            b_i -= (c_i * sg_i + c_j * sg_j) * facet_area

            if sim.conf['method'] == 'newton':     
                sg_dvi = phys.scharfetter_gummel_dVi(c_i, c_j, v_i, v_j, edge.length, mu_edge, temp, type=carrier_type)
                A_iip += sg_dvi * facet_area
                if i_other_node in map_i_node:
                    j_idx = map_idx_of_node_p[i_other_node]
                    A[i_idx, j_idx] = (-sg_dvi) * facet_area

        if carrier_type == 'electron':
            b_i += cons['q'] * R[i_node] * node.box_volume_semi
        elif carrier_type == 'hole':
            b_i -= cons['q'] * R[i_node] * node.box_volume_semi

        if sim.conf['method'] == 'newton':
            ## 202508 - SRH coupling item
            if carrier_type == 'electron':
                ## 202508 - SRH coupling item
                A_ii += -cons['q'] * dR_dn[i_node] * node.box_volume_semi
                if i_node in map_i_node_carrier_h:
                    i_idx_eh=map_idx_of_node_h[i_node]
                    A[i_idx,i_idx_eh] += -cons['q'] * dR_dp[i_node] * node.box_volume_semi
            elif carrier_type == 'hole':
                ## 202508 - SRH coupling item
                A_ii += cons['q'] * dR_dp[i_node] * node.box_volume_semi
                if i_node in map_i_node_carrier_e:
                    i_idx_eh=map_idx_of_node_e[i_node]
                    A[i_idx,i_idx_eh] += cons['q'] * dR_dn[i_node] * node.box_volume_semi

        A[i_idx, i_idx] = A_ii
        b[i_idx] = b_i        
        if sim.conf['method'] == 'newton':  
            A[i_idx, map_idx_of_node_p[i_node]] = A_iip

    return (A, b)

def assemble_heat(sim,quan,A,b):
    # Assemble for heat eaquation
    T=qs.get_temperature(sim)
    potential = qs.get_potential(sim)
    n=qs.get_electron_density(sim)
    p=qs.get_hole_density(sim)
    # power_density=qs.get_power_density(sim)
    semi_node_idx = sim.node_idx_of_material('semiconductor')
    mobility_e=qs.get_mobility(sim,'electron')
    mobility_h=qs.get_mobility(sim,'hole')

    kappa=qs.get_kappa(sim)

    map_idx,map_i_node,map_idx_of_node=sim.mapping_info[quan['name']]

    for i_idx,i_node in zip(map_idx,map_i_node): # loop for all nodes
        node=sim.mesh.nodes[i_node]

        box_volume_node = 0
        A_ii = 0
        b_i = 0

        for i_edge,i_other_node in zip(node.edges, node.other_nodes):
            edge = sim.mesh.edges[i_edge]
            kappa_edge = kappa[i_edge]
            if sim.conf['with_conductor_temperature']:
                facet_area = edge.facet_area_all
            else:
                facet_area = edge.facet_area_semi_ox
            
            if i_other_node in map_i_node:
                j_idx=map_idx_of_node[i_other_node]
                A[i_idx,j_idx]=kappa_edge*facet_area/edge.length
            
            A_ii-=kappa_edge*facet_area/edge.length
            b_i-=kappa_edge*facet_area/edge.length*(T[i_other_node]-T[i_node])

            if (i_node in semi_node_idx) and (i_other_node in semi_node_idx) and (i_other_node in map_i_node):
                # calculate power density only for semiconductor nodes
                v_i, v_j = potential[i_node], potential[i_other_node]
                n_i, n_j = n[i_node], n[i_other_node]
                p_i, p_j = p[i_node], p[i_other_node]
                Tij = (T[i_node] + T[i_other_node])/2
                mu_e = mobility_e[i_edge]
                mu_h = mobility_h[i_edge]

                if sim.conf['with_electron']:
                    Je = phys.scharfetter_gummel(n_i,n_j,v_i,v_j,edge.length,mu_e,Tij,type='electron')
                    dJe_dTi = phys.scharfetter_gummel_dTi(n_i,n_j,v_i,v_j,edge.length,mu_e,Tij,type='electron')
                    dJe_dTj = phys.scharfetter_gummel_dTj(n_i,n_j,v_i,v_j,edge.length,mu_e,Tij,type='electron')
                else:
                    Je = 0.0
                    dJe_dTi = 0.0
                    dJe_dTj = 0.0
                if sim.conf['with_hole']:
                    Jh = phys.scharfetter_gummel(p_i,p_j,v_i,v_j,edge.length,mu_h,Tij,type='hole')
                    dJh_dTi = phys.scharfetter_gummel_dTi(p_i,p_j,v_i,v_j,edge.length,mu_h,Tij,type='hole')
                    dJh_dTj = phys.scharfetter_gummel_dTj(p_i,p_j,v_i,v_j,edge.length,mu_h,Tij,type='hole')
                else:
                    Jh = 0.0
                    dJh_dTi = 0.0
                    dJh_dTj = 0.0

                Qedge = -(Je + Jh)*(v_j - v_i)/edge.length
                b_i-=Qedge * edge.box_volume_semi  

                A_ii+= -(dJe_dTi + dJh_dTi)*(v_j - v_i)/edge.length * edge.box_volume_semi
                j_idx=map_idx_of_node[i_other_node]
                A[i_idx,j_idx]+=-(dJe_dTj + dJh_dTj)*(v_j - v_i)/edge.length * edge.box_volume_semi

        A[i_idx,i_idx]=A_ii
        b[i_idx]=b_i

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

    map_idx,map_i_node,map_idx_of_node=sim.mapping_info[quan['name']]
    _, map_i_node_p, map_idx_of_node_p = sim.mapping_info['potential']

    for i_idx,i_node in zip(map_idx,map_i_node): # loop for all nodes
        node=sim.mesh.nodes[i_node]
        v_i = potential[i_node]
        T_i = T[i_node]
        c_i = c_ion[i_node]
        c_i_last_step = ion_last_step[i_node]
        box_volume = node.box_volume_semi

        A_ii = 0.0
        b_i = 0.0

        A_iip = 0.0  # for newton coupling with potential

        for i_edge,i_other_node in zip(node.edges, node.other_nodes):
            edge = sim.mesh.edges[i_edge]
            v_j = potential[i_other_node]
            T_j = T[i_other_node]
            c_j = c_ion[i_other_node]
            Tij = (T_i + T_j)/2

            facet_area = edge.facet_area_semi
            length = edge.length

            ## with Soret effect
            Delta = 2.0*cons['q']/(cons['k']*Tij)*(v_j - v_i) - Ea[i_edge]/(cons['k']*Tij**2) * (T_j - T_i)

            ## without Soret effect
            # Delta = 2.0*cons['q']/(cons['k']*Tij)*(v_j - v_i)
            
            D_edge = D0[i_edge] * np.exp(-Ea[i_edge]/(cons['k']*Tij))
            # D_edge = D0[i_edge]

            dJij_dci =  D_edge / length * phys.Bern(Delta)
            dJij_dcj = -D_edge / length * phys.Bern(-Delta) 
            Jij = c_i * dJij_dci + c_j * dJij_dcj

            A_ii += dJij_dci * facet_area 
            b_i -= Jij * facet_area

            if i_other_node in map_i_node:
                j_idx=map_idx_of_node[i_other_node]
                A[i_idx,j_idx] = dJij_dcj * facet_area

            if sim.conf['method'] == 'newton':
                # Add newton coupling with potential
                dJij_dvi = - D_edge/length * (c_i * phys.Bern_dx(Delta) + c_j * phys.Bern_dx(-Delta)) * 2.0*cons['q']/(cons['k']*Tij) 
                dJij_dvj = - dJij_dvi
                if (i_node in map_i_node_p):
                    A_iip += dJij_dvi * facet_area
                if (i_other_node in map_i_node_p):
                    j_idx_p = map_idx_of_node_p[i_other_node]
                    A[i_idx,j_idx_p] = dJij_dvj * facet_area
            # if sim.conf['method'] == 'newton' and sim.conf['with_temperature']:
            #     # Add newton coupling with temperature
            #     dDelta_dTi = -cons['q']/(cons['k']*Tij**2)*(v_j - v_i) + Ea[i_edge]/(cons['k']*Tij**3) * (T_j - T_i) + Ea[i_edge]/(cons['k']*Tij**2) 
            #     dDelta_dTj = -cons['q']/(cons['k']*Tij**2)*(v_j - v_i) + Ea[i_edge]/(cons['k']*Tij**3) * (T_j - T_i) - Ea[i_edge]/(cons['k']*Tij**2)

            #     dJij_dti = 1/2 * Ea[i_edge]/(cons['k']*Tij**2) * Jij + D_edge/length * (c_i * phys.Bern_dx(Delta) + c_j * phys.Bern_dx(-Delta)) * dDelta_dTi
            #     dJij_dtj = 1/2 * Ea[i_edge]/(cons['k']*Tij**2) * Jij + D_edge/length * (c_i * phys.Bern_dx(Delta) + c_j * phys.Bern_dx(-Delta)) * dDelta_dTj

            #     _, map_i_node_t, map_idx_of_node_t = sim.mapping_info['temperature']
            #     if (i_node in map_i_node_t):
            #         i_idx_t = map_idx_of_node_t[i_node]
            #         A[i_idx, i_idx_t] += dJij_dti * facet_area
            #     if i_other_node in map_i_node_t:
            #         j_idx_t = map_idx_of_node_t[i_other_node]
            #         A[i_idx, j_idx_t] += dJij_dtj * facet_area  

        A_ii += 1 / time_step * box_volume
        b_i -= (R[i_node] + (c_i - c_i_last_step) / time_step) * box_volume

        A[i_idx,i_idx]=A_ii
        b[i_idx]=b_i

        if sim.conf['method'] == 'newton':
            if (i_node in map_i_node_p):
                j_idx_p = map_idx_of_node_p[i_node]
                A[i_idx,j_idx_p] += A_iip


    return (A,b)