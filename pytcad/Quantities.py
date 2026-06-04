import numpy as np
import scipy as sp
from pytcad.Physics import constant as cons
import pytcad.Physics as phys

def get_coordinate(sim): 
    return sim.mesh.x,sim.mesh.y,sim.mesh.z

def get_potential(sim):
    # potential in [V]
    for quan in sim.quans:
        if quan['name'] == 'potential':
            potential=quan['value']
            break
    return potential

def get_fermi_energy(sim,type='electron'):
    #  [eV]
    n=get_electron_density(sim)
    p=get_hole_density(sim)
    q=cons['q']
    k=cons['k']
    T=cons['T0']

    Ec=get_band_energy(sim,'electron')
    Ev=get_band_energy(sim,'hole')
    Ef=np.zeros(len(sim.mesh.nodes))

    nodes_shared_regions = np.zeros(len(sim.mesh.nodes))
    for reg_num,region in enumerate(sim.mesh.regions):
        node_idx=region.nodes
        if region.type != 'semiconductor':
            # exclude the nodes shared with the semiconductor region
            wo_semi_node_idx = list(set(node_idx)-set(sim.node_idx_of_material('semiconductor'))) 
            Ef[wo_semi_node_idx]=np.NAN
            continue
        material_region=region.material
        if type=='electron':
            Ef[node_idx] = Ec[node_idx]+k*T/q*np.log(n[node_idx]/material_region.Nc)
        elif type=='hole':
            Ef[node_idx] = Ev[node_idx]-k*T/q*np.log(p[node_idx]/material_region.Nv)
        nodes_shared_regions[node_idx] += 1
    shared_node_idx = nodes_shared_regions > 1
    # Ef[shared_node_idx] = Ef[shared_node_idx] / nodes_shared_regions[shared_node_idx]
    return Ef

def get_band_energy(sim,type='electron'):
    #  [eV]
    potential=get_potential(sim)
    E=np.zeros(len(sim.mesh.nodes))

    nodes_shared_regions = np.zeros(len(sim.mesh.nodes))
    for reg_num,region in enumerate(sim.mesh.regions):
        node_idx=region.nodes
        if region.type != 'semiconductor':
            # exclude the nodes shared with the semiconductor region
            wo_semi_node_idx = list(set(node_idx)-set(sim.node_idx_of_material('semiconductor'))) 
            E[wo_semi_node_idx]=np.NAN
            continue
        material_region=region.material
        if type=='electron':
            E[node_idx]=-(potential[node_idx])-material_region.affinity
        elif type=='hole':
            dopping=sim.NA[node_idx]+sim.ND[node_idx]
            E[node_idx]=-(potential[node_idx])-material_region.affinity-phys.band_gap(material_region,dopping)
        nodes_shared_regions[node_idx] += 1
    shared_node_idx = nodes_shared_regions > 1
    # E[shared_node_idx] = E[shared_node_idx] / nodes_shared_regions[shared_node_idx]
    return E

def get_electron_density(sim):
    # electron density in [m^-3]
    for quan in sim.quans:
        if quan['name'] == 'electron_density':
            n=quan['value']
            break
    return n

def get_effective_potential(sim,type='electron'):
    delta_potential = np.zeros(len(sim.mesh.nodes))

    T = get_temperature(sim)
    potential = get_potential(sim)
    nodes_shared_regions = np.zeros(len(sim.mesh.nodes))

    ref_material = sim.ref_material
    if type=='electron':
        delta_potential0 = ref_material.affinity + phys.thermal_potential(cons['T0']) * np.log(ref_material.Nc)
    elif type=='hole':
        delta_potential0 = ref_material.affinity + ref_material.band_gap_parameters['Eg0']- phys.thermal_potential(cons['T0']) * np.log(ref_material.Nv)
    for reg_num,region in enumerate(sim.mesh.regions):
        if region.type == 'semiconductor':
            node_idx=region.nodes
            material_region=region.material
            ni = phys.intrinsic_concentration(material_region,sim.ND[node_idx]+sim.NA[node_idx]) 
            if type=='electron':
                # delta_potential[node_idx] += phys.thermal_potential(T[node_idx]) * np.log(ni)
                delta_potential[node_idx] = material_region.affinity + phys.thermal_potential(T[node_idx]) * np.log(material_region.Nc) - delta_potential0
            elif type=='hole':
                # delta_potential[node_idx] += - phys.thermal_potential(T[node_idx]) * np.log(ni)
                delta_potential[node_idx] =  material_region.affinity + phys.band_gap(material_region, sim.NA[node_idx] + sim.ND[node_idx]) - phys.thermal_potential(T[node_idx]) * np.log(material_region.Nv)  - delta_potential0
            nodes_shared_regions[node_idx] += 1
    
    shared_node_idx = nodes_shared_regions > 1
    # delta_potential[shared_node_idx] = delta_potential[shared_node_idx] / nodes_shared_regions[shared_node_idx]
    # print(type)
    # print(delta_potential)
    return potential+delta_potential
    

def get_hole_density(sim):
    # hole density in [m^-3]
    for quan in sim.quans:
        if quan['name'] == 'hole_density':
            p=quan['value']
            break
    return p


def get_temperature(sim):
    # temperature in [T]
    for quan in sim.quans:
        if quan['name'] == 'temperature':
            T=quan['value']
            break
    return T

def field_cell(sim,fields,normals):
    if sim.mesh.mesh_type=='1D':
        dims=1
    elif sim.mesh.mesh_type in ['2D','cylindrical_2D']:
        dims=2
    elif sim.mesh.mesh_type=='3D':
        dims=3
    A=np.zeros((dims,dims))
    b=np.zeros(dims)
    for i in range(dims):
        for j in range(dims):
            for k,field in enumerate(fields):
                A[i][j]+=normals[k][i]*normals[k][j]
    for i in range(dims):
        for k,field in enumerate(fields):
            b[i]+=normals[k][i]*fields[k]
    field = sp.linalg.solve(A,b)
    if sim.mesh.mesh_type=='1D':
        return [field[0],0.0,0.0]
    elif sim.mesh.mesh_type in ['2D','cylindrical_2D']:
        return [field[0],field[1],0.0]
    elif sim.mesh.mesh_type=='3D':
        return [field[0],field[1],field[2]]
    
def get_electric_field(sim):
    # electric field in [V/m]
    # electric field is calculated in each node
    potential=get_potential(sim)

    Ex=np.zeros(len(sim.mesh.nodes))
    Ey=np.zeros(len(sim.mesh.nodes))
    Ez=np.zeros(len(sim.mesh.nodes))
    for i_node,node0 in enumerate(sim.mesh.nodes):
        potential0=potential[i_node]
        electric_field_edges=[]
        normal_edges=[]
        for i_edge,i_other_node in zip(node0.edges,node0.other_nodes):
            edge=sim.mesh.edges[i_edge]
            potential1=potential[i_other_node]
            node1=sim.mesh.nodes[i_other_node]
            electric_field_edges.append(-(potential1-potential0)/edge.length)
            normal_edges.append([(node1.x-node0.x)/edge.length,(node1.y-node0.y)/edge.length,(node1.z-node0.z)/edge.length])
        if len(electric_field_edges)>0:
            Ex[i_node], Ey[i_node], Ez[i_node] = field_cell(sim,electric_field_edges,normal_edges)

    return (Ex,Ey,Ez)

def get_current_density(sim,type='electron'):
    # current density in [A/m^2]
    # current density is calculated in each node   

    if sim.conf.get('uniform_band', True):
        # (default) for uniform band structure
        potential = get_potential(sim)  
    else:
        # for non-uniform band structure, the flux is calculated based on effective potential
        potential = get_effective_potential(sim,type=type)

    if type=='electron':
        conc=get_electron_density(sim)
    else:
        conc=get_hole_density(sim)
    T=get_temperature(sim)

    Jx=np.zeros(len(sim.mesh.nodes))
    Jy=np.zeros(len(sim.mesh.nodes))
    Jz=np.zeros(len(sim.mesh.nodes))

    semi_node_idx = sim.node_idx_of_material('semiconductor')

    mobility = get_mobility(sim,type)
    for i_node0 in semi_node_idx:
        node0=sim.mesh.nodes[i_node0]
        flux_edges=[]
        normal_edges=[]
        for i_edge,i_node1 in zip(node0.edges,node0.other_nodes):
            edge = sim.mesh.edges[i_edge]
            if i_node1 not in semi_node_idx:
                continue
            mu=mobility[i_edge]
            node1 = sim.mesh.nodes[i_node1]

            c0, c1 = conc[i_node0], conc[i_node1]
            T0, T1 = T[i_node0], T[i_node1]
            potential0, potential1 = potential[i_node0], potential[i_node1]
            temp=(T[i_node0]+T[i_node1])/2

            flux=phys.scharfetter_gummel(c0,c1,potential0,potential1,edge.length,mu,temp,type=type)

            flux_edges.append(flux)
            normal_edges.append([(node1.x-node0.x)/edge.length,(node1.y-node0.y)/edge.length,(node1.z-node0.z)/edge.length])
                
        if len(flux_edges)>0:
            Jx[i_node0], Jy[i_node0], Jz[i_node0] = field_cell(sim,flux_edges,normal_edges)
          

    return (Jx,Jy,Jz)

def get_epsilon(sim):
    # epsilon is calculated in each edge
    epsilon=np.zeros(len(sim.mesh.edges))

    # indicate how many semiconductor or oxide regions share the same edge
    edge_shared_regions = np.zeros(len(sim.mesh.edges))

    for reg_num,region in enumerate(sim.mesh.regions):
        if region.type == 'conductor':
            continue
        edge_idx=region.edges
        epsilon[edge_idx]+=region.material.epsilon
        edge_shared_regions[edge_idx]+=1

    # average the epsilon for edges shared by multiple semiconductor or oxide regions
    shared_edge_idx = edge_shared_regions>1
    epsilon[shared_edge_idx]=epsilon[shared_edge_idx]/edge_shared_regions[shared_edge_idx]
    return epsilon

def get_kappa(sim):
    # kappa is calculated in each edge
    kappa=np.zeros(len(sim.mesh.edges))
    edge_shared_regions = np.zeros(len(sim.mesh.edges))

    for reg_num,region in enumerate(sim.mesh.regions):
        # if region.type == 'conductor':
        #     continue
        edge_idx=region.edges
        material_region=region.material

        kappa[edge_idx]+=material_region.kappa
        edge_shared_regions[edge_idx]+=1
    shared_edge_idx = edge_shared_regions>1
    kappa[shared_edge_idx]=kappa[shared_edge_idx]/edge_shared_regions[shared_edge_idx]
    return kappa

## 20260309
def get_kappa_and_alpha(sim):
    num_edges = len(sim.mesh.edges)
    kappa = np.zeros(num_edges)
    # 新增：用于存储温度系数 (1.0 或 0.01)
    alpha = np.zeros(num_edges) 
    edge_shared_regions = np.zeros(num_edges)

    for reg_num, region in enumerate(sim.mesh.regions):
        edge_idx = region.edges
        edge_shared_regions[edge_idx] += 1
        
        material_region = region.material
        kappa[edge_idx] += material_region.kappa
        alpha[edge_idx] += material_region.kappa_alpha   

    # 处理共享边的平均值
    shared_mask = edge_shared_regions > 0
    kappa[shared_mask] /= edge_shared_regions[shared_mask]
    alpha[shared_mask] /= edge_shared_regions[shared_mask]
    
    return kappa, alpha

def get_D0(sim):
    # diffusion is calculated in each edge
    D0=np.zeros(len(sim.mesh.edges))
    edge_shared_regions = np.zeros(len(sim.mesh.edges))

    for reg_num,region in enumerate(sim.mesh.regions):
        if region.type == 'conductor':
            continue
        edge_idx=region.edges
        material_region=region.material

        D0[edge_idx]+=material_region.D0
        edge_shared_regions[edge_idx]+=1
    shared_edge_idx = edge_shared_regions>1
    D0[shared_edge_idx]=D0[shared_edge_idx]/edge_shared_regions[shared_edge_idx]
    return D0

## 20260311 Ea increased after Forimg (The nature of material changed),Forming effect 
# def get_ion_edge_density_max(sim):
#     ion_density=sim.ion_density_max
#     ion_density_edge=np.zeros(len(sim.mesh.edges))
#     for reg_num,region in enumerate(sim.mesh.regions):
#         if region.type == 'conductor':
#             continue
#         edge_idx=region.edges
#         for i, i_edge in enumerate(edge_idx):
#             edge = sim.mesh.edges[i_edge]
#             ion_density_edge[i] = np.mean(ion_density[edge.nodes])
#     return ion_density_edge

def get_ion_edge_density_max(sim):
    ion_density = sim.ion_density_max
    ion_density_edge = np.zeros(len(sim.mesh.edges))
    
    eps = 1e0
    log10_ion_density = np.log10(ion_density + eps)
    
    for reg_num, region in enumerate(sim.mesh.regions):
        if region.type == 'conductor':
            continue
            
        edge_idx = region.edges
        for i_edge in edge_idx:
            edge = sim.mesh.edges[i_edge]
            log_mean = np.mean(log10_ion_density[edge.nodes])
            ion_density_edge[i_edge] = 10**log_mean
            
    return ion_density_edge

# def get_Ea(sim):
#     # activiation energy is calculated in each edge
#     Ea=np.zeros(len(sim.mesh.edges))
#     ion_density=get_ion_edge_density_max(sim)
#     edge_shared_regions = np.zeros(len(sim.mesh.edges))

#     for reg_num,region in enumerate(sim.mesh.regions):
#         if region.type == 'conductor':
#             continue
#         edge_idx = np.asarray(region.edges)
#         material_region=region.material

#         mask_high = ion_density[edge_idx] >= 1e23   ## 1e25 5e34
#         mask_low = ~mask_high  
#         Ea[edge_idx[mask_low]] += material_region.Ea
#         x = (ion_density[edge_idx[mask_high]] / 5e24)**2   ## 1e25 5e24
#         Ea[edge_idx[mask_high]] += material_region.Ea - 0.4*cons['q'] * (x/(x+1))
#         # Ea[edge_idx[mask_high]] += material_region.Ea - 0.8*cons['q']
#         edge_shared_regions[edge_idx]+=1
#     shared_edge_idx = edge_shared_regions>1
#     Ea[shared_edge_idx]=Ea[shared_edge_idx]/edge_shared_regions[shared_edge_idx]
#     return Ea

def get_gen_Ea(sim):
    # activiation energy is calculated in each edge
    Ea_gen=np.zeros(len(sim.mesh.nodes))
    ion_density=sim.ion_density_max_all

    for reg_num,region in enumerate(sim.mesh.regions):
        if region.type == 'conductor':
            continue
        node_idx = np.asarray(region.nodes)
        material_region=region.material

        mask_high = ion_density[node_idx] >= 1e25   ## 1e25 5e34
        mask_low = ~mask_high  
        Ea_gen[node_idx[mask_low]] += material_region.Ea
        x = (ion_density[node_idx[mask_high]] / 5e24)**2   ## 1e25 5e24
        Ea_gen[node_idx[mask_high]] += material_region.Ea - 0.6*cons['q'] * (x/(x+1))
        # Ea[edge_idx[mask_high]] += material_region.Ea - 0.8*cons['q']
    return Ea_gen


# def get_Ea(sim):
#     # activiation energy is calculated in each edge
#     Ea=np.zeros(len(sim.mesh.edges))
#     edge_shared_regions = np.zeros(len(sim.mesh.edges))

#     for reg_num,region in enumerate(sim.mesh.regions):
#         if region.type == 'conductor':
#             continue
#         edge_idx=region.edges
#         material_region=region.material

#         Ea[edge_idx]+=material_region.Ea
#         edge_shared_regions[edge_idx]+=1
#     shared_edge_idx = edge_shared_regions>1
#     Ea[shared_edge_idx]=Ea[shared_edge_idx]/edge_shared_regions[shared_edge_idx]
#     return Ea

def get_Ea(sim):
    # activiation energy is calculated in each edge
    Ea=np.zeros(len(sim.mesh.edges))
    ion_density=get_ion_edge_density_max(sim)
    edge_shared_regions = np.zeros(len(sim.mesh.edges))

    for reg_num,region in enumerate(sim.mesh.regions):
        if region.type == 'conductor':
            continue
        edge_idx = np.asarray(region.edges)
        material_region=region.material

        mask_high = ion_density[edge_idx] >= 1e25   ## 1e25 5e34
        mask_low = ~mask_high  
        Ea[edge_idx[mask_low]] += material_region.Ea
        x = (ion_density[edge_idx[mask_high]] / 1e25)**5   ## 1e25 5e24
        Ea[edge_idx[mask_high]] += material_region.Ea - 0.015*cons['q'] * (x/(x+1))
        # Ea[edge_idx[mask_high]] += material_region.Ea - 0.04
        edge_shared_regions[edge_idx]+=1
    shared_edge_idx = edge_shared_regions>1
    Ea[shared_edge_idx]=Ea[shared_edge_idx]/edge_shared_regions[shared_edge_idx]
    return Ea

# def get_Ea(sim):
#     # activiation energy is calculated in each edge
#     Ea=np.zeros(len(sim.mesh.edges))
#     ion_density=get_ion_edge_density_max(sim)
#     edge_shared_regions = np.zeros(len(sim.mesh.edges))

#     for reg_num,region in enumerate(sim.mesh.regions):
#         if region.type == 'conductor':
#             continue
#         edge_idx = np.asarray(region.edges)
#         material_region=region.material

#         mask_high = ion_density[edge_idx] >= 9.9e24   ## 1e25 5e34
#         mask_low = ~mask_high  
#         Ea[edge_idx[mask_low]] += material_region.Ea
#         x = (ion_density[edge_idx[mask_high]] / 1e25)**2   ## 1e25 5e24
#         Ea[edge_idx[mask_high]] += material_region.Ea + 0.1*cons['q'] * (x/(x+1))
#         # Ea[edge_idx[mask_high]] += material_region.Ea - 0.04
#         edge_shared_regions[edge_idx]+=1
#     shared_edge_idx = edge_shared_regions>1
#     Ea[shared_edge_idx]=Ea[shared_edge_idx]/edge_shared_regions[shared_edge_idx]
#     return Ea


def get_mobility(sim,type='electron'): 
    # mobility in [m^2/Vs]
    # mobility is calculated in each edge
    mobility_edges=np.zeros(len(sim.mesh.edges))

    # indicate how many semiconductor regions share the same edge
    edge_shared_regions = np.zeros(len(sim.mesh.edges))

    n=get_electron_density(sim)
    p=get_hole_density(sim)
    T=get_temperature(sim)
    NA=sim.NA
    ND=sim.ND
    mob=phys.Mobility()
    for reg_num,region in enumerate(sim.mesh.regions):
        if region.type != 'semiconductor':
            continue
        edge_idx=region.edges
        material_region=region.material

        mobility_model=material_region.mobility_model
        mobility_para=material_region.mobility_parameters

        if mobility_model=='constant':
            mu=mob.const(mobility_para,type=type)
        elif mobility_model=='latice':
            T_region = np.zeros(len(edge_idx))
            for i, i_edge in enumerate(edge_idx):
                edge = sim.mesh.edges[i_edge]
                T_region[i] = np.mean(T[edge.nodes])
            mu=mob.latice(mobility_para,T_region,type=type)
        elif mobility_model=='latice_impurity':
            T_region = np.zeros(len(edge_idx))
            NA_region = np.zeros(len(edge_idx))
            ND_region = np.zeros(len(edge_idx))
            for i, i_edge in enumerate(edge_idx):
                edge = sim.mesh.edges[i_edge]
                T_region[i] = np.mean(T[edge.nodes])
                NA_region[i] = np.mean(NA[edge.nodes])
                ND_region[i] = np.mean(ND[edge.nodes])
            mu=mob.latice_impurity(mobility_para,T_region,NA_region+ND_region,type=type)
        elif mobility_model=='latice_impurity_carrier':
            T_region = np.zeros(len(edge_idx))
            n_region = np.zeros(len(edge_idx))
            p_region = np.zeros(len(edge_idx))
            NA_region = np.zeros(len(edge_idx))
            ND_region = np.zeros(len(edge_idx))
            for i, i_edge in enumerate(edge_idx):
                edge = sim.mesh.edges[i_edge]
                T_region[i] = np.mean(T[edge.nodes])
                n_region[i] = np.mean(n[edge.nodes])
                p_region[i] = np.mean(p[edge.nodes])
                NA_region[i] = np.mean(NA[edge.nodes])
                ND_region[i] = np.mean(ND[edge.nodes])
            mu=mob.latice_impurity_carrier(mobility_para,T_region,NA_region+ND_region,n_region,p_region,type=type)
        ## 20251201 RRAM mobility
        elif mobility_model=='rram_carrier_1':
            ion_density_region = np.zeros(len(edge_idx))
            for i, i_edge in enumerate(edge_idx):
                edge = sim.mesh.edges[i_edge]
                ion_density_region[i] = np.mean(sim.ion_density[edge.nodes])
                # ion_density_region[i] = np.min(sim.ion_density[edge.nodes])
            mu=mob.rram_carrier_1(mobility_para,ion_density_region,type=type)
        elif mobility_model=='rram_carrier_2':
            ion_density_region = np.zeros(len(edge_idx))
            for i, i_edge in enumerate(edge_idx):
                edge = sim.mesh.edges[i_edge]
                ion_density_region[i] = np.mean(sim.ion_density[edge.nodes])
                # ion_density_region[i] = np.min(sim.ion_density[edge.nodes])
            mu=mob.rram_carrier_2(mobility_para,ion_density_region,type=type)

        else:
            raise ValueError('Mobility model type is not defined!')
        mobility_edges[edge_idx]+=mu
        edge_shared_regions[edge_idx]+=1
    
    # average the mobility for edges shared by multiple semiconductor regions
    shared_edge_idx = edge_shared_regions>1
    mobility_edges[shared_edge_idx]=mobility_edges[shared_edge_idx]/edge_shared_regions[shared_edge_idx]
    return mobility_edges


def get_recombination(sim): 
    # recombination in [m^-3s^-1]
    # recombination is calculated in each node
    R=np.zeros(len(sim.mesh.nodes))

    # indicate how many semiconductor regions share the same node
    nodes_shared_regions = np.zeros(len(sim.mesh.nodes))

    ## 202508 - add recombination coupling 
    dR_dn = np.zeros(len(sim.mesh.nodes))
    dR_dp = np.zeros(len(sim.mesh.nodes))
    ## 202508 - add recombination coupling 

    n=get_electron_density(sim)
    p=get_hole_density(sim)
    NA=sim.NA
    ND=sim.ND

    recom=phys.Recombination()
    for reg_num,region in enumerate(sim.mesh.regions):
        if region.type != 'semiconductor':
            continue
        node_idx=region.nodes
        material_region=region.material

        recombination_models=material_region.recombination_models
        recombination_parameters=material_region.recombination_parameters
        if recombination_models==None:
            pass
        else:
            dopping_region = NA[node_idx]+ND[node_idx]
            n_i_region=phys.intrinsic_concentration(material_region,dopping_region)
            for recom_model in recombination_models:  
                if recom_model=='SRH':
                    recombination_para=recombination_parameters['SRH']
                    R[node_idx]+=recom.srh(recombination_para,n_i_region,n[node_idx],p[node_idx])
                    ## 202508 - add recombination coupling 
                    dR_dn[node_idx]+=recom.srh_dR_dn(recombination_para,n_i_region,n[node_idx],p[node_idx])
                    dR_dp[node_idx]+=recom.srh_dR_dp(recombination_para,n_i_region,n[node_idx],p[node_idx])
                elif recom_model=='OPT':
                    recombination_para=recombination_parameters['OPT']
                    R[node_idx]+=recom.optical(recombination_para,n_i_region,n[node_idx],p[node_idx])
                    ## 202508 - add recombination coupling 
                    dR_dn[node_idx]+=recom.optical_dR_dn(recombination_para,n_i_region,n[node_idx],p[node_idx])
                    dR_dp[node_idx]+=recom.optical_dR_dp(recombination_para,n_i_region,n[node_idx],p[node_idx])
                elif recom_model=='AUG':
                    recombination_para=recombination_parameters['AUG']
                    R[node_idx]+=recom.auger(recombination_para,n_i_region,n[node_idx],p[node_idx])
                    ## 202508 - add recombination coupling 
                    dR_dn[node_idx]+=recom.auger_dR_dn(recombination_para,n_i_region,n[node_idx],p[node_idx])
                    dR_dp[node_idx]+=recom.auger_dR_dp(recombination_para,n_i_region,n[node_idx],p[node_idx])

                elif recom_model=='II':
                    Ex,Ey,Ez=get_electric_field(sim) 
                    Jx_n,Jy_n,Jz_n=get_current_density(sim,'electron')
                    Jx_p,Jy_p,Jz_p=get_current_density(sim,'hole')
                    E_region = np.sqrt(Ex[node_idx]**2 + Ey[node_idx]**2 + Ez[node_idx]**2)
                    Jn_region = np.sqrt(Jx_n[node_idx]**2 + Jy_n[node_idx]**2 + Jz_n[node_idx]**2)
                    Jp_region = np.sqrt(Jx_p[node_idx]**2 + Jy_p[node_idx]**2 + Jz_p[node_idx]**2)

                    recombination_para=recombination_parameters['II']
                    R[node_idx]+=recom.impact_ionization(recombination_para,E_region,Jn_region,Jp_region)
                elif recom_model=='BBT':
                    Ex,Ey,Ez=get_electric_field(sim) 
                    E_region = np.sqrt(Ex[node_idx]**2 + Ey[node_idx]**2 + Ez[node_idx]**2)
                    
                    recombination_para=recombination_parameters['BBT']
                    R[node_idx]+=recom.band_to_band(recombination_para,E_region)
                else:
                    raise ValueError('Recommendation model type is not defined!')
            nodes_shared_regions[node_idx]+=1

    # average the recombination rate for nodes shared by multiple semiconductor regions
    shared_node_idx = nodes_shared_regions>1
    R[shared_node_idx]=R[shared_node_idx]/nodes_shared_regions[shared_node_idx]
    dR_dn[shared_node_idx]=dR_dn[shared_node_idx]/nodes_shared_regions[shared_node_idx]
    dR_dp[shared_node_idx]=dR_dp[shared_node_idx]/nodes_shared_regions[shared_node_idx]
            
    return R,dR_dn,dR_dp


def get_ion_recombination(sim,T):
    R=np.zeros(len(sim.mesh.nodes))
    Ea_gen = get_gen_Ea(sim)
    recom=phys.Recombination()
    for reg_num,region in enumerate(sim.mesh.regions):
        if region.type != 'semiconductor':
            continue
        node_idx=region.nodes
        material_region=region.material

        ion_recombination_models=material_region.ion_recombination_models
        ion_recombination_parameters=material_region.ion_recombination_parameters
        if ion_recombination_models==None:
            pass
        else:
            for recom_model in ion_recombination_models:  
                if recom_model=='oxygen_vacancies':
                    Ex,Ey,Ez=get_electric_field(sim) 
                    E_region = np.sqrt(Ex[node_idx]**2 + Ey[node_idx]**2 + Ez[node_idx]**2)
                    recombination_para=ion_recombination_parameters['oxygen_vacancies']
                    R[node_idx]+=recom.ion_recombination(recombination_para,E_region,T[node_idx],Ea_gen[node_idx])
                else:
                    raise ValueError('ion Recommendation model type is not defined!')
            
    return R

# def get_ion_recombination(sim,T):
#     R=np.zeros(len(sim.mesh.nodes))

#     recom=phys.Recombination()
#     for reg_num,region in enumerate(sim.mesh.regions):
#         if region.type != 'semiconductor':
#             continue
#         node_idx=region.nodes
#         material_region=region.material

#         ion_recombination_models=material_region.ion_recombination_models
#         ion_recombination_parameters=material_region.ion_recombination_parameters
#         if ion_recombination_models==None:
#             pass
#         else:
#             for recom_model in ion_recombination_models:  
#                 if recom_model=='oxygen_vacancies':
#                     Ex,Ey,Ez=get_electric_field(sim) 
#                     E_region = np.sqrt(Ex[node_idx]**2 + Ey[node_idx]**2 + Ez[node_idx]**2)
#                     recombination_para=ion_recombination_parameters['oxygen_vacancies']
#                     R[node_idx]+=recom.ion_recombination(recombination_para,E_region,T[node_idx])
#                 else:
#                     raise ValueError('ion Recommendation model type is not defined!')
            
#     return R

def get_power_density(sim):
    # power density in [W/m^3]
    # power density is calculated in each node
    if sim.conf.get('uniform_band', True):
        # (default) for uniform band structure
        potential = get_potential(sim)  
    else:
        # for non-uniform band structure, the flux is calculated based on effective potential
        potential = get_effective_potential(sim,type=type)
    n=get_electron_density(sim)
    p=get_hole_density(sim)
    T=get_temperature(sim)

    power_density=np.zeros(len(sim.mesh.nodes))
    dpower_density_dTi=np.zeros(len(sim.mesh.nodes))
    semi_node_idx = sim.node_idx_of_material('semiconductor')
    mobility_e=get_mobility(sim,'electron')
    mobility_h=get_mobility(sim,'hole')

    for i_node0 in semi_node_idx:
        node0 = sim.mesh.nodes[i_node0]
        power_density_edges = 0
        n_edges = 0
        for i_edge,i_node1 in zip(node0.edges,node0.other_nodes):
            edge = sim.mesh.edges[i_edge]
            if i_node1 not in semi_node_idx:
                continue

            mun=mobility_e[i_edge]
            mup=mobility_h[i_edge]

            n0, n1 = n[i_node0], n[i_node1]
            p0, p1 = p[i_node0], p[i_node1]
            T0, T1 = T[i_node0], T[i_node1]
            potential0, potential1 = potential[i_node0], potential[i_node1]
            temp=(T[i_node0]+T[i_node1])/2

            if sim.conf.get('with_electron', True):
                flux_e=phys.scharfetter_gummel(n0,n1,potential0,potential1,edge.length,mun,temp,type='electron')
                dflux_e_dTi = phys.scharfetter_gummel_dTi(n0,n1,potential0,potential1,
                             edge.length,mun,temp,type='electron')
                dflux_e_dTj = phys.scharfetter_gummel_dTj(n0,n1,potential0,potential1,
                             edge.length,mun,temp,type='electron')
            else:
                flux_e=0
                dflux_e_dTi = 0
                dflux_e_dTj = 0
            if sim.conf.get('with_hole', True):
                flux_h=phys.scharfetter_gummel(p0,p1,potential0,potential1,edge.length,mup,temp,type='hole')
                dflux_h_dTi = phys.scharfetter_gummel_dTi(p0,p1,potential0,potential1,
                             edge.length,mup,temp,type='hole')
                dflux_h_dTj = phys.scharfetter_gummel_dTj(p0,p1,potential0,potential1,
                             edge.length,mup,temp,type='hole')
            else:
                flux_h=0
                dflux_h_dTi = 0
                dflux_h_dTj = 0

            power_density_edges += -(flux_e+flux_h)*(potential1-potential0)/edge.length
            n_edges += 1
            dpower_density_dTi[i_node0] += -(dflux_e_dTi+dflux_h_dTi)*(potential1-potential0)/edge.length
        power_density[i_node0] = power_density_edges / n_edges if n_edges > 0 else 0
        dpower_density_dTi[i_node0] = dpower_density_dTi[i_node0] / n_edges if n_edges > 0 else 0

    return power_density


def get_Jion_diffusion(sim,location='node'):
    # current density in [A/m^2]
    D0=get_D0(sim)
    T=get_temperature(sim)
    carrier_conc = sim.ND

    semi_node_idx = sim.node_idx_of_material('semiconductor')

    if location=='node':
        Jx=np.zeros(len(sim.mesh.nodes))
        Jy=np.zeros(len(sim.mesh.nodes))
        Jz=np.zeros(len(sim.mesh.nodes))

        for i_node0 in semi_node_idx:
            node0=sim.mesh.nodes[i_node0]
            flux_edges=[]
            normal_edges=[]
            c_i = carrier_conc[i_node0]
            box_volume = node0.box_volume_semi_ox
            for i_edge,i_node1 in zip(node0.edges,node0.other_nodes):
                edge = sim.mesh.edges[i_edge]
                facet_area = edge.facet_area_semi_ox

                if i_node1 not in semi_node_idx:
                    continue
                node1 = sim.mesh.nodes[i_node1]
                c_j = carrier_conc[i_node1]
                temp = (T[i_node0]+T[i_node1])/2
                diffusion = D0[i_edge]*np.exp(-cons['q']/cons['k']/temp)
                flux= (facet_area/box_volume)*diffusion * (c_j-c_i)

                flux_edges.append(flux)
                normal_edges.append([(node1.x-node0.x)/edge.length,(node1.y-node0.y)/edge.length,(node1.z-node0.z)/edge.length])
                    
            if len(flux_edges)>0:
                current_density_node=field_cell(sim,flux_edges,normal_edges)
                
                Jx[i_node0]=current_density_node[0]
                Jy[i_node0]=current_density_node[1]
                Jz[i_node0]=current_density_node[2] 

    return (Jx,Jy,Jz)

def get_Jion_drift(sim,location='node'):
    # current density in [A/m^2]
    D0=get_D0(sim)
    T=get_temperature(sim)
    potential=get_potential(sim)
    carrier_conc = sim.ND

    semi_node_idx = sim.node_idx_of_material('semiconductor')

    if location=='node':
        Jx=np.zeros(len(sim.mesh.nodes))
        Jy=np.zeros(len(sim.mesh.nodes))
        Jz=np.zeros(len(sim.mesh.nodes))

        for i_node0 in semi_node_idx:
            node0=sim.mesh.nodes[i_node0]
            flux_edges=[]
            normal_edges=[]
            c_i = carrier_conc[i_node0]
            box_volume = node0.box_volume_semi_ox
            for i_edge,i_node1 in zip(node0.edges,node0.other_nodes):
                edge = sim.mesh.edges[i_edge]
                facet_area = edge.facet_area_semi_ox

                if i_node1 not in semi_node_idx:
                    continue
                node1 = sim.mesh.nodes[i_node1]
                potential0, potential1 = potential[i_node0], potential[i_node1]
                c_j = carrier_conc[i_node1]
                temp = (T[i_node0]+T[i_node1])/2
                diffusion = D0[i_edge]*np.exp(-cons['q']/cons['k']/temp)
                mu = diffusion * 2* phys.constant['q']/(phys.constant['k']*temp)
                flux= -(facet_area/box_volume)*mu * c_i * (potential0-potential1)/edge.length

                flux_edges.append(flux)
                normal_edges.append([(node1.x-node0.x)/edge.length,(node1.y-node0.y)/edge.length,(node1.z-node0.z)/edge.length])
                    
            if len(flux_edges)>0:
                current_density_node=field_cell(sim,flux_edges,normal_edges)
                
                Jx[i_node0]=current_density_node[0]
                Jy[i_node0]=current_density_node[1]
                Jz[i_node0]=current_density_node[2] 

    return (Jx,Jy,Jz)

def get_Jion_soret(sim,location='node'):
    D0=get_D0(sim)
    Ea=get_Ea(sim)
    T=get_temperature(sim)
    carrier_conc = sim.ND

    semi_node_idx = sim.node_idx_of_material('semiconductor')

    if location=='node':
        Jx=np.zeros(len(sim.mesh.nodes))
        Jy=np.zeros(len(sim.mesh.nodes))
        Jz=np.zeros(len(sim.mesh.nodes))

        for i_node0 in semi_node_idx:
            node0=sim.mesh.nodes[i_node0]
            flux_edges=[]
            normal_edges=[]
            c_i = carrier_conc[i_node0]
            box_volume = node0.box_volume_semi_ox
            for i_edge,i_node1 in zip(node0.edges,node0.other_nodes):
                edge = sim.mesh.edges[i_edge]
                facet_area = edge.facet_area_semi_ox
                if i_node1 not in semi_node_idx:
                    continue
                node1 = sim.mesh.nodes[i_node1]

                temp = (T[i_node0]+T[i_node1])/2
                diffusion = D0[i_edge]*np.exp(-cons['q']/cons['k']/temp)

                soret = -Ea[i_edge]*cons['q']/(phys.constant['k']*temp*temp)
                flux= (facet_area/box_volume)*diffusion * soret * c_i * (T[i_node1]-T[i_node0])/edge.length

                flux_edges.append(flux)
                normal_edges.append([(node1.x-node0.x)/edge.length,(node1.y-node0.y)/edge.length,(node1.z-node0.z)/edge.length])
                    
            if len(flux_edges)>0:
                current_density_node=field_cell(sim,flux_edges,normal_edges)
                
                Jx[i_node0]=current_density_node[0]
                Jy[i_node0]=current_density_node[1]
                Jz[i_node0]=current_density_node[2] 

    return (Jx,Jy,Jz)


## 202508
def get_contact_current(sim,contact_name):
    # contact current in [A]
    potential=get_potential(sim)
    n=get_electron_density(sim)
    p=get_hole_density(sim)
    T=get_temperature(sim)
    mobility_e=get_mobility(sim,'electron')
    mobility_h=get_mobility(sim,'hole')

    In=0
    Ip=0

    electrode_region=sim.mesh.regions[sim.device.electrodes[contact_name]]
    electrode_node_idx = set(electrode_region.nodes)
    semi_node_idx = sim.node_idx_of_material('semiconductor')
    intersection_node_idx = list(set(electrode_node_idx) & set(semi_node_idx))
    for i_node0 in intersection_node_idx:
        node = sim.mesh.nodes[i_node0]
        for i_edge,i_node1 in zip(node.edges,node.other_nodes):
            edge = sim.mesh.edges[i_edge]
            if i_node1 not in semi_node_idx:
                continue
            facet_area = edge.facet_area_semi

            mun=mobility_e[i_edge]
            mup=mobility_h[i_edge]

            n0, n1 = n[i_node0], n[i_node1]
            p0, p1 = p[i_node0], p[i_node1]

            T0, T1 = T[i_node0], T[i_node1]
            potential0, potential1 = potential[i_node0], potential[i_node1]

        
            flux_electron=phys.scharfetter_gummel(n0,n1,potential0,potential1,edge.length,mun,(T0+T1)/2,type='electron')
            flux_hole=phys.scharfetter_gummel(p0,p1,potential0,potential1,edge.length,mup,(T0+T1)/2,type='hole')
        
            Jn_edge=flux_electron*facet_area
            Jp_edge=flux_hole*facet_area

            if sim.mesh.mesh_type=='1D':
                Ip+=Jp_edge*sim.device.area
                In+=Jn_edge*sim.device.area
            elif sim.mesh.mesh_type=='2D':
                In+=Jn_edge*sim.device.width
                Ip+=Jp_edge*sim.device.width
            elif sim.mesh.mesh_type=='3D':
                In+=Jn_edge
                Ip+=Jp_edge
            elif sim.mesh.mesh_type=='cylindrical_2D':
                midpoint = edge.midpoint
                if sim.mesh.symmetric_axis=='x-axis':
                    In+=Jn_edge*(midpoint.y)*2*np.pi
                    Ip+=Jp_edge*(midpoint.y)*2*np.pi
                elif sim.mesh.symmetric_axis=='y-axis':
                    In+=Jn_edge*(midpoint.x)*2*np.pi
                    Ip+=Jp_edge*(midpoint.x)*2*np.pi
    return In,Ip