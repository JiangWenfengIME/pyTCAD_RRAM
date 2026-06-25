import numpy as np

constant={'Epsilon0':8.8541878128e-12, #vacuum permitity, unit: [C/(V*m)]
            'q':1.602176634e-19,    # electron charge, unit: [C]
            'k':1.38e-23,           # Boltzmann constant [J/K]}
            'T0':300,
            'hbar':1.0546e-34}   

def thermal_potential(T=constant['T0']):
    return constant['k']*T/constant['q']

def band_gap(material,dopping):
    band_gap_model=material.band_gap_model
    band_gap_para=material.band_gap_parameters
    if band_gap_model=='constant':
        return band_gap_para['Eg0']*np.ones_like(dopping)
    elif band_gap_model=='Slotboom':  # band gap narrowing effect
        delta_Eg=band_gap_para['Ebgn']*(np.log(dopping/band_gap_para['Nbgn'])+(np.log(dopping/band_gap_para['Nbgn'])**2.0+band_gap_para['Cbgn'])**(0.5))
        return band_gap_para['Eg0']-delta_Eg
    else:
        raise ValueError('Band gap model type is not defined!')
        
def intrinsic_concentration(material,dopping):
    return np.sqrt(material.Nc*material.Nv)*np.exp(-band_gap(material,dopping)/(2*thermal_potential(constant['T0'])))

def intrinsic_fermi_energy(material,dopping):
    return band_gap(material,dopping)/2.0

## Wei@2025.09.19
def built_in_potential(ND,NA,material,ref_material):
    # n_eqi=equilibrium_concentration(ND,NA,material)
    # ref_n_i=intrinsic_concentration(ref_material,ND+NA)
    # phi_b=-material.affinity+thermal_potential(constant['T0'])*(np.log(n_eqi/material.Nc)+np.log(ref_material.Nc/ref_n_i))

    ## for homeojunction or uniform materials, equivalent to the following euqation
    n_i=intrinsic_concentration(material,ND+NA)
    phi_b=-material.affinity+thermal_potential(constant['T0'])*np.arcsinh((ND-NA)/n_i/2)

    return phi_b+ref_material.affinity

def equilibrium_concentration(ND,NA,material,type='electron'):
    n_i=intrinsic_concentration(material,ND+NA)
    n=(np.sqrt((ND-NA)**2+4*n_i**2)+(ND-NA))/2
    p=(np.sqrt((ND-NA)**2+4*n_i**2)-(ND-NA))/2
    if type == 'electron':
        n[n<n_i]=(n_i[n<n_i]**2/p[n<n_i])
        return n
    elif type == 'hole':
        p[p<n_i]=(n_i[p<n_i]**2/n[p<n_i])
        return p
    

def potential_band(potential,material,dopping,T=constant['T0'],type='electron'):
    if type == 'electron':
        return potential+material.affinity+thermal_potential(T)*np.log(material.Nc/1e18)
    elif type == 'hole':
        return potential+material.affinity+band_gap(material,dopping)-thermal_potential(T)*np.log(material.Nv/1e18)

def scharfetter_gummel_i(v_i,v_j,dx,mu,T,type='electron'):
    kbT=constant['k']*T
    Vt0=constant['k']*T/constant['q']
    dpsiUt=(v_j-v_i)/Vt0
    if type == 'electron':
        return -mu*kbT/dx*Bern(-dpsiUt)
    elif type == 'hole':
        return mu*kbT/dx*Bern(dpsiUt)
    
def scharfetter_gummel_j(v_i,v_j,dx,mu,T,type='electron'):
    kbT=constant['k']*T
    Vt0=constant['k']*T/constant['q']
    dpsiUt=(v_j-v_i)/Vt0
    if type == 'electron':
        return mu*kbT/dx*Bern(dpsiUt)
    elif type == 'hole':
        return -mu*kbT/dx*Bern(-dpsiUt)

def scharfetter_gummel(np_i,np_j,v_i,v_j,dx,mu,T,type='electron'):
    kbT=constant['k']*T
    Vt0=constant['k']*T/constant['q']
    dpsiUt=(v_j-v_i)/Vt0
    if type == 'electron':
        return mu*kbT/dx*(np_j*Bern(dpsiUt)-np_i*Bern(-dpsiUt))
    elif type == 'hole':
        return -mu*kbT/dx*(np_j*Bern(-dpsiUt)-np_i*Bern(dpsiUt))
    
def scharfetter_gummel_dVi(np_i,np_j,v_i,v_j,dx,mu,T,type='electron'):
    Vt0=constant['k']*T/constant['q']
    dpsiUt=(v_j-v_i)/Vt0
    if type == 'electron':
        return constant['q']*mu/dx*(-np_j*Bern_dx(dpsiUt)-np_i*Bern_dx(-dpsiUt))
    if type == 'hole':
        return -constant['q']*mu/dx*(np_j*Bern_dx(-dpsiUt)+np_i*Bern_dx(dpsiUt))
    
def scharfetter_gummel_dVj(np_i,np_j,v_i,v_j,dx,mu,T,type='electron'):
    Vt0=constant['k']*T/constant['q']
    dpsiUt=(v_j-v_i)/Vt0
    if type == 'electron':
        return constant['q']*mu/dx*(np_j*Bern_dx(dpsiUt)+np_i*Bern_dx(-dpsiUt))
    if type == 'hole':
        return -constant['q']*mu/dx*(-np_j*Bern_dx(-dpsiUt)-np_i*Bern_dx(dpsiUt))
    
## 20250918 assemble thermal coupling    
def scharfetter_gummel_dTi(np_i,np_j,v_i,v_j,dx,mu,T,type='electron'):
    Vt0=constant['k']*T/constant['q']
    dpsiUt=(v_j-v_i)/Vt0
    dpsiUt_dTi = -dpsiUt/(2*T)
    dnp_i_dTi = -np_i*v_i*constant['q']/(2*constant['k']*(T**2))
    dnp_j_dTi = -np_j*v_j*constant['q']/(2*constant['k']*(T**2))    
    if type == 'electron':
        # return (mu/dx)*( (constant['k']/2)*(np_j*Bern(dpsiUt)-np_i*Bern(-dpsiUt)) + constant['k']*T*
        #                 ( dnp_j_dTi*Bern(dpsiUt)+np_j*Bern_dx(dpsiUt)*dpsiUt_dTi-dnp_i_dTi*Bern(-dpsiUt)-np_i*Bern_dx(-dpsiUt)*dpsiUt_dTi ) ) 
        return (mu/dx)*( (constant['k']/2)*(np_j*Bern(dpsiUt)-np_i*Bern(-dpsiUt)) + constant['k']*(-dpsiUt/2)*(np_j*Bern_dx(dpsiUt)+np_i*Bern_dx(-dpsiUt)) ) 
    if type == 'hole':
        return (mu/dx)*( (constant['k']/2)*(np_i*Bern(dpsiUt)-np_j*Bern(-dpsiUt)) + constant['k']*T*dpsiUt_dTi*(np_i*Bern_dx(dpsiUt)+np_j*Bern_dx(-dpsiUt)) )
    
def scharfetter_gummel_dTj(np_i,np_j,v_i,v_j,dx,mu,T,type='electron'):
    Vt0=constant['k']*T/constant['q']
    dpsiUt=(v_j-v_i)/Vt0
    dpsiUt_dTj = -dpsiUt/(2*T)
    dnp_i_dTj = -np_i*v_i*constant['q']/(2*constant['k']*(T**2))
    dnp_j_dTj = -np_j*v_j*constant['q']/(2*constant['k']*(T**2))    
    if type == 'electron':
        # return (mu/dx)*( (constant['k']/2)*(np_j*Bern(dpsiUt)-np_i*Bern(-dpsiUt)) + constant['k']*T*
        #                 ( dnp_j_dTj*Bern(dpsiUt)+np_j*Bern_dx(dpsiUt)*dpsiUt_dTj-dnp_i_dTj*Bern(-dpsiUt)-np_i*Bern_dx(-dpsiUt)*dpsiUt_dTj ) ) 
        return (mu/dx)*( (constant['k']/2)*(np_j*Bern(dpsiUt)-np_i*Bern(-dpsiUt)) + constant['k']*(-dpsiUt/2)*(np_j*Bern_dx(dpsiUt)+np_i*Bern_dx(-dpsiUt)) ) 
    if type == 'hole':
        return (mu/dx)*( (constant['k']/2)*(np_i*Bern(dpsiUt)-np_j*Bern(-dpsiUt)) + constant['k']*T*dpsiUt_dTj*(np_i*Bern_dx(dpsiUt)+np_j*Bern_dx(-dpsiUt)) )

def Bern(x):
    if type(x) == 'float':
        if x>100:
            return 0.0
        elif (np.abs(x) > 0.001):
            return x/(np.exp(x) - 1.0)
        else: # avoids round-off errors due to exp(x) - 1
            return 1.0 / (1.0 + x / 2.0 + x*x / 6.0 + x*x*x/24.0)
    else:
        B = np.zeros_like(x)
        valid_idx = (np.abs(x) > 1.0e-3) & (x < 100)
        B[valid_idx] = x[valid_idx] / (np.exp(x[valid_idx]) - 1.0)

        idx_small = np.abs(x) <= 1.0e-3
        B[idx_small] = 1.0 / (1.0 + x[idx_small] / 2.0 + x[idx_small]**2 / 6.0 + x[idx_small]**3 / 24.0)

        idx_large = x >= 100
        B[idx_large] = 0.0
        return B
    
def Bern_dx(x):
    if type(x) == 'float':
        if x>100:
            return 0.0
        elif (np.abs(x) > 1.0e-3):
            exp_x=np.exp(x)
            return (exp_x - 1.0 - x*exp_x)/(exp_x - 1.0)**2.0
        else: # avoids round-off errors due to exp(x) - 1
            return -(0.5 + x / 3.0 + x*x/8.0)/ (1.0 + x / 2.0 + x*x / 6.0 + x*x*x/24.0)**2
    else:
        B_dx = np.zeros_like(x)
        valid_idx = (np.abs(x) > 1.0e-3) & (x < 100)
        exp_x = np.exp(x[valid_idx])
        B_dx[valid_idx] = (exp_x - 1.0 - x[valid_idx] * exp_x) / (exp_x - 1.0)**2.0

        idx_small = np.abs(x) <= 1.0e-3
        B_dx[idx_small] = -(0.5 + x[idx_small] / 3.0 + x[idx_small]**2 / 8.0) / (1.0 + x[idx_small] / 2.0 + x[idx_small]**2 / 6.0 + x[idx_small]**3 / 24.0)**2

        idx_large = x >= 100
        B_dx[idx_large] = 0.0

        return B_dx

class Mobility:   
    ##20251201
    def rram_carrier_1(self,para,ion_density=2e25,type='electron'):
        if type=='electron':
            mu_L = para['mu0_e_min']+para['mu0_e']*(np.maximum(0,(ion_density-para['ion_percolation'])/para['ion_ref']))**para['alpha']
        elif type=='hole':
            mu_L=para['mu0_h']*para['alpha']*ion_density/para['ion_ref']
        return mu_L 

    def rram_carrier_2(self,para,ion_density=2e25,type='electron'):
        if type=='electron':
            f = ion_density / para['vo_max']
            f = np.clip(f, 0.0, 1.0)
            # -----------------------------
            # Smooth weighting function
            # -----------------------------
            w = 0.5 * (
                1.0 + np.tanh(
                    (f - para['f0']) / para['delta']
                )
            )

            # -----------------------------
            # Effective mobility
            # -----------------------------
            mu_L = (
                para['mu0_e_min']
                + (para['mu0_e'] - para['mu0_e_min']) * w
            )
        elif type=='hole':
            mu_L=para['mu0_h']*para['alpha']*ion_density/para['ion_ref']
        return mu_L 

        
class Recombination():  
    ## 202509 ion recombination
    def ion_recombination(self,para,E,T,Ea_gen):
        R=np.zeros_like(E)

        # print(f"""
        # Parameters:
        # Eag = {para['Eag']}
        # E = {np.abs(E)}
        # P0 = {para['p0']}
        # epsilon_r = {para['epsilon_r']}
        # T = {T}
        # """) 

        # P=1.0/(1.0+1.0/np.exp( -1.0* (para['Eag']*constant['q'] - np.abs(E)*para['p0']*(2+para['epsilon_r'])/3 )/(constant['k']*T) ))
        P=1.0/(1.0+1.0/np.exp( -1.0* (Ea_gen - np.abs(E)*para['p0']*(2+para['epsilon_r'])/3 )/(constant['k']*T) ))
        R=-para['f']*P
        return R