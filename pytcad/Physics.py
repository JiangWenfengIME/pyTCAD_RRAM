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
def Bern_qft(x,y):
    B=np.exp(-x)*Bern(y-x)
    dB_dx = np.exp(-y)*Bern_dx(x-y)
    dB_dy = np.exp(-x)*Bern_dx(y-x)
    return B, dB_dx, dB_dy

def Bern_qft_sg2(x):
    B=np.exp(x)*Bern(2*x)
    dB_dx = np.exp(x)*Bern(2*x)+ np.exp(x)*2*Bern_dx(2*x)
    return B, dB_dx

## 202508 qft coupling item
def dn_dEfn(n_i_node, Ut):
    return -1 * n_i_node/Ut
def dp_dEfp(p_i_node, Ut):
    return p_i_node/Ut

## 202508 trap and interface trap
def trap_occupancy_acceptor(vn,cse,vp,csh,nt0,pt0,n_i_node,p_i_node):
    return  (vn*cse*n_i_node+vp*csh*pt0) / (vn*cse*(n_i_node+nt0) + vp*csh*(p_i_node+pt0))     

def trap_occupancy_donor(vn,cse,vp,csh,nt0,pt0,n_i_node,p_i_node):
    return  (vn*cse*nt0+vp*csh*p_i_node) / (vn*cse*(n_i_node+nt0) + vp*csh*(p_i_node+pt0))

def trap_df_dn(vn,cse,vp,csh,nt0,pt0,n_i_node,p_i_node):
    return  (vn*cse*nt0+vp*csh*p_i_node) / ((vn*cse*(n_i_node+nt0) + vp*csh*(p_i_node+pt0))**2) *vn*cse

def trap_df_dp(vn,cse,vp,csh,nt0,pt0,n_i_node,p_i_node):
    return  (vn*cse*nt0+vp*csh*p_i_node) / ((vn*cse*(n_i_node+nt0) + vp*csh*(p_i_node+pt0))**2) *vp*csh

def heiman_occupancy_acceptor(vn, sign, vp, sigp, nt0, pt0, n_i_node, p_i_node, occupancy_last, lambda1, lambda2, dt):
    return  ( vn*sign*n_i_node + vp*sigp*pt0 - occupancy_last *( vn*sign*n_i_node*lambda1 + vn*sign*nt0*lambda1 \
            + vp*sigp*p_i_node*lambda1 + vp*sigp*pt0*lambda1 - 1/dt ) ) \
            /( 1/dt + vn*sign*n_i_node*lambda2 + vn*sign*nt0*lambda2+vp*sigp*p_i_node*lambda2 + vp*sigp*pt0*lambda2 )

def heiman_occupancy_donor(vn, sign, vp, sigp, nt0, pt0, n_i_node, p_i_node, occupancy_last, lambda1, lambda2, dt):
    return  ( vp*sigp*p_i_node + vn*sign*nt0 - occupancy_last *(vp*sigp*p_i_node*lambda1 + vp*sigp*pt0*lambda1 \
            + vn*sign*n_i_node*lambda1 + vn*sign*nt0*lambda1 - 1/dt ) ) \
            /( 1/dt + vp*sigp*p_i_node*lambda2 + vp*sigp*pt0*lambda2+vn*sign*n_i_node*lambda2 + vn*sign*nt0*lambda2 )

def heiman_df_dn(vn, sign, vp, sigp, nt0, pt0, n_i_node, p_i_node, occupancy_last, lambda1, lambda2, dt):
    return  ( (vn*sign-occupancy_last*vn*sign*lambda1)*(1/dt + vn*sign*n_i_node*lambda2 + vn*sign*nt0*lambda2  \
            +vp*sigp*p_i_node*lambda2 + vp*sigp*pt0*lambda2) - vn*sign*lambda2*(vn*sign*n_i_node + vp*sigp*pt0 - occupancy_last  \
            *( vn*sign*n_i_node*lambda1 + vn*sign*nt0*lambda1 + vp*sigp*p_i_node*lambda1 + vp*sigp*pt0*lambda1 - 1/dt)) ) \
            /np.power( (1/dt + vn*sign*n_i_node*lambda2 + vn*sign*nt0*lambda2+vp*sigp*p_i_node*lambda2 + vp*sigp*pt0*lambda2),2 )

def heiman_df_dp(vn, sign, vp, sigp, nt0, pt0, n_i_node, p_i_node, occupancy_last, lambda1, lambda2, dt):
    return  ( -1*occupancy_last*vp*sigp*lambda1*(1/dt + vn*sign*n_i_node*lambda2 + vn*sign*nt0*lambda2 \
            + vp*sigp*p_i_node*lambda2 + vp*sigp*pt0*lambda2) - vp*sigp*lambda2*( vn*sign*n_i_node + vp*sigp*pt0 - occupancy_last  \
            *( vn*sign*n_i_node*lambda1 + vn*sign*nt0*lambda1 + vp*sigp*p_i_node*lambda1 + vp*sigp*pt0*lambda1 - 1/dt ) ) )  \
            /np.power( (1/dt + vn*sign*n_i_node*lambda2 + vn*sign*nt0*lambda2+vp*sigp*p_i_node*lambda2 + vp*sigp*pt0*lambda2),2 )

# 202508 initial guess for QFT
def cal_fermi_e_with_n(n,phi,ni):
    return phi - thermal_potential(constant['T0'])*np.log(n/ni)

def cal_fermi_h_with_p(p,phi,ni):
    return phi + thermal_potential(constant['T0'])*np.log(p/ni)

class Mobility:
    def const(self,para,type='electron'):
        # constant mobility model
        if type=='electron':
            return para['mu0_e']
        elif type=='hole':
            return para['mu0_h']

    def latice(self,para,temperature=300,type='electron'):  
        # latice temperature dependent (latice scattering) mobility model
        # mu=mu0*(T/300)^(-alpha)
        if type=='electron':
            mu_L=para['mu0_e']*(temperature/300)**(-para['alpha_e'])
        elif type=='hole':
            mu_L=para['mu0_h']*(temperature/300)**(-para['alpha_h'])
        return mu_L

    def latice_impurity(self,para,temperature=300,dopping=1e20,type='electron'): 
        # Caughey and Thomas impurity model considering the scattering by latice and ionized impurity

        mu_L=self.latice(para,temperature=temperature,type=type)
        if type=='electron':
            mu_LI=para['mu_min_e']+(mu_L-para['mu_min_e'])/(1.0+(dopping/para['C_ref_e'])**(para['alpha2_e']))
        elif type=='hole':
            mu_LI=para['mu_min_h']+(mu_L-para['mu_min_h'])/(1.0+(dopping/para['C_ref_h'])**(para['alpha2_h']))
        return mu_LI

    def latice_impurity_carrier(self,para,temperature=300,dopping=1e20,n=1e20,p=1e20,type='electron'): 
        # Caughey and Thomas impurity model considering the scattering by latice, ionized impurity and carrier-carrier
        
        mu_L=self.latice(para,temperature=temperature,type=type)
        if type=='electron':
            mu_LIC=para['mu_min_e']+(mu_L-para['mu_min_e'])      \
                    /(1.0+(dopping/para['C_ref_e'])**(para['alpha2_e'])+(np.sqrt(n*p)/(14.0*para['C_ref_e']))**(para['alpha2_e']))
        elif type=='hole':
            mu_LIC=para['mu_min_h']+(mu_L-para['mu_min_h'])      \
                    /(1.0+(dopping/para['C_ref_h'])**(para['alpha2_h'])+(np.sqrt(n*p)/(14.0*para['C_ref_h']))**(para['alpha2_h']))
        return mu_LIC
    
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
    def srh(self,para,n_i,n,p):
        R=(n*p-n_i**2)/(para['tau_p']*(n+n_i)+para['tau_n']*(p+n_i))
        return R
    ## 202508 - SRH coupling item
    def srh_dR_dn(self,para,n_i,n,p):
        D = (para['tau_p']*(n+n_i)+para['tau_n']*(p+n_i))
        N = (n*p-n_i**2)
        return (p * D - N * para['tau_p']) / (D * D)
    def srh_dR_dp(self,para,n_i,n,p):
        D = (para['tau_p']*(n+n_i)+para['tau_n']*(p+n_i))
        N = (n*p-n_i**2)
        return (n * D - N * para['tau_n']) / (D * D)
    
    def optical(self,para,n_i,n,p):
        R=para['Copt']*(n*p-n_i**2)
        return R
    ## 202508 - optical coupling item    
    def optical_dR_dn(self,para,n_i,n,p):
        return para['Copt'] * p
    def optical_dR_dp(self,para,n_i,n,p):
        return para['Copt'] * n

    def auger(self,para,n_i,n,p):
        R=(para['augn']*n+para['augp']*p)*(n*p-n_i**2)
        return R
    ## 202508 - auger coupling item   
    def auger_dR_dn(self,para,n_i,n,p):
        return para['augn']*(n*p-n_i**2) + (para['augn']*n+para['augp']*p)*p
    def auger_dR_dp(self,para,n_i,n,p):
        return para['augp']*(n*p-n_i**2) + (para['augn']*n+para['augp']*p)*n

    def impact_ionization(self,para,E,Jn,Jp):
        valid_idx=np.abs(E)>para['E_crit_n']/10
        R=np.zeros_like(E)
        alpha_n=para['alpha_inf_n']*np.exp(-(para['E_crit_n']/np.abs(E[valid_idx]))**para['beta_n'])
        alpha_p=para['alpha_inf_p']*np.exp(-(para['E_crit_p']/np.abs(E[valid_idx]))**para['beta_p'])
        R[valid_idx]=-alpha_n*np.abs(Jn[valid_idx])/constant['q']-alpha_p*np.abs(Jp[valid_idx])/constant['q']
        return R
    def band_to_band(self,para,E):
        valid_idx=np.abs(E)>para['Bbbt']/10
        R=np.zeros_like(E)
        R[valid_idx]=-para['D']*para['Abbt']*(np.abs(E[valid_idx])**para['gamma'])*np.exp(-para['Bbbt']/np.abs(E[valid_idx]))
        return R
    
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