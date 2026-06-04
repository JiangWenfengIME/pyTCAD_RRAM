from pytcad.Physics import constant as cons

class Conductor:
    def __init__(self,material_name=None, kappa = 1, kappa_alpha = 0.01) -> None:
        self.material_name=None
        self.kappa=kappa   # Thermal conductivity [W/(mK)]

        ## 20260309
        self.kappa_alpha=kappa_alpha

class Oxide:
    def __init__(self,material_name='sio2',ion_recombination_models=['oxygen_vacancies']) -> None:
        if material_name=='sio2':
            self.material_name='Silicon Oxide'
            self.epsilon=3.9   # Relative permitivity [1]
            self.kappa=1.3     # Thermal conductivity [W/(mK)]

        elif material_name=='hfox':
            self.material_name='hfox'
            self.epsilon=21  # Relative permitivity [1]
            self.kappa=0.4     # Thermal conductivity [W/(mK)]     
            self.D0 = 2e-7 #
            self.Ea = 1*cons['q'] # [J]
            self.rho = 9.62e3  #kg/m^3
            self.cp = 120  #J/(kg*K)
            self.ion_recombination_models=ion_recombination_models
            self.ion_recombination_parameters={}
            if ion_recombination_models==None:
                pass
            else:
                for recom_model in ion_recombination_models:
                    if recom_model=='oxygen_vacancies':  # recombination model 
                        self.ion_recombination_parameters['oxygen_vacancies']={'f':5e28,     # lifetime [s]
                                                                               'Eag':1.0,  #1eV
                                                                               'p0':1.2e-29,  # C*m
                                                                               'epsilon_r':21}   
        else:
            raise ValueError('Material type is not defined!')
        

        
class Semiconductor:
    def __init__(self,material_name='si',band_gap_model='constant',mobility_model='constant',recombination_models=['SRH'],ion_recombination_models=None):
        if material_name in ['si','silicon','Si','Silicon']:
            self.material_name='Silicon'

            if band_gap_model in ['constant','Slotboom']:
                self.band_gap_model=band_gap_model
                self.band_gap_parameters={'Eg0':1.12,       # Energy gap [eV]
                                          'Ebgn':9.0e-3,'Nbgn':1.0e23,'Cbgn':0.5}  # parameters used for Slotboom band gap narrowing effect
            else:
                raise ValueError('Band gap model type is not defined!')
            
            self.affinity=4.17  # affinity [eV]
            self.Nc=2.8e25      # Effective density of states [/m^3]
            self.Nv=1.0e25      # Effective density of states [/m^3]
            self.epsilon=11.7   # Relative permitivity [1]
            self.kappa=148      # Thermal conductivity [W/(mK)]

            if mobility_model in ['constant','latice','latice_impurity','latice_impurity_carrier']:
                self.mobility_model=mobility_model
                self.mobility_parameters={'mu0_e':1400e-4,
                                          'alpha_e':2.33,'mu_min_e':55.24e-4,'C_ref_e':1.072e17*1e6,'alpha2_e':0.733,
                                        'mu0_h':480e-4,
                                        'alpha_h':2.23,'mu_min_h':49.705e-4,'C_ref_h':1.606e17*1e6,'alpha2_h':0.7}
            else:
                raise ValueError('Mobility model type is not defined!')
            
            self.recombination_models=recombination_models
            self.recombination_parameters={}
            if recombination_models==None:
                pass
            else:
                for recom_model in recombination_models:
                    if recom_model=='SRH':  # SRH recombination model 
                        self.recombination_parameters['SRH']={'tau_n':1e-7,     # lifetime [s]
                                                        'tau_p':1e-7,
                                                        'etrap':0}
                    elif recom_model=='OPT': # optical direct recombination model 
                        self.recombination_parameters['OPT']={'Copt':1e-31}
                    elif recom_model=='AUG': # Auger recombination model 
                        self.recombination_parameters['AUG']={'augn':2.8e-31*1e-12,  # m^6/s
                                                              'augp':9.9e-32*1e-12}  
                    elif recom_model=='II':  # Generation by impact ionization
                        self.recombination_parameters['II']={'alpha_inf_n':1.0e8,  # m-1/s
                                                              'E_crit_n':1.66e8,   # V/m
                                                              'beta_n':1.0,
                                                              'alpha_inf_p':2.0e8,  # m-1/s
                                                              'E_crit_p':1.98e8,   # V/m
                                                              'beta_p':1.0}  
                    elif recom_model=='BBT':  # Geneartion by band-to-band tunneling
                        self.recombination_parameters['BBT']={'D':2.5,  
                                                              'Abbt':4.4e13,   # m^(-1/2)V^(-5/2)/s
                                                              'Bbbt':1.9e9,    # V/m
                                                              'gamma':2.5}  
                    else:
                        raise ValueError('Recommendation model type is not defined!')
            
        elif material_name in ['ge','germanium','Ge','Germanium']:
            self.material_name='Germanium'

            if band_gap_model in ['constant']:
                self.band_gap_model=band_gap_model
                self.band_gap_parameters={'Eg0':0.66}    # Energy gap [eV]
            else:
                raise ValueError('Band gap model type is not defined!')
            
            self.affinity=4.0
            self.Nc=1.04e25
            self.Nv=6.0e24
            self.epsilon=16  # Relative permitivity [1]
            self.kappa=60

            if mobility_model in ['constant']:
                self.mobility_model=mobility_model
                self.mobility_parameters={'mu0_e':3900e-4,   # mobility [m^2/V/s]
                                        'mu0_h':1900e-4}
            else:
                raise ValueError('Mobility model type is not defined!')
            
            self.recombination_models=recombination_models
            self.recombination_parameters={}
            if recombination_models==None:
                pass
            else:
                for recom_model in recombination_models:
                    if recom_model=='SRH':
                        self.recombination_parameters['SRH']={'tau_n':1e-7,     # lifetime [s]
                                                        'tau_p':1e-7,
                                                        'etrap':0}
                    elif recom_model=='OPT': # optical direct recombination model 
                        self.recombination_parameters['OPT']={'Copt':1e-31}
                    elif recom_model=='AUG': # Auger recombination model 
                        self.recombination_parameters['AUG']={'augn':2.8e-31*1e-12,  # m^6/s
                                                              'augp':9.9e-32*1e-12}  
                    elif recom_model=='II':  # Generation by impact ionization
                        self.recombination_parameters['II']={'alpha_inf_n':1.0e8,  # m-1/s
                                                              'E_crit_n':1.66e8,   # V/m
                                                              'beta_n':1.0,
                                                              'alpha_inf_p':2.0e8,  # m-1/s
                                                              'E_crit_p':1.98e8,   # V/m
                                                              'beta_p':1.0}  
                    elif recom_model=='BBT':  # Geneartion by band-to-band tunneling
                        self.recombination_parameters['BBT']={'D':2.5,  
                                                              'Abbt':4.4e13,   # m^(-1/2)V^(-5/2)/s
                                                              'Bbbt':1.9e9,    # V/m
                                                              'gamma':2.5}  
                    else:
                        raise ValueError('Recommendation model type is not defined!')
        elif material_name in ['polysi']:
            self.material_name='Poly Silicon'

            if band_gap_model in ['constant']:
                self.band_gap_model=band_gap_model
                self.band_gap_parameters={'Eg0':1.08}    # Energy gap [eV]
            else:
                raise ValueError('Band gap model type is not defined!')
            
            self.affinity=4.17
            self.Nc=2.8e25
            self.Nv=2.8e25
            self.epsilon=11.9  # Relative permitivity [1]
            self.kappa=148

            if mobility_model in ['constant']:
                self.mobility_model=mobility_model
                self.mobility_parameters={'mu0_e':300e-4,   # mobility [m^2/V/s]
                                        'mu0_h':30e-4}
            else:
                raise ValueError('Mobility model type is not defined!')
            
            self.recombination_models=recombination_models
            self.recombination_parameters={}
            if recombination_models==None:
                pass
            else:
                for recom_model in recombination_models:
                    if recom_model=='SRH':
                        self.recombination_parameters['SRH']={'tau_n':1e-7,     # lifetime [s]
                                                        'tau_p':1e-7,
                                                        'etrap':0}
                    elif recom_model=='OPT': # optical direct recombination model 
                        self.recombination_parameters['OPT']={'Copt':1e-31}
                    elif recom_model=='AUG': # Auger recombination model 
                        self.recombination_parameters['AUG']={'augn':2.8e-31*1e-12,  # m^6/s
                                                              'augp':9.9e-32*1e-12}  
                    elif recom_model=='II':  # Generation by impact ionization
                        self.recombination_parameters['II']={'alpha_inf_n':1.0e8,  # m-1/s
                                                              'E_crit_n':1.66e8,   # V/m
                                                              'beta_n':1.0,
                                                              'alpha_inf_p':2.0e8,  # m-1/s
                                                              'E_crit_p':1.98e8,   # V/m
                                                              'beta_p':1.0}  
                    elif recom_model=='BBT':  # Geneartion by band-to-band tunneling
                        self.recombination_parameters['BBT']={'D':2.5,  
                                                              'Abbt':4.4e13,   # m^(-1/2)V^(-5/2)/s
                                                              'Bbbt':1.9e9,    # V/m
                                                              'gamma':2.5}  
                    else:
                        raise ValueError('Recommendation model type is not defined!') 
                    
        elif material_name in ['asi','amorphus si','amorphus silicon','Amorphus Silicon']:
            self.material_name='Amorphus Silicon'

            if band_gap_model in ['constant']:
                self.band_gap_model=band_gap_model
                self.band_gap_parameters={'Eg0':1.9}    # Energy gap [eV]
            else:
                raise ValueError('Band gap model type is not defined!')
            
            # self.affinity=4.17
            self.affinity=0.0
            self.Nc=2.5e26
            self.Nv=2.5e26
            self.epsilon=11.9  # Relative permitivity [1]
            self.kappa=148

            if mobility_model in ['constant']:
                self.mobility_model=mobility_model
                self.mobility_parameters={'mu0_e':20e-4,   # mobility [m^2/V/s]
                                        'mu0_h':1.5e-4}
            else:
                raise ValueError('Mobility model type is not defined!')
            
            self.recombination_models=recombination_models
            self.recombination_parameters={}
            if recombination_models==None:
                pass
            else:
                for recom_model in recombination_models:
                    if recom_model=='SRH':
                        self.recombination_parameters['SRH']={'tau_n':1e-7,     # lifetime [s]
                                                        'tau_p':1e-7,
                                                        'etrap':0}
                    elif recom_model=='OPT': # optical direct recombination model 
                        self.recombination_parameters['OPT']={'Copt':1e-31}
                    elif recom_model=='AUG': # Auger recombination model 
                        self.recombination_parameters['AUG']={'augn':2.8e-31*1e-12,  # m^6/s
                                                              'augp':9.9e-32*1e-12}  
                    elif recom_model=='II':  # Generation by impact ionization
                        self.recombination_parameters['II']={'alpha_inf_n':1.0e8,  # m-1/s
                                                              'E_crit_n':1.66e8,   # V/m
                                                              'beta_n':1.0,
                                                              'alpha_inf_p':2.0e8,  # m-1/s
                                                              'E_crit_p':1.98e8,   # V/m
                                                              'beta_p':1.0}  
                    elif recom_model=='BBT':  # Geneartion by band-to-band tunneling
                        self.recombination_parameters['BBT']={'D':2.5,  
                                                              'Abbt':4.4e13,   # m^(-1/2)V^(-5/2)/s
                                                              'Bbbt':1.9e9,    # V/m
                                                              'gamma':2.5}  
                    else:
                        raise ValueError('Recommendation model type is not defined!')   

        ## for ion transport
        elif material_name in ['HfOx','hfox','HfO2']:
            self.material_name='HfOx'

            if band_gap_model in ['constant']:
                self.band_gap_model=band_gap_model
                self.band_gap_parameters={'Eg0':5.7}    # Energy gap [eV]
            else:
                raise ValueError('Band gap model type is not defined!')
            
            # self.affinity=4.17
            self.affinity=0.0
            self.Nc=5e26
            self.Nv=5e27
            self.epsilon=25  # Relative permitivity [1]
            self.kappa=0.4

            ## 20260309
            self.kappa_alpha=0.01

            # self.D0 = 2.0 #
            self.D0 = 2.0 #
            
            self.Ea = 1.0*cons['q'] # [J] default 1eV
            self.rho = 9.62e3  #kg/m^3
            self.cp = 120  #J/(kg*K)

            if mobility_model in ['constant']:
                self.mobility_model=mobility_model
                self.mobility_parameters={'mu0_e':1.5e-4,   # mobility [m^2/V/s]
                                        'mu0_h':0.1e-4}
            elif mobility_model in ['rram_carrier']:
                self.mobility_model=mobility_model
                # self.mobility_parameters={
                #                         'mu0_e':20e-3,   # mobility [m^2/V/s]
                #                         # 'mu0_e':1.5e-4,   # mobility [m^2/V/s]
                #                         'mu0_h':0.1e-4,
                #                         # 'ion_ref':2e25,
                #                         'ion_ref':5e25,   ## 20260311
                #                         'mu0_e_min':5e-6,
                #                         # 'mu0_e_min':15e-10,
                #                         # 'ion_percolation':1e21,
                #                         # 'ion_percolation':1e22,
                #                         'ion_percolation':1e23,  ##20260311
                #                         'alpha':1.0,
                #                         'ion_low':1e21,
                #                         'ion_high':1e23}

                self.mobility_parameters={'mu0_e':1.5e-4,   # mobility [m^2/V/s]
                                        'mu0_h':0.1e-4,
                                        'ion_ref':2e25,
                                        'mu0_e_min':15e-6,
                                        # 'mu0_e_min':15e-10,
                                        # 'ion_percolation':1e21,
                                        'ion_percolation':1e22,
                                        'alpha':1.0,
                                        'ion_low':1e21,
                                        'ion_high':1e23,
                                        'vo_max':7e25,
                                        'f0':0.02,
                                        'delta':0.008}

                # self.mobility_parameters={'mu0_e':1.5e-4,   # mobility [m^2/V/s]
                #                         'mu0_h':0.1e-4,
                #                         'ion_ref':2e25,
                #                         'mu0_e_min':10e-7,
                #                         # 'mu0_e_min':15e-10,
                #                         # 'ion_percolation':1e21,
                #                         'ion_percolation':1e23,
                #                         'alpha':1.0,
                #                         'ion_low':1e21,
                #                         'ion_high':1e23}

            else:
                raise ValueError('Mobility model type is not defined!')


            self.ion_recombination_models=ion_recombination_models
            self.ion_recombination_parameters={}
            if ion_recombination_models==None:
                pass
            else:
                for recom_model in ion_recombination_models:
                    if recom_model=='oxygen_vacancies':  # recombination model 
                        self.ion_recombination_parameters['oxygen_vacancies']={'f':1e26, #3e25
                                                                            #  'f':8e25, #3e25
                                                                            #    'f':1e28,     # lifetime [s]                                
                                                                            #    'Eag':1.0,  #1eV
                                                                               'Eag':1.0,  #1eV
                                                                            #    'p0':1.2e-29,  # C*m
                                                                               'p0':2.0e-28,        ## constant
                                                                               'epsilon_r':25}       ## constant


            self.recombination_models=recombination_models
            self.recombination_parameters={}
            if recombination_models==None:
                pass
            else:
                for recom_model in recombination_models:
                    if recom_model=='SRH':
                        self.recombination_parameters['SRH']={'tau_n':1e-7,     # lifetime [s]
                                                        'tau_p':1e-7,
                                                        'etrap':0}
                    elif recom_model=='OPT': # optical direct recombination model 
                        self.recombination_parameters['OPT']={'Copt':1e-31}
                    elif recom_model=='AUG': # Auger recombination model 
                        self.recombination_parameters['AUG']={'augn':2.8e-31*1e-12,  # m^6/s
                                                              'augp':9.9e-32*1e-12}  
                    elif recom_model=='II':  # Generation by impact ionization
                        self.recombination_parameters['II']={'alpha_inf_n':1.0e8,  # m-1/s
                                                              'E_crit_n':1.66e8,   # V/m
                                                              'beta_n':1.0,
                                                              'alpha_inf_p':2.0e8,  # m-1/s
                                                              'E_crit_p':1.98e8,   # V/m
                                                              'beta_p':1.0}  
                    elif recom_model=='BBT':  # Geneartion by band-to-band tunneling
                        # self.recombination_parameters['BBT']={'D':2.5,  
                        #                                       'Abbt':4.4e13,   # m^(-1/2)V^(-5/2)/s
                        #                                       'Bbbt':1.9e9,    # V/m
                        #                                       'gamma':2.5}  
                        self.recombination_parameters['BBT']={'D':2.5,  
                                                              'Abbt':1e12,   # m^(-1/2)V^(-5/2)/s
                                                              'Bbbt':1.9e10,    # V/m
                                                              'gamma':2.0}      
                    else:
                        raise ValueError('Recommendation model type is not defined!')    
 

        else:
            raise ValueError('Material type is not defined!')



