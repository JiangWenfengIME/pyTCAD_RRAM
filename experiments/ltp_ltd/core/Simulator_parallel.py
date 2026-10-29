import os
import ctypes
import numpy as np
import scipy as sp
import scipy.sparse as spsp
from scipy.sparse.linalg import spsolve as spsolve
from concurrent.futures import ThreadPoolExecutor
# from experiments.ltp_ltd.core.Assemble import assemble
from experiments.ltp_ltd.core.Assemble_vectorized import assemble
import experiments.ltp_ltd.core.Physics as phys
import experiments.ltp_ltd.core.Quantities as qs
import time
import copy as copy

try:
    import pypardiso
    _HAS_PARDISO = True
except ImportError:
    from scipy.sparse.linalg import spsolve as _scipy_spsolve
    _HAS_PARDISO = False

def _malloc_trim():
    """把 glibc malloc 已释放但未归还给操作系统的内存尽量归还, 缓解碎片化导致的 RSS 增长。"""
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _default_num_threads():
    try:
        import psutil
        physical_cores = psutil.cpu_count(logical=False) or 2
    except ImportError:
        physical_cores = max(1, (os.cpu_count() or 2) // 2)
    return max(1, min(physical_cores, 8))


# ----------------------------------------------------------------------
# parallel assemble
# ----------------------------------------------------------------------
def _assemble_one_quantity(sim, quan, N):
    A_local = spsp.lil_array((N, N))
    b_local = np.zeros(N)
    A_local, b_local = assemble(sim, quan, A_local, b_local)
    return A_local, b_local   
    
# def parallel_assemble(sim, quan_list, N, max_workers=None):
#     if max_workers is None:
#         max_workers = min(len(quan_list), os.cpu_count() or 1)

#     A = spsp.lil_array((N, N))
#     b = np.zeros(N)

#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         futures = [
#             executor.submit(_assemble_one_quantity, sim, quan, N)
#             for quan in quan_list
#         ]
#         for fut in futures:
#             A_local, b_local = fut.result()
#             A = A + A_local
#             b = b + b_local

#     return A, b

class ParallelAssembler:
    """
    持有常驻的 ThreadPoolExecutor。整个仿真(Simulator 实例)生命周期内只创建一次
    线程池, 避免每次牛顿迭代都 new/destroy 一批 OS 线程导致 glibc malloc arena
    碎片化, 这是内存随迭代单调增长的重要原因之一。
    """

    def __init__(self, max_workers=None):
        self._max_workers = max_workers
        self._executor = None

    def _get_executor(self, n_tasks):
        if self._executor is None:
            workers = self._max_workers or min(n_tasks, os.cpu_count() or 1)
            self._executor = ThreadPoolExecutor(max_workers=workers)
        return self._executor

    def assemble(self, sim, quan_list, N):
        # 只对 unknown=True 的量做组装, 与原来串行版本 for quan in self.quans: if
        # quan['unknown']==False: continue 的行为保持一致
        active_quans = [q for q in quan_list if q['unknown']]

        executor = self._get_executor(len(active_quans))

        A = spsp.lil_array((N, N))
        b = np.zeros(N)

        futures = [
            executor.submit(_assemble_one_quantity, sim, quan, N)
            for quan in active_quans
        ]
        for fut in futures:
            A_local, b_local = fut.result()
            A = A + A_local
            b = b + b_local

        return A, b

    def shutdown(self):
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def __getstate__(self):
        """
        序列化时不保存 ThreadPoolExecutor。
        """
        state = self.__dict__.copy()

        state['_executor'] = None

        return state

    def __setstate__(self, state):
        """
        反序列化后线程池保持为空，
        第一次 assemble 时自动重新创建。
        """
        self.__dict__.update(state)

        self._executor = None
# ----------------------------------------------------------------------
# parallel solver
# ----------------------------------------------------------------------
# class ParallelSparseSolver:
#     def __init__(self, num_threads=None):
#         self._pardiso_solver = None
#         if _HAS_PARDISO:
#             if num_threads is not None:
#                 os.environ["MKL_NUM_THREADS"] = str(num_threads)
#                 os.environ["OMP_NUM_THREADS"] = str(num_threads)
#             self._pardiso_solver = pypardiso.PyPardisoSolver()
#     def solve(self, A_csr, b):
#         if _HAS_PARDISO:
#             self._pardiso_solver.factorize(A_csr)
#             return self._pardiso_solver.solve(A_csr, b)
#         else:
#             return _scipy_spsolve(A_csr, b)

# ----------------------------------------------------------------------
# parallel solver —— 常驻 PARDISO 句柄 + 周期性释放内部内存
# ----------------------------------------------------------------------
class ParallelSparseSolver:
    def __init__(self, num_threads=None, release_every=50):
        """
        release_every: 每做多少次 factorize+solve 就主动 free_memory 并重建一次
            PARDISO 内部句柄, 防止稀疏 pattern 细微变化导致内部缓存持续堆积。
            设为 None 关闭周期性释放。
        """
        self._pardiso_solver = None
        self.num_threads = num_threads if num_threads is not None else _default_num_threads()
        self.release_every = release_every
        self._solve_count = 0

        if _HAS_PARDISO:
            self._pardiso_solver = pypardiso.PyPardisoSolver()
            self._set_num_threads(self.num_threads)

    def _set_num_threads(self, n):
        try:
            import mkl
            mkl.set_num_threads(n)
        except ImportError:
            os.environ["MKL_NUM_THREADS"] = str(n)
            os.environ["OMP_NUM_THREADS"] = str(n)

    def set_num_threads(self, n):
        self.num_threads = n
        self._set_num_threads(n)

    def solve(self, A_csr, b):
        if _HAS_PARDISO:
            self._solve_count += 1
            if self.release_every is not None and self._solve_count % self.release_every == 0:
                self._pardiso_solver.free_memory(everything=True)
                self._pardiso_solver = pypardiso.PyPardisoSolver()
                self._set_num_threads(self.num_threads)
                _malloc_trim()

            self._pardiso_solver.factorize(A_csr)
            return self._pardiso_solver.solve(A_csr, b)
        else:
            return _scipy_spsolve(A_csr, b)

    def release(self):
        if _HAS_PARDISO and self._pardiso_solver is not None:
            self._pardiso_solver.free_memory(everything=True)
        _malloc_trim()

    def __getstate__(self):
        """
        dill/pickle 时仅保存配置和状态，
        不保存底层 PARDISO ctypes 句柄。
        """

        state = self.__dict__.copy()

        state['_pardiso_solver'] = None

        return state

    def __setstate__(self, state):
        """
        load 后恢复 Python 状态，
        PARDISO solver 延迟到第一次 solve 时创建。
        """

        self.__dict__.update(state)

        self._pardiso_solver = None

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
        self.temperature = device.temperature.copy()
        self.temperature_last_step = None
        # self.ion_density_last_step = copy.deepcopy(self.ion_density)
        self.ion_boundary_idx=self.device.ion_boundary_idx

        self.ref_material=self.device.ref_material

        # ------------------------------------------------------------
        # 新增: 常驻的并行组装器 / 并行求解器, 整个 Simulator 生命周期内只创建一次,
        # 避免每次牛顿迭代都重建线程池 / 重建 PyPardisoSolver 导致的内存持续增长
        # ------------------------------------------------------------
        self._assembler = ParallelAssembler(
            max_workers=self.conf.get('assemble_num_threads', None)
        )
        self._sparse_solver = ParallelSparseSolver(
            num_threads=self.conf.get('solver_num_threads', None),
            release_every=self.conf.get('solver_release_every', 50),
        )

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
        # 20260909 robin boundary idx
        self.node_idx_of_thremal_robin_boundary=self.device.thremal_robin_boundary_node_idx
        
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
                    all_temperature_node_idx=list((set(self.node_idx_of_material('semiconductor'))|set(self.node_idx_of_material('oxide'))|set(self.node_idx_of_material('conductor')))-set(self.device.thermal_boundary_node_idx))
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
                    print('assemble '+quan['name']+' time ', time.time()-start, 's')
                start = time.time()
                dQ=self.solve(A,b)
                print('solve matrix time ', time.time()-start, 's')
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
        self.temperature_last_step = copy.deepcopy(self.temperature)
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
                # A=sp.sparse.lil_array((self.num_unknowns,self.num_unknowns))
                # b=np.zeros((self.num_unknowns))

                # for quan in self.quans:
                #     if quan['unknown'] == False:
                #         continue
                #     [A,b]=assemble(self,quan,A,b)

                # ------------------------------------------------------------
                # 改动点 1: 用常驻的 self._assembler 代替原来每次都新建线程池的
                # parallel_assemble() 函数调用
                # ------------------------------------------------------------
                # start = time.time()
                # A, b = parallel_assemble(self, self.quans, self.num_unknowns)
                A, b = self._assembler.assemble(self, self.quans, self.num_unknowns)
                # print('assemble time ', time.time()-start, 's')
                # ------------------------------------------------------------
                # 改动点 2: solve_parallel 显式传入常驻的 self._sparse_solver,
                # 不再让它在内部用默认参数新建 ParallelSparseSolver()
                # ------------------------------------------------------------
                # start = time.time()
                # solver = ParallelSparseSolver(num_threads=4)
                # dQ=self.solve_parallel(A,b,solver)
                dQ=self.solve_parallel(A,b,sparse_solver=self._sparse_solver)
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

    # def solve_parallel(self, A, b, sparse_solver=None):
    #     if self.conf['method'] == 'newton':
    #         A = A * np.broadcast_to(np.array(self.scaling_factor).T, A.shape)
    #     A = A.tocsr()
    #     if sparse_solver is None:
    #         sparse_solver = ParallelSparseSolver()
    #     dQ = sparse_solver.solve(A, b)
    #     if self.conf['method'] == 'newton':
    #         dQ = dQ * np.array(self.scaling_factor)
    #     return dQ

    def solve_parallel(self, A, b, sparse_solver):
        """
        sparse_solver 现在是必填参数, 不再提供"不传就新建"的默认行为。
        run_all() 里必须显式传入 self._sparse_solver, 以复用同一个 PARDISO
        句柄, 避免每次牛顿迭代都新建 PyPardisoSolver 造成内存持续增长。
        """
        if self.conf['method'] == 'newton':
            A = A * np.broadcast_to(np.array(self.scaling_factor).T, A.shape)
        A = A.tocsr()
        dQ = sparse_solver.solve(A, b)
        if self.conf['method'] == 'newton':
            dQ = dQ * np.array(self.scaling_factor)
        return dQ

    def __del__(self):
        """
        安全网: 如果调用方忘记显式调用 close(), 至少在对象被垃圾回收时尝试释放
        线程池和 PARDISO 内部内存, 避免忘记调用 close() 导致的资源泄漏。
        注意这不能完全替代显式 close() —— 对象什么时候被垃圾回收是不确定的
        (尤其是存在循环引用、或者在解释器退出阶段), __del__ 只是兜底手段。
        """
        try:
            self.close()
        except Exception:
            # 解释器关闭阶段模块可能已经被清空, 静默忽略即可
            pass

    def close(self):
        """
        整个仿真结束(比如完成所有偏置点扫描)后调用, 显式释放常驻线程池和
        PARDISO 内部内存。不调用也不会有正确性问题, 但长跑批量任务建议调用。
        """
        self._assembler.shutdown()
        self._sparse_solver.release()

    def __deepcopy__(self, memo):
        """
        自定义深拷贝行为: _assembler (ThreadPoolExecutor) 和 _sparse_solver
        (PyPardisoSolver) 都是运行时资源句柄, 内部含有 ctypes 指针 / OS 线程
        等不可 pickle 的对象, 不能也不应该被 copy.deepcopy 递归复制。

        典型用法(瞬态仿真里给每个时间步做快照, 比如
        `simulator_last = copy.deepcopy(simulator)`)通常只是想要一份独立的
        "状态数据"快照(网格量值、mapping 信息等), 并不需要这份快照自己也能
        立刻拿去跑牛顿迭代; 即便后续确实要在快照上继续求解, 让它拥有一份
        全新、独立的线程池/PARDISO 句柄也远比"尝试深拷贝 ctypes 指针"或者
        "两个 Simulator 实例共享同一个 PARDISO 句柄"更安全。
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        for key, value in self.__dict__.items():
            if key in ('_assembler', '_sparse_solver'):
                continue  # 跳过, 下面单独重新创建
            setattr(result, key, copy.deepcopy(value, memo))

        result._assembler = ParallelAssembler(
            max_workers=self.conf.get('assemble_num_threads', None)
        )
        result._sparse_solver = ParallelSparseSolver(
            num_threads=self.conf.get('solver_num_threads', None),
            release_every=self.conf.get('solver_release_every', 50),
        )

        return result


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
            self.ion_density_max = np.maximum(self.ion_density, self.ion_density_max)
            self.ion_density_max_all = np.maximum(self.ion_density, self.ion_density_max_all)
            # self.ion_density_max = np.maximum(self.ion_density, self.ion_density_max).clip(max=1e24)
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