"""Fresh read-only Simulator from static Device plus exactly three ion arrays.
Programming Simulator is never copied into a reader and receives no read-back state.
"""
import copy,hashlib,json,math
import numpy as np
from .common import json_value,voltage_hold_solver,save_state
from .direct_read import Stepper as ContactStepper
ION_FIELDS=('ion_density','ion_density_residual','ion_density_max')

def fingerprint(sim):
    h=hashlib.sha256()
    def array(name,x):
        h.update(name.encode())
        if x is None:h.update(b'None');return
        a=np.asarray(x);h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    for name in ['ion_density','ion_density_residual','ion_density_max','ion_density_max_all','temperature','ion_density_last_step','temperature_last_step','ND','NA']:
        array(name,getattr(sim,name,None))
    for q in sim.quans:
        array('qvalue_'+q['name'],q['value']);array('ql2_'+q['name'],q['l2_norm'])
    for name,value in sorted(vars(sim.device).items()):
        if isinstance(value,np.ndarray):array('device_'+name,value)
    h.update(json.dumps(dict(conf=sim.conf,voltages=sim.voltage_biases,vapplied=getattr(sim,'vapplied',None),time_step=sim.time_step),default=json_value,sort_keys=True).encode())
    return h.hexdigest()

def build_reader(program):
    from .core.Simulator_parallel import Simulator
    # Device contains geometry, materials, doping and boundaries, not the solved
    # Simulator electronic/thermal fields. Deepcopy isolates all mutable objects.
    device_read=copy.deepcopy(program.device)
    device_read.temperature=np.full(len(device_read.mesh.nodes),float(program.conf['heat_robin_T_ambient']))
    device_read.ion_density=copy.deepcopy(program.ion_density)
    device_read.ion_density_residual=copy.deepcopy(program.ion_density_residual)
    conf=copy.deepcopy(program.conf)
    conf.update(compliance=False,with_electrode_voltage=False)
    simulator_read=Simulator(device_read,conf)
    simulator_read.ion_density_max=copy.deepcopy(program.ion_density_max)
    for key in ION_FIELDS:
        np.testing.assert_array_equal(getattr(simulator_read,key),getattr(program,key))
        assert not np.shares_memory(getattr(simulator_read,key),getattr(program,key))
    assert simulator_read.temperature_last_step is None
    assert simulator_read.ion_density_last_step is None
    assert simulator_read.num_unknowns_per_quan['electrode_voltage']==0
    assert all(v==0 for v in simulator_read.voltage_biases.values())
    return simulator_read

class ReadFailure(RuntimeError):
    def __init__(self,message,statistics):super().__init__(message);self.statistics=statistics

def read_conductance(program,*,direction,duration,pulse,attempts,state_path=None,failure_state_path=None):
    if direction not in ('LTP','LTD') or not math.isfinite(duration) or duration<=0:
        raise ValueError('Explicit positive read duration and direction required')
    voltage=.1 if direction=='LTP' else -.1
    before=fingerprint(program);reader=None;stepper=None
    class Log:
        def append(self,row):attempts.append(dict(row,pulse=pulse))
    try:
        reader=build_reader(program);initial=reader.ion_density.copy();initial_residual=reader.ion_density_residual.copy()
        stepper=ContactStepper(reader,Log())
        # Every newly constructed read has its own local clock starting at0.
        voltage_hold_solver(reader,voltage,duration,initial_steps=1,dt_min_ratio=1e-10,
                            name='independent_read',stepper=stepper,pulse=1,max_retries=64)
        reader=stepper.sim
        n,p=reader.get_contact_current(contact_name='TE');i=float(n+p);v=float(reader.voltage_biases['TE'])
        if not all(math.isfinite(x) for x in [i,v]) or abs(v-voltage)>1e-12:
            raise ValueError('Invalid direct read current/contact voltage')
        g=i/voltage
        if not math.isfinite(g):raise ValueError('Nonfinite independent read conductance')
        result=dict(read_voltage_V=voltage,read_TE_voltage_V=v,read_current_A=i,
                    read_electron_current_A=float(n),read_hole_current_A=float(p),conductance_S=g,
                    read_time_s=duration,read_ion_relative_l2=float(np.linalg.norm(reader.ion_density-initial)/max(np.linalg.norm(initial),1e-300)),
                    read_residual_ion_relative_l2=float(np.linalg.norm(reader.ion_density_residual-initial_residual)/max(np.linalg.norm(initial_residual),1e-300)),
                    programming_fingerprint_before=before,**{'read_'+k:v for k,v in stepper.condition.items()})
        if state_path is not None:save_state(reader,state_path)
        return result
    except Exception as exc:
        if stepper is not None:
            reader=stepper.sim
            if failure_state_path is not None:save_state(reader,failure_state_path)
        raise ReadFailure(str(exc),stepper.condition if stepper else {}) from exc
    finally:
        if stepper is not None:stepper.sim.close()
        elif reader is not None:reader.close()
        after=fingerprint(program)
        if after!=before:raise RuntimeError('Independent read mutated programming Simulator; experiment stopped')
