"""Portable constant-hold programming with independent low-bias reads."""
import argparse
import csv
import json
import platform
import time
from datetime import datetime
from pathlib import Path
import dill
import numpy as np
from .common import (CsvLog, Stepper, stats, voltage_hold_solver, load_state,
                     save_state, measurement, write_json, digest)
from .reader import read_conductance, build_reader, ReadFailure
from .core import Post as post
from .core.Simulator_parallel import _HAS_PARDISO
ROOT = Path(__file__).resolve().parents[2]
ATTEMPTS = ['pulse','time_start_s','time_target_s','dt_s','applied_voltage_V','converged','iterations','error']


def settings(direction):
    if direction not in ('LTP', 'LTD'):
        raise ValueError('direction must be LTP or LTD')
    return json.loads(Path(__file__).with_name(direction.lower()+'_settings.json').read_text())


def initial_state(direction):
    """Load the packaged full initial state; no SET/RESET recalculation."""
    source = ROOT/'states/ltp_ltd'/f'{direction}_initial.pkl'
    expected=json.loads((source.parent/'manifest.json').read_text())[direction]['packaged_sha256']
    if digest(source) != expected:
        raise ValueError('Initial state checksum mismatch')
    sim = load_state(source)
    return sim


def run_condition(sim, direction, voltage, output, *, pulses=500, hold_time=1e-6,
                  read_time=1e-10, checkpoint_interval=50, plot_every=10):
    """Takes ownership of sim, closes it on completion/failure; new output required.

    Editing material/conf parameters must be deliberate: the loaded state's
    configuration is authoritative. Unknown-layout changes require reconstruction.
    """
    if direction not in ('LTP','LTD') or not np.isfinite(voltage) or voltage == 0:
        raise ValueError('Invalid direction or voltage')
    if (direction=='LTP') != (voltage>0):raise ValueError('Voltage sign differs from direction')
    if pulses<1 or hold_time<=0 or read_time<=0 or checkpoint_interval<1:
        raise ValueError('Positive pulse/time/checkpoint parameters required')
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    vtk=out/'vtk';vtk.mkdir();sd=out/'states';sd.mkdir()
    pvd=vtk/'pulses.pvd'
    pvd.write_text('<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n<Collection>\n')
    config=dict(direction=direction,hold_voltage_V=voltage,pulses=pulses,
        hold_time_s=hold_time,read_time_s=read_time,read_voltage_V=.1 if direction=='LTP' else -.1,
        program_conf=sim.conf,materials={r.name:vars(r.material) for r in sim.mesh.regions},
        status='running',completed_program_pulses=0,completed_reads=0,
        linear_solver='PARDISO' if _HAS_PARDISO else 'SciPy SuperLU',python=platform.python_version(),
        numpy=np.__version__,created_at=datetime.now().isoformat(),
        source_state_sha256=digest(ROOT/'states/ltp_ltd'/f'{direction}_initial.pkl'),
        initial_steps=1,dt_min_ratio=1e-10,max_retries=64,
        read_definition='I_read / signed read voltage; fresh reader; no read-back to program')
    logs=[];stepper=None;stage='initialization';pulse=0;rows=[];start=time.monotonic()
    try:
        pa=CsvLog(out/'attempts.csv',ATTEMPTS);logs.append(pa)
        ra=CsvLog(out/'read_attempts.csv',ATTEMPTS);logs.append(ra)
        summary=CsvLog(out/'summary.csv',['pulse','stage','converged','attempts','failed_attempts','iterations_total','min_time_step_s','error']);logs.append(summary)
        stepper=Stepper(sim,pa)
        reader=build_reader(sim)
        try:config['read_conf']=reader.conf
        finally:reader.close()
        write_json(out/'config.json',config)
        save_state(sim,sd/'simulator_program_initial.pkl')
        stage='initial_read'
        initial=read_conductance(sim,direction=direction,duration=read_time,pulse=0,attempts=ra,
            state_path=sd/'simulator_read_initial.pkl',failure_state_path=sd/'simulator_read_initial_failed.pkl')
        with (out/'initial_read.csv').open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['pulse']+list(initial));w.writeheader();w.writerow(dict(pulse=0,**initial))
        values=None
        for pulse in range(1,pulses+1):
            stage='program';stepper.condition=stats()
            voltage_hold_solver(stepper.sim,voltage,hold_time,initial_steps=1,dt_min_ratio=1e-10,
                name=direction.lower()+'_hold_state',stepper=stepper,pulse=pulse,max_retries=64)
            config['completed_program_pulses']=pulse
            post.export_vtk(stepper.sim,str(vtk),f'RRAM_{"set" if direction=="LTP" else "reset"}_pulse_{pulse}',
                x_scale=1,y_scale=1,file_name_time_stepped_pvd=str(pvd),timestep=pulse*hold_time)
            pg=measurement(stepper.sim,pulse,pulse*hold_time,voltage)
            summary.append(dict(pulse=pulse,stage=stage,converged=True,error='',**stepper.condition))
            if pulse%checkpoint_interval==0:save_state(stepper.sim,sd/f'program_{pulse:04d}.pkl')
            stage='read'
            rr=read_conductance(stepper.sim,direction=direction,duration=read_time,pulse=pulse,attempts=ra,
                state_path=sd/'simulator_read_end.pkl' if pulse==pulses else None,
                failure_state_path=sd/f'simulator_read_failed_{pulse:04d}.pkl')
            row=dict(pulse=pulse,time_s=pulse*hold_time,program_applied_voltage_V=voltage,
                program_TE_voltage_V=pg['TE_voltage_V'],program_current_A=pg['current_A'],
                program_conductance_S=pg['conductance_S'],**rr)
            if values is None:values=CsvLog(out/'values.csv',list(row));logs.append(values)
            values.append(row);rows.append(row)
            summary.append(dict(pulse=pulse,stage=stage,converged=True,error='',**{k:rr['read_'+k] for k in stats()}))
            config['completed_reads']=pulse;write_json(out/'config.json',config)
            print(f'{direction} {voltage:g} V: {pulse}/{pulses}, read G={rr["conductance_S"]*1e6:.9g} uS',flush=True)
            if plot_every and pulse%plot_every==0:
                from analysis.ltp_ltd.plot import plot_curves
                plot_curves([out],out/'figures',direction)
        save_state(stepper.sim,sd/f'simulator_{direction}_end.pkl')
        for name,column in [('current','read_current_A'),('voltage','read_voltage_V'),('conductance','conductance_S'),('program_current','program_current_A'),('program_voltage','program_applied_voltage_V'),('program_voltage_vte','program_TE_voltage_V')]:
            with (sd/(name+'.pkl')).open('xb') as f:dill.dump([r[column] for r in rows],f)
        config['status']='completed'
    except BaseException as exc:
        config.update(status='failed',failed_pulse=pulse,failed_stage=stage,error=repr(exc))
        if stepper is not None:
            save_state(stepper.sim,sd/f'simulator_partial_{pulse:04d}.pkl')
            detail=exc.statistics if isinstance(exc,ReadFailure) else stepper.condition
            summary.append(dict(pulse=pulse,stage=stage,converged=False,error=repr(exc),**detail))
        raise
    finally:
        with pvd.open('a') as f:f.write('</Collection>\n</VTKFile>\n')
        for log in logs:log.close()
        if stepper is not None:stepper.sim.close()
        else:sim.close()
        config['elapsed_wall_seconds']=time.monotonic()-start
        write_json(out/'config.json',config)
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('direction',choices=['LTP','LTD']);p.add_argument('--voltages',type=float,nargs='+')
    p.add_argument('--pulses',type=int,default=500);p.add_argument('--output',type=Path)
    p.add_argument('--plot-every',type=int,default=10)
    a=p.parse_args();s=settings(a.direction)
    root=a.output or ROOT/'results_ltp_ltd'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')/a.direction
    paths=[]
    for voltage in a.voltages or s['voltages_V']:
        paths.append(run_condition(initial_state(a.direction),a.direction,voltage,root/f'{voltage:g}V',
            pulses=a.pulses,plot_every=a.plot_every))
    from analysis.ltp_ltd.plot import plot_curves
    plot_curves(paths,root/'figures',a.direction)
if __name__=='__main__':main()
