import copy, csv, hashlib, inspect, json, math
from pathlib import Path

def json_value(value):
    """Serialize complete Python configuration, including callable source."""
    if isinstance(value, Path):
        return str(value)
    if callable(value):
        return {'callable': value.__module__ + '.' + value.__name__,
                'source': inspect.getsource(value)}
    if hasattr(value, 'tolist'):
        return value.tolist()
    raise TypeError('Cannot serialize configuration value: ' + repr(value))

def write_json(path, value):
    with path.open('w', encoding='utf-8') as f:
        json.dump(value, f, default=json_value, ensure_ascii=False, indent=2,
                  allow_nan=False)
        f.write('\n')

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def load_state(path):
    """Deepcopy recreates handles excluded by dill, using existing core behavior."""
    import dill
    with path.open('rb') as f:
        loaded = dill.load(f)
    try:
        return copy.deepcopy(loaded)
    finally:
        loaded.close()

class CsvLog:
    def __init__(self, path, columns):
        self.file = path.open('x', encoding='utf-8', newline='')
        self.writer = csv.DictWriter(self.file, fieldnames=columns)
        self.writer.writeheader()
        self.file.flush()

    def append(self, row):
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        self.file.close()

def stats():
    return {'attempts': 0, 'failed_attempts': 0, 'iterations_total': 0,
            'min_time_step_s': None}

class Stepper:
    """Own the current state; every failed attempt restores the full snapshot."""
    def __init__(self, sim, log):
        self.sim = sim
        self.log = log
        self.total = stats()
        self.condition = stats()

    def attempt(self, pulse, time_start, time_target, voltage, time_step=None):
        dt = time_target - time_start if time_step is None else time_step
        if dt <= 0:
            raise RuntimeError('Time step cannot advance floating-point physical time')
        backup = copy.deepcopy(self.sim)
        before = [len(q['l2_norm']) for q in self.sim.quans]
        error = None
        try:
            self.sim.vapplied = voltage
            self.sim.time_step = dt
            converged = bool(self.sim.run_all())
            if not converged:
                error_text = 'max_iter reached without convergence'
            else:
                error_text = ''
        except Exception as exc:
            error = exc
            error_text = type(exc).__name__ + ': ' + str(exc)
            converged = False
        iterations = max((len(q['l2_norm']) - n
                          for q, n in zip(self.sim.quans, before)), default=0)
        self.log.append(dict(pulse=pulse, time_start_s=time_start,
                             time_target_s=time_target, dt_s=dt,
                             applied_voltage_V=voltage, converged=converged,
                             iterations=iterations, error=error_text))
        for counter in (self.total, self.condition):
            counter['attempts'] += 1
            counter['failed_attempts'] += int(not converged)
            counter['iterations_total'] += iterations
            old = counter['min_time_step_s']
            counter['min_time_step_s'] = dt if old is None else min(old, dt)
        if converged:
            backup.close()
        else:
            self.sim.close()
            self.sim = backup
            print('Rollback:', error_text, '; attempted dt=', dt, flush=True)
        # Numerical failures are retried with smaller dt; programming/I/O failures stop.
        if error is not None and not isinstance(error, (ValueError, ArithmeticError)):
            raise RuntimeError('Solver error recorded; experiment stopped') from error
        return converged

def check_retry(dt, dt_min, failures, limit):
    if dt < dt_min or failures > limit:
        raise RuntimeError('Cannot converge within dt_min / max_retries; see attempts.csv')

def voltage_hold_solver(simulator, voltage, hold_time, initial_steps=10,
                        dt_min_ratio=1e-10, name='set_hold_state', *,
                        stepper, pulse, max_retries=64):
    """Notebook hold algorithm, with attempt logging and bounded failure retries.

    The simulator dt is computed from local elapsed time, exactly as in the
    notebook. Global timestamps in attempts.csv do not determine solver dt.
    """
    if stepper.sim is not simulator:
        raise ValueError('Stepper does not own the supplied simulator')
    if initial_steps < 1 or hold_time <= 0:
        raise ValueError('Require positive hold_time and initial_steps')
    start = (pulse - 1) * hold_time
    elapsed = 0.0
    dt_initial = hold_time / initial_steps
    dt = dt_initial
    failures = 0
    print(name, 'hold_voltage=', voltage, 'hold_time=', hold_time, flush=True)
    while elapsed < hold_time:
        dt_current = min(dt, hold_time - elapsed)
        target_elapsed = elapsed + dt_current
        if target_elapsed <= elapsed:
            raise RuntimeError('Hold time step cannot advance local elapsed time')
        target_time = pulse * hold_time if target_elapsed == hold_time else start + target_elapsed
        if stepper.attempt(pulse, start + elapsed, target_time,
                           voltage, time_step=dt_current):
            elapsed = target_elapsed
            failures = 0
            dt = min(dt_initial, dt * 2)
        else:
            failures += 1
            dt /= 2
            check_retry(dt, dt_initial * dt_min_ratio, failures, max_retries)
    return stepper.sim

def measurement(sim, pulse, time_s, voltage):
    In, Ip = sim.get_contact_current(contact_name='TE')
    vte = float(sim.voltage_biases['TE'])
    current = float(In + Ip)
    if vte == 0 or not all(math.isfinite(float(x)) for x in (In, Ip, vte, current)):
        raise ValueError('Nonfinite current or zero TE voltage at converged step')
    return dict(pulse=pulse, time_s=time_s, applied_voltage_V=voltage,
                TE_voltage_V=vte, electron_current_A=float(In),
                hole_current_A=float(Ip), current_A=current, conductance_S=current / vte)

def save_state(sim, path):
    import dill
    # Exclusive creation protects prior runs. Save only after the whole stage succeeds.
    with path.open('xb') as f:
        dill.dump(sim, f)
