import copy
from .common import stats

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
            self.sim.set_contact_voltage('TE', voltage)
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
