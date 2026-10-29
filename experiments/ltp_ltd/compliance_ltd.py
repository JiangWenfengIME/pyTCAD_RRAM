COMPLIANCE = {'Ic': -4e-4, 'Vth': 0.4, 'SS': 0.5, 'beta': 0.7,
              'v0': 0.3, 'gamma': 1, 'epsilon_voltage': 1e-9}

def iv_compliance(v):
    """Exact negative-voltage compliance expression from the RESET notebook."""
    import numpy as np
    p = COMPLIANCE
    v_safe = -v - p['epsilon_voltage']
    f = 1 / (1 + np.exp(-(v_safe - p['Vth']) / p['SS']))
    g = np.exp(-(p['v0'] / v_safe) ** p['gamma'])
    i = p['Ic'] * f ** p['beta'] * g
    df_dv = f * (1 - f) / p['SS']
    dg_dv = g * p['gamma'] * p['v0'] ** p['gamma'] / v_safe ** (p['gamma'] + 1)
    di_dv = -p['Ic'] * (p['beta'] * f ** (p['beta'] - 1) * df_dv * g
                       + f ** p['beta'] * dg_dv)
    return i, di_dv
