# pyTCAD_RRAM

A Physics-Based Fully Coupled TCAD Framework for Oxide-Based RRAM

## 1. Overview

`pyTCAD_RRAM` is a physics-based, fully coupled TCAD framework developed for the multiphysics simulation of oxide-based resistive random-access memory (RRAM) devices.

The framework is designed to self-consistently solve the coupled electrothermal and ionic transport processes involved in RRAM switching, including:

* Poisson equation
* Electron drift-diffusion transport
* Fourier heat conduction
* Oxygen-vacancy transport
* Oxygen-vacancy generation/recombination
* Soret thermophoresis
* Current-compliance-controlled switching

The framework is used to investigate the forming, reset, and set processes of oxide-based RRAM and, in particular, the statistical variability associated with the spatial distribution of oxygen vacancies and electrode/oxide interface morphology.

The repository provides the source code, device mesh, and example Jupyter notebook used for RRAM forming/reset/set simulations.

The simulation framework is currently demonstrated using a **two-dimensional (2D) triangular mesh**, and also support **three-dimensional (3D) tetrahedron mesh**.

---

## 2. Repository Structure

```text
pyTCAD_RRAM/
│
├── device/
│   └── RRAM/
│       └── RRAM.mesh
│
├── pytcad/
│   ├── Assemble.py
│   ├── Assemble_vectorized.py
│   ├── Device.py
│   ├── GRF_fft.py
│   ├── Materials.py
│   ├── Mesh.py
│   ├── Physic.py
│   ├── Post.py
│   ├── Quantities.py
│   └── Simulator.py
│
├── results_vtk/
│
└── RRAM_forming_reset_set.ipynb
```

### 2.1 `device/`

Contains the device geometry and mesh files used by the examples.

```text
device/
└── RRAM/
    └── RRAM.mesh
```

`RRAM.mesh` contains the 2D triangular computational mesh of the RRAM device.

The example device consists of:

* Top electrode (TE)
* HfOx switching layer
* Bottom electrode (BE)

The mesh is imported by the `Mesh` module and subsequently processed by the `Device` module.

---

### 2.2 `pytcad/`

This directory contains the core components of the TCAD framework.

#### `Mesh.py`

Handles mesh reading and mesh-related preprocessing.

Main functions include:

* Reading the computational mesh
* Processing triangular elements and tetrahedron elements
* Handling mesh regions
* Defining geometric information
* Supporting the 2D and 3D simulation domain

---

#### `Device.py`

Handles device-level preprocessing and configuration.

The module is responsible for:

* Defining device regions
* Assigning electrodes
* Assigning material properties
* Defining ionic distributions
* Defining thermal boundaries
* Setting device-level physical parameters

---

#### `Materials.py`

Contains material definitions and material-related parameters.

The example uses:

* Conductive materials for the top and bottom electrodes
* Semiconductor-like transport treatment for Oxides
* Semiconductor materials

Material parameters can be modified according to the device and physical model under investigation.

---

#### `Physic.py`

Contains the physical parameters and physical models used by the simulator.

The module provides the parameters required by the coupled multiphysics equations, including electrical, thermal, carrier-transport, and ionic-transport parameters.

---

#### `Quantities.py`

Handles the physical quantities that are solved and updated during the simulation.

Examples include:

* Electrostatic potential
* Electron density
* Temperature
* Oxygen-vacancy concentration

---

#### `Assemble.py`

Provides the matrix/vector assembly procedures required for solving the discretized governing equations.

---

#### `Assemble_vectorized.py`

Provides a vectorized implementation of the assembly procedures to improve computational efficiency and facilitate parallel numerical operations.

---

#### `Simulator.py`

Contains the main simulation framework.

The module handles:

* Physical parameter transfer
* Boundary-condition setup
* Nonlinear equation solving
* Newton/Gummel-type iterations
* Variable updates
* Coupled multiphysics solution

The simulator supports different nonlinear solution configurations, including Newton-based solution procedures.

---

#### `GRF_fft.py` 

Provides the generation and visualization of spatially correlated random fields.

The module is used to generate spatially correlated Gaussian random fields that can be used to represent nonuniform initial oxygen-vacancy distributions. The spatially correlated random-field approach provides a statistical description of the initial ionic disorder used in the RRAM simulations.

---

#### `Post.py`

Contains post-processing and data-export functions.

The example uses the module to:

* Extract terminal current
* Process simulation variables
* Export simulation results
* Generate VTK files
* Generate time-dependent VTK collections for visualization

The exported VTK files can be visualized using ParaView or other VTK-compatible visualization software.

---

## 3. Example: RRAM Forming, Reset and Set Simulation

The main example is provided in:

```text
RRAM_forming_reset_set.ipynb
```

The notebook demonstrates a complete RRAM simulation workflow.

The general simulation sequence is:

```text
Mesh
  ↓
Device preprocessing
  ↓
Material assignment
  ↓
Initial oxygen-vacancy distribution
  ↓
Boundary-condition setup
  ↓
Initial equilibrium solution
  ↓
Voltage-controlled forming
  ↓
Current compliance
  ↓
Reset
  ↓
Set
  ↓
Post-processing and visualization
```

---

## 4. Loading the Device Mesh

The example uses a 2D triangular mesh:

```python
mesh = Mesh(
    mesh_type='2D',
    cell_type='triangular',
    filename='device/RRAM/RRAM.mesh',
    scale=1e-9,
    symmetric_axis='y-axis'
)
```

The mesh is scaled from nanometer-based geometry to SI units.

The device regions are then assigned as:

```python
mesh.set_region_name(reg_num=0, reg_name='hfox')
mesh.set_region_name(reg_num=1, reg_name='BE')
mesh.set_region_name(reg_num=2, reg_name='TE')
```

where:

* `hfox`: HfOx switching layer
* `BE`: bottom electrode
* `TE`: top electrode

---

## 5. Device and Material Definition

The electrodes are defined as:

```python
device.set_electrode(
    reg_name='BE',
    electrode_name='BE'
)

device.set_electrode(
    reg_name='TE',
    electrode_name='TE'
)
```

The HfOx region is assigned a semiconductor-like transport model with oxygen-vacancy transport:

```python
hfox_semi = Semiconductor(
    'hfox',
    mobility_model='rram_carrier_1',
    recombination_models=None,
    ion_recombination_models=['oxygen_vacancies']
)
```

The electrodes are treated as conductive regions.

---

## 6. Initial Oxygen-Vacancy Distribution

The example allows the initial ionic distribution to be specified explicitly.

A uniform background oxygen-vacancy distribution can be defined using:

```python
device.set_uniform_ion(
    reg_name='hfox',
    conc=10,
    type="acceptor"
)
```

A localized Gaussian oxygen-vacancy distribution can also be introduced:

```python
device.set_gaussian_ion(
    reg_name='hfox',
    conc=1e25,
    type="donor",
    char=2e-9,
    x_min=0e-9,
    x_max=5e-9,
    y_peak=20e-9
)
```

The interface ionic distribution can be specified using:

```python
device.set_ion_boundary_region_interface(
    reg_name_1='hfox',
    reg_name_2='TE',
    conc=1e25,
    type="donor"
)
```

For statistical studies, the `GRF_fft.py` module can be used to generate spatially correlated Gaussian random fields and construct nonuniform initial oxygen-vacancy distributions.

---

## 7. Thermal Boundary Conditions

Thermal boundaries are explicitly defined in the device setup.

For example:

```python
device.set_thermal_boundary(
    x_min=-1,
    x_max=1,
    y_min=-0.01e-9,
    y_max=0.01e-9
)

device.set_thermal_boundary(
    x_min=-1,
    x_max=1,
    y_min=29.99e-9,
    y_max=30.01e-9
)
```

These boundaries are used to define the thermal contacts of the simulated structure.

---

## 8. Coupled Simulation Configuration

The simulator is configured to solve the coupled electrothermal and ionic transport problem.

A representative configuration is:

```python
conf = {
    'method': 'newton',
    'max_iter': 100,
    'damping': 0.8,
    'l2_resolution': 1e-7,
    'with_electron': True,
    'with_hole': False,
    'with_temperature': True,
    'with_ion': True,
    'with_conductor_temperature': True,
    'uniform_band': True,
    'compliance': False,
    'with_soret': True
}
```

The main enabled physical quantities include:

* Electron transport
* Temperature
* Oxygen-vacancy transport
* Conductor temperature
* Soret thermophoresis

The nonlinear equations are solved using a Newton-based iterative procedure in the example.

---

## 9. Initial Equilibrium Solution

Before applying the forming voltage, an initial coupled solution is obtained:

```python
simulator_initial = Simulator(device, conf)

simulator_initial.run_all()
```

The resulting state provides the initial electrothermal and carrier state for subsequent voltage-driven simulations.

The terminal current can be extracted using:

```python
Jn, Jp = simulator_initial.get_contact_current(
    contact_name='TE'
)
```

---

## 10. Forming Simulation

The forming process is simulated using a piecewise-linear voltage waveform:

```python
pwl_list = {
    "time": [0.0, 0.020, 0.04],
    "voltage": [0.001, 2.0, 0.001]
}
```

The applied voltage is updated as a function of time.

The simulation uses adaptive time stepping.

The time step is reduced when the nonlinear solver fails to converge:

```text
Non-convergence
      ↓
Restore previous converged state
      ↓
Reduce time step
      ↓
Retry simulation
```

After several consecutive converged steps, the time step can be increased within the predefined maximum time step.

This procedure improves numerical robustness during rapidly changing forming and switching processes.

---

## 11. Current Compliance

A voltage-dependent current-compliance function is used in the example:

```python
def iv_compliance(v):
    Ic = 1e-4
    alpha = 3.0

    i = Ic * (1 - np.exp(-alpha * v))

    di_dv = (
        Ic
        * (-np.exp(-alpha * v))
        * (-alpha)
    )

    return i, di_dv
```

The compliance condition can be enabled through the simulator configuration:

```python
conf['compliance'] = True
conf['compliance_method'] = 'iv_func'
conf['func'] = iv_compliance
conf['compliance_electrode'] = 'TE'
```

The compliance condition limits the current during the voltage-driven switching process.

---

## 12. Time-Dependent Output

During the transient simulation, the following quantities are recorded:

* Applied voltage
* Device terminal voltage
* Terminal current
* Simulation time

Representative variables are:

```python
time_all
voltage
current
voltage_vte
```

The evolving physical state can also be exported as VTK files.

For example:

```python
post.export_vtk(
    simulator,
    vtk_foldername,
    ...,
    file_name_time_stepped_pvd=file_name_time_stepped_pvd,
    timestep=pvd_step
)
```

This allows the spatial and temporal evolution of the simulated RRAM state to be visualized.

---

## 13. Simulation Results

Simulation results are stored under:

```text
results_vtk/
```

The example automatically creates date-stamped directories such as:

```text
results_vtk/
└── RRAM_forming_YYYYMMDD/
```

The output may include:

* VTK field data
* Time-dependent VTK collections (`.pvd`)
* Current-voltage curves
* Time-dependent voltage data
* Intermediate simulation states

The VTK results can be opened using ParaView.

---

## 14. Visualization

The generated VTK/PVD files can be visualized using ParaView.

A typical workflow is:

```text
Run RRAM_forming_reset_set.ipynb
             ↓
      Generate VTK/PVD files
             ↓
          Open in
          ParaView
             ↓
Visualize spatial evolution of
potential / carrier / temperature /
oxygen-vacancy-related quantities
```

The notebook also generates current-voltage plots for the simulated switching process.

---

## 15. Spatially Correlated Gaussian Random Field

The repository includes:

```text
pytcad/GRF_fft.py
```

for generating spatially correlated Gaussian random fields.

The generated field can be used to introduce spatial nonuniformity into the initial oxygen-vacancy distribution.

The key statistical parameters include:

* `σ`: fluctuation amplitude / standard deviation of the generated random field
* `λ`: spatial correlation length

Conceptually:

```text
Gaussian random field
        ↓
Spatial correlation
        ↓
Nonuniform oxygen-vacancy distribution
        ↓
RRAM forming simulation
        ↓
Device-to-device variability
```

By changing the random-field parameters, statistically different initial ionic configurations can be generated for device-to-device variability studies.

---

## 16. Reproducibility

To reproduce a simulation:

1. Clone or download this repository.
2. Install the required Python dependencies.
3. Open:

```text
RRAM_forming_reset_set.ipynb
```

4. Run the notebook from the beginning.
5. Ensure that the mesh path points to:

```text
device/RRAM/RRAM.mesh
```

6. Check the generated results under:

```text
results_vtk/
```

For statistical simulations, each device can be initialized using a different realization of the spatially correlated Gaussian random field.

For exact reproducibility of a particular random realization, the corresponding random-number seed should be recorded and reused.

---

## 17. Main Software Components

The overall architecture of `pyTCAD_RRAM` can be summarized as follows:

```text
                 pyTCAD_RRAM
                      │
          ┌───────────┴───────────┐
          │                       │
       Device                    Mesh
          │                       │
          └───────────┬───────────┘
                      │
               Material / Physics
                      │
          ┌───────────┴───────────┐
          │                       │
       Quantities              GRF_fft
          │                       │
          └───────────┬───────────┘
                      │
                 Simulator
                      │
              ┌───────┴───────┐
              │               │
        Numerical Solver   Time Stepping
              │               │
              └───────┬───────┘
                      │
                    Post
                      │
              ┌───────┴───────┐
              │               │
             VTK          IV Curves
```

---

## 18. Scope and Current Limitations

The present example demonstrates a **two-dimensional cross-sectional TCAD implementation** for oxide-based RRAM.

The statistical variability demonstrated in the accompanying work is introduced through variations in the initial ionic/oxygen-vacancy distribution and interface morphology.

The current framework should therefore be interpreted within the assumptions and physical models implemented in the repository.

In particular, the present implementation does not imply that the subsequent ionic evolution is intrinsically stochastic for an identical initial state. For a fixed initial configuration, material parameters, boundary conditions, and applied voltage waveform, the governing numerical model produces a deterministic solution.

---

## 19. Citation

If you use this code or the associated models in academic research, please cite the corresponding publication:

```text
[DOI: 10.1109/TED.2026.3737722]
```

---

## 20. License

Please refer to the repository license for the terms governing the use, modification, and redistribution of the source code.

---

## 21. Acknowledgement

The authors appreciate the open-source community and provide this repository to facilitate reproducibility, further development, and independent research on physics-based TCAD simulation of oxide-based RRAM.
