# Packaged programming initial states

LTP_initial.pkl: full 0.83 V SET endpoint.
LTD_initial.pkl: fresh BE50/TE50 LTD state prepared from the selected partial RESET snapshot by importing only the three ion arrays.

Load through `experiments.ltp_ltd.run.initial_state` from the repository root; checksums are verified. The package namespace was converted for portability, without recomputing physical fields. See manifest.json and LTP_LTD_README.md. These are dill files: load only trusted files and use the documented environment.
