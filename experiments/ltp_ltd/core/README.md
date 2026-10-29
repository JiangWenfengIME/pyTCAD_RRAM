# Isolated research-run backend

These eight modules were copied from the research-run `pytcad` tree. Only `pytcad.` import prefixes were changed to `experiments.ltp_ltd.core.`. No equations, material defaults or Newton updates were edited. Source/package hashes are in `validation/ltp_ltd/source_manifest.json`.

This namespace is used only by the new LTP/LTD examples and their converted initial states. The existing repository `pytcad/` is deliberately retained unchanged for `RRAM_forming_reset_set.ipynb`. Do not replace it with this directory without a separate legacy regression study.

The sparse solver selects optional PARDISO when installed, otherwise SciPy SuperLU. This choice is recorded in run config. The existing state serialization omits native solver/thread handles; loading/deepcopy recreates those resources.
