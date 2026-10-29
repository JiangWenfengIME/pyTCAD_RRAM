# Archived figure data, not a new full-run validation

LTP: `ltp_read01_20261018_130753`, excluding +1.5 V.
LTD: `ltd_oldRESET_snapshot_TE50_20261018_164225`, excluding −1.2 V.

Each direction contains seven 500-pulse CSV sequences and the original `read_*_full500.png`. `values.csv`, `initial_read.csv` and `summary.csv` are copied byte-for-byte. Public `config.json` files retain the relevant physical/material/boundary and solver parameters while omitting process IDs, machine paths and legacy supervisor bookkeeping; compliance callable paths point to the packaged equivalent functions. CSV source hashes are in the source manifest.

Plot G from `conductance_S`, not `program_conductance_S`. G is measured with an independent ±0.1 V, 100 ps read. Pulse zero is not included in the figure. Files are not interpolated, smoothed, or extended to fill missing data.
