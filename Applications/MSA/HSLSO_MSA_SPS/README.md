# HSLSO-based MSA: SPS comparison code

This package contains the SPS-based MSA implementation used to compare HSLSO with the comparison algorithms shown in the manuscript figure.

## Included algorithms

- HSLSO
- SLPSO
- CSO
- WGA
- EAPSO
- APSO-DEE (`APSO_DEE.m`)
- DE
- EO

`initialization.m` is retained because it is required by EO.

## Included DNA data

All A/C/G/T-only sequence datasets present in the supplied MSA source package are retained:

- `1aab_ref1`
- `1aboA_ref1`
- `1ad2_ref1`
- `1hfh_ref1`
- `1ivy_ref5`
- `451c_ref1`
- `arp_ref1`
- `kinase_ref1`

The protein file `BBA0001.tfa` is not included because this package is limited to the DNA/SPS version.

## Core files

- `fitness.m`: SPS-based fitness wrapper
- `Viterbi.m`: decoding routine
- `SPS.m`: SPS calculation

## Run

Open `run_MSA_SPS.m`, set `dataset`, `algorithm`, and `maxiter`, then run the script in MATLAB.

MATLAB Bioinformatics Toolbox is required for `fastaread`.

The original optimization algorithm files and SPS/MSA core routines are retained without changing their algorithmic logic. Historical result files, figures, backup files, unrelated algorithms, RDG/INTERACT utilities, and the protein-data version are omitted from this public package.
