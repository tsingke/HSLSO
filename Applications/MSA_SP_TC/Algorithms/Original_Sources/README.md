# Original level-based optimizer source snapshots

The three `.m.txt` files in this folder contain source snapshots corresponding
to LLSO, DLLSO, and RLLPSO. They are retained for provenance and comparison;
the executable MSA adaptations are:

- `Algorithms/LLSO.m`
- `Algorithms/DLLSO.m`
- `Algorithms/RLLPSO.m`

The executable versions adapt the algorithms to maximize the internal SPS
objective, use the Profile-HMM decision-vector representation and MSA bounds,
and obey the common FE budget specified in `config_msa.m`.
