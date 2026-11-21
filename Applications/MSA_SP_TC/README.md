# HSLSO Profile-HMM MSA Reproducibility Package

## 1. Purpose and scope

This package contains the MATLAB source code and the six BALI2DNA nucleotide
instances used for the Profile-HMM multiple sequence alignment experiments.
It includes the 11 optimizers compared in the manuscript:

1. HSLSO
2. SLPSO
3. CSO
4. WGA
5. EAPSO
6. APSO-DEE
7. DE
8. EO
9. LLSO
10. DLLSO
11. RLLPSO

The experiment driver performs HMM parameter optimization, Viterbi decoding,
and the per-run calculation of SPS, SP, and TC. It writes raw run-level
observations and descriptive summaries (mean and standard deviation).

All experiments in this package use a common function-evaluation (FE) budget.
Results obtained under different evaluation-budget protocols should be reported
separately.

## 2. Contents

- `Algorithms/`: the 11 optimizers and their MSA-specific implementations.
- `Core/`: Profile-HMM construction, initialization, fitness evaluation,
  Viterbi decoding, SPS, SP/TC scoring, and descriptive result aggregation.
- `Data/Input/`: the six unaligned BALI2DNA nucleotide FASTA inputs.
- `Data/BALI2DNA/ref/`: the six matching reference nucleotide alignments.
- `Data/Dataset_Manifest.csv`: dataset names, input/reference paths, sequence
  counts, sequence lengths, and reference-file SHA-256 hashes.
- `config_msa.m`: experiment configuration.
- `run_reproduce_all.m`: complete experiment entry point.
- `validate_reproducibility_package.m`: checks source files and bundled data.
- `validate_bali2dna_data.m`: verifies input/reference sequence correspondence.

## 3. Datasets and reference alignments

Each `Data/Input/` file contains unaligned nucleotide sequences. The matching
`Data/BALI2DNA/ref/` file contains the reference nucleotide alignment used
only for the per-run SP/TC evaluation after Viterbi decoding.

| Dataset | Input file | Reference file | Sequences |
|---|---|---|---:|
| 1hfh | `1hfh_ref1` | `1hfh_ref1` | 5 |
| 451c | `451c_ref1` | `451c_ref1` | 5 |
| 1ivy | `1ivy_ref5` | `1ivy_ref5` | 7 |
| kinase | `kinase_ref1` | `kinase_ref1` | 5 |
| 1ad2 | `1ad2_ref1` | `1ad2_ref1` | 4 |
| 1aab | `1aab_ref1` | `1aab_ref1` | 4 |

The reference files are the bundled files from the BALI2DNA benchmark
distribution associated with BAliBASE. The original benchmark source is:

https://www.drive5.com/bench/bench.tar.gz

The exact reference files used here are included in the package. The scripts
do not download, replace, or select alternative reference alignments.

## 4. Optimizer population sizes and evaluation budgets

All 11 optimizers use the same total objective-evaluation budget:

**MaxFEs = 10,050 per run, including all initialization evaluations.**

Each optimizer run is stopped when the fitness-evaluation counter reaches
10,050. The experiment driver independently counts calls to `Core/fitness.m`
and checks that the observed count equals the configured budget before
accepting a run.

| Optimizer | Initial population | Minimum active population | Total FEs per run |
|---|---:|---:|---:|
| HSLSO | 50 | 50 | 10,050 |
| SLPSO | 50 | 50 | 10,050 |
| CSO | 50 | 50 | 10,050 |
| WGA | 120 | 30 | 10,050 |
| EAPSO | 50 | 50 | 10,050 |
| APSO-DEE | 50 | 50 | 10,050 |
| DE | 50 | 50 | 10,050 |
| EO | 30 | 30 | 10,050 |
| LLSO | 50 | 50 | 10,050 |
| DLLSO | 50 | 50 | 10,050 |
| RLLPSO | 50 | 50 | 10,050 |

WGA reduces its active population from 120 to 30 according to the consumed
evaluation-budget ratio. The other optimizers use fixed population sizes.
For EO, the initial candidate population is evaluated during its first
generation; those evaluations are included in the same total budget.

To enforce the common budget without overshooting, algorithms may terminate
partway through their final update sweep/generation. WGA's population-size
schedule and APSO-DEE's original group-size stages are indexed by the consumed
FE ratio in this MSA budget-controlled implementation. EO's decay schedule
uses the number of generations implied by the configured FE budget. These
budget-control changes are part of the executable MSA adaptations and are
visible in the corresponding optimizer source files.

`Results/<experiment>/Optimizer_Budget_Specification.csv` records the
configured population sizes and FE budget. Every run MAT file and every
`RawResults.csv` also records the actual FEs consumed.

The default configuration uses 30 independent runs for each optimizer and
each dataset. Convergence histories contain 200 samples on the FE axis;
these 200 history points are not an iteration limit.

## 5. HSLSO implementation consistency

The MSA-specific `Algorithms/HSLSO.m` implements the two progress-dependent
mechanisms using the consumed fitness-evaluation ratio:

- Shrinking superior-level sampling range:

  `k = max(2, ceil((level-1)*(1-(FEs/MaxFEs)^2)))`

  Two distinct superior levels are sampled from `1:k`.

- Probability stage-aware update control:

  `PL(level) = PLinit(level) + (FEs/MaxFEs) * (PLfinal(level)-PLinit(level))`

The initial population evaluations are included in `FEs`. The stopping
condition is based on the total evaluation budget, not the number of outer
iterations. This MSA implementation receives `MaxFEs` explicitly from the
experiment driver.

## 6. Model and per-run evaluation workflow

1. Read the unaligned BALI2DNA nucleotide sequences.
2. Translate nucleotide sequences into proteins.
3. Calculate pairwise protein identity distances.
4. Construct a UPGMA guide tree from the identity distances.
5. Build a progressive protein pre-alignment and project it back to
   nucleotide sequences by codon.
6. Estimate the Profile-HMM structure and parameter prior from this
   input-only pre-alignment.
7. Optimize the HMM parameter vector using the internal SPS fitness.
8. Decode the optimized Profile-HMM using Viterbi to obtain the final MSA.
9. Calculate SPS for the final decoded alignment.
10. Evaluate the final alignment against the corresponding reference
    alignment using SP and TC.

**Reference alignments are not used** for guide-tree construction,
pre-alignment, Profile-HMM initialization, parameter optimization, or
Viterbi decoding. They are read only when calculating SP and TC after
optimization.

### Metric distinction

- `Core/SPS.m`: reference-free affine sum-of-pairs objective used by the
  optimizer. It is not the BAliBASE reference SP/Q score.
- `Core/score_sp_tc.m`: reference-based SP and TC evaluation against the
  matching bundled BALI2DNA reference alignment.

## 7. Run the complete experiment

Open MATLAB and set this package directory as the current working directory.
Run the package preflight check:

```matlab
validate_reproducibility_package
```

Then execute:

```matlab
run_reproduce_all
```

The entry point:

1. validates all six input/reference pairs;
2. runs all 11 optimizers on all six datasets for 30 independent runs;
3. saves the final Viterbi-decoded alignment for each run;
4. writes per-run MAT files, raw CSV files, descriptive mean/standard-deviation
   summaries, and convergence histories.

Outputs are written under:

```text
Results/MSA_11_Algorithms_30runs_10050FEs/
```

For each dataset and optimizer, the main outputs include:

- `run_XX.mat`: run-level results, including SPS, SP, TC, population sizes,
  configured evaluation budget, actual evaluations consumed, seed, and time.
- `RawResults.csv`: run-level numerical results.
- `Summary.csv`: descriptive statistics for that optimizer's runs.
- `Convergence.csv`: mean and median best-so-far histories on the FE axis.
- `PaperTable_MeanStd.csv`: compact mean ± standard-deviation values for
  SPS, SP, and TC across the compared optimizers.

The dataset-level files include `RawResults_AllAlgorithms.csv` and
`Summary_AllAlgorithms.csv`. The experiment-level
`Optimizer_Budget_Specification.csv` records the population sizes and
evaluation budget used for each optimizer.

## 8. Reproducibility settings

- Runs per optimizer and dataset: 30.
- Common total FE budget: 10,050 per run, initialization included.
- Convergence-history samples: 200, indexed by FEs.
- Base random seed: `20260827`.
- Per-run seed:
  `baseSeed + (datasetIndex-1)*100000 + (algorithmIndex-1)*1000 + runIndex`.

Keep the input files, reference files, optimizer sources, and configuration
unchanged when reproducing a result. The package starts without a `Results/`
directory. If an existing `Results/` directory is reused after changing the
configuration or source code, remove the old experiment output first or set
`cfg.overwrite = true` and `cfg.resume = false` in `config_msa.m`.

## 9. Software requirements

The scripts are written in MATLAB. The FASTA reader, identity-based
progressive pre-alignment, Profile-HMM construction, Viterbi decoder, SPS
objective, and SP/TC scoring routines are included in this package. No
external MSA program is invoked by the experiment pipeline.
