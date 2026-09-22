# PlatLSGO — Large-Scale Global Optimization Benchmark Platform

A MATLAB platform for running HSLSO and 21 comparison algorithms on the
CEC2010 and CEC2013 Large-Scale Global Optimization (LSGO) benchmark suites,
with `D = 1000` and `MaxFEs = 3e6` under the protocol used in the paper.

The platform is the code that produced the LSGO comparison tables of:

> **Heterogeneous selection learning swarm optimization for large-scale global
> optimization and biological multiple sequence alignment**
> Qingke Zhang\*, Guanghui Zhou, Xingchen Dong, Kaitong Fu, Xiaolin Wang, Sichen Tao, Huaxiang Zhang
> *Swarm and Evolutionary Computation* (under revision)

---

## Requirements

- **MATLAB** — tested with R2025b. No toolbox is required for either suite.
- MATLAB is *not* required to be a specific version, but algorithms that use the
  legacy `rand('seed',...)` syntax (e.g. `SLPSO`, `PCLSO`) will emit a deprecation
  warning on recent releases. This is harmless.

---

## Quick start

Open MATLAB, `cd` into this folder (`PlatLSGO/`), and run one of the following.

| Script | What it does | Rough cost |
|---|---|---|
| `run_demo.m` | Self-check: HSLSO on CEC2010 F1, **1 run**. Confirms data, paths and algorithms all resolve, and prints what a single run costs on your machine. | ~4 min (measured) |
| `run_CEC2010.m` | Full CEC2010 suite (F1–F20), default settings | days–weeks |
| `run_CEC2013.m` | Full CEC2013 suite (F1–F15), default settings | days–weeks |
| `main.m` | Both suites back to back — the complete paper experiment | — |

**Always run `run_demo.m` first.** It is the intended way to verify that a new
machine is set up correctly before committing to a long experiment.

Every entry script switches the working directory to the platform root itself,
so you may launch them from anywhere:

```matlab
run('/path/to/PlatLSGO/run_demo.m')
```

### Cost warning

A single run is priced by the algorithm, not by `config.m`: each algorithm file
fixes its own budget at `MaxFEs = 3e6` evaluations of a 1000-dimensional
objective. `run_demo.m` runs one of them and reports the elapsed time, which is
what you need to extrapolate: `numRuns × numFunctions × numAlgorithms`.

For scale, HSLSO on CEC2010 F1 took **≈238 s (4 min)** for one run on the machine
this was prepared on. The `config.m` defaults are `30 runs × 20 functions ×
22 algorithms = 13,200 runs` for CEC2010 — on the order of **weeks** of
single-machine wall-clock at that rate, before counting CEC2013. Plan
accordingly, and start with a reduced `cfg.algorithms` / `cfg.funcIds` if you
are exploring rather than reproducing.

Per-run cost varies widely between algorithms (swarm sizes range from 50 to 1000,
and some do extra work such as the DG2 grouping), so treat any single-run
measurement as an order of magnitude, not a schedule.

---

## Configuration

All experiments are driven by `config.m`:

```matlab
function cfg = config()
cfg.suite     = 'CEC2010';      % 'CEC2010' or 'CEC2013'
cfg.funcIds   = 1:20;           % which functions
cfg.dimension = 1000;           % base dimension
cfg.numRuns   = 30;             % independent runs
cfg.baseSeed  = 1;              % seeds are derived deterministically from this
cfg.algorithms = {...};         % 22 algorithms, see below
cfg.resultDir = fullfile(pwd,'results');
end
```

Seeding is deterministic and reproducible: run *r* of algorithm *a* on function
*f* uses

```matlab
rng(cfg.baseSeed + 100000*indexOf(a) + 1000*f + r, 'twister')
```

so re-running with the same `baseSeed` reproduces the same numbers regardless of
execution order.

> **Note on `maxiter`.** The platform passes `maxiter = ceil(3e6/popsize)` to each
> algorithm, but **no** algorithm sizes its budget from it: every file declares its
> own evaluation budget internally and ignores the passed value. `MLSHADE_SPA.m`
> says so outright (`maxiter = maxiter; %#ok<NASGU>`), and `RCIPSO.m` carries a
> comment that `maxiter` is ignorable because the algorithm is driven by `MaxFEs`.
> `config.m` therefore has no `maxFEs` knob: the evaluation budget is a property of
> each algorithm file, matching the paper protocol.

---

## Repository layout

```
PlatLSGO/
├── main.m                  Run both suites (CEC2010 + CEC2013)
├── run_CEC2010.m           Run the CEC2010 suite only
├── run_CEC2013.m           Run the CEC2013 suite only
├── run_demo.m              One algorithm / one function / one run — environment self-check
├── config.m                Suite, function list, runs, seeds, algorithm list
│
├── core/                   Platform support functions (not algorithms)
│   ├── platform_root.m       Resolve the platform root from any location
│   ├── setup_paths.m         Put the folders a suite needs on the MATLAB path
│   ├── run_suite.m           Loop over algorithms × functions × runs, write CSVs
│   ├── run_single.m          One independent run: ranges, seeding, timing
│   ├── run_algorithm.m       Dispatch to an algorithm through the common interface
│   ├── run_ccos.m            CCOS wrapper (handles its suite-switched decomposition file)
│   ├── get_benchmark.m       Function handle for a suite
│   ├── get_ranges.m          Search range and dimension per function
│   ├── get_population.m      Swarm size per algorithm (paper settings)
│   ├── check_mdg_data.m      Stage the DECC-MDG decomposition file for the active suite
│   ├── build_suite_summary.m Collate per-function summaries into one CSV
│   └── summary_stats.m       Mean/std/median/best/worst of a runs CSV
│
├── benchmarks/
│   ├── CEC2010/
│   │   ├── benchmark_func_2010_LSGO.m    Objective function file (F1–F20)
│   │   └── docs/                         Official CEC2010 LSGO definition + reference code
│   └── CEC2013/
│       ├── benchmark_func_2013_LSGO.m    Objective function file (F1–F15)
│       └── docs/                         Official CEC2013 LSGO tech report + C++ reference
│
├── Algorithms/             22 comparison algorithms + supporting decomposition data
│   ├── HSLSO/              HSLSO (the proposed algorithm) — platform experiment version
│   ├── TPCSO/, HCLPSO/, SCLDPSO/, DCSO/, WGA/, CSO/, SLPSO/, APSO_DEE/, EAPSO/,
│   │   LLSO/, DLLSO/, RLLPSO/, AHLSO/, PCLSO/, DPCLSO/, RCIPSO/, MOS/,
│   │   MLSHADE_SPA/, CCOS/, DECC_MDG/, DECC_DG2/
│   └── MDG/DECC-MDG/       Precomputed DECC-MDG groupings + upstream MDG sources
│                           (data dependency of Algorithms/DECC_MDG)
│
├── input_data_10_LSGO/     CEC2010 shift/rotation data (f01_o.mat … f20_o.mat)
├── input_data_13_LSGO/     CEC2013 shift/rotation data
├── results/                Experiment output (created on first run)
└── .gitignore
```

### Why the data folders sit at the root

The benchmark files and three of the algorithms (`CCOS`, `DECC_MDG`, `DECC_DG2`)
load their data through **CWD-relative** paths, baked into the original sources:

```matlab
% benchmarks/CEC2010/benchmark_func_2010_LSGO.m
load 'input_data_10_LSGO/f01_o.mat'
% Algorithms/CCOS/CCOS.m
load './Algorithms/CCOS/CEC2013/rdg/results/F01'
```

The relative positions of `input_data_10_LSGO/`, `input_data_13_LSGO/`,
`Algorithms/CCOS/*/rdg/results/` and
`Algorithms/MDG/DECC-MDG/MergedDifferentialGrouping/results20{10,13}/` are
therefore **frozen**. The entry scripts handle this for you by `cd`-ing to the
platform root; if you call the platform functions by hand, you must do the same.

---

## Algorithms

22 algorithms, all reached through one interface. Parameters an algorithm does
not need are declared `~` in its own signature, never passed as placeholders:

```matlab
[gbestx, bestever, gbesthistory] =
    Algorithm(popsize, dimension, xmax, xmin, vmax, vmin, ...
              maxiter, fCalculation, FuncId)
```

Grouped by family, so the comparison is legible at a glance — the baseline swarm
methods in the middle of the table are all the same lineage as HSLSO, and the
cooperative-coevolution block at the bottom is the other main line of attack on
LSGO:

| Family | Algorithm | Swarm size | Reference |
|---|---|---|---|
| **Proposed** | **HSLSO** | 400 | This paper |
| **PSO variants** | HCLPSO | 500 | Heterogeneous comprehensive learning PSO |
| | SCLDPSO | 400 | Advantage-combined learning distributed PSO (2023) |
| | SLPSO | 200 | Social learning PSO (swarm size self-adapts internally) |
| | APSO_DEE | 1000 ¹ | Adaptive PSO with decoupled exploration/exploitation |
| | EAPSO | 100 | Enhanced adaptive PSO |
| | PCLSO | 200 | PSO with a population/component learning variant |
| | DPCLSO | 600 | Distributed PCLSO |
| | RCIPSO | 800 ¹ | Rank/contribution-informed PSO |
| **CSO variants** | CSO | 500 | Competitive swarm optimizer |
| | TPCSO | 500 | Two-stage/three-population CSO variant |
| | DCSO | 500 | CSO with a diversity/`varphi` variant |
| **Level-based learning swarm** (LLSO family) | LLSO | 500 | Yang et al., level-based learning swarm optimizer, IEEE TEC 2018 |
| | DLLSO | 500 | Dynamic LLSO |
| | RLLPSO | 500 | Reinforcement-learning level-based PSO |
| | AHLSO | 500 | Agent-assisted heterogeneous learning swarm optimizer (2024) |
| **Cooperative coevolution** | CCOS | 100 | Sun et al., contribution-based CC framework with optimizer selection |
| | DECC_MDG | 50 | Omidvar et al., CC with differential grouping (MDG) |
| | DECC_DG2 | 50 | Omidvar et al., DG2: differential grouping 2, IEEE TEC 2017 |
| **Memetic / hybrid / other** | WGA | 120 | Ghasemi et al., *Array* 11 (2021) 100074 |
| | MOS | 400 | LaTorre et al., MOS-based hybrid algorithms, IEEE CEC 2013 |
| | MLSHADE_SPA | 250 | Hadi et al., LSHADE-SPA memetic framework, *Complex & Intelligent Systems* 5 (2019) 25–40 |

Each algorithm lives in `Algorithms/<Algorithm>/` and is dispatched by name in
`core/run_algorithm.m`. The entry file is `<Algorithm>.m` for all but two —
`Algorithms/DCSO/D_CSO.m` and `Algorithms/AHLSO/zAHLSO.m` — which
`run_algorithm.m` knows about. `Algorithms/MDG/` is not an algorithm: it holds
the decomposition data that `DECC_MDG` reads at run time.

¹ The swarm-size column lists the value each algorithm **actually runs with**,
which for most of them is the value `core/get_population.m` passes in. Three
algorithm files hardcode their own swarm size and silently override the passed
value: `APSO_DEE.m` sets `Npop = 1000` where the platform passes 100, `RCIPSO.m`
sets `popsize = 800` where the platform passes 900, and `SCLDPSO.m` sets
`popsize = 400` where the platform passes 500. The sources are reproduced as
published, so the override stands; the table reports the effective size. If you
change `get_population.m` expecting it to take effect for these three, it will
not. (Several other files restate their swarm size too, but to the same value the
platform passes, so nothing changes.)

### Decomposition-based algorithms

`CCOS`, `DECC_MDG` and `DECC_DG2` need a variable-grouping result before they can
run. The platform stages these automatically:

- **CCOS** ships precomputed RDG decompositions under
  `Algorithms/CCOS/CEC20{10,13}/rdg/results/F*.mat`. `CCOS.m` carries **two**
  hardcoded, suite-agnostic paths: the main body loads the *CEC2013* tree
  (`CCOS.m:13`) and its local `grouping()` loads the *CEC2010* tree (`CCOS.m:94`).
  `core/run_ccos.m` therefore copies the active suite's file over the *other*
  tree's path, runs, and restores the original afterwards.
- **DECC_MDG** ships precomputed merges under
  `Algorithms/MDG/DECC-MDG/MergedDifferentialGrouping/results20{10,13}/`.
  `Algorithms/DECC_MDG/DECC_MDG.m` contains a path that is hardcoded to the
  *CEC2013* tree, so `core/check_mdg_data.m` stages the active suite's file into
  place, runs, and restores the original afterwards — the same stage-and-restore
  pattern `core/run_ccos.m` uses. Two consequences are worth knowing:

  - The `DECC-MDG/MergedDifferentialGrouping/` nesting is **not** decorative. It
    is the literal relative path inside the algorithm file, so the data cannot be
    moved up a level without editing the algorithm itself.
  - `DECC_MDG.m` is **self-contained**: `diff_grouping` and `sansde` are defined
    inside it as local functions, so no companion `.m` file is needed at run
    time — only the grouping data. The folder additionally keeps the upstream MDG
    implementation (`MDG.m`, `mergeGroup.m`, `biSearch.m`,
    `mergeInteractionGroup.m`, `analyze20{10,13}.m`) for reference. The standalone
    driver scripts that originally *generated* the groupings are not
    redistributed, because they depend on the MDG authors' own copies of the CEC
    benchmark and trace files; the generated groupings themselves are included,
    which is what the platform actually needs.
- **DECC_DG2** performs the DG2 grouping on first use and caches it under
  `Algorithms/DECC_DG2/DG2_results/`. The grouping cost is counted inside the
  3e6 budget on every run, including cache hits, matching the paper protocol.

Decomposition data is **not** regenerated and must not be deleted — copying these
files is the only reason a fresh checkout can reproduce the results.

---

## Output

Results are written under `results/<SUITE>/<ALGORITHM>/F<nn>/`:

```
results/CEC2010/HSLSO/F01/
├── run_01.mat … run_30.mat   Full per-run record: algorithm, funcId, runId,
│                             dimension, bestFitness, bestX, bestHistory, runtime
├── runs.csv                  One row per run: Run, Dimension, BestFitness,
│                             Runtime, Status
└── summary.csv               Aggregates over successful runs: Function, Dimension,
                              Runs, Mean, Std, Median, Best, Worst, MeanRuntime
```

plus two roll-ups written per suite:

- `results/<SUITE>/summary_all.csv` — every algorithm × function row in one table
- `results/<SUITE>/errors.csv` — only if some runs failed; one row per failure with
  the MATLAB error message. A failed run is recorded as `NaN`/`ERROR` in
  `runs.csv` rather than aborting the sweep, so a long experiment always finishes.

`results/` is regenerated by the run scripts and is git-ignored. Only
`results/.gitkeep` is tracked.

---

## Notes and caveats

1. **Working directory.** The platform must run with the `PlatLSGO/` root as CWD
   (see *Why the data folders sit at the root*). The supplied entry scripts do
   this automatically.
2. **Stray `results_*.csv`.** `benchmarks/CEC2013/benchmark_func_2013_LSGO.m`
   unconditionally calls its progress logger, which writes `results_<n>.csv` into
   the current directory during CEC2013 runs. These files are git-ignored and can
   be deleted at any time.
3. **`DECC_DG2` runtime cache.** `Algorithms/DECC_DG2/DG2_results/` is a live
   cache, not source. Deleting it only costs a re-run of the DG2 grouping; it is
   git-ignored.
4. **Absolute paths.** No absolute path appears anywhere in the platform — it is
   portable across machines and operating systems.
5. **Reproducibility.** Runs use MATLAB's `twister` generator with a fixed derived
   seed, so the same `baseSeed`, algorithm and function reproduce the same run.
   Note that `SLPSO` and `PCLSO` additionally call `rand('seed', sum(100*clock))`
   in their sources, which re-seeds the legacy generator from the wall clock and
   overrides the platform seed — a property of the original implementations, kept
   unchanged for fidelity.
6. **Encoding.** Every text file here is UTF-8, so it renders correctly on GitHub.
   A few files inherited from the official CEC distribution were originally GBK
   and have been transcoded; see *Third-party code* below for exactly what that
   did and did not change.

---

## Third-party code

The comparison algorithms are the authors' own reference implementations,
platform-adapted to the common interface above. Their search logic is reproduced
as published. Where a source file carries its own header and license notice, that
notice governs the file and has been preserved.

Three changes were made across the algorithm files. None of them touches the
search: no evaluation counter, fitness value, branch or loop is altered anywhere.

**One common interface.** Every algorithm now declares the same nine-argument
signature `(popsize,dimension,xmax,xmin,vmax,vmin,maxiter,fCalculation,FuncId)`.
The unused leading `mainHandle` and trailing `VisualSwitch` placeholders that some
files carried have been dropped, so `core/run_algorithm.m` reaches all of them
through a single `feval`.

**English throughout.** Comments and progress messages that were written in
Chinese are now in English. Only comment text and `fprintf` format strings
changed; each message still carries the same numbers in the same order.

**Print throttling.** In the working copies used for the paper, several
algorithms printed a line on **every** function evaluation
(`fprintf(... FE %d  best = %e ...)`) — some 3·10⁶ lines of stdout per run, which makes
the log of a full sweep unusable and slows it down. Each of those prints is now
wrapped in a `mod` guard and fires every 10% of the budget: 10 progress lines per
run. The algorithms that already shipped behind a throttle (`SLPSO`, `LLSO`,
`DLLSO`, `MOS`, `AHLSO`) are untouched. The guard wraps the `fprintf` only — no
evaluation counter, no fitness value and no branch is changed — so the search is
unaffected.

- `Algorithms/CCOS/CCOS.m` — © Yuan Sun; `sansde.m`, `slpso.m` retained as dependencies
- `Algorithms/DECC_MDG/DECC_MDG.m`, `Algorithms/MDG/DECC-MDG/MergedDifferentialGrouping/*`
  and `Algorithms/DECC_DG2/DECC_DG2.m` — © Mohammad Nabi Omidvar (GPL, as stated in
  the file headers), with SaNSDE from Yang et al., CEC 2008. `DECC_MDG.m` is
  self-contained: `diff_grouping` and `sansde` are included in it as local
  functions, so only the precomputed grouping data is needed at run time.
- `Algorithms/MOS/MOS.m` — ported from A. LaTorre, S. Muelas, J.-M. Peña, IEEE CEC 2013
- `Algorithms/MLSHADE_SPA/MLSHADE_SPA.m` — A. A. Hadi, A. W. Mohamed, K. M. Jambi,
  *Complex & Intelligent Systems* 5 (2019) 25–40
- `Algorithms/WGA/WGA.m` — Ghasemi et al., *Array* 11 (2021) 100074 (keeps its
  source's own evaluation cap)
- `Algorithms/LLSO/`, `Algorithms/DLLSO/` — Q. Yang et al., IEEE TEC 2018

Apart from the three changes listed above, no algorithm file in this release has
been modified.

Benchmark code and definition documents under `benchmarks/*/docs/` are the
official CEC2010/CEC2013 LSGO competition materials. Their content is unchanged.
A few files that the original distribution shipped in a legacy Chinese encoding
(GBK) have been re-encoded to UTF-8 so that they display correctly on GitHub;
this changes the byte encoding of comments and one apostrophe, not the code.

Algorithms without a header notice follow the citation given in the table above;
consult the file for implementation details rather than assuming a canonical
public source.

---

## License

Platform code (`main.m`, `run_*.m`, `config.m`, `core/`) is released under the MIT
License — see `../LICENSE`. Third-party algorithm files retain their own terms as
noted above.
