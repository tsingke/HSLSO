# HSLSO

## Heterogeneous Selection Learning Swarm Optimization

**Title**: Heterogeneous selection learning swarm optimization for large-scale global optimization and biological multiple sequence alignment

```
Authors: Qingke Zhang*, Guanghui Zhou, Xingchen Dong, Kaitong Fu, Xiaolin Wang, Sichen Tao, Huaxiang Zhang
```

> 1. School of Computer Science and Artificial Intelligence, Shandong Normal University, Jinan 250358, China
> 2. Cyberscience Center, Tohoku University, Sendai-shi 980-8578, Japan

> Corresponding Author: **Qingke Zhang**, Email: tsingke@sdnu.edu.cn, Tel: +86-13953128163

---

## 1. Introduction

Large-scale global optimization (LSGO) is difficult due to high dimensionality, complex search spaces, and the risk of premature convergence in swarm methods. This paper proposes Heterogeneous Selection Learning Swarm Optimization (HSLSO), a PSO-based algorithm designed to improve global exploration while maintaining fast convergence.

HSLSO introduces:

- **(i) A hierarchical selection learning strategy** that controls the generation of learning exemplars, to preserve population diversity and raise search efficiency.
- **(ii) A heterogeneous learning mechanism** that adaptively adjusts particle update probabilities at different levels, to balance exploration and exploitation and reduce premature convergence.

Comprehensive tests on the CEC'2010 and CEC'2013 LSGO suites show that HSLSO consistently outperforms advanced PSO variants and recent cooperative coevolution methods, with gains confirmed by standard statistical tests. A real-world study on multiple sequence alignment modeled with Hidden Markov Models further indicates better solution quality than competing metaheuristics.

### Schematic diagram of HSLSO

<img width="414" height="345" alt="image" src="https://github.com/user-attachments/assets/04659f99-634e-4b52-84a2-b768e8dedc87" />

<img width="481" height="330" alt="image" src="https://github.com/user-attachments/assets/ee1508ae-32a6-480d-bdd3-6ac7c98f0ceb" />

### Pseudocode of the HSLSO optimizer

<img width="602" height="285" alt="image" src="https://github.com/user-attachments/assets/06072750-adcb-4c7e-9e24-cd6d14f2a492" />

---

## 2. Repository structure

```
HSLSO/
├── HSLSO.m                             The algorithm, standalone — parameters set inside the file
├── PlatLSGO/                           Benchmark platform — the code behind the paper's experiments
│   ├── config.m                        All experiment settings in one place
│   ├── run_demo.m                      Self-check: HSLSO on one function, one run
│   ├── run_CEC2010.m                   Full CEC2010 suite (F1–F20)
│   ├── run_CEC2013.m                   Full CEC2013 suite (F1–F15)
│   ├── main.m                          Both suites back to back
│   ├── core/                           Platform plumbing: paths, run loop, result summaries
│   ├── benchmarks/                     CEC2010 / CEC2013 objective functions, plus docs/
│   ├── input_data_10_LSGO/             CEC2010 data — loaded by relative path, keep the layout
│   ├── input_data_13_LSGO/             CEC2013 data
│   ├── Algorithms/                     22 algorithms behind one interface, plus MDG/ grouping data
│   │   ├── HSLSO/                      The proposed algorithm as the platform calls it
│   │   └── …                           the 21 comparison algorithms
│   └── results/                        Run output, one folder per suite
└── Applications/                       Two case studies
    ├── MSA/HSLSO_MSA_SPS/              Multiple sequence alignment
    └── UAV/UAV_HSLSO_Supplementary/    UAV 3-D path planning
```

- **`HSLSO.m`** (225 lines) — the algorithm standalone, runnable on your own objective. Population size and evaluation budget are set inside the file, and the numbered comment sections tie the main steps to the paper's equations.
- **`PlatLSGO/Algorithms/HSLSO/HSLSO.m`** (225 lines) — the copy the platform actually calls, written to the platform's shared interface so HSLSO can be swapped for any comparison algorithm.

The two are identical — same parameter list, same algorithm body — so either can be used as a drop-in comparison algorithm on the platform. The first is for reading, the second is for reproducing the experiments. `PlatLSGO/` and `Applications/` each have their own README with the full layout and details.

---

## 3. Quick start

**Requirements:** MATLAB (tested with R2025b). No toolbox is needed for the LSGO benchmark or the UAV application; the MSA application needs the Bioinformatics Toolbox.

The fastest way to confirm a working setup is the platform self-check, which runs HSLSO once on a single benchmark function:

```matlab
cd PlatLSGO
run_demo
```

It prints the elapsed time of one run, which is what you need to extrapolate the cost of a full experiment. See `PlatLSGO/README.md` for the full experiment scripts.

---

## 4. Reproducing the paper's experiments

Everything lives under `PlatLSGO/`; its README has the complete documentation.

```matlab
cd PlatLSGO
run_CEC2010     % CEC2010 suite, F1–F20
run_CEC2013     % CEC2013 suite, F1–F15
main            % both suites back to back
```

CEC2010 (F1–F20) and CEC2013 (F1–F15), D = 1000, MaxFEs = 3e6 per run, 30 independent runs. A suite is a multi-week job on one machine, so run `run_demo` first to measure a single run on your own hardware.

### Comparison algorithms

HSLSO is compared against **21 algorithms**. Each is a folder under `PlatLSGO/Algorithms/`, all reached through the same interface; `PlatLSGO/Algorithms/MDG/` holds the decomposition data that `DECC_MDG` reads. Swarm sizes and references are tabulated in [`PlatLSGO/README.md`](PlatLSGO/README.md).

| | | | | | | |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| TPCSO | HCLPSO | SCLDPSO | DCSO | WGA | CSO | SLPSO |
| APSO_DEE | EAPSO | CCOS | DECC_MDG | LLSO | DLLSO | RLLPSO |
| AHLSO | PCLSO | DPCLSO | RCIPSO | DECC_DG2 | MOS | MLSHADE_SPA |

**HSLSO** is the proposed algorithm; the 21 entries above are the baselines it is compared against.

---

## 5. Applications

### 5.1 Multiple sequence alignment — `Applications/MSA/HSLSO_MSA_SPS/`

Sequence alignment scored through a profile Hidden Markov Model, optimized with HSLSO against `SLPSO`, `CSO`, `WGA`, `EAPSO`, `APSO-DEE`, `DE` and `EO`.

```matlab
cd Applications/MSA/HSLSO_MSA_SPS
run_MSA_SPS     % set dataset / algorithm / maxiter at the top of the script
```

Eight A/C/G/T-only DNA datasets are included. **Requires the MATLAB Bioinformatics Toolbox** (`fastaread`).

### 5.2 UAV 3-D path planning — `Applications/UAV/UAV_HSLSO_Supplementary/`

Terrain-aware UAV path planning with HSLSO against `SCLDPSO`, `EAPSO`, `SLPSO` and `CSO`.

```matlab
cd Applications/UAV/UAV_HSLSO_Supplementary
run_UAV_comparison
```

The terrain is generated synthetically under a fixed seed, so no external dataset is needed.

---

## 6. Citation

If you use this code, please cite the paper:

```bibtex
@article{zhanghslso,
  title   = {Heterogeneous selection learning swarm optimization for large-scale
             global optimization and biological multiple sequence alignment},
  author  = {Zhang, Qingke and Zhou, Guanghui and Dong, Xingchen and Fu, Kaitong
             and Wang, Xiaolin and Tao, Sichen and Zhang, Huaxiang},
  journal = {Swarm and Evolutionary Computation},
  note    = {under revision}
}
```

The bibliographic details will be updated once the paper is published.

---

## 7. License and acknowledgements

This project is released under the MIT License — see [`LICENSE`](LICENSE).

The 21 comparison algorithms under `PlatLSGO/Algorithms/` are the original authors' reference implementations, ported to the platform's common interface; their search logic is unchanged, and files that carry their own header and license keep it. See `PlatLSGO/README.md` §"Third-party code" for the three changes made across those files. We thank the authors for making them available.

**We would like to express our sincere gratitude to the editors and the anonymous reviewers for taking the time to review our paper.**

This work is supported by the National Natural Science Foundation of China (Grant Nos. 62006144).
