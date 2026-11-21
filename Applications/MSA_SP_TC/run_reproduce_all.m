% RUN_REPRODUCE_ALL
% Run the six-dataset, 11-optimizer Profile-HMM MSA experiment.
% Each optimizer receives exactly cfg.MaxFEs objective evaluations per run.
clc;
close all;

rootDir = fileparts(mfilename('fullpath'));
addpath(rootDir);
addpath(fullfile(rootDir,'Core'));
addpath(fullfile(rootDir,'Algorithms'));

cfg = config_msa();
validate_reproducibility_package();

fprintf('\n=== Running all 11 optimizers with a common FE budget ===\n');
run_msa_experiment(cfg);

fprintf('\nAll optimization runs and per-run SPS/SP/TC evaluations completed.\n');
