function validate_reproducibility_package()
%VALIDATE_REPRODUCIBILITY_PACKAGE Check required source files and bundled data.
rootDir = fileparts(mfilename('fullpath'));
addpath(rootDir);
addpath(fullfile(rootDir,'Core'));
addpath(fullfile(rootDir,'Algorithms'));
cfg = config_msa();

required = {'run_reproduce_all.m','config_msa.m', ...
    'Core/fitness.m','Core/SPS.m','Core/Viterbi.m', ...
    'Core/score_sp_tc.m','Core/run_msa_experiment.m', ...
    'Core/summarize_dataset.m','Core/summarize_algorithm_runs.m'};

for i = 1:numel(required)
    if exist(fullfile(rootDir,required{i}),'file') ~= 2
        error('Missing required package file: %s',required{i});
    end
end

if numel(cfg.algorithms) ~= 11
    error('Expected 11 optimizers, found %d.',numel(cfg.algorithms));
end
if cfg.MaxRun ~= 30
    error('Expected 30 independent runs; config specifies %d.',cfg.MaxRun);
end
if cfg.MaxFEs ~= 10050
    error('Expected common MaxFEs=10050; config specifies %d.',cfg.MaxFEs);
end

for i = 1:numel(cfg.algorithms)
    alg = cfg.algorithms{i};
    if exist(fullfile(rootDir,'Algorithms',[alg '.m']),'file') ~= 2
        error('Missing optimizer implementation: %s.m',alg);
    end
    if cfg.populationSizes.(alg) < 1 || cfg.minimumPopulationSizes.(alg) < 1
        error('Invalid population-size specification for %s.',alg);
    end
end

validate_bali2dna_data();
fprintf('\nPackage preflight passed: %d optimizers, %d datasets, %d runs, %d FEs/run.\n', ...
    numel(cfg.algorithms),numel(cfg.datasets),cfg.MaxRun,cfg.MaxFEs);
end
