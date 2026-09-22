% RUN_DEMO  Platform self-check: one algorithm, one function, one run.
%
%   Runs HSLSO on CEC2010 F1 for a single run. Use this to confirm that the
%   benchmark data, the paths and the algorithm files all resolve on your
%   setup before launching a full experiment.
%
%   Expect minutes rather than seconds: each algorithm file fixes its own
%   budget at MaxFEs = 3e6 to match the paper protocol, so the cost of a run
%   is set by the algorithm, not by config.m. The elapsed time printed below
%   tells you what a single run costs, which is what you need to estimate a
%   full main.m run (numRuns x functions x algorithms).
%
%   Results land in results/CEC2010/HSLSO/F01/ (run_01.mat, runs.csv,
%   summary.csv). The working directory is switched to the platform root.
clear; clc;

root = fileparts(mfilename('fullpath'));
cd(root);
addpath(root);
addpath(fullfile(root,'core'));

cfg = config();
cfg.suite      = 'CEC2010';
cfg.algorithms = {'HSLSO'};
cfg.funcIds    = 1;
cfg.numRuns    = 1;

fprintf('Platform self-check: HSLSO on CEC2010 F1, 1 run.\n');
fprintf('Running from: %s\n\n', root);

t = tic;
run_suite(cfg);
elapsed = toc(t);

full = config();
runsPerSuite = numel(full.funcIds) * numel(full.algorithms) * full.numRuns;

fprintf('\nFinished in %.1f s (%.1f min) for this single run.\n', elapsed, elapsed/60);
fprintf('A run therefore costs about %.1f s on this machine. The config.m\n', elapsed);
fprintf('defaults are %d functions x %d algorithms x %d runs = %d runs per suite,\n', ...
        numel(full.funcIds), numel(full.algorithms), full.numRuns, runsPerSuite);
fprintf('which is on the order of %.0f h at that rate.\n', runsPerSuite*elapsed/3600);
fprintf('Treat that as an order of magnitude only: each algorithm fixes its own\n');
fprintf('swarm size and budget, so per-run cost varies widely between them.\n');
