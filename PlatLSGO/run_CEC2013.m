% RUN_CEC2013  Run every algorithm on the CEC2013 LSGO suite (F1-F15).
%
%   Uses the settings in config.m: 1000 dimensions and cfg.numRuns runs per
%   function. The working directory is switched to the platform root.
clear; clc;

root = fileparts(mfilename('fullpath'));
cd(root);
addpath(root);
addpath(fullfile(root,'core'));

cfg = config();
cfg.suite     = 'CEC2013';
cfg.funcIds   = 1:15;
cfg.dimension = 1000;
run_suite(cfg);
