% MAIN  Run the complete HSLSO comparison experiment (CEC2010 + CEC2013).
%
%   Runs every algorithm listed in cfg.algorithms on all functions of both
%   suites, with cfg.numRuns independent runs each. This is the full protocol
%   used in the paper and takes hours to days on a single machine -- prefer
%   run_demo.m first to confirm the platform runs on your setup.
%
%   Note: the working directory is switched to the platform root, because the
%   benchmark suites and several algorithms load their data through
%   root-relative paths.
clear; clc;

root = fileparts(mfilename('fullpath'));
cd(root);
addpath(root);
addpath(fullfile(root,'core'));

cfg = config();

cfg.suite     = 'CEC2010';
cfg.funcIds   = 1:20;
cfg.dimension = 1000;
run_suite(cfg);

cfg.suite   = 'CEC2013';
cfg.funcIds = 1:15;
run_suite(cfg);
