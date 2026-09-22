% HSLSO-MSA SPS comparison demo
% Select one DNA dataset and one algorithm below, then run this script.
% Requires MATLAB Bioinformatics Toolbox (fastaread).

clc;
clear;

rootDir = fileparts(mfilename('fullpath'));
addpath(fullfile(rootDir,'Algorithms'));
addpath(fullfile(rootDir,'Core'));

% ---------------- User settings ----------------
dataset = '451c_ref1';
algorithm = 'HSLSO';
maxiter = 200;
% ------------------------------------------------

dataDir = fullfile(rootDir,'Data');
validDatasets = {'1aab_ref1','1aboA_ref1','1ad2_ref1','1hfh_ref1', ...
    '1ivy_ref5','451c_ref1','arp_ref1','kinase_ref1'};
validAlgorithms = {'HSLSO','SLPSO','CSO','WGA','EAPSO','APSO_DEE','DE','EO'};

assert(ismember(dataset,validDatasets),'Unknown dataset: %s',dataset);
assert(ismember(algorithm,validAlgorithms),'Unknown algorithm: %s',algorithm);

sequence = fastaread(fullfile(dataDir,dataset));
lengthdata = length(sequence);
a = zeros(1,lengthdata);
for i = 1:lengthdata
    a(i) = length(sequence(i).Sequence);
end
maxlength = max(a);
L = ceil(maxlength + 0.2*maxlength);
dimension = 3*(3*L+1) + 4*(2*L+1);

fprintf('Dataset: %s | sequences: %d | L: %d | D: %d\n', ...
    dataset,lengthdata,L,dimension);
fprintf('Algorithm: %s | iterations: %d\n',algorithm,maxiter);

rng('shuffle');
tStart = tic;
switch algorithm
    case 'HSLSO'
        [gbestx,gbestfitness,gbesthistory] = HSLSO(sequence,a,lengthdata,L,maxiter,dimension);
    case 'SLPSO'
        [gbestx,gbestfitness,gbesthistory] = SLPSO(sequence,a,lengthdata,L,maxiter,dimension);
    case 'CSO'
        [gbestx,gbestfitness,gbesthistory] = CSO(sequence,a,lengthdata,L,maxiter,dimension);
    case 'WGA'
        [gbestx,gbestfitness,gbesthistory] = WGA(sequence,a,lengthdata,L,maxiter,dimension);
    case 'EAPSO'
        [gbestx,gbestfitness,gbesthistory] = EAPSO(sequence,a,lengthdata,L,maxiter,dimension);
    case 'APSO_DEE'
        [gbestx,gbestfitness,gbesthistory] = APSO_DEE(sequence,a,lengthdata,L,maxiter,dimension);
    case 'DE'
        [gbestx,gbestfitness,gbesthistory] = DE(sequence,a,lengthdata,L,maxiter,dimension);
    case 'EO'
        [gbestx,gbestfitness,gbesthistory] = EO(sequence,a,lengthdata,L,maxiter,dimension);
end
elapsed = toc(tStart);

fprintf('\nBest SPS fitness: %.12g\n',gbestfitness);
fprintf('Elapsed time: %.3f s\n',elapsed);
