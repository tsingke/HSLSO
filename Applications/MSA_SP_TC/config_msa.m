function cfg = config_msa()
%CONFIG_MSA Configuration for the reproducible BALI2DNA Profile-HMM study.
% All 11 optimizers receive the same total fitness-evaluation budget.

rootDir=fileparts(mfilename('fullpath'));
cfg.MaxRun=30;
cfg.historyPoints=200;       % Number of points stored in convergence histories.
cfg.maxiter=cfg.historyPoints; % Backward-compatible alias used by summary helpers.
cfg.MaxFEs=10050;             % Includes initialization evaluations for every optimizer.
cfg.baseSeed=20260827;

cfg.originalAlgorithms={'HSLSO','SLPSO','CSO','WGA','EAPSO', ...
                        'APSO_DEE','DE','EO'};
cfg.levelAlgorithms={'LLSO','DLLSO','RLLPSO'};
cfg.allAlgorithms=[cfg.originalAlgorithms,cfg.levelAlgorithms];
cfg.algorithms=cfg.allAlgorithms;
cfg.experimentName='MSA_11_Algorithms_30runs_10050FEs';

cfg.datasets={'1hfh_ref1','451c_ref1','1ivy_ref5', ...
               'kinase_ref1','1ad2_ref1','1aab_ref1'};

% Initial population sizes. WGA reduces its active population from 120 to 30.
cfg.populationSizes=struct( ...
    'HSLSO',50,'SLPSO',50,'CSO',50,'WGA',120,'EAPSO',50, ...
    'APSO_DEE',50,'DE',50,'EO',30,'LLSO',50,'DLLSO',50,'RLLPSO',50);
cfg.minimumPopulationSizes=cfg.populationSizes;
cfg.minimumPopulationSizes.WGA=30;

% The level-based baselines use the same common FE budget.
cfg.levelBaselinePopulation=50;
cfg.levelBaselineMaxFEs=cfg.MaxFEs;

cfg.rootDir=rootDir;
cfg.inputDir=fullfile(rootDir,'Data','Input');
cfg.referenceDir=fullfile(rootDir,'Data','BALI2DNA','ref');
cfg.resultsDir=fullfile(rootDir,'Results');

cfg.usePreAlignmentHMMInitialization=true;
cfg.preAlignmentProteinMatch=1.0;
cfg.preAlignmentProteinMismatch=-1.0;
cfg.preAlignmentProteinGapOpen=-4.0;
cfg.preAlignmentProteinGapExtend=-0.5;

cfg.hmmMatchOccupancy=0.50;
cfg.hmmTransitionPseudoCount=0.5;
cfg.hmmEmissionPseudoCount=0.5;
cfg.hmmPerturbationTemperature=0.75;
cfg.profileLengthMode='median';

cfg.resume=true;
cfg.overwrite=false;
cfg.saveAlignments=true;
cfg.saveBestVectors=false;
cfg.saveConvergence=true;
cfg.stopOnError=true;

cfg.autoDownloadReference=false;
cfg.benchmarkURL='https://www.drive5.com/bench/bench.tar.gz';
cfg.requireReference=true;
cfg.verbose=true;
end
