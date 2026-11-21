function run_msa_experiment(cfg)
%RUN_MSA_EXPERIMENT HMM-based BALI2DNA MSA experiment driver.
%
% Alignment stages:
%   (1) PreAlignment_*: the coarse UPGMA-guided identity protein alignment used
%       only to estimate the Profile-HMM prior.
%   (2) InitialHMM_*: alignment obtained by Viterbi decoding of the
%       UNOPTIMIZED Profile-HMM (decision vector x=0.5).
%   (3) SP/TC: alignment obtained after optimizer search of HMM parameters.
%
% DeltaSP/DeltaTC are computed relative to InitialHMM, not relative to the
% pre-alignment. This keeps the reported refinement gain inside the HMM
% formulation used by the manuscript.

addpath(cfg.rootDir);
addpath(fullfile(cfg.rootDir,'Core'));
addpath(fullfile(cfg.rootDir,'Algorithms'));
validate_cfg(cfg);
if cfg.requireReference
    setup_bali2dna(cfg);
end

expRoot=fullfile(cfg.resultsDir,cfg.experimentName);
if ~exist(expRoot,'dir'), mkdir(expRoot); end
write_config_snapshot(cfg,expRoot);
write_optimizer_budget_specification(expRoot,cfg);

for d=1:numel(cfg.datasets)
    dataset=cfg.datasets{d};
    inputFile=fullfile(cfg.inputDir,dataset);
    if ~exist(inputFile,'file'), error('Input dataset not found: %s',inputFile); end
    refFile=find_reference_file(cfg.referenceDir,dataset);
    if cfg.requireReference && isempty(refFile)
        error('Reference alignment not found for %s.',dataset);
    end

    sequence=read_fasta_records(inputFile);
    lengthdata=numel(sequence);
    a=zeros(1,lengthdata);
    for i=1:lengthdata, a(i)=length(sequence(i).Sequence); end

    usePreInit=isfield(cfg,'usePreAlignmentHMMInitialization') && cfg.usePreAlignmentHMMInitialization;
    if usePreInit
        % Reference-free preprocessing: only the input sequences are used.
        [preAligned,preInfo]=build_initial_alignment(sequence,cfg);
        hmmModel=build_seed_profile(preAligned,cfg);
        L=hmmModel.K;
        msa_seed_context('set',hmmModel);
    else
        preAligned=[]; preInfo=struct(); hmmModel=[];
        L=select_profile_length(a,cfg);
        msa_seed_context('clear');
    end
    dimension=17*L+7;

    datasetDir=fullfile(expRoot,dataset);
    if ~exist(datasetDir,'dir'), mkdir(datasetDir); end

    % -------- Pre-alignment diagnostics (not the HMM baseline) --------
    preSPS=NaN; preSP=NaN; preTC=NaN;
    preCP=NaN; preRP=NaN; preCC=NaN; preRC=NaN;
    if usePreInit
        preFile=fullfile(datasetDir,'PreAlignment_UPGMA_Identity.fasta');
        write_fasta_alignment(preFile,sequence,preAligned);
        preSPS=SPS(preAligned);
        [preSP,preTC,preCP,preRP,preCC,preRC]=score_if_available(preFile,refFile);
        save(fullfile(datasetDir,'Initial_ProfileHMM_Prior.mat'),'hmmModel','preInfo','-v7');
    end

    % -------- Actual unoptimized HMM baseline --------
    % x=0.5 causes every perturbation multiplier to equal one, so the HMM
    % parameters are exactly the empirical prior estimated above.
    neutralVector=0.5*ones(1,dimension);
    initialHMMAligned=decode_solution(sequence,a,lengthdata,L,neutralVector);
    initialHMMSPS=SPS(initialHMMAligned);
    initialHMMFile=fullfile(datasetDir,'Initial_ProfileHMM_Alignment.fasta');
    write_fasta_alignment(initialHMMFile,sequence,initialHMMAligned);
    [initialHMMSP,initialHMMTC,initialCP,initialRP,initialCC,initialRC]=score_if_available(initialHMMFile,refFile);

    write_initialization_scores(datasetDir,preSPS,preSP,preTC,preCP,preRP,preCC,preRC, ...
        initialHMMSPS,initialHMMSP,initialHMMTC,initialCP,initialRP,initialCC,initialRC,preInfo,hmmModel,usePreInit);
    write_dataset_info(datasetDir,dataset,sequence,a,L,dimension,refFile,cfg,usePreInit);

    fprintf('\n============================================================\n');
    fprintf('Dataset: %s | NSEQ=%d | K=%d | D=%d | runs=%d | MaxFEs=%d\n', ...
        dataset,lengthdata,L,dimension,cfg.MaxRun,cfg.MaxFEs);
    if usePreInit
        fprintf('Pre-alignment (HMM initialization only): SPS=%.6g | SP=%.6f | TC=%.6f\n',preSPS,preSP,preTC);
    end
    fprintf('Initial unoptimized Profile-HMM:        SPS=%.6g | SP=%.6f | TC=%.6f\n', ...
        initialHMMSPS,initialHMMSP,initialHMMTC);
    fprintf('============================================================\n');

    for ai=1:numel(cfg.algorithms)
        alg=cfg.algorithms{ai};
        algDir=fullfile(datasetDir,alg);
        if ~exist(algDir,'dir'), mkdir(algDir); end
        if cfg.saveAlignments
            alnDir=fullfile(algDir,'Alignments');
            if ~exist(alnDir,'dir'), mkdir(alnDir); end
        end
        fprintf('\n---- %s / %s ----\n',dataset,alg);

        for r=1:cfg.MaxRun
            runMat=fullfile(algDir,sprintf('run_%02d.mat',r));
            if cfg.resume && exist(runMat,'file') && ~cfg.overwrite
                fprintf('[resume] %s run %02d already exists.\n',alg,r);
                continue;
            end

            seed=cfg.baseSeed + (d-1)*100000 + (ai-1)*1000 + r;
            rng(seed,'twister');
            fprintf('%s run %02d/%02d (seed=%d)\n',alg,r,cfg.MaxRun,seed);

            try
                t0=tic;
                msa_fitness_counter('reset');
                [gbestx,optimizerFitnessReported,history]=call_optimizer(alg,sequence,a,lengthdata,L,cfg.historyPoints,dimension,cfg);
                actualFEs=msa_fitness_counter('get');
                msa_fitness_counter('off');
                elapsed=toc(t0);
                if actualFEs~=cfg.MaxFEs
                    error('%s consumed %d FEs, but the configured budget is %d.',alg,actualFEs,cfg.MaxFEs);
                end
                history=normalize_history(history,cfg.historyPoints);
                if isempty(gbestx) || numel(gbestx)~=dimension
                    error('%s returned an invalid gbest vector.',alg);
                end

                % Verify that the reported objective belongs to returned gbest.
                optimizerFitness=fitness(sequence,a,lengthdata,L,gbestx);
                fitDelta=optimizerFitness-optimizerFitnessReported;
                tol=1e-9*(1+abs(optimizerFitness));
                if ~isfinite(optimizerFitnessReported) || abs(fitDelta)>tol
                    warning('%s returned fitness %.12g but gbest decodes to %.12g (delta %.3g).', ...
                        alg,optimizerFitnessReported,optimizerFitness,fitDelta);
                end

                finalAligned=decode_solution(sequence,a,lengthdata,L,gbestx);
                finalSPS=SPS(finalAligned);
                alnFile=fullfile(algDir,'Alignments',sprintf('run_%02d.fasta',r));
                if cfg.saveAlignments
                    write_fasta_alignment(alnFile,sequence,finalAligned);
                else
                    alnFile=fullfile(tempdir,sprintf('platmsa_%s_%s_%02d.fasta',dataset,alg,r));
                    write_fasta_alignment(alnFile,sequence,finalAligned);
                end
                [SP,TC,CorrectPairs,ReferencePairs,CorrectColumns,ReferenceColumns]=score_if_available(alnFile,refFile);

                result=struct();
                result.Dataset=dataset;
                result.Algorithm=alg;
                result.Run=r;
                result.DecoderVersion='upgma_identity_prealignment_global_profile_hmm_v7';
                result.ObjectiveVersion='affine_sum_of_pairs_sps_v4';
                result.Seed=seed;
                result.PopulationSizeInitial=optimizer_population_size(alg,cfg);
                result.PopulationSizeMinimum=optimizer_min_population_size(alg,cfg);
                result.MaxFEs=cfg.MaxFEs;
                result.FitnessEvaluations=actualFEs;
                result.ActualFEs=actualFEs;
                result.InitialEvaluationsIncluded=true;
                result.EvaluationBudgetPolicy='common exact FE budget; initialization included';
                result.BudgetUtilization=actualFEs/cfg.MaxFEs;
                result.HistoryAxis='fitness_evaluations';
                result.OptimizerFitnessReported=optimizerFitnessReported;
                result.OptimizerFitness=optimizerFitness;
                result.OptimizerFitnessDelta=fitDelta;
                result.DecodedSPS=finalSPS;

                % Keep pre-alignment and HMM baseline explicitly separate.
                result.PreAlignmentSPS=preSPS;
                result.PreAlignmentSP=preSP;
                result.PreAlignmentTC=preTC;
                result.InitialHMMSPS=initialHMMSPS;
                result.InitialHMMSP=initialHMMSP;
                result.InitialHMMTC=initialHMMTC;
                % Backward-compatible aliases now mean the initial HMM.
                result.InitialSPS=initialHMMSPS;
                result.InitialSP=initialHMMSP;
                result.InitialTC=initialHMMTC;

                result.SP=SP;
                result.TC=TC;
                result.DeltaSPFromInitialHMM=SP-initialHMMSP;
                result.DeltaTCFromInitialHMM=TC-initialHMMTC;
                result.DeltaSPFromPreAlignment=SP-preSP;
                result.DeltaTCFromPreAlignment=TC-preTC;
                result.TimeSeconds=elapsed;
                result.CorrectPairs=CorrectPairs;
                result.ReferencePairs=ReferencePairs;
                result.CorrectColumns=CorrectColumns;
                result.ReferenceColumns=ReferenceColumns;
                result.History=history;
                if cfg.saveBestVectors, result.GbestX=gbestx; end
                save(runMat,'result','-v7');

                fprintf('  SPS=%.6g | SP=%.6f (dSP_HMM=%+.6f) | TC=%.6f (dTC_HMM=%+.6f) | FEs=%d | %.2fs\n', ...
                    finalSPS,SP,SP-initialHMMSP,TC,TC-initialHMMTC,actualFEs,elapsed);
            catch ME
                msa_fitness_counter('off');
                fprintf(2,'ERROR: %s / run %d: %s\n',alg,r,ME.message);
                errFile=fullfile(algDir,sprintf('run_%02d_ERROR.txt',r));
                fid=fopen(errFile,'w');
                if fid>=0
                    fprintf(fid,'%s\n',getReport(ME,'extended','hyperlinks','off'));
                    fclose(fid);
                end
                if cfg.stopOnError, rethrow(ME); end
            end
        end

        summarize_algorithm_runs(datasetDir,alg,cfg);
    end
    summarize_dataset(datasetDir,cfg);
end

msa_seed_context('clear');
fprintf('\nCompleted. Results: %s\n',expRoot);
end

function [sp,tc,cp,rp,cc,rc]=score_if_available(alnFile,refFile)
if ~isempty(refFile)
    q=score_sp_tc(alnFile,refFile);
    sp=q.SP; tc=q.TC;
    cp=q.CorrectPairs; rp=q.ReferencePairs;
    cc=q.CorrectColumns; rc=q.ReferenceColumns;
else
    sp=NaN; tc=NaN; cp=NaN; rp=NaN; cc=NaN; rc=NaN;
end
end

function [x,f,h]=call_optimizer(name,sequence,a,lengthdata,L,historyPoints,dimension,cfg)
if exist(name,'file')~=2, error('Algorithm file not found: %s.m',name); end
if is_level_baseline(name)
    [x,f,h]=feval(name,sequence,a,lengthdata,L,historyPoints,dimension,cfg);
else
    [x,f,h]=feval(name,sequence,a,lengthdata,L,historyPoints,dimension,cfg.MaxFEs);
end
x=x(:)';
end

function n=optimizer_population_size(name,cfg)
n=cfg.populationSizes.(name);
end

function n=optimizer_min_population_size(name,cfg)
n=cfg.minimumPopulationSizes.(name);
end

function tf=is_level_baseline(name)
tf=any(strcmp(name,{'LLSO','DLLSO','RLLPSO'}));
end

function validate_cfg(cfg)
req={'MaxRun','maxiter','historyPoints','MaxFEs','baseSeed','algorithms','datasets','rootDir','inputDir','referenceDir','resultsDir','populationSizes','minimumPopulationSizes'};
for i=1:numel(req)
    if ~isfield(cfg,req{i}), error('Missing cfg.%s',req{i}); end
end
if cfg.MaxRun<1 || cfg.historyPoints<1 || cfg.MaxFEs<1, error('MaxRun/historyPoints/MaxFEs must be positive.'); end
end

function write_config_snapshot(cfg,expRoot)
file=fullfile(expRoot,'Experiment_Config.txt');
fid=fopen(file,'w'); if fid<0, return; end
cleanup=onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid,'ExperimentName: %s\n',cfg.experimentName);
fprintf(fid,'MaxRun: %d\nHistoryPoints: %d\nMaxFEs: %d\nbaseSeed: %d\n', ...
    cfg.MaxRun,cfg.historyPoints,cfg.MaxFEs,cfg.baseSeed);
fprintf(fid,'Algorithms: %s\n',strjoin(cfg.algorithms,', '));
fprintf(fid,'BudgetPolicy: exact common total fitness evaluations, initialization included\n');
for ai=1:numel(cfg.algorithms)
    alg=cfg.algorithms{ai};
    fprintf(fid,'Population_%s: initial=%d, minimum=%d\n',alg, ...
        cfg.populationSizes.(alg),cfg.minimumPopulationSizes.(alg));
end
fprintf(fid,'Datasets: %s\n',strjoin(cfg.datasets,', '));
fprintf(fid,'UsePreAlignmentHMMInitialization: %d\n',cfg.usePreAlignmentHMMInitialization);
fprintf(fid,'PreAlignmentProteinMatch: %.6g\n',cfg.preAlignmentProteinMatch);
fprintf(fid,'PreAlignmentProteinMismatch: %.6g\n',cfg.preAlignmentProteinMismatch);
fprintf(fid,'PreAlignmentProteinGapOpen: %.6g\n',cfg.preAlignmentProteinGapOpen);
fprintf(fid,'PreAlignmentProteinGapExtend: %.6g\n',cfg.preAlignmentProteinGapExtend);
fprintf(fid,'HMMMatchOccupancy: %.6g\n',cfg.hmmMatchOccupancy);
fprintf(fid,'HMMTransitionPseudoCount: %.6g\n',cfg.hmmTransitionPseudoCount);
fprintf(fid,'HMMEmissionPseudoCount: %.6g\n',cfg.hmmEmissionPseudoCount);
fprintf(fid,'HMMPerturbationTemperature: %.6g\n',cfg.hmmPerturbationTemperature);
fprintf(fid,'Reference source: %s\n',cfg.benchmarkURL);
fprintf(fid,'Created: %s\n',datestr(now,31));
end

function write_optimizer_budget_specification(expRoot,cfg)
Algorithm=cfg.algorithms(:);
InitialPopulationSize=zeros(numel(Algorithm),1);
MinimumPopulationSize=zeros(numel(Algorithm),1);
PopulationPolicy=cell(numel(Algorithm),1);
MaxFEs=repmat(cfg.MaxFEs,numel(Algorithm),1);
InitializationIncluded=true(numel(Algorithm),1);
for i=1:numel(Algorithm)
    alg=Algorithm{i};
    InitialPopulationSize(i)=cfg.populationSizes.(alg);
    MinimumPopulationSize(i)=cfg.minimumPopulationSizes.(alg);
    if InitialPopulationSize(i)==MinimumPopulationSize(i)
        PopulationPolicy{i}='fixed';
    else
        PopulationPolicy{i}='decreases with consumed FE ratio';
    end
end
T=table(Algorithm,InitialPopulationSize,MinimumPopulationSize,PopulationPolicy, ...
    MaxFEs,InitializationIncluded);
writetable(T,fullfile(expRoot,'Optimizer_Budget_Specification.csv'));
end

function write_initialization_scores(dirp,preSPS,preSP,preTC,preCP,preRP,preCC,preRC, ...
    hmmSPS,hmmSP,hmmTC,hmmCP,hmmRP,hmmCC,hmmRC,preInfo,hmmModel,usePreInit)

Stage={'PreAlignment';'InitialProfileHMM'};
SPSValue=[preSPS;hmmSPS]; SP=[preSP;hmmSP]; TC=[preTC;hmmTC];
CorrectPairs=[preCP;hmmCP]; ReferencePairs=[preRP;hmmRP];
CorrectColumns=[preCC;hmmCC]; ReferenceColumns=[preRC;hmmRC];
T=table(Stage,SPSValue,SP,TC,CorrectPairs,ReferencePairs,CorrectColumns,ReferenceColumns);
writetable(T,fullfile(dirp,'Initialization_Scores.csv'));

fid=fopen(fullfile(dirp,'HMM_Initialization_Info.txt'),'w'); if fid<0, return; end
cleanup=onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid,'MSA model: global nucleotide Profile-HMM\n');
fprintf(fid,'Pre-alignment role: HMM parameter initialization only\n');
fprintf(fid,'Final alignment source: Viterbi decoding of optimized Profile-HMM\n');
fprintf(fid,'Reference use: SP/TC evaluation only\n');
if usePreInit
    fprintf(fid,'PreAlignmentMethod: %s\n',preInfo.Method);
    fprintf(fid,'GuideTree: %s\n',preInfo.GuideTree);
    fprintf(fid,'ProgressiveMerge: %s\n',preInfo.ProgressiveMerge);
    fprintf(fid,'ProteinMatch: %.6g\nProteinMismatch: %.6g\n',preInfo.ProteinMatch,preInfo.ProteinMismatch);
    fprintf(fid,'ProteinGapOpen: %.6g\nProteinGapExtend: %.6g\n',preInfo.ProteinGapOpen,preInfo.ProteinGapExtend);
    fprintf(fid,'ProteinAlignmentWidth: %d\nNucleotideAlignmentWidth: %d\n',preInfo.ProteinAlignmentWidth,preInfo.NucleotideAlignmentWidth);
    fprintf(fid,'ProfileMatchStatesK: %d\nMatchOccupancyThreshold: %.6g\n',hmmModel.K,hmmModel.MatchOccupancyThreshold);
    fprintf(fid,'PerturbationTemperature: %.6g\n',hmmModel.PerturbationTemperature);
end
fprintf(fid,'PreAlignmentSPS: %.12g\nPreAlignmentSP: %.12g\nPreAlignmentTC: %.12g\n',preSPS,preSP,preTC);
fprintf(fid,'InitialHMMSPS: %.12g\nInitialHMMSP: %.12g\nInitialHMMTC: %.12g\n',hmmSPS,hmmSP,hmmTC);
end

function write_dataset_info(dirp,dataset,sequence,a,L,D,refFile,cfg,usePreInit)
fid=fopen(fullfile(dirp,'Dataset_Info.txt'),'w'); if fid<0, return; end
cleanup=onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid,'Dataset: %s\n',dataset);
fprintf(fid,'Data representation: BALI2DNA nucleotide (A/C/G/T)\n');
fprintf(fid,'NSEQ: %d\n',numel(sequence));
fprintf(fid,'Lengths: %s\n',mat2str(a));
if usePreInit
    fprintf(fid,'HMM initialization: protein identity distances -> UPGMA guide tree -> progressive pre-alignment -> codon projection -> Profile-HMM parameter estimation\n');
else
    fprintf(fid,'HMM initialization: compatibility fallback (%s profile length)\n',cfg.profileLengthMode);
end
fprintf(fid,'Final MSA: Viterbi decoding of optimized global Profile-HMM\n');
fprintf(fid,'K (Profile-HMM Match states): %d\n',L);
fprintf(fid,'D = 17K+7 = %d\n',D);
fprintf(fid,'MaxRun: %d\nHistoryPoints: %d\nMaxFEsPerOptimizerRun: %d\n',cfg.MaxRun,cfg.historyPoints,cfg.MaxFEs);
fprintf(fid,'Reference: %s\n',refFile);
fprintf(fid,'Reference usage: evaluation only (SP/TC)\n');
fprintf(fid,'DecoderVersion: upgma_identity_prealignment_global_profile_hmm_v7\n');
fprintf(fid,'ObjectiveVersion: affine_sum_of_pairs_sps_v4\n');
end
